from __future__ import annotations

import html
import re
from collections.abc import Mapping
from dataclasses import dataclass

from app.import_process.agent.nodes.safe_node_contracts import (
    SafeNodeContractError,
    require_metadata,
    require_safe_text,
)
from app.import_process.agent.state import ImportGraphState

_HTML_IMAGE = re.compile(r"<img\b[^>]*>", flags=re.IGNORECASE | re.DOTALL)
_HTML_IMAGE_START = re.compile(r"<img\b", flags=re.IGNORECASE)
_HTML_ALT = re.compile(
    r"\balt\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>[^\s>]+))",
    flags=re.IGNORECASE | re.DOTALL,
)
_REFERENCE_DEFINITION = re.compile(r"^[ ]{0,3}\[([^\]]+)]\s*:")
_MARKDOWN_ESCAPE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")


@dataclass(frozen=True, slots=True)
class MarkdownImagePolicy:
    max_document_characters: int = 8_000_000
    max_document_bytes: int = 16_000_000
    max_images: int = 2_000
    max_image_syntax_characters: int = 16_384
    max_alt_characters: int = 512
    max_alt_bytes: int = 2_048

    def __post_init__(self) -> None:
        limits = (
            self.max_document_characters,
            self.max_document_bytes,
            self.max_images,
            self.max_image_syntax_characters,
            self.max_alt_characters,
            self.max_alt_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in limits):
            raise ValueError("invalid Markdown image policy")


def _is_escaped(text: str, position: int) -> bool:
    backslashes = 0
    position -= 1
    while position >= 0 and text[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def _find_balanced_end(
    text: str,
    start: int,
    *,
    opening: str,
    closing: str,
    maximum_span: int,
) -> int | None:
    depth = 1
    limit = min(len(text), start + maximum_span + 1)
    for position in range(start, limit):
        character = text[position]
        if _is_escaped(text, position):
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return position
    return None


def _reference_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _image_label(raw_alt: str, policy: MarkdownImagePolicy) -> str:
    unescaped = _MARKDOWN_ESCAPE.sub(r"\1", html.unescape(raw_alt))
    collapsed = " ".join(unescaped.split())
    alt = require_metadata(
        collapsed,
        field="image alternative text",
        max_characters=policy.max_alt_characters,
        max_bytes=policy.max_alt_bytes,
    )
    return "[image]" if not alt else f"[image: {alt}]"


def _replace_markdown_images(
    text: str,
    policy: MarkdownImagePolicy,
) -> tuple[str, set[str], int]:
    output: list[str] = []
    references: set[str] = set()
    cursor = 0
    count = 0
    while True:
        start = text.find("![", cursor)
        if start < 0:
            output.append(text[cursor:])
            break
        if _is_escaped(text, start):
            output.append(text[cursor : start + 2])
            cursor = start + 2
            continue
        output.append(text[cursor:start])
        alt_end = _find_balanced_end(
            text,
            start + 2,
            opening="[",
            closing="]",
            maximum_span=policy.max_image_syntax_characters,
        )
        if alt_end is None:
            raise SafeNodeContractError("invalid Markdown image syntax")
        alt = text[start + 2 : alt_end]
        next_position = alt_end + 1
        if next_position < len(text) and text[next_position] == "(":
            target_end = _find_balanced_end(
                text,
                next_position + 1,
                opening="(",
                closing=")",
                maximum_span=policy.max_image_syntax_characters,
            )
            if target_end is None:
                raise SafeNodeContractError("invalid Markdown image syntax")
            cursor = target_end + 1
        elif next_position < len(text) and text[next_position] == "[":
            reference_end = _find_balanced_end(
                text,
                next_position + 1,
                opening="[",
                closing="]",
                maximum_span=policy.max_image_syntax_characters,
            )
            if reference_end is None:
                raise SafeNodeContractError("invalid Markdown image syntax")
            reference = text[next_position + 1 : reference_end] or alt
            references.add(_reference_key(reference))
            cursor = reference_end + 1
        else:
            references.add(_reference_key(alt))
            cursor = next_position
        output.append(_image_label(alt, policy))
        count += 1
        if count > policy.max_images:
            raise SafeNodeContractError("too many Markdown images")
    return "".join(output), references, count


def _replace_html_images(
    text: str,
    policy: MarkdownImagePolicy,
    prior_count: int,
) -> str:
    count = prior_count

    def replacement(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        if count > policy.max_images:
            raise SafeNodeContractError("too many Markdown images")
        tag = match.group(0)
        if len(tag) > policy.max_image_syntax_characters:
            raise SafeNodeContractError("invalid HTML image syntax")
        alt_match = _HTML_ALT.search(tag)
        if alt_match is None:
            alt = ""
        else:
            alt = next(value for value in alt_match.group("double", "single", "bare") if value is not None)
        return _image_label(alt, policy)

    replaced = _HTML_IMAGE.sub(replacement, text)
    if _HTML_IMAGE_START.search(replaced) is not None:
        raise SafeNodeContractError("invalid HTML image syntax")
    return replaced


def _remove_image_reference_definitions(text: str, references: set[str]) -> str:
    if not references:
        return text
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        match = _REFERENCE_DEFINITION.match(line)
        if match is None or _reference_key(match.group(1)) not in references:
            output.append(line)
    return "".join(output)


class MarkdownImageTextNode:
    """Reduce Markdown/HTML images to bounded alt text without doing any I/O."""

    def __init__(self, policy: MarkdownImagePolicy | None = None) -> None:
        self._policy = policy or MarkdownImagePolicy()

    def __call__(self, state: ImportGraphState) -> Mapping[str, object]:
        markdown = require_safe_text(
            state.get("md_content"),
            field="Markdown content",
            allow_empty=False,
            multiline=True,
            max_characters=self._policy.max_document_characters,
            max_bytes=self._policy.max_document_bytes,
        )
        markdown, references, image_count = _replace_markdown_images(markdown, self._policy)
        markdown = _replace_html_images(markdown, self._policy, image_count)
        markdown = _remove_image_reference_definitions(markdown, references)
        normalized = require_safe_text(
            markdown,
            field="processed Markdown content",
            allow_empty=False,
            multiline=True,
            max_characters=self._policy.max_document_characters,
            max_bytes=self._policy.max_document_bytes,
        )
        return {"md_content": normalized}


__all__ = ["MarkdownImagePolicy", "MarkdownImageTextNode"]
