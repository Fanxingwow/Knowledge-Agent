from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import regex  # type: ignore[import-untyped]

from app.import_process.agent.nodes.safe_node_contracts import (
    SafeNodeContractError,
    TextChunk,
    require_metadata,
    require_safe_text,
)
from app.import_process.agent.state import ImportGraphState

_GRAPHEME = regex.compile(r"\X")
_HEADING = re.compile(r"^[ ]{0,3}(#{1,6})[\t ]+(.+?)[\t ]*\n?$")
_FENCE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class MarkdownSplitPolicy:
    max_document_characters: int = 8_000_000
    max_document_bytes: int = 16_000_000
    max_chunk_characters: int = 1_600
    max_chunk_bytes: int = 6_400
    max_chunks: int = 10_000

    def __post_init__(self) -> None:
        limits = (
            self.max_document_characters,
            self.max_document_bytes,
            self.max_chunk_characters,
            self.max_chunk_bytes,
            self.max_chunks,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in limits):
            raise ValueError("invalid Markdown split policy")
        if self.max_chunk_characters > self.max_document_characters or self.max_chunk_bytes > self.max_document_bytes:
            raise ValueError("Markdown chunk limit exceeds document limit")


@dataclass(frozen=True, slots=True)
class _Section:
    title: str
    parent_title: str
    content: str


def _heading_text(raw: str) -> str:
    trimmed = re.sub(r"[\t ]+#+[\t ]*$", "", raw).strip()
    return require_metadata(
        trimmed,
        field="Markdown heading",
        allow_empty=False,
    )


def _sections(markdown: str, *, file_title: str) -> tuple[_Section, ...]:
    sections: list[_Section] = []
    heading_stack = [""] * 6
    current_title = file_title
    current_parent = ""
    buffer: list[str] = []
    fence_character = ""
    fence_length = 0

    def flush() -> None:
        content = "".join(buffer).strip()
        if content:
            sections.append(
                _Section(
                    title=current_title,
                    parent_title=current_parent,
                    content=content,
                )
            )
        buffer.clear()

    for line in markdown.splitlines(keepends=True):
        fence_match = _FENCE.match(line)
        if fence_character:
            buffer.append(line)
            if fence_match is not None:
                marker = fence_match.group(1)
                if marker[0] == fence_character and len(marker) >= fence_length:
                    fence_character = ""
                    fence_length = 0
            continue
        if fence_match is not None:
            marker = fence_match.group(1)
            fence_character = marker[0]
            fence_length = len(marker)
            buffer.append(line)
            continue

        heading_match = _HEADING.match(line)
        if heading_match is None:
            buffer.append(line)
            continue
        flush()
        level = len(heading_match.group(1))
        title = _heading_text(heading_match.group(2))
        parent = next(
            (candidate for candidate in reversed(heading_stack[: level - 1]) if candidate),
            "",
        )
        heading_stack[level - 1] = title
        for index in range(level, len(heading_stack)):
            heading_stack[index] = ""
        current_title = title
        current_parent = parent
        buffer.append(line)
    flush()
    return tuple(sections)


def _preferred_break(buffer: list[str]) -> int:
    minimum = max(1, len(buffer) // 2)
    for index in range(len(buffer), minimum - 1, -1):
        if buffer[index - 1].isspace():
            return index
    return len(buffer)


def _bounded_pieces(content: str, policy: MarkdownSplitPolicy) -> tuple[str, ...]:
    pieces: list[str] = []
    buffer: list[str] = []
    character_count = 0
    byte_count = 0

    def discard_leading_space() -> None:
        nonlocal character_count, byte_count
        while buffer and buffer[0].isspace():
            removed = buffer.pop(0)
            character_count -= len(removed)
            byte_count -= len(removed.encode("utf-8"))

    def emit_prefix(index: int) -> None:
        nonlocal character_count, byte_count
        piece = "".join(buffer[:index]).strip()
        removed = buffer[:index]
        del buffer[:index]
        character_count -= sum(len(cluster) for cluster in removed)
        byte_count -= sum(len(cluster.encode("utf-8")) for cluster in removed)
        discard_leading_space()
        if piece:
            pieces.append(piece)

    for match in _GRAPHEME.finditer(content):
        cluster = match.group(0)
        cluster_characters = len(cluster)
        cluster_bytes = len(cluster.encode("utf-8"))
        if cluster_characters > policy.max_chunk_characters or cluster_bytes > policy.max_chunk_bytes:
            raise SafeNodeContractError("Markdown grapheme exceeds chunk limit")
        while buffer and (
            character_count + cluster_characters > policy.max_chunk_characters
            or byte_count + cluster_bytes > policy.max_chunk_bytes
        ):
            emit_prefix(_preferred_break(buffer))
        buffer.append(cluster)
        character_count += cluster_characters
        byte_count += cluster_bytes
    if buffer:
        emit_prefix(len(buffer))
    return tuple(pieces)


class MarkdownSplitterNode:
    """Deterministically split normalized Markdown on Unicode grapheme boundaries."""

    def __init__(self, policy: MarkdownSplitPolicy | None = None) -> None:
        self._policy = policy or MarkdownSplitPolicy()

    def __call__(self, state: ImportGraphState) -> Mapping[str, object]:
        markdown = require_safe_text(
            state.get("md_content"),
            field="Markdown content",
            allow_empty=False,
            multiline=True,
            max_characters=self._policy.max_document_characters,
            max_bytes=self._policy.max_document_bytes,
        )
        file_title = require_metadata(
            state.get("file_title"),
            field="file title",
            allow_empty=False,
        )
        item_name = require_metadata(
            state.get("item_name", ""),
            field="item name",
        )
        chunks: list[TextChunk] = []
        for section in _sections(markdown, file_title=file_title):
            for piece in _bounded_pieces(section.content, self._policy):
                chunks.append(
                    {
                        "content": piece,
                        "title": section.title,
                        "parent_title": section.parent_title,
                        "file_title": file_title,
                        "item_name": item_name,
                    }
                )
                if len(chunks) > self._policy.max_chunks:
                    raise SafeNodeContractError("Markdown chunk limit exceeded")
        if not chunks:
            raise SafeNodeContractError("Markdown produced no chunks")
        return {"chunks": chunks}


__all__ = ["MarkdownSplitPolicy", "MarkdownSplitterNode"]
