from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from typing import TypedDict, cast


class SafeNodeContractError(ValueError):
    """A caller or injected dependency violated a pure-node contract."""


class TextChunk(TypedDict):
    """The only fields emitted before embedding."""

    content: str
    title: str
    parent_title: str
    file_title: str
    item_name: str


class EmbeddedChunk(TextChunk):
    """The only fields accepted by the tenant-scoped index commit node."""

    dense_vector: list[float]
    sparse_vector: dict[int, float]


TEXT_CHUNK_FIELDS = frozenset({"content", "title", "parent_title", "file_title", "item_name"})
EMBEDDED_CHUNK_FIELDS = frozenset({*TEXT_CHUNK_FIELDS, "dense_vector", "sparse_vector"})

_BIDI_CONTROL_CODEPOINTS = frozenset(
    {
        0x061C,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2066,
        0x2067,
        0x2068,
        0x2069,
    }
)


def normalize_unicode(value: str) -> str:
    """Normalize line endings and canonical Unicode without changing graphemes."""

    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def require_safe_text(
    value: object,
    *,
    field: str,
    allow_empty: bool,
    multiline: bool,
    max_characters: int,
    max_bytes: int,
) -> str:
    """Return normalized UTF-8 text or raise a value-free contract error."""

    if not isinstance(value, str):
        raise SafeNodeContractError(f"invalid {field}")
    normalized = normalize_unicode(value)
    if not normalized and not allow_empty:
        raise SafeNodeContractError(f"invalid {field}")
    if len(normalized) > max_characters:
        raise SafeNodeContractError(f"invalid {field}")
    try:
        encoded = normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise SafeNodeContractError(f"invalid {field}") from None
    if len(encoded) > max_bytes:
        raise SafeNodeContractError(f"invalid {field}")
    for character in normalized:
        codepoint = ord(character)
        category = unicodedata.category(character)
        if (
            0xD800 <= codepoint <= 0xDFFF
            or codepoint & 0xFFFF in {0xFFFE, 0xFFFF}
            or codepoint in _BIDI_CONTROL_CODEPOINTS
            or (category == "Cc" and character not in {"\n", "\t"})
            or (not multiline and character in {"\n", "\t"})
        ):
            raise SafeNodeContractError(f"invalid {field}")
    return normalized


def require_metadata(
    value: object,
    *,
    field: str,
    allow_empty: bool = True,
    max_characters: int = 4_096,
    max_bytes: int = 16_384,
) -> str:
    return require_safe_text(
        value,
        field=field,
        allow_empty=allow_empty,
        multiline=False,
        max_characters=max_characters,
        max_bytes=max_bytes,
    )


def require_text_chunk(
    value: object,
    *,
    max_content_characters: int,
    max_content_bytes: int,
) -> TextChunk:
    if not isinstance(value, Mapping) or set(value) != TEXT_CHUNK_FIELDS:
        raise SafeNodeContractError("invalid text chunk shape")
    normalized: TextChunk = {
        "content": require_safe_text(
            value.get("content"),
            field="chunk content",
            allow_empty=False,
            multiline=True,
            max_characters=max_content_characters,
            max_bytes=max_content_bytes,
        ),
        "title": require_metadata(value.get("title"), field="chunk title"),
        "parent_title": require_metadata(
            value.get("parent_title"),
            field="chunk parent title",
        ),
        "file_title": require_metadata(
            value.get("file_title"),
            field="chunk file title",
            allow_empty=False,
        ),
        "item_name": require_metadata(
            value.get("item_name"),
            field="chunk item name",
        ),
    }
    return normalized


def require_text_chunks(
    value: object,
    *,
    max_chunks: int,
    max_content_characters: int,
    max_content_bytes: int,
) -> list[TextChunk]:
    if not isinstance(value, list) or not value or len(value) > max_chunks:
        raise SafeNodeContractError("invalid text chunks")
    return [
        require_text_chunk(
            chunk,
            max_content_characters=max_content_characters,
            max_content_bytes=max_content_bytes,
        )
        for chunk in value
    ]


def copy_text_chunk(chunk: TextChunk, *, item_name: str | None = None) -> TextChunk:
    copied = dict(chunk)
    if item_name is not None:
        copied["item_name"] = item_name
    return cast(TextChunk, copied)


__all__ = [
    "EMBEDDED_CHUNK_FIELDS",
    "TEXT_CHUNK_FIELDS",
    "EmbeddedChunk",
    "SafeNodeContractError",
    "TextChunk",
    "copy_text_chunk",
    "normalize_unicode",
    "require_metadata",
    "require_safe_text",
    "require_text_chunk",
    "require_text_chunks",
]
