from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from app.import_process.agent.nodes.safe_node_contracts import (
    SafeNodeContractError,
    copy_text_chunk,
    require_metadata,
    require_text_chunks,
)
from app.import_process.agent.state import ImportGraphState


@dataclass(frozen=True, slots=True)
class ItemNameContext:
    """The bounded, content-free context exposed to an injected pure recognizer."""

    file_title: str
    titles: tuple[str, ...]
    parent_titles: tuple[str, ...]
    chunk_count: int


class PureItemNameRecognizer(Protocol):
    """A deterministic recognizer; implementations must not perform external I/O."""

    def __call__(self, context: ItemNameContext) -> str: ...


@dataclass(frozen=True, slots=True)
class ItemNamePolicy:
    max_chunks: int = 10_000
    max_chunk_characters: int = 1_600
    max_chunk_bytes: int = 6_400
    max_name_characters: int = 512
    max_name_bytes: int = 2_048

    def __post_init__(self) -> None:
        limits = (
            self.max_chunks,
            self.max_chunk_characters,
            self.max_chunk_bytes,
            self.max_name_characters,
            self.max_name_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in limits):
            raise ValueError("invalid item-name policy")


def _unique_nonempty(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


class SafeItemNameNode:
    """Derive an item name locally or call one explicitly injected pure function."""

    def __init__(
        self,
        *,
        explicit_name: str | None = None,
        recognizer: PureItemNameRecognizer | None = None,
        policy: ItemNamePolicy | None = None,
    ) -> None:
        if explicit_name is not None and recognizer is not None:
            raise ValueError("item name and recognizer are mutually exclusive")
        self._policy = policy or ItemNamePolicy()
        self._recognizer = recognizer
        self._explicit_name = (
            require_metadata(
                explicit_name,
                field="explicit item name",
                allow_empty=False,
                max_characters=self._policy.max_name_characters,
                max_bytes=self._policy.max_name_bytes,
            )
            if explicit_name is not None
            else None
        )

    def __call__(self, state: ImportGraphState) -> Mapping[str, object]:
        existing_item_name = require_metadata(
            state.get("item_name", ""),
            field="existing item name",
        )
        chunks = require_text_chunks(
            state.get("chunks"),
            max_chunks=self._policy.max_chunks,
            max_content_characters=self._policy.max_chunk_characters,
            max_content_bytes=self._policy.max_chunk_bytes,
        )
        if existing_item_name or any(chunk["item_name"] for chunk in chunks):
            raise SafeNodeContractError("item name was populated before recognition")
        file_title = require_metadata(
            state.get("file_title"),
            field="file title",
            allow_empty=False,
        )
        context = ItemNameContext(
            file_title=file_title,
            titles=_unique_nonempty([chunk["title"] for chunk in chunks]),
            parent_titles=_unique_nonempty([chunk["parent_title"] for chunk in chunks]),
            chunk_count=len(chunks),
        )
        if self._explicit_name is not None:
            raw_name = self._explicit_name
        elif self._recognizer is not None:
            raw_name = self._recognizer(context)
        else:
            raw_name = context.file_title
        item_name = require_metadata(
            raw_name,
            field="recognized item name",
            allow_empty=False,
            max_characters=self._policy.max_name_characters,
            max_bytes=self._policy.max_name_bytes,
        )
        return {
            "item_name": item_name,
            "chunks": [copy_text_chunk(chunk, item_name=item_name) for chunk in chunks],
        }


__all__ = [
    "ItemNameContext",
    "ItemNamePolicy",
    "PureItemNameRecognizer",
    "SafeItemNameNode",
]
