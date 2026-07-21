from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_ATTRIBUTE_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")


def _canonical_uuid(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid {field}")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"invalid {field}") from None
    if str(parsed) != value:
        raise ValueError(f"invalid {field}")
    return value


def _valid_identity(value: str, *, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= maximum
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _valid_content(value: str, *, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= maximum
        and "\x00" not in value
        and not any(ord(character) < 32 and character not in {"\t", "\n", "\r"} for character in value)
    )


@dataclass(frozen=True, slots=True)
class ChatOwnerScope:
    tenant_id: str
    owner_subject: str

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or _TENANT_ID.fullmatch(self.tenant_id) is None:
            raise ValueError("invalid chat tenant")
        if not _valid_identity(self.owner_subject, maximum=256):
            raise ValueError("invalid chat owner")


@dataclass(frozen=True, slots=True)
class ChatSessionScope:
    """An authorization-complete key for every single-session operation."""

    tenant_id: str
    owner_subject: str
    session_id: str

    def __post_init__(self) -> None:
        ChatOwnerScope(self.tenant_id, self.owner_subject)
        _canonical_uuid(self.session_id, field="chat session identifier")

    @property
    def owner_scope(self) -> ChatOwnerScope:
        return ChatOwnerScope(self.tenant_id, self.owner_subject)


class ChatMessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class NewChatSession:
    scope: ChatSessionScope
    title: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ChatSessionScope):
            raise ValueError("invalid chat session scope")
        if self.title is not None and not _valid_content(self.title, maximum=200):
            raise ValueError("invalid chat session title")


@dataclass(frozen=True, slots=True)
class ChatSessionSnapshot:
    scope: ChatSessionScope
    title: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ChatSessionPage:
    items: tuple[ChatSessionSnapshot, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class NewChatMessage:
    message_id: str
    role: ChatMessageRole
    content: str
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _canonical_uuid(self.message_id, field="chat message identifier")
        if not isinstance(self.role, ChatMessageRole):
            raise ValueError("invalid chat message role")
        if not _valid_content(self.content, maximum=200_000):
            raise ValueError("invalid chat message content")
        if not isinstance(self.attributes, tuple) or len(self.attributes) > 32:
            raise ValueError("invalid chat message attributes")
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in self.attributes:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("invalid chat message attributes")
            key, value = item
            if (
                not isinstance(key, str)
                or _ATTRIBUTE_KEY.fullmatch(key) is None
                or key in seen
                or not _valid_content(value, maximum=2_000)
            ):
                raise ValueError("invalid chat message attributes")
            seen.add(key)
            normalized.append((key, value))
        object.__setattr__(self, "attributes", tuple(sorted(normalized)))


@dataclass(frozen=True, slots=True)
class ChatMessageSnapshot:
    scope: ChatSessionScope
    message_id: str
    role: ChatMessageRole
    content: str
    attributes: tuple[tuple[str, str], ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ChatMessagePage:
    items: tuple[ChatMessageSnapshot, ...]
    next_cursor: str | None


class ChatHistoryRepository(Protocol):
    durable: bool

    def create_session(self, session: NewChatSession) -> ChatSessionSnapshot: ...

    def get_session(self, scope: ChatSessionScope) -> ChatSessionSnapshot: ...

    def list_sessions(
        self,
        owner: ChatOwnerScope,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ChatSessionPage: ...

    def update_session_title(
        self,
        scope: ChatSessionScope,
        title: str | None,
    ) -> ChatSessionSnapshot: ...

    def append_message(
        self,
        scope: ChatSessionScope,
        message: NewChatMessage,
    ) -> ChatMessageSnapshot: ...

    def get_message(
        self,
        scope: ChatSessionScope,
        message_id: str,
    ) -> ChatMessageSnapshot: ...

    def list_messages(
        self,
        scope: ChatSessionScope,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ChatMessagePage: ...

    def delete_message(self, scope: ChatSessionScope, message_id: str) -> None: ...

    def delete_session(self, scope: ChatSessionScope) -> None: ...
