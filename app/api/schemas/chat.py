from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.chat_history import (
    ChatMessagePage,
    ChatMessageRole,
    ChatMessageSnapshot,
    ChatSessionPage,
    ChatSessionSnapshot,
    NewChatMessage,
)
from app.services.knowledge_search import MAX_QUERY_CHARACTERS, MAX_SEARCH_TOP_K


class ChatSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: UUID | None = None
    title: str | None = Field(default=None, min_length=1, max_length=200)


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_service(cls, session: ChatSessionSnapshot) -> ChatSessionResponse:
        return cls(
            session_id=UUID(session.scope.session_id),
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )


class ChatSessionPageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[ChatSessionResponse, ...]
    next_cursor: str | None

    @classmethod
    def from_service(cls, page: ChatSessionPage) -> ChatSessionPageResponse:
        return cls(
            items=tuple(ChatSessionResponse.from_service(item) for item in page.items),
            next_cursor=page.next_cursor,
        )


class ChatMessageCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: UUID
    content: str = Field(min_length=1, max_length=200_000)
    attributes: dict[str, str] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def validate_domain_contract(self) -> ChatMessageCreateRequest:
        NewChatMessage(
            message_id=str(self.message_id),
            role=ChatMessageRole.USER,
            content=self.content,
            attributes=tuple(self.attributes.items()),
        )
        return self


class AgentResponseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: UUID
    content: str = Field(min_length=1, max_length=MAX_QUERY_CHARACTERS)
    top_k: int = Field(default=5, ge=1, le=MAX_SEARCH_TOP_K)


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: UUID
    role: ChatMessageRole
    content: str
    attributes: dict[str, str]
    created_at: datetime

    @classmethod
    def from_service(cls, message: ChatMessageSnapshot) -> ChatMessageResponse:
        return cls(
            message_id=UUID(message.message_id),
            role=message.role,
            content=message.content,
            attributes=dict(message.attributes),
            created_at=message.created_at,
        )


class ChatMessagePageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[ChatMessageResponse, ...]
    next_cursor: str | None

    @classmethod
    def from_service(cls, page: ChatMessagePage) -> ChatMessagePageResponse:
        return cls(
            items=tuple(ChatMessageResponse.from_service(item) for item in page.items),
            next_cursor=page.next_cursor,
        )


__all__ = [
    "AgentResponseCreateRequest",
    "ChatMessageCreateRequest",
    "ChatMessagePageResponse",
    "ChatMessageResponse",
    "ChatSessionCreateRequest",
    "ChatSessionPageResponse",
    "ChatSessionResponse",
]
