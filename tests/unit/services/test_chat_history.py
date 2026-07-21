from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.chat_history import (
    ChatMessagePage,
    ChatMessageRole,
    ChatMessageSnapshot,
    ChatOwnerScope,
    ChatSessionPage,
    ChatSessionScope,
    ChatSessionSnapshot,
    NewChatMessage,
    NewChatSession,
)

_SESSION_ID = "00000000-0000-4000-8000-000000000001"
_MESSAGE_ID = "10000000-0000-4000-8000-000000000001"


def test_scope_is_an_immutable_complete_authorization_key() -> None:
    scope = ChatSessionScope(
        tenant_id="tenant-a",
        owner_subject="user:42",
        session_id=_SESSION_ID,
    )

    assert hash(scope)
    assert scope == ChatSessionScope("tenant-a", "user:42", _SESSION_ID)
    assert scope.owner_scope == ChatOwnerScope("tenant-a", "user:42")
    with pytest.raises(AttributeError):
        scope.tenant_id = "tenant-b"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("tenant_id", "owner", "session_id"),
    [
        ("../tenant", "user:42", _SESSION_ID),
        ("tenant-a", "", _SESSION_ID),
        ("tenant-a", "user\n42", _SESSION_ID),
        ("tenant-a", "user:42", "not-a-uuid"),
        ("tenant-a", "user:42", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa".upper()),
    ],
)
def test_scope_rejects_noncanonical_or_unsafe_identity(
    tenant_id: str,
    owner: str,
    session_id: str,
) -> None:
    with pytest.raises(ValueError):
        ChatSessionScope(tenant_id, owner, session_id)


def test_message_canonicalizes_attributes_for_idempotency() -> None:
    message = NewChatMessage(
        message_id=_MESSAGE_ID,
        role=ChatMessageRole.ASSISTANT,
        content="line one\nline two",
        attributes=(("trace.id", "abc"), ("model", "example-v1")),
    )

    assert message.attributes == (("model", "example-v1"), ("trace.id", "abc"))


@pytest.mark.parametrize(
    "message",
    [
        lambda: NewChatMessage("bad", ChatMessageRole.USER, "hello"),
        lambda: NewChatMessage(_MESSAGE_ID, ChatMessageRole.USER, ""),
        lambda: NewChatMessage(_MESSAGE_ID, ChatMessageRole.USER, "bad\x00content"),
        lambda: NewChatMessage(
            _MESSAGE_ID,
            ChatMessageRole.USER,
            "hello",
            (("bad key", "value"),),
        ),
        lambda: NewChatMessage(
            _MESSAGE_ID,
            ChatMessageRole.USER,
            "hello",
            (("model", "one"), ("model", "two")),
        ),
    ],
)
def test_message_rejects_unsafe_or_ambiguous_input(message: object) -> None:
    with pytest.raises(ValueError):
        message()  # type: ignore[operator]


def test_service_snapshots_and_page_are_transport_neutral() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scope = ChatSessionScope("tenant-a", "user:42", _SESSION_ID)
    session = ChatSessionSnapshot(scope=scope, title="Support", created_at=now, updated_at=now)
    message = ChatMessageSnapshot(
        scope=scope,
        message_id=_MESSAGE_ID,
        role=ChatMessageRole.USER,
        content="hello",
        attributes=(),
        created_at=now,
    )

    assert NewChatSession(scope, "Support").scope == session.scope
    assert ChatSessionPage(items=(session,), next_cursor="sessions").items == (session,)
    assert ChatMessagePage(items=(message,), next_cursor="opaque").items == (message,)
