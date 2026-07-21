from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
from threading import Event as ThreadEvent
from typing import Any, cast

import anyio
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.routes.chat import _repository_call, _sse_stream, router
from app.api.routes.search import get_knowledge_search
from app.core.errors import AppError, DependencyError, ErrorCode, register_exception_handlers
from app.core.security import Principal, get_current_principal
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
from app.services.knowledge_search import (
    KnowledgeSearchHit,
    KnowledgeSearchMetadata,
    KnowledgeSearchResult,
)
from app.services.sse_broker import SseEvent

_SESSION_ID = "00000000-0000-4000-8000-000000000001"
_MESSAGE_ID = "10000000-0000-4000-8000-000000000001"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _not_found() -> AppError:
    return AppError(ErrorCode.RESOURCE_NOT_FOUND, "Chat resource not found", 404)


def _conflict() -> AppError:
    return AppError(ErrorCode.RESOURCE_CONFLICT, "Chat resource conflict", 409)


class FakeChatRepository:
    durable = False

    def __init__(self) -> None:
        self.sessions: dict[ChatSessionScope, ChatSessionSnapshot] = {}
        self.messages: dict[ChatSessionScope, dict[str, ChatMessageSnapshot]] = {}
        self.calls: list[tuple[str, object]] = []

    def create_session(self, session: NewChatSession) -> ChatSessionSnapshot:
        self.calls.append(("create_session", session.scope))
        existing = self.sessions.get(session.scope)
        if existing is not None:
            if existing.title == session.title:
                return existing
            raise _conflict()
        snapshot = ChatSessionSnapshot(
            scope=session.scope,
            title=session.title,
            created_at=_NOW,
            updated_at=_NOW,
        )
        self.sessions[session.scope] = snapshot
        return snapshot

    def get_session(self, scope: ChatSessionScope) -> ChatSessionSnapshot:
        self.calls.append(("get_session", scope))
        try:
            return self.sessions[scope]
        except KeyError:
            raise _not_found() from None

    def list_sessions(
        self,
        owner: ChatOwnerScope,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ChatSessionPage:
        self.calls.append(("list_sessions", (owner, limit, cursor)))
        if cursor == "invalid-service-cursor":
            raise ValueError("private parser detail")
        sessions = tuple(session for session in self.sessions.values() if session.scope.owner_scope == owner)
        return ChatSessionPage(items=sessions[:limit], next_cursor="session-next" if sessions else None)

    def update_session_title(
        self,
        scope: ChatSessionScope,
        title: str | None,
    ) -> ChatSessionSnapshot:
        current = self.get_session(scope)
        updated = ChatSessionSnapshot(scope, title, current.created_at, _NOW)
        self.sessions[scope] = updated
        return updated

    def append_message(
        self,
        scope: ChatSessionScope,
        message: NewChatMessage,
    ) -> ChatMessageSnapshot:
        self.calls.append(("append_message", scope))
        self.get_session(scope)
        scoped = self.messages.setdefault(scope, {})
        existing = scoped.get(message.message_id)
        if existing is not None:
            if (
                existing.role is message.role
                and existing.content == message.content
                and existing.attributes == message.attributes
            ):
                return existing
            raise _conflict()
        snapshot = ChatMessageSnapshot(
            scope=scope,
            message_id=message.message_id,
            role=message.role,
            content=message.content,
            attributes=message.attributes,
            created_at=_NOW,
        )
        scoped[message.message_id] = snapshot
        return snapshot

    def get_message(
        self,
        scope: ChatSessionScope,
        message_id: str,
    ) -> ChatMessageSnapshot:
        self.calls.append(("get_message", scope))
        try:
            return self.messages[scope][message_id]
        except KeyError:
            raise _not_found() from None

    def list_messages(
        self,
        scope: ChatSessionScope,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ChatMessagePage:
        self.calls.append(("list_messages", (scope, limit, cursor)))
        self.get_session(scope)
        messages = tuple(self.messages.get(scope, {}).values())
        return ChatMessagePage(items=messages[:limit], next_cursor="message-next" if messages else None)

    def delete_message(self, scope: ChatSessionScope, message_id: str) -> None:
        self.calls.append(("delete_message", (scope, message_id)))
        self.messages.get(scope, {}).pop(message_id, None)

    def delete_session(self, scope: ChatSessionScope) -> None:
        self.calls.append(("delete_session", scope))
        self.messages.pop(scope, None)
        self.sessions.pop(scope, None)


class FakeSubscription:
    def __init__(self, scope: ChatSessionScope, events: tuple[SseEvent, ...]) -> None:
        self.scope = scope
        self.events = deque(events)
        self.close_count = 0

    async def receive(self) -> SseEvent:
        if not self.events:
            raise StopAsyncIteration
        return self.events.popleft()

    async def aclose(self) -> None:
        self.close_count += 1


class FakeBroker:
    distributed = False

    def __init__(self, events: tuple[SseEvent, ...] = ()) -> None:
        self.events = events
        self.subscribe_calls: list[tuple[ChatSessionScope, int | None]] = []
        self.subscriptions: list[FakeSubscription] = []

    async def subscribe(
        self,
        scope: ChatSessionScope,
        *,
        last_event_id: int | None = None,
    ) -> FakeSubscription:
        self.subscribe_calls.append((scope, last_event_id))
        subscription = FakeSubscription(scope, self.events)
        self.subscriptions.append(subscription)
        return subscription

    async def publish(
        self,
        scope: ChatSessionScope,
        *,
        event_type: str,
        data: str,
        terminal: bool = False,
    ) -> SseEvent:
        return SseEvent(1, event_type, data, terminal=terminal)

    async def terminate(
        self,
        scope: ChatSessionScope,
        *,
        event_type: str,
        data: str,
    ) -> SseEvent:
        return await self.publish(scope, event_type=event_type, data=data, terminal=True)

    async def aclose(self) -> None:
        return None


class FakeAgentBackend:
    async def stream_answer(self, *, question, history, hits):
        assert question == "库存灯是什么颜色？"
        assert history == ()
        assert len(hits) == 1
        yield "库存灯是"
        yield "蓝色。[1]"


class FakeKnowledgeSearch:
    async def search(self, *, tenant_id: str, query_text: str, top_k: int) -> KnowledgeSearchResult:
        assert (tenant_id, query_text, top_k) == ("tenant-a", "库存灯是什么颜色？", 5)
        return KnowledgeSearchResult(
            hits=(
                KnowledgeSearchHit(
                    citation_id="cit_" + "a" * 64,
                    content="库存灯为蓝色。",
                    metadata=KnowledgeSearchMetadata(
                        title="库存指南",
                        parent_title="",
                        file_title="inventory",
                        item_name="库存灯",
                    ),
                    score=0.91,
                ),
            )
        )


def build_client(
    *,
    principal: Principal | None = None,
    repository: FakeChatRepository | None = None,
    broker: FakeBroker | None = None,
) -> tuple[TestClient, FastAPI, FakeChatRepository, FakeBroker]:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    resolved_repository = repository or FakeChatRepository()
    resolved_broker = broker or FakeBroker()
    app.state.chat_history_repository = resolved_repository
    app.state.session_event_broker = resolved_broker
    resolved_principal = principal or Principal(
        "user:42",
        "tenant-a",
        frozenset({"reader", "editor"}),
    )
    app.dependency_overrides[get_current_principal] = lambda: resolved_principal
    return TestClient(app), app, resolved_repository, resolved_broker


def owned_scope() -> ChatSessionScope:
    return ChatSessionScope("tenant-a", "user:42", _SESSION_ID)


def create_owned_session(repository: FakeChatRepository) -> ChatSessionSnapshot:
    return repository.create_session(NewChatSession(owned_scope(), "Support"))


def test_create_and_get_session_derive_scope_only_from_principal() -> None:
    client, _app, repository, _broker = build_client()

    created = client.post(
        "/chat/sessions",
        json={"session_id": _SESSION_ID, "title": "Support"},
    )
    fetched = client.get(f"/chat/sessions/{_SESSION_ID}")

    assert created.status_code == 201
    assert fetched.status_code == 200
    assert created.json()["session_id"] == _SESSION_ID
    assert created.json()["title"] == "Support"
    assert owned_scope() in repository.sessions
    assert all(
        value.tenant_id == "tenant-a" and value.owner_subject == "user:42"
        for operation, value in repository.calls
        if operation in {"create_session", "get_session"} and isinstance(value, ChatSessionScope)
    )


def test_agent_response_streams_grounded_answer_and_persists_assistant_message() -> None:
    client, app, repository, _broker = build_client()
    create_owned_session(repository)
    app.state.agent_chat_backend = FakeAgentBackend()
    app.dependency_overrides[get_knowledge_search] = lambda: FakeKnowledgeSearch()

    response = client.post(
        f"/chat/sessions/{_SESSION_ID}/responses",
        json={
            "message_id": _MESSAGE_ID,
            "content": "库存灯是什么颜色？",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: run.started" in response.text
    assert "event: retrieval.completed" in response.text
    assert response.text.count("event: message.delta") == 2
    assert "event: message.completed" in response.text
    assert "库存灯是蓝色。[1]" in response.text
    stored = tuple(repository.messages[owned_scope()].values())
    assert [message.role for message in stored] == [
        ChatMessageRole.USER,
        ChatMessageRole.ASSISTANT,
    ]
    assert stored[-1].content == "库存灯是蓝色。[1]"
    assert dict(stored[-1].attributes)["citation_ids"] == "cit_" + "a" * 64


def test_agent_response_fails_before_writing_when_generation_backend_is_missing() -> None:
    client, _app, repository, _broker = build_client()
    create_owned_session(repository)

    response = client.post(
        f"/chat/sessions/{_SESSION_ID}/responses",
        json={"message_id": _MESSAGE_ID, "content": "库存灯是什么颜色？"},
    )

    assert response.status_code == 503
    assert response.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert repository.messages.get(owned_scope()) is None


def test_session_list_is_owner_scoped_and_forwards_cursor() -> None:
    repository = FakeChatRepository()
    create_owned_session(repository)
    repository.create_session(NewChatSession(ChatSessionScope("tenant-b", "user:42", _SESSION_ID), "Foreign"))
    repository.create_session(
        NewChatSession(
            ChatSessionScope("tenant-a", "user:43", _SESSION_ID),
            "Another user's private conversation",
        )
    )
    client, _app, _repository, _broker = build_client(repository=repository)

    response = client.get("/chat/sessions?limit=7&cursor=opaque")

    assert response.status_code == 200
    assert [item["title"] for item in response.json()["items"]] == ["Support"]
    owner, limit, cursor = cast(
        tuple[ChatOwnerScope, int, str | None],
        next(value for operation, value in repository.calls if operation == "list_sessions"),
    )
    assert owner == ChatOwnerScope("tenant-a", "user:42")
    assert (limit, cursor) == (7, "opaque")


def test_history_is_scoped_and_forwards_opaque_pagination() -> None:
    client, _app, repository, _broker = build_client()
    create_owned_session(repository)
    repository.append_message(
        owned_scope(),
        NewChatMessage(_MESSAGE_ID, role=ChatMessageRole.USER, content="hello"),
    )

    response = client.get(f"/chat/sessions/{_SESSION_ID}/history?limit=9&cursor=history-cursor")

    assert response.status_code == 200
    assert response.json()["items"][0]["message_id"] == _MESSAGE_ID
    scope, limit, cursor = cast(
        tuple[ChatSessionScope, int, str | None],
        next(value for operation, value in repository.calls if operation == "list_messages"),
    )
    assert scope == owned_scope()
    assert (limit, cursor) == (9, "history-cursor")


def test_message_post_is_idempotent_and_payload_conflict_is_generic() -> None:
    client, _app, repository, _broker = build_client()
    create_owned_session(repository)
    payload = {
        "message_id": _MESSAGE_ID,
        "content": "How do I reset it?",
        "attributes": {"channel": "web"},
    }

    first = client.post(f"/chat/sessions/{_SESSION_ID}/messages", json=payload)
    repeated = client.post(f"/chat/sessions/{_SESSION_ID}/messages", json=payload)
    conflict = client.post(
        f"/chat/sessions/{_SESSION_ID}/messages",
        json={**payload, "content": "different"},
    )

    assert first.status_code == repeated.status_code == 200
    assert first.json() == repeated.json()
    assert conflict.status_code == 409
    assert conflict.json()["message"] == "Chat resource conflict"
    assert len(repository.messages[owned_scope()]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"message_id": _MESSAGE_ID, "content": "hello", "unexpected": True},
        {"message_id": _MESSAGE_ID, "content": "bad\u0000content"},
        {"message_id": _MESSAGE_ID, "content": "hello", "attributes": {"bad key": "x"}},
    ],
)
def test_message_schema_rejects_extra_and_unsafe_input(payload: dict[str, Any]) -> None:
    client, _app, repository, _broker = build_client()
    create_owned_session(repository)

    response = client.post(f"/chat/sessions/{_SESSION_ID}/messages", json=payload)

    assert response.status_code == 422
    assert not any(operation == "append_message" for operation, _value in repository.calls)


def test_delete_responses_do_not_disclose_prior_existence() -> None:
    client, _app, repository, _broker = build_client()
    create_owned_session(repository)

    missing_message = client.delete(f"/chat/sessions/{_SESSION_ID}/messages/{_MESSAGE_ID}")
    existing_session = client.delete(f"/chat/sessions/{_SESSION_ID}")
    missing_session = client.delete(f"/chat/sessions/{_SESSION_ID}")

    assert [missing_message.status_code, existing_session.status_code, missing_session.status_code] == [
        204,
        204,
        204,
    ]
    assert missing_message.content == existing_session.content == missing_session.content == b""


def test_reader_cannot_mutate_chat_state() -> None:
    principal = Principal("user:42", "tenant-a", frozenset({"reader"}))
    client, _app, repository, _broker = build_client(principal=principal)

    response = client.post(
        "/chat/sessions",
        json={"session_id": _SESSION_ID, "title": "Denied"},
    )

    assert response.status_code == 403
    assert repository.sessions == {}


def test_chat_user_can_create_only_an_owner_scoped_session() -> None:
    principal = Principal("user:42", "tenant-a", frozenset({"chat-user"}))
    client, _app, repository, _broker = build_client(principal=principal)

    response = client.post(
        "/chat/sessions",
        json={"session_id": _SESSION_ID, "title": "Private conversation"},
    )

    assert response.status_code == 201
    assert set(repository.sessions) == {owned_scope()}


def test_missing_app_state_services_fail_closed() -> None:
    client, app, _repository, _broker = build_client()
    del app.state.chat_history_repository

    response = client.get("/chat/sessions")

    assert response.status_code == 503
    assert response.json()["message"] == "Chat history is unavailable"


def test_repository_cursor_errors_are_mapped_without_internal_details() -> None:
    client, _app, _repository, _broker = build_client()

    response = client.get("/chat/sessions?cursor=invalid-service-cursor")

    assert response.status_code == 422
    assert response.json()["message"] == "Invalid chat request"
    assert "private parser" not in response.text


def test_unexpected_repository_errors_are_mapped_without_internal_details() -> None:
    leaked_detail = "mongodb://admin:credential@private-host"

    class FailingRepository(FakeChatRepository):
        def list_sessions(
            self,
            owner: ChatOwnerScope,
            *,
            limit: int = 50,
            cursor: str | None = None,
        ) -> ChatSessionPage:
            raise RuntimeError(leaked_detail)

    client, _app, _repository, _broker = build_client(repository=FailingRepository())

    response = client.get("/chat/sessions")

    assert response.status_code == 503
    assert response.json()["message"] == "Chat history is unavailable"
    assert response.json()["details"] == {}
    assert leaked_detail not in response.text


@pytest.mark.asyncio
async def test_repository_thread_does_not_delay_request_cancellation() -> None:
    started = asyncio.Event()
    release = ThreadEvent()
    loop = asyncio.get_running_loop()

    def blocking_operation() -> str:
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=1)
        return "late result"

    task = asyncio.create_task(_repository_call(blocking_operation))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0.02)
    completed_before_release = task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert completed_before_release is True


@pytest.mark.asyncio
async def test_repository_thread_is_abandoned_when_anyio_scope_is_cancelled() -> None:
    worker_started = asyncio.Event()
    scope_ready = asyncio.Event()
    release = ThreadEvent()
    loop = asyncio.get_running_loop()
    scopes: list[anyio.CancelScope] = []

    def blocking_operation() -> None:
        loop.call_soon_threadsafe(worker_started.set)
        release.wait(timeout=1)

    async def invoke() -> None:
        with anyio.CancelScope() as cancel_scope:
            scopes.append(cancel_scope)
            scope_ready.set()
            await _repository_call(blocking_operation)

    task = asyncio.create_task(invoke())
    await scope_ready.wait()
    await worker_started.wait()
    scopes[0].cancel()
    await asyncio.sleep(0.02)
    completed_before_release = task.done()
    release.set()
    await task

    assert completed_before_release is True


@pytest.mark.asyncio
async def test_repository_timeout_is_bounded_and_sanitized() -> None:
    leaked_detail = "private Mongo socket timeout detail"
    started = asyncio.Event()
    release = ThreadEvent()
    loop = asyncio.get_running_loop()

    def blocking_operation() -> None:
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=1)
        raise RuntimeError(leaked_detail)

    task = asyncio.create_task(_repository_call(blocking_operation, timeout_seconds=0.02))
    await started.wait()
    try:
        with pytest.raises(DependencyError) as raised:
            await task
    finally:
        release.set()
    await asyncio.sleep(0.02)

    assert raised.value.message == "Chat history is unavailable"
    assert raised.value.details == {}
    assert raised.value.__cause__ is None
    assert leaked_detail not in str(raised.value)


def test_out_of_scope_repository_result_fails_closed() -> None:
    class OutOfScopeRepository(FakeChatRepository):
        def list_sessions(
            self,
            owner: ChatOwnerScope,
            *,
            limit: int = 50,
            cursor: str | None = None,
        ) -> ChatSessionPage:
            foreign = ChatSessionSnapshot(
                ChatSessionScope("tenant-b", owner.owner_subject, _SESSION_ID),
                "Foreign",
                _NOW,
                _NOW,
            )
            return ChatSessionPage((foreign,), None)

    client, _app, _repository, _broker = build_client(repository=OutOfScopeRepository())

    response = client.get("/chat/sessions")

    assert response.status_code == 503
    assert response.json()["message"] == "Chat history is unavailable"


def test_sse_forwards_last_event_id_and_always_closes_subscription() -> None:
    event = SseEvent(
        event_id=8,
        event_type="answer.complete",
        data='{"answer":"done"}',
        terminal=True,
    )
    broker = FakeBroker((event,))
    client, _app, repository, _broker = build_client(broker=broker)
    create_owned_session(repository)

    response = client.get(
        f"/chat/sessions/{_SESSION_ID}/events",
        headers={"Last-Event-ID": "7"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert "id: 8\nevent: answer.complete" in response.text
    assert broker.subscribe_calls == [(owned_scope(), 7)]
    assert broker.subscriptions[0].close_count == 1


@pytest.mark.parametrize("last_event_id", ["-1", "+1", "1.0", "9223372036854775808"])
def test_sse_rejects_invalid_last_event_id_before_subscribing(last_event_id: str) -> None:
    client, _app, repository, broker = build_client()
    create_owned_session(repository)

    response = client.get(
        f"/chat/sessions/{_SESSION_ID}/events",
        headers={"Last-Event-ID": last_event_id},
    )

    assert response.status_code == 422
    assert broker.subscribe_calls == []


def test_sse_sanitizes_semantically_invalid_replay_position() -> None:
    class RejectingBroker(FakeBroker):
        async def subscribe(
            self,
            scope: ChatSessionScope,
            *,
            last_event_id: int | None = None,
        ) -> FakeSubscription:
            raise ValueError("private event watermark")

    broker = RejectingBroker()
    client, _app, repository, _broker = build_client(broker=broker)
    create_owned_session(repository)

    response = client.get(
        f"/chat/sessions/{_SESSION_ID}/events",
        headers={"Last-Event-ID": "99"},
    )

    assert response.status_code == 422
    assert response.json()["message"] == "Invalid Last-Event-ID"
    assert "watermark" not in response.text


def test_sse_checks_owned_session_before_allocating_subscription() -> None:
    client, _app, _repository, broker = build_client()

    response = client.get(f"/chat/sessions/{_SESSION_ID}/events")

    assert response.status_code == 404
    assert broker.subscribe_calls == []


def test_sse_rejects_subscription_bound_to_another_tenant() -> None:
    class OutOfScopeBroker(FakeBroker):
        async def subscribe(
            self,
            scope: ChatSessionScope,
            *,
            last_event_id: int | None = None,
        ) -> FakeSubscription:
            self.subscribe_calls.append((scope, last_event_id))
            subscription = FakeSubscription(
                ChatSessionScope("tenant-b", scope.owner_subject, scope.session_id),
                (),
            )
            self.subscriptions.append(subscription)
            return subscription

    broker = OutOfScopeBroker()
    client, _app, repository, _broker = build_client(broker=broker)
    create_owned_session(repository)

    response = client.get(f"/chat/sessions/{_SESSION_ID}/events")

    assert response.status_code == 503
    assert response.json()["message"] == "Event streaming is unavailable"
    assert broker.subscriptions[0].close_count == 1


@pytest.mark.asyncio
async def test_sse_disconnect_closes_subscription_without_waiting_for_events() -> None:
    subscription = FakeSubscription(owned_scope(), ())

    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    chunks = [
        chunk
        async for chunk in _sse_stream(
            cast(Request, DisconnectedRequest()),
            subscription,
        )
    ]

    assert chunks == []
    assert subscription.close_count == 1
