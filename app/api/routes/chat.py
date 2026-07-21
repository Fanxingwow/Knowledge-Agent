from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Annotated, Any, TypeVar, cast

import anyio
from fastapi import APIRouter, Depends, Header, Path, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from app.api.routes.search import get_knowledge_search
from app.api.schemas.chat import (
    AgentResponseCreateRequest,
    ChatMessageCreateRequest,
    ChatMessagePageResponse,
    ChatMessageResponse,
    ChatSessionCreateRequest,
    ChatSessionPageResponse,
    ChatSessionResponse,
)
from app.api.schemas.common import ErrorResponse
from app.core.errors import AppError, DependencyError, ErrorCode
from app.core.logger import logger
from app.core.security import Principal, require_roles
from app.services.agent_chat import AgentChatBackend, AgentHistoryMessage
from app.services.chat_history import (
    ChatHistoryRepository,
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
from app.services.knowledge_search import KnowledgeSearchService
from app.services.sse_broker import SessionEventBroker, SseSubscription, parse_last_event_id

router = APIRouter(prefix="/chat", tags=["chat"])
chat_reader = require_roles("chat-user", "reader", "editor", "admin")
chat_writer = require_roles("chat-user", "editor", "admin")
_T = TypeVar("_T")
_REPOSITORY_CALL_TIMEOUT_SECONDS = 30.0
_AGENT_RESPONSE_TIMEOUT_SECONDS = 120.0
_ASSISTANT_MESSAGE_NAMESPACE = uuid.UUID("6eec9788-20d8-48fb-8816-7654f38ddd85")
_ERROR_RESPONSE: dict[str, Any] = {"model": ErrorResponse}
_READ_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: _ERROR_RESPONSE for code in (401, 403, 404, 409, 422, 500, 503)
}
_WRITE_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: _ERROR_RESPONSE for code in (401, 403, 404, 409, 422, 500, 503)
}


def get_chat_history_repository(request: Request) -> ChatHistoryRepository:
    _require_chat_readiness(request)
    repository = getattr(request.app.state, "chat_history_repository", None)
    if repository is None:
        raise DependencyError("Chat history is unavailable")
    return cast(ChatHistoryRepository, repository)


def get_session_event_broker(request: Request) -> SessionEventBroker:
    _require_chat_readiness(request)
    broker = getattr(request.app.state, "session_event_broker", None)
    if broker is None:
        raise DependencyError("Event streaming is unavailable")
    return cast(SessionEventBroker, broker)


def get_agent_chat_backend(request: Request) -> AgentChatBackend:
    backend = getattr(request.app.state, "agent_chat_backend", None)
    if not callable(getattr(backend, "stream_answer", None)):
        raise DependencyError("Agent generation is unavailable")
    return cast(AgentChatBackend, backend)


def _require_chat_readiness(request: Request) -> None:
    managed_until = getattr(request.app.state, "chat_ready_until", None)
    if managed_until is None:
        return
    if (
        getattr(request.app.state, "chat_ready", False) is not True
        or isinstance(managed_until, bool)
        or not isinstance(managed_until, (int, float))
        or not math.isfinite(managed_until)
        or time.monotonic() >= managed_until
    ):
        raise DependencyError("Chat runtime is unavailable")


def _session_scope(principal: Principal, session_id: uuid.UUID) -> ChatSessionScope:
    return ChatSessionScope(
        tenant_id=principal.tenant_id,
        owner_subject=principal.subject,
        session_id=str(session_id),
    )


async def _repository_call(
    operation: Callable[[], _T],
    *,
    timeout_seconds: float = _REPOSITORY_CALL_TIMEOUT_SECONDS,
) -> _T:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError("invalid chat repository timeout")
    try:
        with anyio.fail_after(float(timeout_seconds)):
            return await anyio.to_thread.run_sync(
                operation,
                abandon_on_cancel=True,
            )
    except AppError:
        raise
    except ValueError:
        raise AppError(ErrorCode.REQUEST_INVALID, "Invalid chat request", 422) from None
    except TimeoutError:
        raise DependencyError("Chat history is unavailable") from None
    except Exception:
        raise DependencyError("Chat history is unavailable") from None


def _verified_session(
    snapshot: ChatSessionSnapshot,
    expected: ChatSessionScope,
) -> ChatSessionSnapshot:
    if not isinstance(snapshot, ChatSessionSnapshot) or snapshot.scope != expected:
        raise DependencyError("Chat history is unavailable")
    return snapshot


def _verified_session_page(
    page: ChatSessionPage,
    expected: ChatOwnerScope,
) -> ChatSessionPage:
    if not isinstance(page, ChatSessionPage) or any(item.scope.owner_scope != expected for item in page.items):
        raise DependencyError("Chat history is unavailable")
    return page


def _verified_message(
    snapshot: ChatMessageSnapshot,
    expected: ChatSessionScope,
) -> ChatMessageSnapshot:
    if not isinstance(snapshot, ChatMessageSnapshot) or snapshot.scope != expected:
        raise DependencyError("Chat history is unavailable")
    return snapshot


def _verified_message_page(
    page: ChatMessagePage,
    expected: ChatSessionScope,
) -> ChatMessagePage:
    if not isinstance(page, ChatMessagePage) or any(item.scope != expected for item in page.items):
        raise DependencyError("Chat history is unavailable")
    return page


async def _sse_stream(
    request: Request,
    subscription: SseSubscription,
) -> AsyncIterator[str]:
    try:
        while not await request.is_disconnected():
            try:
                event = await subscription.receive()
            except StopAsyncIteration:
                return
            yield event.encode()
    finally:
        with anyio.CancelScope(shield=True):
            await subscription.aclose()


def _agent_event(event_id: int, event_type: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id}\nevent: {event_type}\ndata: {data}\n\n"


async def _publish_agent_event(
    broker: SessionEventBroker,
    scope: ChatSessionScope,
    *,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    try:
        await broker.publish(
            scope,
            event_type=event_type,
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
    except Exception:
        logger.bind(event_type=event_type).warning("Agent event publication failed")


async def _optional_message(
    repository: ChatHistoryRepository,
    scope: ChatSessionScope,
    message_id: str,
) -> ChatMessageSnapshot | None:
    try:
        snapshot = await _repository_call(partial(repository.get_message, scope, message_id))
    except AppError as exc:
        if exc.code is ErrorCode.RESOURCE_NOT_FOUND:
            return None
        raise
    return _verified_message(snapshot, scope)


def _history_for_agent(
    page: ChatMessagePage,
    *,
    current_message_id: str,
) -> tuple[AgentHistoryMessage, ...]:
    return tuple(
        AgentHistoryMessage(item.role, item.content)
        for item in page.items
        if item.message_id != current_message_id and item.role in {ChatMessageRole.USER, ChatMessageRole.ASSISTANT}
    )


@router.post(
    "/sessions",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_WRITE_RESPONSES,
    summary="Create an owned chat session",
)
async def create_chat_session(
    payload: ChatSessionCreateRequest,
    principal: Annotated[Principal, Depends(chat_writer)],
    repository: Annotated[ChatHistoryRepository, Depends(get_chat_history_repository)],
) -> ChatSessionResponse:
    session_id = payload.session_id or uuid.uuid4()
    try:
        new_session = NewChatSession(
            scope=_session_scope(principal, session_id),
            title=payload.title,
        )
    except ValueError:
        raise AppError(ErrorCode.REQUEST_INVALID, "Invalid chat request", 422) from None
    snapshot = await _repository_call(partial(repository.create_session, new_session))
    return ChatSessionResponse.from_service(_verified_session(snapshot, new_session.scope))


@router.get(
    "/sessions",
    response_model=ChatSessionPageResponse,
    responses=_READ_RESPONSES,
    summary="List the authenticated owner's chat sessions",
)
async def list_chat_sessions(
    principal: Annotated[Principal, Depends(chat_reader)],
    repository: Annotated[ChatHistoryRepository, Depends(get_chat_history_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(min_length=1, max_length=512)] = None,
) -> ChatSessionPageResponse:
    owner = ChatOwnerScope(principal.tenant_id, principal.subject)
    page = _verified_session_page(
        await _repository_call(partial(repository.list_sessions, owner, limit=limit, cursor=cursor)),
        owner,
    )
    return ChatSessionPageResponse.from_service(page)


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionResponse,
    responses=_READ_RESPONSES,
    summary="Get an owned chat session",
)
async def get_chat_session(
    session_id: Annotated[uuid.UUID, Path()],
    principal: Annotated[Principal, Depends(chat_reader)],
    repository: Annotated[ChatHistoryRepository, Depends(get_chat_history_repository)],
) -> ChatSessionResponse:
    scope = _session_scope(principal, session_id)
    snapshot = _verified_session(
        await _repository_call(partial(repository.get_session, scope)),
        scope,
    )
    return ChatSessionResponse.from_service(snapshot)


@router.get(
    "/sessions/{session_id}/history",
    response_model=ChatMessagePageResponse,
    responses=_READ_RESPONSES,
    summary="Page through an owned chat session's history",
)
async def list_chat_history(
    session_id: Annotated[uuid.UUID, Path()],
    principal: Annotated[Principal, Depends(chat_reader)],
    repository: Annotated[ChatHistoryRepository, Depends(get_chat_history_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(min_length=1, max_length=512)] = None,
) -> ChatMessagePageResponse:
    scope = _session_scope(principal, session_id)
    page = _verified_message_page(
        await _repository_call(
            partial(
                repository.list_messages,
                scope,
                limit=limit,
                cursor=cursor,
            )
        ),
        scope,
    )
    return ChatMessagePageResponse.from_service(page)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatMessageResponse,
    responses=_WRITE_RESPONSES,
    summary="Idempotently append a user message to an owned chat session",
)
async def create_chat_message(
    payload: ChatMessageCreateRequest,
    session_id: Annotated[uuid.UUID, Path()],
    principal: Annotated[Principal, Depends(chat_writer)],
    repository: Annotated[ChatHistoryRepository, Depends(get_chat_history_repository)],
) -> ChatMessageResponse:
    message = NewChatMessage(
        message_id=str(payload.message_id),
        role=ChatMessageRole.USER,
        content=payload.content,
        attributes=tuple(payload.attributes.items()),
    )
    scope = _session_scope(principal, session_id)
    snapshot = _verified_message(
        await _repository_call(
            partial(
                repository.append_message,
                scope,
                message,
            )
        ),
        scope,
    )
    return ChatMessageResponse.from_service(snapshot)


@router.post(
    "/sessions/{session_id}/responses",
    responses=_WRITE_RESPONSES,
    summary="Run a retrieval-grounded Agent response as an SSE stream",
)
async def create_agent_response(
    request: Request,
    payload: AgentResponseCreateRequest,
    session_id: Annotated[uuid.UUID, Path()],
    principal: Annotated[Principal, Depends(chat_writer)],
    repository: Annotated[ChatHistoryRepository, Depends(get_chat_history_repository)],
    broker: Annotated[SessionEventBroker, Depends(get_session_event_broker)],
    search: Annotated[KnowledgeSearchService, Depends(get_knowledge_search)],
    backend: Annotated[AgentChatBackend, Depends(get_agent_chat_backend)],
) -> StreamingResponse:
    scope = _session_scope(principal, session_id)
    session = _verified_session(
        await _repository_call(partial(repository.get_session, scope)),
        scope,
    )
    assistant_message_id = str(
        uuid.uuid5(
            _ASSISTANT_MESSAGE_NAMESPACE,
            f"{scope.session_id}:{payload.message_id}",
        )
    )
    existing_assistant = await _optional_message(
        repository,
        scope,
        assistant_message_id,
    )
    user_message = NewChatMessage(
        message_id=str(payload.message_id),
        role=ChatMessageRole.USER,
        content=payload.content,
        attributes=(("source", "knowledge-console"),),
    )
    _verified_message(
        await _repository_call(partial(repository.append_message, scope, user_message)),
        scope,
    )
    if session.title is None:
        title = payload.content.strip().replace("\r", " ").replace("\n", " ")[:60]
        if title:
            await _repository_call(partial(repository.update_session_title, scope, title))
    history_page = _verified_message_page(
        await _repository_call(partial(repository.list_messages, scope, limit=50)),
        scope,
    )
    history = _history_for_agent(
        history_page,
        current_message_id=str(payload.message_id),
    )
    run_id = str(uuid.uuid5(_ASSISTANT_MESSAGE_NAMESPACE, f"run:{assistant_message_id}"))

    async def event_stream() -> AsyncIterator[str]:
        event_id = 0

        async def emit(event_type: str, data: dict[str, Any]) -> str:
            nonlocal event_id
            event_id += 1
            await _publish_agent_event(
                broker,
                scope,
                event_type=event_type,
                payload=data,
            )
            return _agent_event(event_id, event_type, data)

        if existing_assistant is not None:
            attributes = dict(existing_assistant.attributes)
            replay_citation_ids = tuple(filter(None, attributes.get("citation_ids", "").split(",")))
            yield await emit(
                "message.completed",
                {
                    "run_id": run_id,
                    "replayed": True,
                    "message": ChatMessageResponse.from_service(existing_assistant).model_dump(mode="json"),
                    "citations": [{"citation_id": item} for item in replay_citation_ids],
                },
            )
            return

        try:
            yield await emit(
                "run.started",
                {
                    "run_id": run_id,
                    "message_id": str(payload.message_id),
                },
            )
            search_result = await search.search(
                tenant_id=principal.tenant_id,
                query_text=payload.content,
                top_k=payload.top_k,
            )
            citation_ids = [hit.citation_id for hit in search_result.hits]
            citations = [
                {
                    "citation_id": hit.citation_id,
                    "index": index,
                    "content": hit.content,
                    "score": hit.score,
                    "title": hit.metadata.title,
                    "parent_title": hit.metadata.parent_title,
                    "file_title": hit.metadata.file_title,
                    "item_name": hit.metadata.item_name,
                }
                for index, hit in enumerate(search_result.hits, start=1)
            ]
            yield await emit(
                "retrieval.completed",
                {
                    "run_id": run_id,
                    "citations": citations,
                },
            )
            fragments: list[str] = []
            with anyio.fail_after(_AGENT_RESPONSE_TIMEOUT_SECONDS):
                async for delta in backend.stream_answer(
                    question=payload.content,
                    history=history,
                    hits=search_result.hits,
                ):
                    if not isinstance(delta, str) or not delta:
                        continue
                    fragments.append(delta)
                    yield await emit(
                        "message.delta",
                        {
                            "run_id": run_id,
                            "delta": delta,
                        },
                    )
            answer = "".join(fragments).strip()
            if not answer:
                raise DependencyError("Agent generation is unavailable")
            assistant_message = NewChatMessage(
                message_id=assistant_message_id,
                role=ChatMessageRole.ASSISTANT,
                content=answer,
                attributes=(
                    ("citation_ids", ",".join(citation_ids)),
                    ("run_id", run_id),
                    ("source", "knowledge-agent"),
                ),
            )
            assistant_snapshot = _verified_message(
                await _repository_call(partial(repository.append_message, scope, assistant_message)),
                scope,
            )
            yield await emit(
                "message.completed",
                {
                    "run_id": run_id,
                    "replayed": False,
                    "message": ChatMessageResponse.from_service(assistant_snapshot).model_dump(mode="json"),
                    "citations": citations,
                },
            )
        except AppError as exc:
            logger.bind(error_code=exc.code.value, run_id=run_id).warning("Agent response failed")
            yield await emit(
                "run.failed",
                {
                    "run_id": run_id,
                    "code": exc.code.value,
                    "message": "Agent 暂时不可用，请稍后重试。",
                },
            )
        except TimeoutError:
            logger.bind(run_id=run_id).warning("Agent response timed out")
            yield await emit(
                "run.failed",
                {
                    "run_id": run_id,
                    "code": ErrorCode.DEPENDENCY_UNAVAILABLE.value,
                    "message": "Agent 响应超时，请稍后重试。",
                },
            )
        except Exception:
            logger.bind(run_id=run_id).exception("Unexpected Agent response failure")
            yield await emit(
                "run.failed",
                {
                    "run_id": run_id,
                    "code": ErrorCode.INTERNAL_ERROR.value,
                    "message": "Agent 响应失败，请稍后重试。",
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete(
    "/sessions/{session_id}/messages/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=_WRITE_RESPONSES,
    summary="Delete an owned message without disclosing prior existence",
)
async def delete_chat_message(
    session_id: Annotated[uuid.UUID, Path()],
    message_id: Annotated[uuid.UUID, Path()],
    principal: Annotated[Principal, Depends(chat_writer)],
    repository: Annotated[ChatHistoryRepository, Depends(get_chat_history_repository)],
) -> Response:
    await _repository_call(
        partial(
            repository.delete_message,
            _session_scope(principal, session_id),
            str(message_id),
        )
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=_WRITE_RESPONSES,
    summary="Delete an owned session without disclosing prior existence",
)
async def delete_chat_session(
    session_id: Annotated[uuid.UUID, Path()],
    principal: Annotated[Principal, Depends(chat_writer)],
    repository: Annotated[ChatHistoryRepository, Depends(get_chat_history_repository)],
) -> Response:
    await _repository_call(partial(repository.delete_session, _session_scope(principal, session_id)))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/sessions/{session_id}/events",
    responses=_READ_RESPONSES,
    summary="Resume an owned session event stream",
)
async def stream_chat_events(
    request: Request,
    session_id: Annotated[uuid.UUID, Path()],
    principal: Annotated[Principal, Depends(chat_reader)],
    repository: Annotated[ChatHistoryRepository, Depends(get_chat_history_repository)],
    broker: Annotated[SessionEventBroker, Depends(get_session_event_broker)],
    last_event_id_header: Annotated[
        str | None,
        Header(alias="Last-Event-ID", min_length=1, max_length=20),
    ] = None,
) -> StreamingResponse:
    scope = _session_scope(principal, session_id)
    _verified_session(await _repository_call(partial(repository.get_session, scope)), scope)
    try:
        last_event_id = parse_last_event_id(last_event_id_header)
    except ValueError:
        raise AppError(ErrorCode.REQUEST_INVALID, "Invalid Last-Event-ID", 422) from None
    try:
        subscription = await broker.subscribe(scope, last_event_id=last_event_id)
    except AppError:
        raise
    except ValueError:
        raise AppError(ErrorCode.REQUEST_INVALID, "Invalid Last-Event-ID", 422) from None
    if subscription.scope != scope:
        with anyio.CancelScope(shield=True):
            await subscription.aclose()
        raise DependencyError("Event streaming is unavailable")
    return StreamingResponse(
        _sse_stream(request, subscription),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = [
    "create_agent_response",
    "create_chat_message",
    "create_chat_session",
    "delete_chat_message",
    "delete_chat_session",
    "get_chat_history_repository",
    "get_agent_chat_backend",
    "get_chat_session",
    "get_session_event_broker",
    "list_chat_history",
    "list_chat_sessions",
    "router",
    "stream_chat_events",
]
