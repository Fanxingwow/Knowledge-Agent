from __future__ import annotations

import asyncio
import re
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Protocol

from app.core.errors import AppError, DependencyError, ErrorCode
from app.services.chat_history import ChatSessionScope

_EVENT_TYPE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_BROKER_UNAVAILABLE = "Event stream broker is unavailable"


def _stream_conflict(message: str = "Event stream state conflict") -> AppError:
    return AppError(ErrorCode.RESOURCE_CONFLICT, message, 409)


def _valid_event_data(data: object) -> bool:
    if not isinstance(data, str) or "\x00" in data:
        return False
    try:
        return len(data.encode("utf-8")) <= 1_000_000
    except UnicodeEncodeError:
        return False


def parse_last_event_id(value: str | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 20 or not value.isascii() or not value.isdecimal():
        raise ValueError("invalid Last-Event-ID")
    parsed = int(value)
    if parsed > 9_223_372_036_854_775_807:
        raise ValueError("invalid Last-Event-ID")
    return parsed


@dataclass(frozen=True, slots=True)
class SseEvent:
    event_id: int | None
    event_type: str
    data: str
    terminal: bool = False
    heartbeat: bool = False

    def __post_init__(self) -> None:
        if self.heartbeat:
            if self.event_id is not None or self.event_type != "heartbeat" or self.data or self.terminal:
                raise ValueError("invalid SSE heartbeat")
            return
        if (
            isinstance(self.event_id, bool)
            or not isinstance(self.event_id, int)
            or self.event_id < 1
            or not isinstance(self.event_type, str)
            or _EVENT_TYPE.fullmatch(self.event_type) is None
            or self.event_type == "heartbeat"
            or not _valid_event_data(self.data)
        ):
            raise ValueError("invalid SSE event")

    @classmethod
    def heartbeat_event(cls) -> SseEvent:
        return cls(
            event_id=None,
            event_type="heartbeat",
            data="",
            heartbeat=True,
        )

    def encode(self) -> str:
        if self.heartbeat:
            return ": heartbeat\n\n"
        data_lines = self.data.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        fields = [f"id: {self.event_id}", f"event: {self.event_type}"]
        fields.extend(f"data: {line}" for line in data_lines)
        return "\n".join(fields) + "\n\n"


class SseSubscription(Protocol):
    scope: ChatSessionScope

    async def receive(self) -> SseEvent: ...

    async def aclose(self) -> None: ...


class SessionEventBroker(Protocol):
    distributed: bool

    async def publish(
        self,
        scope: ChatSessionScope,
        *,
        event_type: str,
        data: str,
        terminal: bool = False,
    ) -> SseEvent: ...

    async def subscribe(
        self,
        scope: ChatSessionScope,
        *,
        last_event_id: int | None = None,
    ) -> SseSubscription: ...

    async def terminate(
        self,
        scope: ChatSessionScope,
        *,
        event_type: str,
        data: str,
    ) -> SseEvent: ...

    async def aclose(self) -> None: ...


@dataclass(slots=True)
class _StreamState:
    history: deque[SseEvent]
    next_event_id: int = 1
    terminal: bool = False
    subscribers: set[_InMemorySseSubscription] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _CloseSignal:
    error: AppError | None


class _InMemorySseSubscription:
    def __init__(
        self,
        broker: InMemorySseBroker,
        scope: ChatSessionScope,
        replay: tuple[SseEvent, ...],
        *,
        queue_capacity: int,
        heartbeat_seconds: float,
    ) -> None:
        self._broker = broker
        self.scope = scope
        self._replay = deque(replay, maxlen=len(replay) or 1)
        self._queue: asyncio.Queue[SseEvent | _CloseSignal] = asyncio.Queue(maxsize=queue_capacity)
        self._heartbeat_seconds = heartbeat_seconds
        self._closed = False
        self._close_error: AppError | None = None
        self._terminal_consumed = False

    def __aiter__(self) -> _InMemorySseSubscription:
        return self

    async def __anext__(self) -> SseEvent:
        try:
            return await self.receive()
        except StopAsyncIteration:
            raise

    async def __aenter__(self) -> _InMemorySseSubscription:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    def _offer(self, event: SseEvent) -> bool:
        if self._closed or self._queue.full():
            return False
        self._queue.put_nowait(event)
        return True

    def _detach(self, error: AppError | None) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_error = error
        self._replay.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(_CloseSignal(error=error))

    async def _finish_terminal(self) -> None:
        self._terminal_consumed = True
        await self._broker._unsubscribe(self)

    async def receive(self) -> SseEvent:
        if self._close_error is not None:
            raise self._close_error
        if self._terminal_consumed or self._closed:
            raise StopAsyncIteration
        if self._replay:
            event = self._replay.popleft()
            if event.terminal:
                await self._finish_terminal()
            return event
        try:
            item = await asyncio.wait_for(
                self._queue.get(),
                timeout=self._heartbeat_seconds,
            )
        except TimeoutError:
            return SseEvent.heartbeat_event()
        except asyncio.CancelledError:
            await self.aclose()
            raise
        if isinstance(item, _CloseSignal):
            if item.error is not None:
                raise item.error
            raise StopAsyncIteration
        if item.terminal:
            await self._finish_terminal()
        return item

    async def aclose(self) -> None:
        if self._closed:
            return
        await self._broker._unsubscribe(self)


class InMemorySseBroker:
    """Bounded single-process SSE transport keyed only by complete ownership scopes."""

    distributed = False

    def __init__(
        self,
        *,
        max_sessions: int = 1_000,
        replay_capacity: int = 256,
        max_replay_events: int = 128,
        subscriber_queue_capacity: int = 64,
        max_subscribers_per_session: int = 8,
        heartbeat_seconds: float = 15.0,
    ) -> None:
        integer_options = {
            "max sessions": max_sessions,
            "replay capacity": replay_capacity,
            "max replay events": max_replay_events,
            "subscriber queue capacity": subscriber_queue_capacity,
            "max subscribers": max_subscribers_per_session,
        }
        for label, value in integer_options.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"invalid SSE {label}")
        if max_replay_events > replay_capacity:
            raise ValueError("SSE replay limit exceeds capacity")
        if (
            isinstance(heartbeat_seconds, bool)
            or not isinstance(heartbeat_seconds, (int, float))
            or not 0.01 <= heartbeat_seconds <= 300
        ):
            raise ValueError("invalid SSE heartbeat interval")
        self._max_sessions = max_sessions
        self._replay_capacity = replay_capacity
        self._max_replay_events = max_replay_events
        self._subscriber_queue_capacity = subscriber_queue_capacity
        self._max_subscribers_per_session = max_subscribers_per_session
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._streams: OrderedDict[ChatSessionScope, _StreamState] = OrderedDict()
        self._lock = asyncio.Lock()
        self._closed = False

    @staticmethod
    def _validate_scope(scope: ChatSessionScope) -> None:
        if not isinstance(scope, ChatSessionScope):
            raise ValueError("invalid SSE stream scope")

    def _require_open(self) -> None:
        if self._closed:
            raise DependencyError(_BROKER_UNAVAILABLE)

    def _state_locked(self, scope: ChatSessionScope) -> _StreamState:
        state = self._streams.get(scope)
        if state is not None:
            self._streams.move_to_end(scope)
            return state
        if len(self._streams) >= self._max_sessions:
            raise _stream_conflict("Event stream capacity reached")
        state = _StreamState(history=deque(maxlen=self._replay_capacity))
        self._streams[scope] = state
        return state

    @staticmethod
    def _validate_event(event_type: str, data: str) -> None:
        if (
            not isinstance(event_type, str)
            or _EVENT_TYPE.fullmatch(event_type) is None
            or event_type == "heartbeat"
            or not _valid_event_data(data)
        ):
            raise ValueError("invalid SSE event")

    def _detach_locked(
        self,
        state: _StreamState,
        subscription: _InMemorySseSubscription,
        error: AppError | None,
    ) -> None:
        state.subscribers.discard(subscription)
        subscription._detach(error)

    def _discard_terminal_locked(
        self,
        scope: ChatSessionScope,
        state: _StreamState,
    ) -> None:
        if (
            state.terminal
            and not state.subscribers
            and self._streams.get(scope) is state
        ):
            del self._streams[scope]

    async def publish(
        self,
        scope: ChatSessionScope,
        *,
        event_type: str,
        data: str,
        terminal: bool = False,
    ) -> SseEvent:
        self._validate_scope(scope)
        self._validate_event(event_type, data)
        if not isinstance(terminal, bool):
            raise ValueError("invalid SSE terminal flag")
        async with self._lock:
            self._require_open()
            state = self._state_locked(scope)
            if state.terminal:
                raise _stream_conflict()
            if state.next_event_id > 9_223_372_036_854_775_807:
                raise _stream_conflict("Event stream identifier exhausted")
            event = SseEvent(
                event_id=state.next_event_id,
                event_type=event_type,
                data=data,
                terminal=terminal,
            )
            state.next_event_id += 1
            state.history.append(event)
            if terminal:
                state.terminal = True
            had_subscribers = bool(state.subscribers)
            backpressure = _stream_conflict("Event stream consumer fell behind")
            for subscription in tuple(state.subscribers):
                if not subscription._offer(event):
                    self._detach_locked(state, subscription, backpressure)
            if terminal and had_subscribers:
                self._discard_terminal_locked(scope, state)
            return event

    async def subscribe(
        self,
        scope: ChatSessionScope,
        *,
        last_event_id: int | None = None,
    ) -> _InMemorySseSubscription:
        self._validate_scope(scope)
        if last_event_id is not None and (
            isinstance(last_event_id, bool)
            or not isinstance(last_event_id, int)
            or last_event_id < 0
            or last_event_id > 9_223_372_036_854_775_807
        ):
            raise ValueError("invalid Last-Event-ID")
        async with self._lock:
            self._require_open()
            state = self._state_locked(scope)
            current_event_id = state.next_event_id - 1
            if last_event_id is not None and last_event_id > current_event_id:
                raise ValueError("invalid Last-Event-ID")
            replay: tuple[SseEvent, ...] = ()
            replay_after = 0 if last_event_id is None else last_event_id
            if replay_after < current_event_id:
                if not state.history:
                    raise _stream_conflict("Event replay window expired")
                oldest_event_id = state.history[0].event_id
                if oldest_event_id is None or replay_after < oldest_event_id - 1:
                    raise _stream_conflict("Event replay window expired")
                replay = tuple(
                    event for event in state.history if event.event_id is not None and event.event_id > replay_after
                )
                if len(replay) > self._max_replay_events:
                    raise _stream_conflict("Event replay window expired")
            if state.terminal and (not replay or not replay[-1].terminal):
                raise _stream_conflict("Event stream already terminated")
            if len(state.subscribers) >= self._max_subscribers_per_session:
                raise _stream_conflict("Event stream subscriber capacity reached")
            subscription = _InMemorySseSubscription(
                self,
                scope,
                replay,
                queue_capacity=self._subscriber_queue_capacity,
                heartbeat_seconds=self._heartbeat_seconds,
            )
            state.subscribers.add(subscription)
            return subscription

    async def terminate(
        self,
        scope: ChatSessionScope,
        *,
        event_type: str,
        data: str,
    ) -> SseEvent:
        return await self.publish(
            scope,
            event_type=event_type,
            data=data,
            terminal=True,
        )

    async def discard_terminal(self, scope: ChatSessionScope) -> None:
        """Release replay state only after every subscriber consumed or left the terminal stream."""

        self._validate_scope(scope)
        async with self._lock:
            state = self._streams.get(scope)
            if state is None:
                return
            if not state.terminal or state.subscribers:
                raise _stream_conflict()
            del self._streams[scope]

    async def _unsubscribe(self, subscription: _InMemorySseSubscription) -> None:
        async with self._lock:
            state = self._streams.get(subscription.scope)
            if state is None:
                subscription._detach(None)
                return
            self._detach_locked(state, subscription, None)
            self._discard_terminal_locked(subscription.scope, state)

    async def subscriber_count(self, scope: ChatSessionScope) -> int:
        self._validate_scope(scope)
        async with self._lock:
            state = self._streams.get(scope)
            return 0 if state is None else len(state.subscribers)

    async def stream_count(self) -> int:
        async with self._lock:
            return len(self._streams)

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            unavailable = DependencyError(_BROKER_UNAVAILABLE)
            for state in self._streams.values():
                for subscription in tuple(state.subscribers):
                    self._detach_locked(state, subscription, unavailable)
            self._streams.clear()
