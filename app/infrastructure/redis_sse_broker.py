from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib
import inspect
import math
import re
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast
from urllib.parse import urlsplit

from app.core.errors import AppError, DependencyError, ErrorCode
from app.services.chat_history import ChatSessionScope
from app.services.sse_broker import SseEvent

_MAX_EVENT_ID = 9_223_372_036_854_775_807
_KEY_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9:_.-]{0,63}$")
_STREAM_ID = re.compile(r"^(0|[1-9][0-9]{0,18})-0$")
_BROKER_UNAVAILABLE = "Event stream broker is unavailable"

_PUBLISH_SCRIPT = """
local terminal = redis.call('GET', KEYS[3])
if terminal then
    return {'CONFLICT', 'TERMINATED'}
end
local current = redis.call('GET', KEYS[1])
if current then
    if not string.match(current, '^[1-9][0-9]*$') then
        return {'ERROR', 'CORRUPT'}
    end
    if string.len(current) > string.len(ARGV[6]) or
       (string.len(current) == string.len(ARGV[6]) and current >= ARGV[6]) then
        return {'CONFLICT', 'EXHAUSTED'}
    end
end
redis.call('INCR', KEYS[1])
-- GET preserves Redis' exact int64 decimal representation. Converting the
-- INCR result through Lua would round large IDs because Lua 5.1 uses doubles.
local event_id = redis.call('GET', KEYS[1])
local stream_id = event_id .. '-0'
redis.call(
    'XADD', KEYS[2], 'MAXLEN', '=', ARGV[4], stream_id,
    'event_type', ARGV[1], 'data', ARGV[2], 'terminal', ARGV[3]
)
redis.call('EXPIRE', KEYS[1], ARGV[5])
redis.call('EXPIRE', KEYS[2], ARGV[5])
if ARGV[3] == '1' then
    redis.call('SET', KEYS[3], event_id, 'EX', ARGV[5])
end
return {'OK', event_id}
""".strip()

_SNAPSHOT_SCRIPT = """
local current = redis.call('GET', KEYS[1])
local terminal = redis.call('GET', KEYS[3])
local events = redis.call('XRANGE', KEYS[2], '-', '+', 'COUNT', ARGV[1])
return {current or '', terminal or '', events}
""".strip()


def _stream_conflict(message: str = "Event stream state conflict") -> AppError:
    return AppError(ErrorCode.RESOURCE_CONFLICT, message, 409)


class AsyncRedisStreamsClient(Protocol):
    async def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: object,
    ) -> object: ...

    async def xread(
        self,
        streams: Mapping[str, str],
        *,
        count: int,
        block: int,
    ) -> object: ...

    async def ping(self) -> object: ...

    async def aclose(self) -> object: ...


@dataclass(frozen=True, slots=True)
class RedisStreamsBrokerConfig:
    url: str = field(repr=False)
    key_hmac_secret: bytes = field(repr=False)
    key_prefix: str = "knowledge-agent:sse:v1"
    replay_capacity: int = 256
    max_replay_events: int = 128
    read_batch_size: int = 64
    max_subscribers_per_session: int = 32
    heartbeat_seconds: float = 15.0
    retention_seconds: int = 86_400
    connect_timeout_seconds: float = 3.0
    socket_timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url) if isinstance(self.url, str) else None
        if (
            parsed is None
            or parsed.scheme not in {"redis", "rediss"}
            or not parsed.hostname
            or parsed.fragment
        ):
            raise ValueError("invalid Redis event broker URL")
        if (
            not isinstance(self.key_hmac_secret, bytes)
            or not 32 <= len(self.key_hmac_secret) <= 4_096
        ):
            raise ValueError("invalid Redis event broker key secret")
        if (
            not isinstance(self.key_prefix, str)
            or _KEY_PREFIX.fullmatch(self.key_prefix) is None
        ):
            raise ValueError("invalid Redis event broker key prefix")
        integers = (
            self.replay_capacity,
            self.max_replay_events,
            self.read_batch_size,
            self.max_subscribers_per_session,
            self.retention_seconds,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integers
        ):
            raise ValueError("invalid Redis event broker limit")
        if self.max_replay_events > self.replay_capacity:
            raise ValueError("Redis event replay limit exceeds capacity")
        if self.read_batch_size > self.replay_capacity:
            raise ValueError("Redis event read batch exceeds capacity")
        if (
            isinstance(self.heartbeat_seconds, bool)
            or not isinstance(self.heartbeat_seconds, (int, float))
            or not math.isfinite(float(self.heartbeat_seconds))
            or not 0.01 <= float(self.heartbeat_seconds) <= 300
        ):
            raise ValueError("invalid Redis event heartbeat interval")
        if (
            isinstance(self.connect_timeout_seconds, bool)
            or not isinstance(self.connect_timeout_seconds, (int, float))
            or not math.isfinite(float(self.connect_timeout_seconds))
            or not 0 < float(self.connect_timeout_seconds) <= 30
        ):
            raise ValueError("invalid Redis event connect timeout")
        if (
            isinstance(self.socket_timeout_seconds, bool)
            or not isinstance(self.socket_timeout_seconds, (int, float))
            or not math.isfinite(float(self.socket_timeout_seconds))
            or float(self.socket_timeout_seconds) <= float(self.heartbeat_seconds)
        ):
            raise ValueError("invalid Redis event socket timeout")


@dataclass(frozen=True, slots=True)
class _ScopeKeys:
    stream: str
    sequence: str
    terminal: str


RedisClientFactory = Callable[
    [RedisStreamsBrokerConfig],
    AsyncRedisStreamsClient | Awaitable[AsyncRedisStreamsClient],
]


def _default_client_factory(config: RedisStreamsBrokerConfig) -> AsyncRedisStreamsClient:
    module = importlib.import_module("redis.asyncio")
    redis_type = getattr(module, "Redis", None)
    from_url = getattr(redis_type, "from_url", None)
    if not callable(from_url):
        raise RuntimeError("redis.asyncio is unavailable")
    client = from_url(
        config.url,
        decode_responses=False,
        socket_connect_timeout=float(config.connect_timeout_seconds),
        socket_timeout=float(config.socket_timeout_seconds),
    )
    return cast(AsyncRedisStreamsClient, client)


def _validate_scope(scope: ChatSessionScope) -> None:
    if not isinstance(scope, ChatSessionScope):
        raise ValueError("invalid SSE stream scope")


def _scope_material(scope: ChatSessionScope) -> bytes:
    material = bytearray(b"chat-session-event-stream-key-v1\x00")
    for value in (scope.tenant_id, scope.owner_subject, scope.session_id):
        encoded = value.encode("utf-8", errors="strict")
        material.extend(len(encoded).to_bytes(8, "big"))
        material.extend(encoded)
    return bytes(material)


def _keys_for_scope(
    scope: ChatSessionScope,
    config: RedisStreamsBrokerConfig,
) -> _ScopeKeys:
    _validate_scope(scope)
    digest = hmac.new(
        config.key_hmac_secret,
        _scope_material(scope),
        hashlib.sha256,
    ).hexdigest()
    # The hash tag keeps all Lua keys in one Redis Cluster slot.
    base = f"{config.key_prefix}:{{{digest}}}"
    return _ScopeKeys(
        stream=f"{base}:events",
        sequence=f"{base}:sequence",
        terminal=f"{base}:terminal",
    )


def _decode_text(value: object) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise DependencyError(_BROKER_UNAVAILABLE) from None
    if isinstance(value, str):
        return value
    raise DependencyError(_BROKER_UNAVAILABLE)


def _parse_decimal(value: object, *, allow_empty: bool = False) -> int | None:
    text = _decode_text(value)
    if allow_empty and not text:
        return None
    if (
        not text
        or len(text) > 19
        or not text.isascii()
        or not text.isdecimal()
    ):
        raise DependencyError(_BROKER_UNAVAILABLE)
    parsed = int(text)
    if not 0 <= parsed <= _MAX_EVENT_ID:
        raise DependencyError(_BROKER_UNAVAILABLE)
    return parsed


def _parse_stream_id(value: object) -> int:
    text = _decode_text(value)
    if _STREAM_ID.fullmatch(text) is None:
        raise DependencyError(_BROKER_UNAVAILABLE)
    event_id = int(text[:-2])
    if not 1 <= event_id <= _MAX_EVENT_ID:
        raise DependencyError(_BROKER_UNAVAILABLE)
    return event_id


def _field_mapping(value: object) -> dict[str, str]:
    raw_items: Sequence[tuple[object, object]]
    if isinstance(value, Mapping):
        raw_items = tuple(value.items())
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) % 2 == 0
    ):
        raw_items = tuple(
            (value[index], value[index + 1]) for index in range(0, len(value), 2)
        )
    else:
        raise DependencyError(_BROKER_UNAVAILABLE)
    fields: dict[str, str] = {}
    for raw_key, raw_value in raw_items:
        key = _decode_text(raw_key)
        if key in fields:
            raise DependencyError(_BROKER_UNAVAILABLE)
        fields[key] = _decode_text(raw_value)
    if set(fields) != {"event_type", "data", "terminal"}:
        raise DependencyError(_BROKER_UNAVAILABLE)
    return fields


def _parse_entry(value: object) -> SseEvent:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        raise DependencyError(_BROKER_UNAVAILABLE)
    event_id = _parse_stream_id(value[0])
    fields = _field_mapping(value[1])
    terminal_text = fields["terminal"]
    if terminal_text not in {"0", "1"}:
        raise DependencyError(_BROKER_UNAVAILABLE)
    try:
        return SseEvent(
            event_id=event_id,
            event_type=fields["event_type"],
            data=fields["data"],
            terminal=terminal_text == "1",
        )
    except ValueError:
        raise DependencyError(_BROKER_UNAVAILABLE) from None


def _parse_entries(value: object, *, maximum: int) -> tuple[SseEvent, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) > maximum
    ):
        raise DependencyError(_BROKER_UNAVAILABLE)
    events = tuple(_parse_entry(item) for item in value)
    previous = 0
    for event in events:
        event_id = cast(int, event.event_id)
        if previous and event_id != previous + 1:
            raise DependencyError(_BROKER_UNAVAILABLE)
        previous = event_id
    if any(event.terminal for event in events[:-1]):
        raise DependencyError(_BROKER_UNAVAILABLE)
    return events


def _validate_last_event_id(last_event_id: int | None) -> None:
    if last_event_id is not None and (
        isinstance(last_event_id, bool)
        or not isinstance(last_event_id, int)
        or not 0 <= last_event_id <= _MAX_EVENT_ID
    ):
        raise ValueError("invalid Last-Event-ID")


def _validate_event(event_type: str, data: str, terminal: bool) -> None:
    if not isinstance(terminal, bool):
        raise ValueError("invalid SSE terminal flag")
    try:
        SseEvent(
            event_id=1,
            event_type=event_type,
            data=data,
            terminal=terminal,
        )
    except ValueError:
        raise ValueError("invalid SSE event") from None


class RedisSseSubscription:
    def __init__(
        self,
        broker: RedisStreamsSseBroker,
        scope: ChatSessionScope,
        keys: _ScopeKeys,
        replay: tuple[SseEvent, ...],
        *,
        cursor: int,
    ) -> None:
        self._broker = broker
        self.scope = scope
        self._keys = keys
        self._replay = deque(replay)
        self._pending: deque[SseEvent] = deque()
        self._cursor = cursor
        self._closed = False
        self._close_error: AppError | None = None
        self._terminal_consumed = False
        self._active_receive: asyncio.Task[object] | None = None

    def __aiter__(self) -> RedisSseSubscription:
        return self

    async def __anext__(self) -> SseEvent:
        return await self.receive()

    async def __aenter__(self) -> RedisSseSubscription:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    def _detach(self, error: AppError | None) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_error = error
        self._replay.clear()
        self._pending.clear()
        active = self._active_receive
        if active is not None and active is not asyncio.current_task():
            active.cancel()

    async def _finish_terminal(self) -> None:
        self._terminal_consumed = True
        await self._broker._unsubscribe(self)

    async def _next_buffered(self) -> SseEvent | None:
        source = self._replay if self._replay else self._pending
        if not source:
            return None
        event = source.popleft()
        if event.terminal:
            await self._finish_terminal()
        return event

    async def receive(self) -> SseEvent:
        if self._close_error is not None:
            raise self._close_error
        if self._terminal_consumed or self._closed:
            raise StopAsyncIteration
        if self._active_receive is not None:
            raise ValueError("concurrent SSE receive is invalid")
        task = asyncio.current_task()
        if task is None:
            raise DependencyError(_BROKER_UNAVAILABLE)
        self._active_receive = cast(asyncio.Task[object], task)
        try:
            buffered = await self._next_buffered()
            if buffered is not None:
                return buffered
            events = await self._broker._read(self._keys.stream, self._cursor)
            if self._closed:
                if self._close_error is not None:
                    raise self._close_error
                raise StopAsyncIteration
            if not events:
                return SseEvent.heartbeat_event()
            first_id = cast(int, events[0].event_id)
            if first_id != self._cursor + 1:
                conflict = _stream_conflict("Event stream consumer fell behind")
                await self._broker._unsubscribe(self, error=conflict)
                raise conflict
            self._cursor = cast(int, events[-1].event_id)
            self._pending.extend(events)
            buffered = await self._next_buffered()
            if buffered is None:  # pragma: no cover - guarded by non-empty events
                raise DependencyError(_BROKER_UNAVAILABLE)
            return buffered
        except asyncio.CancelledError:
            if self._close_error is not None:
                raise self._close_error from None
            await self.aclose()
            raise
        except DependencyError as error:
            await self._broker._unsubscribe(self, error=error)
            raise
        except AppError:
            raise
        except Exception:
            await self._broker._unsubscribe(
                self,
                error=DependencyError(_BROKER_UNAVAILABLE),
            )
            raise DependencyError(_BROKER_UNAVAILABLE) from None
        finally:
            if self._active_receive is task:
                self._active_receive = None

    async def aclose(self) -> None:
        if self._closed:
            return
        await self._broker._unsubscribe(self)


class RedisStreamsSseBroker:
    """Distributed, replay-bounded SSE broker backed by Redis Streams."""

    distributed = True

    def __init__(
        self,
        config: RedisStreamsBrokerConfig,
        *,
        client: AsyncRedisStreamsClient | None = None,
        client_factory: RedisClientFactory | None = None,
        manage_client: bool = True,
    ) -> None:
        if not isinstance(config, RedisStreamsBrokerConfig):
            raise ValueError("Redis event broker config is required")
        if client is not None and client_factory is not None:
            raise ValueError("Redis event broker client sources are mutually exclusive")
        if not isinstance(manage_client, bool):
            raise ValueError("invalid Redis event broker ownership policy")
        if client is not None:
            self._validate_client(client)
        self._config = config
        self._client = client
        self._client_factory = client_factory or _default_client_factory
        self._manage_client = manage_client
        self._client_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._subscriptions: dict[str, set[RedisSseSubscription]] = {}
        self._closed = False

    @staticmethod
    def _validate_client(client: object) -> None:
        if any(
            not callable(getattr(client, method, None))
            for method in ("eval", "xread", "ping", "aclose")
        ):
            raise ValueError("invalid async Redis event broker client")

    def stream_key(self, scope: ChatSessionScope) -> str:
        """Return the opaque HMAC-derived stream key for observability/tests."""

        return _keys_for_scope(scope, self._config).stream

    async def _require_open(self) -> None:
        async with self._state_lock:
            if self._closed:
                raise DependencyError(_BROKER_UNAVAILABLE)

    async def _get_client(self) -> AsyncRedisStreamsClient:
        async with self._client_lock:
            if self._closed:
                raise DependencyError(_BROKER_UNAVAILABLE)
            if self._client is not None:
                return self._client
            try:
                created = self._client_factory(self._config)
                if inspect.isawaitable(created):
                    created = await created
                self._validate_client(created)
                self._client = created
            except asyncio.CancelledError:
                raise
            except Exception:
                raise DependencyError(_BROKER_UNAVAILABLE) from None
            return self._client

    async def _eval(self, script: str, keys: _ScopeKeys, *arguments: object) -> object:
        client = await self._get_client()
        try:
            return await client.eval(
                script,
                3,
                keys.sequence,
                keys.stream,
                keys.terminal,
                *arguments,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise DependencyError(_BROKER_UNAVAILABLE) from None

    async def publish(
        self,
        scope: ChatSessionScope,
        *,
        event_type: str,
        data: str,
        terminal: bool = False,
    ) -> SseEvent:
        _validate_scope(scope)
        _validate_event(event_type, data, terminal)
        await self._require_open()
        keys = _keys_for_scope(scope, self._config)
        raw = await self._eval(
            _PUBLISH_SCRIPT,
            keys,
            event_type,
            data,
            "1" if terminal else "0",
            self._config.replay_capacity,
            self._config.retention_seconds,
            str(_MAX_EVENT_ID),
        )
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes, bytearray))
            or len(raw) != 2
        ):
            raise DependencyError(_BROKER_UNAVAILABLE)
        status = _decode_text(raw[0])
        value = _decode_text(raw[1])
        if status == "CONFLICT":
            if value == "EXHAUSTED":
                raise _stream_conflict("Event stream identifier exhausted")
            if value == "TERMINATED":
                raise _stream_conflict()
            raise DependencyError(_BROKER_UNAVAILABLE)
        if status != "OK":
            raise DependencyError(_BROKER_UNAVAILABLE)
        event_id = _parse_decimal(value)
        if event_id is None or event_id < 1:
            raise DependencyError(_BROKER_UNAVAILABLE)
        return SseEvent(
            event_id=event_id,
            event_type=event_type,
            data=data,
            terminal=terminal,
        )

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

    async def _snapshot(
        self,
        keys: _ScopeKeys,
    ) -> tuple[int, int | None, tuple[SseEvent, ...]]:
        raw = await self._eval(
            _SNAPSHOT_SCRIPT,
            keys,
            self._config.replay_capacity + 1,
        )
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes, bytearray))
            or len(raw) != 3
        ):
            raise DependencyError(_BROKER_UNAVAILABLE)
        current = _parse_decimal(raw[0], allow_empty=True) or 0
        terminal_id = _parse_decimal(raw[1], allow_empty=True)
        events = _parse_entries(raw[2], maximum=self._config.replay_capacity)
        if current == 0:
            if events or terminal_id is not None:
                raise DependencyError(_BROKER_UNAVAILABLE)
            return current, terminal_id, events
        if not events or events[-1].event_id != current:
            raise DependencyError(_BROKER_UNAVAILABLE)
        if terminal_id is None:
            if events[-1].terminal:
                raise DependencyError(_BROKER_UNAVAILABLE)
        elif (
            terminal_id != current
            or not events[-1].terminal
            or events[-1].event_id != terminal_id
        ):
            raise DependencyError(_BROKER_UNAVAILABLE)
        return current, terminal_id, events

    async def subscribe(
        self,
        scope: ChatSessionScope,
        *,
        last_event_id: int | None = None,
    ) -> RedisSseSubscription:
        _validate_scope(scope)
        _validate_last_event_id(last_event_id)
        await self._require_open()
        keys = _keys_for_scope(scope, self._config)
        current, terminal_id, history = await self._snapshot(keys)
        if last_event_id is not None and last_event_id > current:
            if current == 0:
                raise _stream_conflict("Event replay window expired")
            raise ValueError("invalid Last-Event-ID")
        replay_after = 0 if last_event_id is None else last_event_id
        replay: tuple[SseEvent, ...] = ()
        if replay_after < current:
            if not history:
                raise _stream_conflict("Event replay window expired")
            oldest = cast(int, history[0].event_id)
            if replay_after < oldest - 1:
                raise _stream_conflict("Event replay window expired")
            replay = tuple(
                event
                for event in history
                if cast(int, event.event_id) > replay_after
            )
            if len(replay) > self._config.max_replay_events:
                raise _stream_conflict("Event replay window expired")
        if terminal_id is not None and (not replay or not replay[-1].terminal):
            raise _stream_conflict("Event stream already terminated")
        subscription = RedisSseSubscription(
            self,
            scope,
            keys,
            replay,
            cursor=current,
        )
        async with self._state_lock:
            if self._closed:
                raise DependencyError(_BROKER_UNAVAILABLE)
            subscribers = self._subscriptions.setdefault(keys.stream, set())
            if len(subscribers) >= self._config.max_subscribers_per_session:
                if not subscribers:
                    self._subscriptions.pop(keys.stream, None)
                raise _stream_conflict("Event stream subscriber capacity reached")
            subscribers.add(subscription)
        return subscription

    async def _read(self, stream_key: str, cursor: int) -> tuple[SseEvent, ...]:
        client = await self._get_client()
        try:
            raw = await client.xread(
                {stream_key: f"{cursor}-0"},
                count=self._config.read_batch_size,
                block=max(1, round(self._config.heartbeat_seconds * 1_000)),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise DependencyError(_BROKER_UNAVAILABLE) from None
        if raw is None or raw == [] or raw == ():
            return ()
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes, bytearray))
            or len(raw) != 1
        ):
            raise DependencyError(_BROKER_UNAVAILABLE)
        stream = raw[0]
        if (
            not isinstance(stream, Sequence)
            or isinstance(stream, (str, bytes, bytearray))
            or len(stream) != 2
            or _decode_text(stream[0]) != stream_key
        ):
            raise DependencyError(_BROKER_UNAVAILABLE)
        events = _parse_entries(stream[1], maximum=self._config.read_batch_size)
        if any(cast(int, event.event_id) <= cursor for event in events):
            raise DependencyError(_BROKER_UNAVAILABLE)
        return events

    async def _unsubscribe(
        self,
        subscription: RedisSseSubscription,
        *,
        error: AppError | None = None,
    ) -> None:
        async with self._state_lock:
            subscribers = self._subscriptions.get(subscription._keys.stream)
            if subscribers is not None:
                subscribers.discard(subscription)
                if not subscribers:
                    self._subscriptions.pop(subscription._keys.stream, None)
            subscription._detach(error)

    async def subscriber_count(self, scope: ChatSessionScope) -> int:
        keys = _keys_for_scope(scope, self._config)
        async with self._state_lock:
            return len(self._subscriptions.get(keys.stream, ()))

    async def probe(self) -> None:
        await self._require_open()
        client = await self._get_client()
        try:
            result = await client.ping()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise DependencyError(_BROKER_UNAVAILABLE) from None
        if not (
            result is True
            or (isinstance(result, str) and result == "PONG")
            or (isinstance(result, bytes) and result == b"PONG")
        ):
            raise DependencyError(_BROKER_UNAVAILABLE)

    async def aclose(self) -> None:
        async with self._state_lock:
            if self._closed:
                return
            self._closed = True
            subscriptions = tuple(
                subscription
                for subscribers in self._subscriptions.values()
                for subscription in subscribers
            )
            self._subscriptions.clear()
        unavailable = DependencyError(_BROKER_UNAVAILABLE)
        for subscription in subscriptions:
            subscription._detach(unavailable)
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None and self._manage_client:
            try:
                await client.aclose()
            except asyncio.CancelledError:
                raise
            except Exception:
                raise DependencyError(_BROKER_UNAVAILABLE) from None


__all__ = [
    "AsyncRedisStreamsClient",
    "RedisClientFactory",
    "RedisSseSubscription",
    "RedisStreamsBrokerConfig",
    "RedisStreamsSseBroker",
]
