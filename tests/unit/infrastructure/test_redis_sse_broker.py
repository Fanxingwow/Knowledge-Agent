from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import pytest

from app.core.errors import AppError, DependencyError, ErrorCode
from app.infrastructure.redis_sse_broker import (
    AsyncRedisStreamsClient,
    RedisStreamsBrokerConfig,
    RedisStreamsSseBroker,
)
from app.services.chat_history import ChatSessionScope
from app.services.sse_broker import SessionEventBroker, SseEvent

_SESSION_ID = "00000000-0000-4000-8000-000000000001"
_OTHER_SESSION_ID = "00000000-0000-4000-8000-000000000002"
_UNSET = object()


def scope(
    tenant_id: str = "tenant-a",
    owner_subject: str = "user:42",
    session_id: str = _SESSION_ID,
) -> ChatSessionScope:
    return ChatSessionScope(tenant_id, owner_subject, session_id)


def config(**overrides: object) -> RedisStreamsBrokerConfig:
    values: dict[str, object] = {
        "url": "rediss://operator:credential@redis.internal:6380/0",  # pragma: allowlist secret
        "key_hmac_secret": b"k" * 32,
        "key_prefix": "test:sse:v1",
        "replay_capacity": 4,
        "max_replay_events": 4,
        "read_batch_size": 4,
        "max_subscribers_per_session": 4,
        "heartbeat_seconds": 0.01,
        "retention_seconds": 600,
        "socket_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return RedisStreamsBrokerConfig(**values)  # type: ignore[arg-type]


@dataclass
class _FakeStream:
    sequence: int = 0
    terminal: int | None = None
    entries: list[tuple[bytes, dict[bytes, bytes]]] = field(default_factory=list)


class FakeRedis:
    def __init__(self) -> None:
        self.streams: dict[str, _FakeStream] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail_operation: str | None = None
        self.publish_override: object = _UNSET
        self.snapshot_override: object = _UNSET
        self.xread_override: object = _UNSET
        self.ping_result: object = True
        self.closed = False
        self._condition = asyncio.Condition()
        self.xread_started = asyncio.Event()

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode()
        assert isinstance(value, str)
        return value

    def _fail(self, operation: str) -> None:
        if self.fail_operation == operation:
            raise RuntimeError(
                "rediss://admin:secret@private-redis/0 internal dependency detail"  # pragma: allowlist secret
            )

    async def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: object,
    ) -> object:
        self._fail("eval")
        assert numkeys == 3
        sequence_key, stream_key, terminal_key = (
            self._text(value) for value in keys_and_args[:3]
        )
        assert sequence_key.endswith(":sequence")
        assert stream_key.endswith(":events")
        assert terminal_key.endswith(":terminal")
        arguments = keys_and_args[3:]
        if "XADD" in script:
            self.calls.append(
                (
                    "publish",
                    {
                        "keys": keys_and_args[:3],
                        "arguments": arguments,
                    },
                )
            )
            if self.publish_override is not _UNSET:
                return self.publish_override
            event_type = self._text(arguments[0])
            data = self._text(arguments[1])
            terminal = self._text(arguments[2]) == "1"
            maxlen = int(cast(int | str, arguments[3]))
            maximum_id = int(cast(str, arguments[5]))
            async with self._condition:
                state = self.streams.setdefault(stream_key, _FakeStream())
                if state.terminal is not None:
                    return [b"CONFLICT", b"TERMINATED"]
                if state.sequence == maximum_id:
                    return [b"CONFLICT", b"EXHAUSTED"]
                state.sequence += 1
                event_id = state.sequence
                state.entries.append(
                    (
                        f"{event_id}-0".encode(),
                        {
                            b"event_type": event_type.encode(),
                            b"data": data.encode(),
                            b"terminal": b"1" if terminal else b"0",
                        },
                    )
                )
                del state.entries[:-maxlen]
                if terminal:
                    state.terminal = event_id
                self._condition.notify_all()
                return [b"OK", str(event_id).encode()]
        self.calls.append(
            (
                "snapshot",
                {
                    "keys": keys_and_args[:3],
                    "arguments": arguments,
                },
            )
        )
        if self.snapshot_override is not _UNSET:
            return self.snapshot_override
        count = int(cast(int | str, arguments[0]))
        async with self._condition:
            state = self.streams.get(stream_key, _FakeStream())
            return [
                str(state.sequence).encode() if state.sequence else b"",
                str(state.terminal).encode() if state.terminal is not None else b"",
                list(state.entries[:count]),
            ]

    async def xread(
        self,
        streams: Mapping[str, str],
        *,
        count: int,
        block: int,
    ) -> object:
        self._fail("xread")
        self.calls.append(
            (
                "xread",
                {"streams": dict(streams), "count": count, "block": block},
            )
        )
        self.xread_started.set()
        if self.xread_override is not _UNSET:
            return self.xread_override
        stream_key, raw_cursor = next(iter(streams.items()))
        cursor = int(raw_cursor.split("-", 1)[0])

        def available() -> list[tuple[bytes, dict[bytes, bytes]]]:
            state = self.streams.get(stream_key, _FakeStream())
            return [
                entry
                for entry in state.entries
                if int(entry[0].split(b"-", 1)[0]) > cursor
            ][:count]

        async with self._condition:
            entries = available()
            if not entries and not self.closed:
                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=block / 1_000,
                    )
                except TimeoutError:
                    return []
                entries = available()
            if self.closed:
                raise RuntimeError("private closed Redis client")
            return [] if not entries else [(stream_key.encode(), entries)]

    async def ping(self) -> object:
        self._fail("ping")
        self.calls.append(("ping", {}))
        return self.ping_result

    async def aclose(self) -> object:
        self._fail("aclose")
        self.calls.append(("aclose", {}))
        async with self._condition:
            self.closed = True
            self._condition.notify_all()
        return None


def as_broker(value: SessionEventBroker) -> SessionEventBroker:
    return value


def test_scope_key_is_complete_hmac_opaque_and_collision_resistant() -> None:
    broker = RedisStreamsSseBroker(config(), client=FakeRedis())
    owned = scope()
    keys = {
        broker.stream_key(owned),
        broker.stream_key(scope(tenant_id="tenant-b")),
        broker.stream_key(scope(owner_subject="user:other")),
        broker.stream_key(scope(session_id=_OTHER_SESSION_ID)),
    }

    assert len(keys) == 4
    key = broker.stream_key(owned)
    assert key == broker.stream_key(owned)
    assert key.startswith("test:sse:v1:") and key.endswith(":events")
    hash_tag = key.split(":")[-2]
    assert hash_tag.startswith("{") and hash_tag.endswith("}")
    assert len(hash_tag[1:-1]) == 64
    assert all(
        raw not in key
        for raw in (owned.tenant_id, owned.owner_subject, owned.session_id)
    )
    other_secret = RedisStreamsSseBroker(
        config(key_hmac_secret=b"z" * 32),
        client=FakeRedis(),
    )
    assert other_secret.stream_key(owned) != key
    assert as_broker(broker) is broker
    assert broker.distributed is True


def test_config_repr_hides_url_credentials_and_hmac_secret() -> None:
    configured = config()
    rendered = repr(configured)

    assert "credential" not in rendered
    assert "rediss://" not in rendered
    assert "kkkk" not in rendered


@pytest.mark.parametrize(
    "overrides",
    [
        {"url": "http://redis.invalid"},
        {"key_hmac_secret": b"short"},
        {"key_prefix": "bad prefix"},
        {"replay_capacity": 0},
        {"max_replay_events": 5},
        {"read_batch_size": 5},
        {"heartbeat_seconds": float("nan")},
        {"connect_timeout_seconds": 0},
        {"connect_timeout_seconds": 31},
        {"socket_timeout_seconds": 0.01},
    ],
)
def test_invalid_config_fails_before_client_access(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Redis event"):
        config(**overrides)


@pytest.mark.asyncio
async def test_factory_is_lazy_probe_is_explicit_and_async_factory_is_supported() -> None:
    client = FakeRedis()
    factory_calls = 0

    async def factory(
        configured: RedisStreamsBrokerConfig,
    ) -> AsyncRedisStreamsClient:
        nonlocal factory_calls
        factory_calls += 1
        assert configured.key_prefix == "test:sse:v1"
        return client

    broker = RedisStreamsSseBroker(config(), client_factory=factory)
    assert factory_calls == 0

    await broker.probe()
    await broker.probe()

    assert factory_calls == 1
    assert [name for name, _ in client.calls].count("ping") == 2
    await broker.aclose()
    assert client.closed is True


@pytest.mark.asyncio
async def test_default_factory_import_is_deferred_and_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_import(name: str) -> object:
        calls.append(name)
        raise ImportError("private Python path")

    monkeypatch.setattr(
        "app.infrastructure.redis_sse_broker.importlib.import_module",
        fail_import,
    )
    broker = RedisStreamsSseBroker(config())
    assert calls == []

    with pytest.raises(DependencyError) as raised:
        await broker.probe()

    assert calls == ["redis.asyncio"]
    assert str(raised.value) == "Event stream broker is unavailable"
    assert "private" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_default_factory_uses_independent_connect_and_socket_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeRedis()
    factory_calls: list[tuple[str, dict[str, object]]] = []

    class RedisType:
        @staticmethod
        def from_url(url: str, **options: object) -> FakeRedis:
            factory_calls.append((url, options))
            return client

    class RedisModule:
        Redis = RedisType

    def import_module(name: str) -> object:
        assert name == "redis.asyncio"
        return RedisModule

    monkeypatch.setattr(
        "app.infrastructure.redis_sse_broker.importlib.import_module",
        import_module,
    )
    configured = config(
        connect_timeout_seconds=2.5,
        socket_timeout_seconds=7.5,
    )
    broker = RedisStreamsSseBroker(configured)
    assert factory_calls == []

    await broker.probe()

    assert factory_calls == [
        (
            configured.url,
            {
                "decode_responses": False,
                "socket_connect_timeout": 2.5,
                "socket_timeout": 7.5,
            },
        )
    ]
    await broker.aclose()


@pytest.mark.asyncio
async def test_atomic_ids_are_monotonic_per_scope_and_stream_is_exactly_bounded() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(config(replay_capacity=5), client=client)
    owned = scope()

    events = await asyncio.gather(
        *(
            broker.publish(
                owned,
                event_type="answer.delta",
                data=str(number),
            )
            for number in range(20)
        )
    )

    assert sorted(cast(int, event.event_id) for event in events) == list(range(1, 21))
    state = client.streams[broker.stream_key(owned)]
    assert len(state.entries) == 5
    assert [entry_id for entry_id, _ in state.entries] == [
        b"16-0",
        b"17-0",
        b"18-0",
        b"19-0",
        b"20-0",
    ]
    publish_call = next(values for name, values in client.calls if name == "publish")
    keys = publish_call["keys"]
    assert isinstance(keys, tuple)
    hash_tags = {
        key.split("{", 1)[1].split("}", 1)[0]
        for key in keys
        if isinstance(key, str)
    }
    assert len(hash_tags) == 1
    arguments = publish_call["arguments"]
    assert isinstance(arguments, tuple)
    assert arguments[3] == 5
    assert arguments[4] == 600
    assert arguments[5] == "9223372036854775807"


@pytest.mark.asyncio
async def test_complete_scopes_have_independent_event_sequences() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(config(), client=client)

    events = [
        await broker.publish(value, event_type="ready", data="{}")
        for value in (
            scope(),
            scope(tenant_id="tenant-b"),
            scope(owner_subject="user:other"),
            scope(session_id=_OTHER_SESSION_ID),
        )
    ]

    assert [event.event_id for event in events] == [1, 1, 1, 1]
    assert len(client.streams) == 4


@pytest.mark.asyncio
async def test_replay_last_event_id_and_heartbeat_match_session_broker_contract() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(config(), client=client)
    owned = scope()
    events = [
        await broker.publish(
            owned,
            event_type="answer.delta",
            data=str(number),
        )
        for number in range(1, 4)
    ]

    subscription = await broker.subscribe(owned, last_event_id=1)

    assert await subscription.receive() == events[1]
    assert await subscription.receive() == events[2]
    heartbeat = await subscription.receive()
    assert heartbeat == SseEvent.heartbeat_event()
    assert heartbeat.event_id is None
    await subscription.aclose()


@pytest.mark.asyncio
async def test_replay_window_and_batch_expiration_fail_closed() -> None:
    client = FakeRedis()
    expired = RedisStreamsSseBroker(
        config(replay_capacity=2, max_replay_events=2, read_batch_size=2),
        client=client,
    )
    owned = scope()
    for number in range(3):
        await expired.publish(
            owned,
            event_type="answer.delta",
            data=str(number),
        )

    with pytest.raises(AppError) as raised:
        await expired.subscribe(owned, last_event_id=0)
    assert raised.value.code is ErrorCode.RESOURCE_CONFLICT
    assert raised.value.message == "Event replay window expired"

    oversized = RedisStreamsSseBroker(
        config(max_replay_events=1),
        client=FakeRedis(),
    )
    await oversized.publish(owned, event_type="answer.delta", data="one")
    await oversized.publish(owned, event_type="answer.delta", data="two")
    with pytest.raises(AppError, match="replay window"):
        await oversized.subscribe(owned, last_event_id=0)


@pytest.mark.asyncio
async def test_last_event_id_validation_and_absent_expired_stream_fail_closed() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(config(), client=client)
    owned = scope()

    for invalid in (-1, True, 9_223_372_036_854_775_808):
        with pytest.raises(ValueError, match="Last-Event-ID"):
            await broker.subscribe(owned, last_event_id=invalid)
    assert client.calls == []

    with pytest.raises(AppError, match="replay window"):
        await broker.subscribe(owned, last_event_id=1)


@pytest.mark.asyncio
async def test_terminal_event_closes_live_and_replay_subscriptions() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(config(), client=client)
    owned = scope()
    live = await broker.subscribe(owned)
    delta = await broker.publish(
        owned,
        event_type="answer.delta",
        data="hello",
    )
    terminal = await broker.terminate(
        owned,
        event_type="answer.complete",
        data='{"ok":true}',
    )

    assert await live.receive() == delta
    assert await live.receive() == terminal
    assert terminal.terminal is True
    with pytest.raises(StopAsyncIteration):
        await live.receive()
    assert await broker.subscriber_count(owned) == 0
    with pytest.raises(AppError, match="state conflict"):
        await broker.publish(owned, event_type="answer.delta", data="late")

    replay = await broker.subscribe(owned, last_event_id=0)
    assert await replay.receive() == delta
    assert await replay.receive() == terminal
    with pytest.raises(StopAsyncIteration):
        await replay.receive()
    with pytest.raises(AppError, match="already terminated"):
        await broker.subscribe(owned, last_event_id=cast(int, terminal.event_id))


@pytest.mark.asyncio
async def test_slow_consumer_gap_is_detected_instead_of_silently_skipped() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(
        config(replay_capacity=2, max_replay_events=2, read_batch_size=2),
        client=client,
    )
    owned = scope()
    slow = await broker.subscribe(owned)
    for number in range(3):
        await broker.publish(
            owned,
            event_type="answer.delta",
            data=str(number),
        )

    with pytest.raises(AppError) as raised:
        await slow.receive()

    assert raised.value.code is ErrorCode.RESOURCE_CONFLICT
    assert raised.value.message == "Event stream consumer fell behind"
    assert await broker.subscriber_count(owned) == 0


@pytest.mark.asyncio
async def test_cancellation_explicit_close_and_capacity_cleanup_subscriptions() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(
        config(
            heartbeat_seconds=30,
            socket_timeout_seconds=31,
            max_subscribers_per_session=1,
        ),
        client=client,
    )
    owned = scope()
    cancelled = await broker.subscribe(owned)
    waiting = asyncio.create_task(cancelled.receive())
    await client.xread_started.wait()

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert await broker.subscriber_count(owned) == 0

    explicit = await broker.subscribe(owned)
    with pytest.raises(AppError, match="subscriber capacity"):
        await broker.subscribe(owned)
    await explicit.aclose()
    assert await broker.subscriber_count(owned) == 0


@pytest.mark.asyncio
async def test_broker_shutdown_wakes_waiter_and_is_terminal() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(
        config(heartbeat_seconds=30, socket_timeout_seconds=31),
        client=client,
    )
    owned = scope()
    subscription = await broker.subscribe(owned)
    waiting = asyncio.create_task(subscription.receive())
    await client.xread_started.wait()

    await broker.aclose()

    with pytest.raises(DependencyError, match="broker is unavailable"):
        await waiting
    assert client.closed is True
    with pytest.raises(DependencyError, match="broker is unavailable"):
        await broker.subscribe(owned)
    with pytest.raises(DependencyError, match="broker is unavailable"):
        await broker.publish(owned, event_type="ready", data="{}")


@pytest.mark.asyncio
async def test_dependency_failures_are_sanitized_and_subscription_is_removed() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(config(), client=client)
    owned = scope()
    client.fail_operation = "eval"

    with pytest.raises(DependencyError) as publish_error:
        await broker.publish(owned, event_type="ready", data="{}")
    assert "secret" not in str(publish_error.value)
    assert publish_error.value.__cause__ is None

    client.fail_operation = None
    subscription = await broker.subscribe(owned)
    client.fail_operation = "xread"
    with pytest.raises(DependencyError) as read_error:
        await subscription.receive()
    assert "secret" not in str(read_error.value)
    assert read_error.value.__cause__ is None
    assert await broker.subscriber_count(owned) == 0

    client.fail_operation = "ping"
    with pytest.raises(DependencyError) as probe_error:
        await broker.probe()
    assert "secret" not in str(probe_error.value)


@pytest.mark.asyncio
async def test_invalid_probe_result_and_close_failure_are_sanitized() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(config(), client=client)
    client.ping_result = []

    with pytest.raises(DependencyError) as probe_error:
        await broker.probe()
    assert str(probe_error.value) == "Event stream broker is unavailable"
    assert probe_error.value.__cause__ is None

    client.fail_operation = "aclose"
    with pytest.raises(DependencyError) as close_error:
        await broker.aclose()
    assert "secret" not in str(close_error.value)
    assert close_error.value.__cause__ is None

    # A failed dependency close must not reopen the broker or repeat side effects.
    await broker.aclose()
    with pytest.raises(DependencyError, match="broker is unavailable"):
        await broker.probe()


@pytest.mark.asyncio
async def test_corrupt_dependency_responses_fail_closed_without_internal_details() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(config(), client=client)
    owned = scope()
    client.publish_override = [b"OK", b"not-an-id"]

    with pytest.raises(DependencyError, match="broker is unavailable"):
        await broker.publish(owned, event_type="ready", data="{}")

    client.publish_override = _UNSET
    client.snapshot_override = [b"2", b"", []]
    with pytest.raises(DependencyError, match="broker is unavailable"):
        await broker.subscribe(owned)


@pytest.mark.asyncio
async def test_invalid_events_never_reach_redis_and_exhaustion_is_conflict() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(config(), client=client)
    owned = scope()

    for event_type, data, terminal in (
        ("heartbeat", "", False),
        ("Bad Type", "data", False),
        ("ready", "bad\x00data", False),
        ("ready", "data", cast(bool, "yes")),
    ):
        with pytest.raises(ValueError):
            await broker.publish(
                owned,
                event_type=event_type,
                data=data,
                terminal=terminal,
            )
    assert client.calls == []

    key = broker.stream_key(owned)
    client.streams[key] = _FakeStream(sequence=9_223_372_036_854_775_806)
    maximum = await broker.publish(owned, event_type="ready", data="{}")
    assert maximum.event_id == 9_223_372_036_854_775_807
    with pytest.raises(AppError, match="identifier exhausted"):
        await broker.publish(owned, event_type="ready", data="{}")


@pytest.mark.asyncio
async def test_unmanaged_injected_client_is_not_closed() -> None:
    client = FakeRedis()
    broker = RedisStreamsSseBroker(
        config(),
        client=client,
        manage_client=False,
    )
    await broker.probe()

    await broker.aclose()

    assert client.closed is False
