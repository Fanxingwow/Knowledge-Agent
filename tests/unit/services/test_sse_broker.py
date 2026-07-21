from __future__ import annotations

import asyncio

import pytest

from app.core.errors import AppError, DependencyError, ErrorCode
from app.services.chat_history import ChatSessionScope
from app.services.sse_broker import InMemorySseBroker, SseEvent, parse_last_event_id

_SESSION_ID = "00000000-0000-4000-8000-000000000001"


def scope(
    tenant_id: str = "tenant-a",
    owner_subject: str = "user:42",
) -> ChatSessionScope:
    return ChatSessionScope(tenant_id, owner_subject, _SESSION_ID)


@pytest.mark.parametrize(
    ("header", "expected"),
    [(None, None), ("0", 0), ("42", 42), ("9223372036854775807", 9_223_372_036_854_775_807)],
)
def test_last_event_id_parser_accepts_only_bounded_decimal(
    header: str | None,
    expected: int | None,
) -> None:
    assert parse_last_event_id(header) == expected


@pytest.mark.parametrize("header", ["", "-1", "+1", "1.0", " 1", "１", "9223372036854775808"])
def test_last_event_id_parser_rejects_ambiguous_values(header: str) -> None:
    with pytest.raises(ValueError, match="Last-Event-ID"):
        parse_last_event_id(header)


def test_event_encoding_is_sse_safe_for_multiline_data() -> None:
    event = SseEvent(
        event_id=7,
        event_type="answer.delta",
        data="first\r\nsecond\nthird",
    )

    assert event.encode() == ("id: 7\nevent: answer.delta\ndata: first\ndata: second\ndata: third\n\n")
    assert SseEvent.heartbeat_event().encode() == ": heartbeat\n\n"


@pytest.mark.asyncio
async def test_streams_are_bound_to_tenant_owner_and_session_together() -> None:
    broker = InMemorySseBroker(heartbeat_seconds=0.01)
    tenant_a = scope()
    tenant_b = scope(tenant_id="tenant-b")
    other_owner = scope(owner_subject="user:other")
    subscription_a = await broker.subscribe(tenant_a)
    subscription_b = await broker.subscribe(tenant_b)
    subscription_owner = await broker.subscribe(other_owner)

    event_a = await broker.publish(tenant_a, event_type="answer.delta", data="A")
    event_b = await broker.publish(tenant_b, event_type="answer.delta", data="B")
    event_owner = await broker.publish(other_owner, event_type="answer.delta", data="C")

    assert await subscription_a.receive() == event_a
    assert await subscription_b.receive() == event_b
    assert await subscription_owner.receive() == event_owner
    assert event_a.event_id == event_b.event_id == event_owner.event_id == 1
    await subscription_a.aclose()
    await subscription_b.aclose()
    await subscription_owner.aclose()


@pytest.mark.asyncio
async def test_event_ids_are_monotonic_and_replay_respects_last_event_id() -> None:
    broker = InMemorySseBroker(replay_capacity=4, max_replay_events=3)
    owned = scope()
    events = [await broker.publish(owned, event_type="answer.delta", data=str(number)) for number in range(1, 4)]

    subscription = await broker.subscribe(owned, last_event_id=1)

    assert [events[0].event_id, events[1].event_id, events[2].event_id] == [1, 2, 3]
    assert await subscription.receive() == events[1]
    assert await subscription.receive() == events[2]
    await subscription.aclose()


@pytest.mark.asyncio
async def test_initial_subscription_replays_bounded_preconnect_events() -> None:
    broker = InMemorySseBroker(replay_capacity=2, max_replay_events=2)
    owned = scope()
    published = await broker.publish(owned, event_type="ready", data="{}")

    subscription = await broker.subscribe(owned)

    assert await subscription.receive() == published
    await subscription.aclose()


@pytest.mark.asyncio
async def test_replay_window_and_replay_batch_are_bounded_fail_closed() -> None:
    owned = scope()
    expired = InMemorySseBroker(replay_capacity=2, max_replay_events=2)
    for number in range(3):
        await expired.publish(owned, event_type="answer.delta", data=str(number))

    with pytest.raises(AppError) as old_cursor:
        await expired.subscribe(owned, last_event_id=0)
    assert old_cursor.value.code is ErrorCode.RESOURCE_CONFLICT
    assert old_cursor.value.message == "Event replay window expired"

    oversized = InMemorySseBroker(replay_capacity=4, max_replay_events=1)
    await oversized.publish(owned, event_type="answer.delta", data="one")
    await oversized.publish(owned, event_type="answer.delta", data="two")
    with pytest.raises(AppError, match="replay window"):
        await oversized.subscribe(owned, last_event_id=0)


@pytest.mark.asyncio
async def test_subscription_emits_heartbeat_without_advancing_event_id() -> None:
    broker = InMemorySseBroker(heartbeat_seconds=0.01)
    owned = scope()
    subscription = await broker.subscribe(owned)

    heartbeat = await subscription.receive()
    event = await broker.publish(owned, event_type="ready", data="{}")

    assert heartbeat.heartbeat is True
    assert heartbeat.event_id is None
    assert event.event_id == 1
    assert await subscription.receive() == event
    await subscription.aclose()


@pytest.mark.asyncio
async def test_slow_consumer_is_detached_instead_of_silently_dropping_events() -> None:
    broker = InMemorySseBroker(subscriber_queue_capacity=1)
    owned = scope()
    slow = await broker.subscribe(owned)
    fast = await broker.subscribe(owned)

    first = await broker.publish(owned, event_type="answer.delta", data="one")
    assert await fast.receive() == first
    second = await broker.publish(owned, event_type="answer.delta", data="two")

    assert await fast.receive() == second
    with pytest.raises(AppError) as raised:
        await slow.receive()
    assert raised.value.code is ErrorCode.RESOURCE_CONFLICT
    assert raised.value.message == "Event stream consumer fell behind"
    assert await broker.subscriber_count(owned) == 1
    await fast.aclose()


@pytest.mark.asyncio
async def test_terminal_event_discards_stream_after_last_live_subscriber_exits() -> None:
    broker = InMemorySseBroker(replay_capacity=4, max_replay_events=4)
    owned = scope()
    live = await broker.subscribe(owned)
    delta = await broker.publish(owned, event_type="answer.delta", data="hello")
    terminal = await broker.terminate(owned, event_type="answer.complete", data='{"ok":true}')

    assert await live.receive() == delta
    assert await live.receive() == terminal
    assert terminal.terminal is True
    with pytest.raises(StopAsyncIteration):
        await live.receive()
    assert await broker.subscriber_count(owned) == 0
    assert await broker.stream_count() == 0


@pytest.mark.asyncio
async def test_preterminated_stream_remains_replayable_until_last_subscriber_exits() -> None:
    broker = InMemorySseBroker(replay_capacity=4, max_replay_events=4)
    owned = scope()
    delta = await broker.publish(owned, event_type="answer.delta", data="hello")
    terminal = await broker.terminate(owned, event_type="answer.complete", data='{"ok":true}')

    assert await broker.stream_count() == 1
    replay = await broker.subscribe(owned, last_event_id=0)
    assert await replay.receive() == delta
    assert await replay.receive() == terminal
    assert await broker.stream_count() == 0


@pytest.mark.asyncio
async def test_terminal_stream_waits_for_every_subscriber_before_discard() -> None:
    broker = InMemorySseBroker()
    owned = scope()
    first = await broker.subscribe(owned)
    second = await broker.subscribe(owned)
    terminal = await broker.terminate(owned, event_type="answer.complete", data="{}")

    assert await first.receive() == terminal
    assert await broker.subscriber_count(owned) == 1
    assert await broker.stream_count() == 1
    await second.aclose()
    assert await broker.stream_count() == 0


@pytest.mark.asyncio
async def test_terminal_backpressure_discards_after_last_slow_subscriber_is_detached() -> None:
    broker = InMemorySseBroker(subscriber_queue_capacity=1)
    owned = scope()
    slow = await broker.subscribe(owned)
    await broker.publish(owned, event_type="answer.delta", data="queued")

    await broker.terminate(owned, event_type="answer.complete", data="{}")

    with pytest.raises(AppError, match="consumer fell behind"):
        await slow.receive()
    assert await broker.stream_count() == 0


@pytest.mark.asyncio
async def test_nonterminal_disconnect_preserves_replay_state() -> None:
    broker = InMemorySseBroker(replay_capacity=4, max_replay_events=4)
    owned = scope()
    disconnected = await broker.subscribe(owned)
    event = await broker.publish(owned, event_type="answer.delta", data="keep")

    await disconnected.aclose()

    assert await broker.stream_count() == 1
    replay = await broker.subscribe(owned, last_event_id=0)
    assert await replay.receive() == event
    await replay.aclose()
    assert await broker.stream_count() == 1


@pytest.mark.asyncio
async def test_explicit_terminal_discard_is_idempotent_after_auto_cleanup() -> None:
    broker = InMemorySseBroker(replay_capacity=4, max_replay_events=4)
    owned = scope()
    live = await broker.subscribe(owned)
    terminal = await broker.terminate(owned, event_type="answer.complete", data="{}")

    assert await live.receive() == terminal
    await broker.discard_terminal(owned)
    assert await broker.stream_count() == 0


@pytest.mark.asyncio
async def test_explicit_discard_releases_preterminated_stream_without_subscribers() -> None:
    broker = InMemorySseBroker(replay_capacity=4, max_replay_events=4)
    owned = scope()
    await broker.terminate(owned, event_type="answer.complete", data="{}")

    await broker.discard_terminal(owned)
    assert await broker.stream_count() == 0


@pytest.mark.asyncio
async def test_cancellation_and_explicit_disconnect_remove_subscription() -> None:
    broker = InMemorySseBroker(heartbeat_seconds=30)
    owned = scope()
    cancelled = await broker.subscribe(owned)
    waiting = asyncio.create_task(cancelled.receive())
    await asyncio.sleep(0)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert await broker.subscriber_count(owned) == 0

    explicit = await broker.subscribe(owned)
    assert await broker.subscriber_count(owned) == 1
    await explicit.aclose()
    assert await broker.subscriber_count(owned) == 0


@pytest.mark.asyncio
async def test_session_and_subscriber_capacities_are_bounded() -> None:
    broker = InMemorySseBroker(max_sessions=1, max_subscribers_per_session=1)
    first = await broker.subscribe(scope())

    with pytest.raises(AppError, match="subscriber capacity"):
        await broker.subscribe(scope())
    with pytest.raises(AppError, match="stream capacity"):
        await broker.subscribe(scope(tenant_id="tenant-b"))

    await first.aclose()


@pytest.mark.asyncio
async def test_broker_shutdown_wakes_subscribers_and_fails_closed() -> None:
    broker = InMemorySseBroker(heartbeat_seconds=30)
    owned = scope()
    subscription = await broker.subscribe(owned)
    waiting = asyncio.create_task(subscription.receive())
    await asyncio.sleep(0)

    await broker.aclose()

    with pytest.raises(DependencyError, match="broker is unavailable"):
        await waiting
    with pytest.raises(DependencyError, match="broker is unavailable"):
        await broker.subscribe(owned)
    with pytest.raises(DependencyError, match="broker is unavailable"):
        await broker.publish(owned, event_type="ready", data="{}")
