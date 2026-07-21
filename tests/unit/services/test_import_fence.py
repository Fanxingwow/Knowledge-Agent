from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.core.errors import AppError, ErrorCode
from app.infrastructure.minio_store import build_object_ref
from app.services.import_fence import RegistryCommitFence
from app.services.import_registry import InMemoryImportTaskRegistry, NewImportTask


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


def _new_task() -> NewImportTask:
    task_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    source_ref = build_object_ref(
        "tenant-a",
        "documents",
        uuid.UUID(document_id).hex,
        "imports",
        uuid.UUID(task_id).hex,
        "source.md",
    )
    return NewImportTask(
        task_id=task_id,
        document_id=document_id,
        tenant_id="tenant-a",
        owner_subject="owner-a",
        source_ref=source_ref.uri,
        original_name="guide.md",
        content_type="text/markdown",
        size=5,
        sha256="a" * 64,
    )


def test_fence_exposes_only_stable_identity_and_survives_heartbeat() -> None:
    clock = MutableClock(datetime(2026, 7, 16, tzinfo=UTC))
    registry = InMemoryImportTaskRegistry(clock=clock)
    task = _new_task()
    registry.create_many((task,))
    lease = registry.claim_next("worker-a", lease_seconds=30, max_attempts=3)
    assert lease is not None
    fence = RegistryCommitFence(registry, lease)

    assert fence.task_id == task.task_id
    assert fence.document_id == task.document_id
    assert fence.tenant_id == task.tenant_id
    assert fence.import_version == task.task_id
    assert fence.attempt_count == 1
    assert len(fence.fence_id) == 64
    assert lease.lease_token not in fence.fence_id
    assert copy.deepcopy(fence) is fence

    fence.assert_current()
    clock.advance(5)
    renewed = registry.renew_lease(lease, lease_seconds=30)
    fence.assert_current()
    assert renewed.expires_at > lease.expires_at


def test_fence_fails_after_attempt_is_requeued() -> None:
    registry = InMemoryImportTaskRegistry()
    task = _new_task()
    registry.create_many((task,))
    lease = registry.claim_next("worker-a", lease_seconds=30, max_attempts=3)
    assert lease is not None
    fence = RegistryCommitFence(registry, lease)

    registry.retry(
        lease,
        error_code=ErrorCode.DEPENDENCY_UNAVAILABLE.value,
        delay_seconds=0,
    )
    replacement = registry.claim_next("worker-b", lease_seconds=30, max_attempts=3)
    assert replacement is not None

    with pytest.raises(AppError) as stale:
        fence.assert_current()
    assert stale.value.code is ErrorCode.RESOURCE_CONFLICT


def test_fence_fails_after_lease_expiry() -> None:
    clock = MutableClock(datetime(2026, 7, 16, tzinfo=UTC))
    registry = InMemoryImportTaskRegistry(clock=clock)
    task = _new_task()
    registry.create_many((task,))
    lease = registry.claim_next("worker-a", lease_seconds=1, max_attempts=3)
    assert lease is not None
    fence = RegistryCommitFence(registry, lease)

    clock.advance(1)

    with pytest.raises(AppError) as expired:
        fence.assert_current()
    assert expired.value.code is ErrorCode.RESOURCE_CONFLICT
