from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pytest

from app.core.errors import AppError, ErrorCode
from app.infrastructure.minio_store import build_object_ref
from app.services.import_registry import (
    ImportTaskLease,
    ImportTaskStatus,
    InMemoryImportTaskRegistry,
    NewImportTask,
)


@dataclass
class MutableClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def new_task(
    *,
    task_id: str | None = None,
    document_id: str | None = None,
    tenant_id: str = "tenant-a",
    owner_subject: str = "user-a",
) -> NewImportTask:
    resolved_task_id = task_id or str(uuid.uuid4())
    resolved_document_id = document_id or str(uuid.uuid4())
    ref = build_object_ref(
        tenant_id,
        "documents",
        uuid.UUID(resolved_document_id).hex,
        "imports",
        uuid.UUID(resolved_task_id).hex,
        "source.md",
    )
    return NewImportTask(
        task_id=resolved_task_id,
        document_id=resolved_document_id,
        tenant_id=tenant_id,
        owner_subject=owner_subject,
        source_ref=ref.uri,
        original_name="guide.md",
        content_type="text/markdown",
        size=5,
        sha256="a" * 64,
    )


def test_create_many_is_atomic_and_returns_immutable_snapshots() -> None:
    registry = InMemoryImportTaskRegistry()
    first = new_task()
    second = new_task()

    snapshots = registry.create_many([first, second])

    assert tuple(item.task_id for item in snapshots) == (first.task_id, second.task_id)
    assert all(item.status is ImportTaskStatus.QUEUED for item in snapshots)
    assert snapshots[0].completed_steps == ()
    assert isinstance(snapshots[0].completed_steps, tuple)


def test_duplicate_batch_does_not_create_any_record() -> None:
    registry = InMemoryImportTaskRegistry()
    item = new_task()

    with pytest.raises(AppError) as captured:
        registry.create_many([item, item])

    assert captured.value.code is ErrorCode.RESOURCE_CONFLICT
    with pytest.raises(AppError) as missing:
        registry.get_owned(item.task_id, item.tenant_id, item.owner_subject)
    assert missing.value.code is ErrorCode.RESOURCE_NOT_FOUND


@pytest.mark.parametrize(
    ("tenant_id", "owner_subject"),
    [("tenant-b", "user-a"), ("tenant-a", "user-b")],
)
def test_cross_tenant_and_cross_user_tasks_are_indistinguishable_from_missing(
    tenant_id: str,
    owner_subject: str,
) -> None:
    registry = InMemoryImportTaskRegistry()
    item = new_task()
    registry.create_many([item])

    with pytest.raises(AppError) as captured:
        registry.get_owned(item.task_id, tenant_id, owner_subject)

    assert captured.value.code is ErrorCode.RESOURCE_NOT_FOUND
    assert str(captured.value) == "Import task not found"


def test_valid_lifecycle_records_steps_and_commit_count() -> None:
    registry = InMemoryImportTaskRegistry()
    item = new_task()
    registry.create_many([item])

    registry.begin(item.task_id, item.tenant_id, item.owner_subject)
    registry.start_step(item.task_id, item.tenant_id, item.owner_subject, "parse_document")
    registry.complete_step(item.task_id, item.tenant_id, item.owner_subject, "parse_document")
    completed = registry.complete(
        item.task_id,
        item.tenant_id,
        item.owner_subject,
        imported_chunk_count=12,
        commit_token="milvus-revision-12",  # noqa: S106 - opaque revision, not a credential
    )

    assert completed.status is ImportTaskStatus.COMPLETED
    assert completed.completed_steps == ("parse_document",)
    assert completed.imported_chunk_count == 12
    assert completed.commit_token == "milvus-revision-12"  # noqa: S105 - opaque revision
    assert completed.error_code is None


def test_terminal_task_cannot_be_changed() -> None:
    registry = InMemoryImportTaskRegistry()
    item = new_task()
    registry.create_many([item])
    registry.begin(item.task_id, item.tenant_id, item.owner_subject)
    registry.complete(
        item.task_id,
        item.tenant_id,
        item.owner_subject,
        imported_chunk_count=0,
        commit_token="milvus-revision-0",  # noqa: S106 - opaque revision, not a credential
    )

    with pytest.raises(AppError) as captured:
        registry.fail(
            item.task_id,
            item.tenant_id,
            item.owner_subject,
            error_code=ErrorCode.INTERNAL_ERROR.value,
        )

    assert captured.value.code is ErrorCode.RESOURCE_CONFLICT


def test_failed_transition_is_idempotent() -> None:
    registry = InMemoryImportTaskRegistry()
    item = new_task()
    registry.create_many([item])

    first = registry.fail(
        item.task_id,
        item.tenant_id,
        item.owner_subject,
        error_code=ErrorCode.DEPENDENCY_UNAVAILABLE.value,
    )
    second = registry.fail(
        item.task_id,
        item.tenant_id,
        item.owner_subject,
        error_code=ErrorCode.INTERNAL_ERROR.value,
    )

    assert first == second
    assert second.error_code == ErrorCode.DEPENDENCY_UNAVAILABLE.value


def test_rollback_is_atomic_when_any_task_has_started() -> None:
    registry = InMemoryImportTaskRegistry()
    first = new_task()
    second = new_task()
    registry.create_many([first, second])
    registry.begin(second.task_id, second.tenant_id, second.owner_subject)

    with pytest.raises(AppError) as captured:
        registry.rollback_queued(first.tenant_id, first.owner_subject, [first.task_id, second.task_id])

    assert captured.value.code is ErrorCode.RESOURCE_CONFLICT
    assert registry.get_owned(first.task_id, first.tenant_id, first.owner_subject).status is ImportTaskStatus.QUEUED
    second_snapshot = registry.get_owned(second.task_id, second.tenant_id, second.owner_subject)
    assert second_snapshot.status is ImportTaskStatus.PROCESSING


def test_only_one_concurrent_worker_can_begin_a_task() -> None:
    registry = InMemoryImportTaskRegistry()
    item = new_task()
    registry.create_many([item])

    def begin() -> ErrorCode | None:
        try:
            registry.begin(item.task_id, item.tenant_id, item.owner_subject)
        except AppError as exc:
            return exc.code
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: begin(), range(8)))

    assert results.count(None) == 1
    assert results.count(ErrorCode.RESOURCE_CONFLICT) == 7


def test_only_one_worker_can_claim_and_stale_fencing_token_cannot_commit() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    registry = InMemoryImportTaskRegistry(clock=clock)
    item = new_task()
    registry.create_many([item])

    def claim(index: int) -> ImportTaskLease | None:
        return registry.claim_next(
            f"worker-{index}",
            lease_seconds=30,
            max_attempts=3,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        leases = [lease for lease in executor.map(claim, range(8)) if lease is not None]

    assert len(leases) == 1
    first = leases[0]
    clock.advance(30)
    second = registry.claim_next("replacement", lease_seconds=30, max_attempts=3)
    assert second is not None
    assert second.lease_token != first.lease_token

    with pytest.raises(AppError) as stale:
        registry.complete(
            item.task_id,
            item.tenant_id,
            item.owner_subject,
            imported_chunk_count=1,
            commit_token="old-worker-commit",  # noqa: S106 - opaque index revision
            lease_token=first.lease_token,
        )
    assert stale.value.code is ErrorCode.RESOURCE_CONFLICT

    completed = registry.complete(
        item.task_id,
        item.tenant_id,
        item.owner_subject,
        imported_chunk_count=1,
        commit_token="new-worker-commit",  # noqa: S106 - opaque index revision
        lease_token=second.lease_token,
    )
    assert completed.status is ImportTaskStatus.COMPLETED


def test_lease_renewal_and_retry_are_cas_guarded() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    registry = InMemoryImportTaskRegistry(clock=clock)
    item = new_task()
    registry.create_many([item])
    lease = registry.claim_next("worker", lease_seconds=30, max_attempts=3)
    assert lease is not None

    clock.advance(10)
    renewed = registry.renew_lease(lease, lease_seconds=30)
    assert renewed.expires_at == clock.value + timedelta(seconds=30)
    with pytest.raises(AppError):
        registry.renew_lease(lease, lease_seconds=30)

    queued = registry.retry(
        renewed,
        error_code=ErrorCode.DEPENDENCY_UNAVAILABLE.value,
        delay_seconds=20,
    )
    assert queued.status is ImportTaskStatus.QUEUED
    assert queued.available_at == clock.value + timedelta(seconds=20)
    assert registry.claim_next("too-early", lease_seconds=30, max_attempts=3) is None
    clock.advance(20)
    retried = registry.claim_next("retry", lease_seconds=30, max_attempts=3)
    assert retried is not None
    assert retried.attempt_count == 2


@pytest.mark.parametrize(
    "change",
    [
        {"task_id": "not-a-uuid"},
        {"owner_subject": ""},
        {"size": 0},
        {"sha256": "not-a-digest"},
        {"original_name": "x\nsecret.md"},
    ],
)
def test_invalid_generated_records_are_rejected_before_mutation(change: dict[str, object]) -> None:
    registry = InMemoryImportTaskRegistry()
    item = new_task()

    with pytest.raises((AppError, ValueError)):
        registry.create_many([replace(item, **change)])  # type: ignore[arg-type]
