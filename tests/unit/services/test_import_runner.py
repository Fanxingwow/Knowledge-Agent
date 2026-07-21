from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from app.core.errors import DependencyError, ErrorCode
from app.infrastructure.minio_store import build_object_ref
from app.services.import_fence import CommitFence
from app.services.import_registry import (
    ImportTaskStatus,
    InMemoryImportTaskRegistry,
    NewImportTask,
)
from app.services.import_runner import (
    ImportCommitEvent,
    ImportCommitReceipt,
    ImportJob,
    ImportStepEvent,
    run_import_batch,
    run_import_task,
)


class Processor:
    def __init__(
        self,
        events: tuple[ImportStepEvent, ...] = (),
        *,
        failure: Exception | None = None,
        prove_commit: bool = True,
        imported_chunk_count: int = 3,
    ) -> None:
        self.events = events
        self.failure = failure
        self.prove_commit = prove_commit
        self.imported_chunk_count = imported_chunk_count
        self.jobs: list[ImportJob] = []

    def stream(self, job: ImportJob):
        self.jobs.append(job)
        if self.failure is not None:
            raise self.failure
        yield from self.events
        if self.prove_commit:
            yield ImportCommitEvent(
                ImportCommitReceipt(
                    task_id=job.task_id,
                    document_id=job.document_id,
                    tenant_id=job.tenant_id,
                    imported_chunk_count=self.imported_chunk_count,
                    commit_token=f"milvus-{job.task_id}",
                )
            )


class Workspace:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.cleaned: list[str] = []

    def cleanup_task(self, task_id: str) -> None:
        self.cleaned.append(task_id)
        if self.fail:
            raise OSError("sensitive-local-path")


def task_and_job(tmp_path: Path):
    task_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    ref = build_object_ref(
        "tenant-a",
        "documents",
        uuid.UUID(document_id).hex,
        "imports",
        uuid.UUID(task_id).hex,
        "source.md",
    )
    task = NewImportTask(
        task_id=task_id,
        document_id=document_id,
        tenant_id="tenant-a",
        owner_subject="user-a",
        source_ref=ref.uri,
        original_name="guide.md",
        content_type="text/markdown",
        size=5,
        sha256="a" * 64,
    )
    job = ImportJob(
        task_id=task_id,
        document_id=document_id,
        tenant_id="tenant-a",
        owner_subject="user-a",
        local_dir=tmp_path,
        local_file_path=tmp_path / "source.md",
        source_ref=ref.uri,
        original_name="guide.md",
        content_type="text/markdown",
        size=5,
        sha256="a" * 64,
    )
    return task, job


def test_runner_completes_only_after_explicit_index_commit(tmp_path: Path) -> None:
    task, job = task_and_job(tmp_path)
    registry = InMemoryImportTaskRegistry()
    registry.create_many([task])
    processor = Processor(
        (ImportStepEvent("parse_document"),),
    )
    workspace = Workspace()

    run_import_task(job, processor, registry, workspace)

    snapshot = registry.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert snapshot.status is ImportTaskStatus.COMPLETED
    assert snapshot.completed_steps == ("parse_document", "commit_index")
    assert snapshot.imported_chunk_count == 3
    assert workspace.cleaned == [task.task_id]


def test_missing_commit_proof_fails_closed(tmp_path: Path) -> None:
    task, job = task_and_job(tmp_path)
    registry = InMemoryImportTaskRegistry()
    registry.create_many([task])

    run_import_task(
        job,
        Processor((ImportStepEvent("parse_document"),), prove_commit=False),
        registry,
        Workspace(),
    )

    snapshot = registry.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert snapshot.status is ImportTaskStatus.FAILED
    assert snapshot.error_code == ErrorCode.INTERNAL_ERROR.value


def test_dependency_failure_is_reduced_to_stable_error_code(tmp_path: Path) -> None:
    task, job = task_and_job(tmp_path)
    registry = InMemoryImportTaskRegistry()
    registry.create_many([task])

    run_import_task(
        job,
        Processor(failure=DependencyError("https://secret.invalid/?token=leak")),
        registry,
        Workspace(),
    )

    snapshot = registry.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert snapshot.status is ImportTaskStatus.FAILED
    assert snapshot.error_code == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert "secret" not in snapshot.error_code.lower()


def test_duplicate_delivery_does_not_clean_active_or_completed_workspace_twice(tmp_path: Path) -> None:
    task, job = task_and_job(tmp_path)
    registry = InMemoryImportTaskRegistry()
    registry.create_many([task])
    processor = Processor(
        imported_chunk_count=0,
    )
    workspace = Workspace()

    run_import_task(job, processor, registry, workspace)
    run_import_task(job, processor, registry, workspace)

    assert len(processor.jobs) == 1
    assert workspace.cleaned == [task.task_id]
    assert registry.get_owned(task.task_id, task.tenant_id, task.owner_subject).status is ImportTaskStatus.COMPLETED


def test_cleanup_failure_cannot_change_a_committed_task(tmp_path: Path) -> None:
    task, job = task_and_job(tmp_path)
    registry = InMemoryImportTaskRegistry()
    registry.create_many([task])

    run_import_task(
        job,
        Processor(imported_chunk_count=1),
        registry,
        Workspace(fail=True),
    )

    snapshot = registry.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert snapshot.status is ImportTaskStatus.COMPLETED
    assert snapshot.error_code is None


def test_mismatched_commit_receipt_fails_closed(tmp_path: Path) -> None:
    task, job = task_and_job(tmp_path)
    registry = InMemoryImportTaskRegistry()
    registry.create_many([task])

    class MismatchedProcessor:
        def stream(self, current: ImportJob):
            yield ImportCommitEvent(
                ImportCommitReceipt(
                    task_id=str(uuid.uuid4()),
                    document_id=current.document_id,
                    tenant_id=current.tenant_id,
                    imported_chunk_count=1,
                    commit_token="forged-receipt",  # noqa: S106 - opaque revision, not a credential
                )
            )

    run_import_task(job, MismatchedProcessor(), registry, Workspace())

    snapshot = registry.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert snapshot.status is ImportTaskStatus.FAILED
    assert snapshot.commit_token is None


def test_batch_cancellation_fails_and_cleans_current_and_pending_tasks(tmp_path: Path) -> None:
    first_task, first_job = task_and_job(tmp_path / "first")
    second_task, second_job = task_and_job(tmp_path / "second")
    registry = InMemoryImportTaskRegistry()
    registry.create_many([first_task, second_task])
    workspace = Workspace()

    class CancelProcessor:
        def stream(self, job: ImportJob):
            del job
            raise asyncio.CancelledError
            yield

    try:
        run_import_batch((first_job, second_job), CancelProcessor(), registry, workspace)
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancellation must propagate after state is persisted")

    first = registry.get_owned(first_task.task_id, first_task.tenant_id, first_task.owner_subject)
    second = registry.get_owned(second_task.task_id, second_task.tenant_id, second_task.owner_subject)
    assert first.status is ImportTaskStatus.FAILED
    assert second.status is ImportTaskStatus.FAILED
    assert workspace.cleaned == [first_task.task_id, second_task.task_id]


def test_claimed_runner_uses_fencing_token_and_completes(tmp_path: Path) -> None:
    task, job = task_and_job(tmp_path)
    registry = InMemoryImportTaskRegistry()
    registry.create_many([task])
    lease = registry.claim_next("worker-a", lease_seconds=30, max_attempts=3)
    assert lease is not None

    run_import_task(
        job,
        Processor(imported_chunk_count=2),
        registry,
        Workspace(),
        lease=lease,
        retry_delay_seconds=1,
    )

    completed = registry.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert completed.status is ImportTaskStatus.COMPLETED
    assert completed.imported_chunk_count == 2
    assert completed.attempt_count == 1


def test_claimed_runner_requeues_retryable_dependency_failure(tmp_path: Path) -> None:
    task, job = task_and_job(tmp_path)
    registry = InMemoryImportTaskRegistry()
    registry.create_many([task])
    lease = registry.claim_next("worker-a", lease_seconds=30, max_attempts=3)
    assert lease is not None

    run_import_task(
        job,
        Processor(failure=DependencyError("credential-secret")),
        registry,
        Workspace(),
        lease=lease,
        retry_delay_seconds=0,
    )

    queued = registry.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert queued.status is ImportTaskStatus.QUEUED
    assert queued.error_code == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert registry.claim_next("worker-b", lease_seconds=30, max_attempts=3) is not None


def test_claimed_runner_uses_fenced_stream_for_external_commit(tmp_path: Path) -> None:
    task, job = task_and_job(tmp_path)
    registry = InMemoryImportTaskRegistry()
    registry.create_many([task])
    lease = registry.claim_next("worker-a", lease_seconds=30, max_attempts=3)
    assert lease is not None

    class FencedProcessor:
        idempotent_commits = True
        fenced_commits = True

        def __init__(self) -> None:
            self.fence_ids: list[str] = []

        def stream(self, current: ImportJob):
            del current
            raise AssertionError("leased imports must use the fenced stream")
            yield

        def stream_fenced(self, current: ImportJob, fence: CommitFence):
            fence.assert_current()
            self.fence_ids.append(fence.fence_id)
            assert fence.task_id == current.task_id
            assert fence.document_id == current.document_id
            assert fence.tenant_id == current.tenant_id
            assert fence.import_version == current.task_id
            yield ImportCommitEvent(
                ImportCommitReceipt(
                    task_id=current.task_id,
                    document_id=current.document_id,
                    tenant_id=current.tenant_id,
                    imported_chunk_count=2,
                    commit_token=f"milvus-{fence.fence_id}",
                )
            )

    processor = FencedProcessor()
    run_import_task(job, processor, registry, Workspace(), lease=lease)

    completed = registry.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert completed.status is ImportTaskStatus.COMPLETED
    assert completed.imported_chunk_count == 2
    assert len(processor.fence_ids) == 1


def test_fenced_processor_without_fenced_stream_fails_closed(tmp_path: Path) -> None:
    task, job = task_and_job(tmp_path)
    registry = InMemoryImportTaskRegistry()
    registry.create_many([task])
    lease = registry.claim_next("worker-a", lease_seconds=30, max_attempts=3)
    assert lease is not None

    class InvalidFencedProcessor:
        idempotent_commits = True
        fenced_commits = True

        def stream(self, current: ImportJob):
            del current
            raise AssertionError("unfenced stream must not be used")
            yield

    run_import_task(job, InvalidFencedProcessor(), registry, Workspace(), lease=lease)

    failed = registry.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert failed.status is ImportTaskStatus.FAILED
    assert failed.error_code == ErrorCode.INTERNAL_ERROR.value
