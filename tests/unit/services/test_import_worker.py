from __future__ import annotations

import hashlib
import os
import uuid
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from threading import Event

import anyio

from app.core.errors import DependencyError, ErrorCode
from app.infrastructure.minio_store import StoredObject, build_object_ref
from app.services.import_admission import AdmissionBatchIntent, AdmissionObjectIntent
from app.services.import_coordinator import InMemoryImportCoordinator
from app.services.import_registry import (
    ImportTaskLease,
    ImportTaskStatus,
    InMemoryImportTaskRegistry,
    NewImportTask,
)
from app.services.import_runner import ImportCommitEvent, ImportCommitReceipt, ImportJob
from app.services.import_worker import ImportWorker, ImportWorkerConfig
from app.services.import_workspace import ImportWorkspace


class Store:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.failure: Exception | None = None
        self.opened_versions: list[str | None] = []

    @contextmanager
    def open_stored(self, tenant_id: str, stored: StoredObject):
        assert tenant_id == stored.ref.tenant_id
        if self.failure is not None:
            raise self.failure
        self.opened_versions.append(stored.version_id)
        yield BytesIO(self.body)

    def inspect_object(self, tenant_id: str, ref):
        del tenant_id, ref
        return None

    def delete_stored(self, tenant_id: str, stored: StoredObject) -> None:
        del tenant_id, stored


class Processor:
    idempotent_commits = True

    def probe(self) -> None:
        return None

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.jobs: list[ImportJob] = []

    def stream(self, job: ImportJob):
        self.jobs.append(job)
        if self.failure is not None:
            raise self.failure
        yield ImportCommitEvent(
            ImportCommitReceipt(
                task_id=job.task_id,
                document_id=job.document_id,
                tenant_id=job.tenant_id,
                imported_chunk_count=1,
                commit_token=f"commit-{job.task_id}",
            )
        )


class BlockingProcessor(Processor):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def stream(self, job: ImportJob):
        self.jobs.append(job)
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test processor was not released")
        yield ImportCommitEvent(
            ImportCommitReceipt(
                task_id=job.task_id,
                document_id=job.document_id,
                tenant_id=job.tenant_id,
                imported_chunk_count=1,
                commit_token=f"commit-{job.task_id}",
            )
        )


class BlockingClaimRegistry(InMemoryImportTaskRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.claim_started = Event()
        self.release_claim = Event()

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        max_attempts: int,
    ) -> ImportTaskLease | None:
        self.claim_started.set()
        if not self.release_claim.wait(timeout=5):
            raise TimeoutError("test claim was not released")
        return super().claim_next(
            worker_id,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
        )


class CountingHeartbeatRegistry(InMemoryImportTaskRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.renew_calls = 0
        self.renewed = Event()

    def renew_lease(
        self,
        lease: ImportTaskLease,
        *,
        lease_seconds: int,
    ) -> ImportTaskLease:
        self.renew_calls += 1
        self.renewed.set()
        return super().renew_lease(lease, lease_seconds=lease_seconds)


def runtime(
    tmp_path: Path,
    *,
    body: bytes = b"hello",
    processor: Processor | None = None,
    registry: InMemoryImportTaskRegistry | None = None,
    heartbeat_seconds: float = 5,
    dependency_probe_interval_seconds: float = 0.02,
    dependency_probe_timeout_seconds: float = 0.05,
):
    coordinator = InMemoryImportCoordinator(tasks=registry or InMemoryImportTaskRegistry())
    workspace = ImportWorkspace(tmp_path / "imports")
    store = Store(body)
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
    digest = hashlib.sha256(body).hexdigest()
    item = AdmissionObjectIntent(
        object_id=task_id,
        task_id=task_id,
        document_id=document_id,
        ordinal=0,
        ref=ref,
        original_name="guide.md",
        content_type="text/markdown",
        size=len(body),
        sha256=digest,
    )
    admission = AdmissionBatchIntent(
        batch_id=str(uuid.uuid4()),
        tenant_id="tenant-a",
        owner_subject="user-a",
        idempotency_key="request-1",
        payload_hash="a" * 64,
        objects=(item,),
    )
    task = NewImportTask(
        task_id=task_id,
        document_id=document_id,
        tenant_id="tenant-a",
        owner_subject="user-a",
        source_ref=ref.uri,
        original_name="guide.md",
        content_type="text/markdown",
        size=len(body),
        sha256=digest,
    )
    coordinator.admissions.prepare_batch(admission)
    started = coordinator.admissions.mark_write_started(
        admission.batch_id,
        task_id,
        admission.tenant_id,
        admission.owner_subject,
    )
    coordinator.admissions.record_stored(
        admission.batch_id,
        task_id,
        admission.tenant_id,
        admission.owner_subject,
        StoredObject(ref=ref, etag="etag-1", version_id="version-1"),
        expected_revision=started.revision,
    )
    ready = coordinator.admissions.get_owned(
        admission.batch_id,
        admission.tenant_id,
        admission.owner_subject,
    )
    coordinator.accept_and_enqueue(
        admission.batch_id,
        admission.tenant_id,
        admission.owner_subject,
        (task,),
        expected_revision=ready.revision,
    )
    resolved_processor = processor or Processor()
    worker = ImportWorker(
        registry=coordinator.tasks,
        admissions=coordinator.admissions,
        processor=resolved_processor,
        store=store,  # type: ignore[arg-type]
        workspace=workspace,
        config=ImportWorkerConfig(
            poll_interval_seconds=0.01,
            cleanup_poll_interval_seconds=1,
            lease_seconds=30,
            heartbeat_seconds=heartbeat_seconds,
            max_attempts=3,
            retry_base_seconds=0.1,
            retry_max_seconds=1,
            dependency_probe_interval_seconds=dependency_probe_interval_seconds,
            dependency_probe_timeout_seconds=dependency_probe_timeout_seconds,
        ),
        worker_id="worker-a",
    )
    return worker, coordinator, workspace, store, resolved_processor, task


class RecoveringProbeProcessor(Processor):
    def __init__(self) -> None:
        super().__init__()
        self.healthy = False
        self.probe_calls = 0

    def probe(self) -> None:
        self.probe_calls += 1
        if not self.healthy:
            raise DependencyError("secret processor endpoint")


class BlockingProbeProcessor(Processor):
    def __init__(self) -> None:
        super().__init__()
        self.probe_started = Event()
        self.release_probe = Event()
        self.probe_calls = 0

    def probe(self) -> None:
        self.probe_calls += 1
        self.probe_started.set()
        self.release_probe.wait(timeout=2)


async def test_worker_downloads_exact_version_and_completes_task(tmp_path: Path) -> None:
    worker, coordinator, workspace, store, processor, task = runtime(tmp_path)

    assert await worker.run_once() is True

    snapshot = coordinator.tasks.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert snapshot.status is ImportTaskStatus.COMPLETED
    assert snapshot.imported_chunk_count == 1
    assert store.opened_versions == ["version-1"]
    assert len(processor.jobs) == 1
    assert not any(path.is_file() for path in workspace.root.rglob("*"))


async def test_worker_does_not_claim_until_processor_probe_recovers(tmp_path: Path) -> None:
    processor = RecoveringProbeProcessor()
    worker, coordinator, _, _, _, task = runtime(
        tmp_path,
        processor=processor,
        dependency_probe_interval_seconds=0.02,
        dependency_probe_timeout_seconds=0.05,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.run)
        await worker.wait_started()
        await anyio.sleep(0.08)
        queued = coordinator.tasks.get_owned(
            task.task_id,
            task.tenant_id,
            task.owner_subject,
        )
        assert queued.status is ImportTaskStatus.QUEUED
        assert processor.jobs == []
        assert worker.is_ready() is False

        processor.healthy = True
        with anyio.fail_after(2):
            while True:
                completed = coordinator.tasks.get_owned(
                    task.task_id,
                    task.tenant_id,
                    task.owner_subject,
                )
                if completed.status is ImportTaskStatus.COMPLETED:
                    break
                await anyio.sleep(0.01)
        worker.request_stop()
        await worker.wait_stopped()
        task_group.cancel_scope.cancel()

    assert processor.probe_calls >= 2
    assert len(processor.jobs) == 1


async def test_hanging_processor_probe_never_stacks_or_claims(tmp_path: Path) -> None:
    processor = BlockingProbeProcessor()
    worker, coordinator, _, _, _, task = runtime(
        tmp_path,
        processor=processor,
        dependency_probe_interval_seconds=0.01,
        dependency_probe_timeout_seconds=0.02,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.run)
        await worker.wait_started()
        started = await anyio.to_thread.run_sync(processor.probe_started.wait, 1)
        assert started is True
        await anyio.sleep(0.12)

        queued = coordinator.tasks.get_owned(
            task.task_id,
            task.tenant_id,
            task.owner_subject,
        )
        assert queued.status is ImportTaskStatus.QUEUED
        assert processor.probe_calls == 1
        assert processor.jobs == []

        worker.request_stop()
        await worker.wait_stopped()
        processor.release_probe.set()
        task_group.cancel_scope.cancel()


async def test_worker_sweeps_old_marked_orphan_before_polling_task(tmp_path: Path) -> None:
    worker, _, workspace, _, _, _ = runtime(tmp_path)
    crashed_process = ImportWorkspace(workspace.root)
    orphan_task_id = str(uuid.uuid4())
    orphan_dir = crashed_process.prepare(orphan_task_id)
    os.utime(orphan_dir / ".import-workspace-owner-v1", (1, 1))
    os.utime(orphan_dir, (1, 1))

    assert await worker.run_once() is True

    assert not orphan_dir.exists()


async def test_worker_requeues_transient_download_failure(tmp_path: Path) -> None:
    worker, coordinator, _, store, processor, task = runtime(tmp_path)
    store.failure = DependencyError("credential-secret")

    assert await worker.run_once() is True

    snapshot = coordinator.tasks.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert snapshot.status is ImportTaskStatus.QUEUED
    assert snapshot.error_code == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert processor.jobs == []


async def test_worker_marks_non_retryable_source_conflict_failed(tmp_path: Path) -> None:
    worker, coordinator, _, _, processor, task = runtime(tmp_path)
    record = coordinator.admissions._records[  # type: ignore[attr-defined]
        next(iter(coordinator.admissions._records))  # type: ignore[attr-defined]
    ]
    record.objects[task.task_id].intent = replace(
        record.objects[task.task_id].intent,
        sha256="f" * 64,
    )

    assert await worker.run_once() is True

    snapshot = coordinator.tasks.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert snapshot.status is ImportTaskStatus.FAILED
    assert snapshot.error_code == ErrorCode.RESOURCE_CONFLICT.value
    assert processor.jobs == []


async def test_run_loop_exposes_health_and_stops_cleanly(tmp_path: Path) -> None:
    worker, _, _, _, _, _ = runtime(tmp_path)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.run)
        with anyio.fail_after(1):
            await worker.wait_started()
        worker.require_ready()
        worker.request_stop()
        assert worker.is_ready() is False
        with anyio.fail_after(1):
            await worker.wait_stopped()

    assert worker.is_ready() is False
    try:
        worker.require_ready()
    except DependencyError:
        pass
    else:
        raise AssertionError("stopped worker must not remain ready")


async def test_stop_finishes_active_task_without_claiming_another(tmp_path: Path) -> None:
    processor = BlockingProcessor()
    worker, coordinator, _, _, _, first = runtime(tmp_path, processor=processor)
    second = replace(
        first,
        task_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        source_ref=f"{first.source_ref}-second",
    )
    coordinator.tasks.create_many((second,))

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.run)
        with anyio.fail_after(1):
            await worker.wait_started()
            started = await anyio.to_thread.run_sync(processor.started.wait, 1)
        assert started is True

        worker.request_stop()
        assert worker.is_ready() is False
        processor.release.set()
        with anyio.fail_after(1):
            await worker.wait_stopped()

    first_snapshot = coordinator.tasks.get_owned(
        first.task_id,
        first.tenant_id,
        first.owner_subject,
    )
    second_snapshot = coordinator.tasks.get_owned(
        second.task_id,
        second.tenant_id,
        second.owner_subject,
    )
    assert first_snapshot.status is ImportTaskStatus.COMPLETED
    assert second_snapshot.status is ImportTaskStatus.QUEUED
    assert second_snapshot.attempt_count == 0
    assert len(processor.jobs) == 1


async def test_cancellation_abandons_hanging_sync_processor(tmp_path: Path) -> None:
    processor = BlockingProcessor()
    worker, _, _, _, _, _ = runtime(tmp_path, processor=processor)

    try:
        with anyio.fail_after(0.5):
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(worker.run)
                with anyio.fail_after(1):
                    await worker.wait_started()
                    started = await anyio.to_thread.run_sync(processor.started.wait, 1)
                assert started is True
                task_group.cancel_scope.cancel()
    finally:
        processor.release.set()

    assert worker.is_ready() is False


async def test_stop_racing_with_claim_requeues_without_processing(tmp_path: Path) -> None:
    registry = BlockingClaimRegistry()
    worker, coordinator, _, _, processor, task = runtime(tmp_path, registry=registry)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.run)
        with anyio.fail_after(1):
            await worker.wait_started()
            started = await anyio.to_thread.run_sync(registry.claim_started.wait, 1)
        assert started is True

        worker.request_stop()
        registry.release_claim.set()
        with anyio.fail_after(1):
            await worker.wait_stopped()

    snapshot = coordinator.tasks.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert snapshot.status is ImportTaskStatus.QUEUED
    assert snapshot.attempt_count == 1
    assert snapshot.error_code == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert processor.jobs == []


async def test_stop_keeps_active_lease_heartbeat_during_grace(tmp_path: Path) -> None:
    registry = CountingHeartbeatRegistry()
    processor = BlockingProcessor()
    worker, coordinator, _, _, _, task = runtime(
        tmp_path,
        processor=processor,
        registry=registry,
        heartbeat_seconds=0.02,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(worker.run)
        with anyio.fail_after(1):
            await worker.wait_started()
            started = await anyio.to_thread.run_sync(processor.started.wait, 1)
        assert started is True

        worker.request_stop()
        registry.renewed.clear()
        renewals_after_stop = registry.renew_calls
        try:
            renewed = await anyio.to_thread.run_sync(registry.renewed.wait, 0.5)
            assert renewed is True
            assert registry.renew_calls > renewals_after_stop
        finally:
            processor.release.set()
        with anyio.fail_after(1):
            await worker.wait_stopped()

    snapshot = coordinator.tasks.get_owned(task.task_id, task.tenant_id, task.owner_subject)
    assert snapshot.status is ImportTaskStatus.COMPLETED
