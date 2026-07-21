from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from threading import Event, Lock, Thread

import anyio

from app.core.errors import AppError, DependencyError, ErrorCode
from app.core.logger import logger
from app.infrastructure.minio_store import MinioStore
from app.services.import_admission import (
    AdmissionStatus,
    ImportAdmissionRepository,
    ObjectStatus,
)
from app.services.import_cleanup import run_cleanup_cycle
from app.services.import_registry import ImportTaskLease, ImportTaskRegistry
from app.services.import_runner import ImportJob, ImportProcessor, run_import_task
from app.services.import_source import materialize_import_source
from app.services.import_workspace import ImportWorkspace


@dataclass(frozen=True, slots=True)
class ImportWorkerConfig:
    poll_interval_seconds: float
    cleanup_poll_interval_seconds: float
    lease_seconds: int
    heartbeat_seconds: float
    max_attempts: int
    retry_base_seconds: float
    retry_max_seconds: float
    dependency_probe_interval_seconds: float = 5.0
    dependency_probe_timeout_seconds: float = 5.0


class _ProcessorProbe:
    """One non-stacking probe for a potentially blocking sync processor."""

    def __init__(self, processor: ImportProcessor) -> None:
        self._processor = processor
        self.completion = Event()
        self.succeeded = False

    def run(self) -> None:
        try:
            self._processor.probe()
        except Exception:
            self.succeeded = False
        else:
            self.succeeded = True
        finally:
            self.completion.set()


class ImportWorker:
    """At-least-once import consumer with fenced task leases and exact source versions."""

    def __init__(
        self,
        *,
        registry: ImportTaskRegistry,
        admissions: ImportAdmissionRepository,
        processor: ImportProcessor,
        store: MinioStore,
        workspace: ImportWorkspace,
        config: ImportWorkerConfig,
        worker_id: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        for value, name in (
            (config.dependency_probe_interval_seconds, "dependency probe interval"),
            (config.dependency_probe_timeout_seconds, "dependency probe timeout"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"invalid import worker {name}")
        self._registry = registry
        self._admissions = admissions
        self._processor = processor
        self._store = store
        self._workspace = workspace
        self._config = config
        self.worker_id = worker_id or f"import-worker-{uuid.uuid4()}"
        self._monotonic = monotonic
        self._state_lock = Lock()
        self._running = False
        self._fatal = False
        self._stop_requested = False
        self._last_poll_at = 0.0
        self._last_cleanup_at = 0.0
        self._last_workspace_sweep_at = 0.0
        self._started_event = anyio.Event()
        self._stop_event = anyio.Event()
        self._stopped_event = anyio.Event()
        self._processor_probe: _ProcessorProbe | None = None
        self._processor_probe_succeeded = False
        self._processor_probe_logged = False
        self._last_processor_probe_at = float("-inf")

    def _mark_running(self, value: bool) -> None:
        with self._state_lock:
            self._running = value
            if value:
                self._last_poll_at = self._monotonic()

    def _mark_poll(self) -> None:
        with self._state_lock:
            self._last_poll_at = self._monotonic()

    def _mark_fatal(self) -> None:
        with self._state_lock:
            self._fatal = True

    def _mark_healthy(self) -> None:
        with self._state_lock:
            self._fatal = False

    def is_ready(self) -> bool:
        with self._state_lock:
            age = self._monotonic() - self._last_poll_at
            return (
                self._running
                and not self._fatal
                and not self._stop_requested
                and math.isfinite(age)
                and age <= self._config.lease_seconds
            )

    def require_ready(self) -> None:
        if not self.is_ready():
            raise DependencyError("Import worker is unavailable")

    async def wait_started(self) -> None:
        await self._started_event.wait()

    def request_stop(self) -> None:
        """Stop accepting work while allowing the active fenced lease to finish."""
        with self._state_lock:
            self._stop_requested = True
        self._stop_event.set()

    async def wait_stopped(self) -> None:
        await self._stopped_event.wait()

    def _should_stop(self) -> bool:
        with self._state_lock:
            return self._stop_requested

    async def _wait_for_poll_or_stop(self) -> None:
        with anyio.move_on_after(self._config.poll_interval_seconds):
            await self._stop_event.wait()

    def _record_processor_probe(self, probe: _ProcessorProbe) -> bool:
        self._processor_probe = None
        self._last_processor_probe_at = self._monotonic()
        self._processor_probe_succeeded = probe.succeeded
        if probe.succeeded:
            self._processor_probe_logged = False
        elif not self._processor_probe_logged:
            logger.warning("Import processor dependency probe failed")
            self._processor_probe_logged = True
        return probe.succeeded

    async def _processor_is_ready(self) -> bool:
        inflight = self._processor_probe
        if inflight is not None:
            if inflight.completion.is_set():
                return self._record_processor_probe(inflight)
            return False

        now = self._monotonic()
        if (
            now - self._last_processor_probe_at
            < self._config.dependency_probe_interval_seconds
        ):
            return self._processor_probe_succeeded

        probe = _ProcessorProbe(self._processor)
        self._processor_probe = probe
        try:
            Thread(
                target=probe.run,
                name="import-worker-processor-probe",
                daemon=True,
            ).start()
        except Exception:
            probe.completion.set()
            return self._record_processor_probe(probe)

        deadline = now + self._config.dependency_probe_timeout_seconds
        while not probe.completion.is_set() and not self._should_stop():
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return False
            await anyio.sleep(min(0.01, remaining))
        if probe.completion.is_set():
            return self._record_processor_probe(probe)
        return False

    def _retry_delay(self, attempt_count: int) -> float:
        exponent = max(0, attempt_count - 1)
        return min(
            self._config.retry_base_seconds * (2**exponent),
            self._config.retry_max_seconds,
        )

    @staticmethod
    def _stable_error(exc: BaseException) -> str:
        if isinstance(exc, AppError):
            return exc.code.value
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return ErrorCode.DEPENDENCY_UNAVAILABLE.value
        return ErrorCode.INTERNAL_ERROR.value

    @staticmethod
    def _retryable(exc: BaseException) -> bool:
        return (
            isinstance(exc, (ConnectionError, TimeoutError))
            or isinstance(exc, AppError)
            and exc.code is ErrorCode.DEPENDENCY_UNAVAILABLE
        )

    def _job_for_lease(self, lease: ImportTaskLease) -> ImportJob:
        task = lease.task
        admission = self._admissions.get_by_task_id(task.task_id)
        if (
            admission.status is not AdmissionStatus.ACCEPTED
            or admission.tenant_id != task.tenant_id
            or admission.owner_subject != task.owner_subject
        ):
            raise AppError(ErrorCode.RESOURCE_CONFLICT, "Import task source conflict", 409)
        matches = tuple(item for item in admission.objects if item.task_id == task.task_id)
        if len(matches) != 1:
            raise AppError(ErrorCode.RESOURCE_CONFLICT, "Import task source conflict", 409)
        source = matches[0]
        if (
            source.status is not ObjectStatus.WRITE_CONFIRMED
            or source.stored_object is None
            or source.document_id != task.document_id
            or source.ref.uri != task.source_ref
            or source.size != task.size
            or source.sha256 != task.sha256
            or source.original_name != task.original_name
            or source.content_type != task.content_type
        ):
            raise AppError(ErrorCode.RESOURCE_CONFLICT, "Import task source conflict", 409)

        local_dir, local_file = materialize_import_source(
            self._store,
            self._workspace,
            task_id=task.task_id,
            tenant_id=task.tenant_id,
            source=source.stored_object,
            original_name=task.original_name,
            expected_size=task.size,
            expected_sha256=task.sha256,
        )
        return ImportJob(
            task_id=task.task_id,
            document_id=task.document_id,
            tenant_id=task.tenant_id,
            owner_subject=task.owner_subject,
            local_dir=local_dir,
            local_file_path=local_file,
            source_ref=task.source_ref,
            original_name=task.original_name,
            content_type=task.content_type,
            size=task.size,
            sha256=task.sha256,
        )

    def _handle_preprocessing_failure(self, lease: ImportTaskLease, exc: BaseException) -> None:
        try:
            if self._retryable(exc) and lease.attempt_count < self._config.max_attempts:
                self._registry.retry(
                    lease,
                    error_code=self._stable_error(exc),
                    delay_seconds=self._retry_delay(lease.attempt_count),
                )
            else:
                self._registry.fail(
                    lease.task.task_id,
                    lease.task.tenant_id,
                    lease.task.owner_subject,
                    error_code=self._stable_error(exc),
                    lease_token=lease.lease_token,
                )
        except Exception as state_error:
            logger.bind(
                task_id=lease.task.task_id,
                tenant_id=lease.task.tenant_id,
                error_type=type(state_error).__name__,
            ).warning("Import worker could not persist preprocessing failure")
        try:
            self._workspace.cleanup_task(lease.task.task_id)
        except Exception as cleanup_error:
            logger.bind(
                task_id=lease.task.task_id,
                tenant_id=lease.task.tenant_id,
                error_type=type(cleanup_error).__name__,
            ).warning("Import worker preprocessing workspace cleanup failed")

    async def _heartbeat(
        self,
        lease_state: list[ImportTaskLease],
        done: anyio.Event,
    ) -> None:
        current = lease_state[0]
        while True:
            with anyio.move_on_after(self._config.heartbeat_seconds):
                await done.wait()
            if done.is_set():
                return
            try:
                renew = partial(
                    self._registry.renew_lease,
                    current,
                    lease_seconds=self._config.lease_seconds,
                )
                current = await anyio.to_thread.run_sync(
                    renew,
                    abandon_on_cancel=True,
                )
                lease_state[0] = current
                await anyio.to_thread.run_sync(
                    self._workspace.heartbeat_task,
                    current.task.task_id,
                    abandon_on_cancel=True,
                )
            except Exception as heartbeat_error:
                self._mark_fatal()
                logger.bind(
                    task_id=current.task.task_id,
                    tenant_id=current.task.tenant_id,
                    error_type=type(heartbeat_error).__name__,
                ).error("Import worker lease heartbeat failed")
                return

    async def _process(self, lease: ImportTaskLease) -> None:
        lease_state = [lease]
        preprocessing_error: Exception | None = None
        job: ImportJob | None = None
        preprocessing_done = anyio.Event()
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(self._heartbeat, lease_state, preprocessing_done)
            try:
                job = await anyio.to_thread.run_sync(
                    self._job_for_lease,
                    lease,
                    abandon_on_cancel=True,
                )
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise
                preprocessing_error = exc
            finally:
                preprocessing_done.set()
                task_group.cancel_scope.cancel()

        if preprocessing_error is not None:
            await anyio.to_thread.run_sync(
                self._handle_preprocessing_failure,
                lease,
                preprocessing_error,
                abandon_on_cancel=True,
            )
            return
        if job is None:
            raise AppError(ErrorCode.INTERNAL_ERROR, "Import job preparation failed", 500)

        done = anyio.Event()
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(self._heartbeat, lease_state, done)
            try:
                await anyio.to_thread.run_sync(
                    lambda: run_import_task(
                        job,
                        self._processor,
                        self._registry,
                        self._workspace,
                        lease=lease_state[0],
                        retry_delay_seconds=self._retry_delay(lease_state[0].attempt_count),
                    ),
                    abandon_on_cancel=True,
                )
            finally:
                done.set()
                task_group.cancel_scope.cancel()

    async def run_once(self) -> bool:
        if self._should_stop():
            return False
        await self._sweep_workspace_if_due()
        if self._should_stop():
            return False
        lease = await anyio.to_thread.run_sync(
            lambda: self._registry.claim_next(
                self.worker_id,
                lease_seconds=self._config.lease_seconds,
                max_attempts=self._config.max_attempts,
            ),
            abandon_on_cancel=True,
        )
        self._mark_poll()
        if lease is None:
            return False
        if self._should_stop():
            try:
                await anyio.to_thread.run_sync(
                    lambda: self._registry.retry(
                        lease,
                        error_code=ErrorCode.DEPENDENCY_UNAVAILABLE.value,
                        delay_seconds=0.0,
                    ),
                    abandon_on_cancel=True,
                )
            except Exception as release_error:
                logger.bind(
                    task_id=lease.task.task_id,
                    tenant_id=lease.task.tenant_id,
                    error_type=type(release_error).__name__,
                ).warning("Import worker could not release shutdown-raced lease")
            return False
        await self._process(lease)
        return True

    async def _sweep_workspace_if_due(self) -> None:
        now = self._monotonic()
        if now - self._last_workspace_sweep_at < self._config.cleanup_poll_interval_seconds:
            return
        self._last_workspace_sweep_at = now
        report = await anyio.to_thread.run_sync(
            self._workspace.sweep_orphans,
            abandon_on_cancel=True,
        )
        if report.failed:
            logger.bind(
                scanned=report.scanned,
                removed=report.removed,
                failed=report.failed,
            ).warning("Import workspace orphan sweep completed with failures")
        elif report.removed:
            logger.bind(
                scanned=report.scanned,
                removed=report.removed,
            ).info("Import workspace orphans removed")

    async def _cleanup_if_due(self) -> None:
        now = self._monotonic()
        if now - self._last_cleanup_at < self._config.cleanup_poll_interval_seconds:
            return
        self._last_cleanup_at = now
        cutoff = datetime.now(UTC) - timedelta(seconds=self._config.lease_seconds * 2)
        await anyio.to_thread.run_sync(
            lambda: self._admissions.recover_stale_preparing(cutoff, limit=20),
            abandon_on_cancel=True,
        )
        await anyio.to_thread.run_sync(
            lambda: run_cleanup_cycle(
                self._admissions,
                self._store,
                self._workspace,
                worker_id=f"{self.worker_id}-cleanup",
                limit=20,
                lease_seconds=self._config.lease_seconds,
            ),
            abandon_on_cancel=True,
        )

    async def run(self) -> None:
        self._mark_running(True)
        self._started_event.set()
        try:
            while not self._should_stop():
                try:
                    processor_ready = await self._processor_is_ready()
                    worked = await self.run_once() if processor_ready else False
                    if self._should_stop():
                        break
                    await self._cleanup_if_due()
                except Exception as worker_error:
                    self._mark_fatal()
                    logger.bind(error_type=type(worker_error).__name__).error("Import worker polling failed")
                    await self._wait_for_poll_or_stop()
                    continue
                if not processor_ready:
                    self._mark_fatal()
                    await self._wait_for_poll_or_stop()
                    continue
                self._mark_healthy()
                if not worked:
                    await self._wait_for_poll_or_stop()
        finally:
            self._mark_running(False)
            self._stopped_event.set()
