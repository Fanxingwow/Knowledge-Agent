from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from app.core.errors import AppError, ErrorCode
from app.core.logger import logger
from app.services.import_fence import CommitFence, RegistryCommitFence
from app.services.import_registry import ImportTaskLease, ImportTaskRegistry

_COMMIT_TOKEN = re.compile(r"^[!-~]{1,256}$")
IMPORT_STEP_NAMES = frozenset(
    {
        "parse_document",
        "extract_images",
        "split_document",
        "recognize_item",
        "embed_chunks",
        "commit_index",
    }
)


@dataclass(frozen=True, slots=True)
class ImportJob:
    task_id: str
    document_id: str
    tenant_id: str
    owner_subject: str
    local_dir: Path
    local_file_path: Path
    source_ref: str
    original_name: str
    content_type: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ImportStepEvent:
    step: str


@dataclass(frozen=True, slots=True)
class ImportCommitReceipt:
    task_id: str
    document_id: str
    tenant_id: str
    imported_chunk_count: int
    commit_token: str


@dataclass(frozen=True, slots=True)
class ImportCommitEvent:
    receipt: ImportCommitReceipt


class ImportProcessor(Protocol):
    idempotent_commits: bool

    def probe(self) -> None: ...

    def stream(self, job: ImportJob) -> Iterable[ImportStepEvent | ImportCommitEvent]: ...


class FencedImportProcessor(ImportProcessor, Protocol):
    fenced_commits: bool

    def stream_fenced(
        self,
        job: ImportJob,
        fence: CommitFence,
    ) -> Iterable[ImportStepEvent | ImportCommitEvent]: ...


class ImportWorkspaceCleaner(Protocol):
    def cleanup_task(self, task_id: str) -> None: ...


def _classify_error(exc: BaseException) -> str:
    if isinstance(exc, AppError):
        return exc.code.value
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return ErrorCode.DEPENDENCY_UNAVAILABLE.value
    return ErrorCode.INTERNAL_ERROR.value


def _valid_receipt(job: ImportJob, receipt: ImportCommitReceipt) -> bool:
    return (
        receipt.task_id == job.task_id
        and receipt.document_id == job.document_id
        and receipt.tenant_id == job.tenant_id
        and not isinstance(receipt.imported_chunk_count, bool)
        and isinstance(receipt.imported_chunk_count, int)
        and receipt.imported_chunk_count >= 0
        and isinstance(receipt.commit_token, str)
        and _COMMIT_TOKEN.fullmatch(receipt.commit_token) is not None
    )


def _lease_matches_job(job: ImportJob, lease: ImportTaskLease) -> bool:
    task = lease.task
    return (
        task.task_id == job.task_id
        and task.document_id == job.document_id
        and task.tenant_id == job.tenant_id
        and task.owner_subject == job.owner_subject
        and task.source_ref == job.source_ref
        and task.sha256 == job.sha256
        and task.size == job.size
    )


def _is_retryable(exc: BaseException) -> bool:
    return (
        isinstance(exc, (ConnectionError, TimeoutError))
        or isinstance(exc, AppError)
        and exc.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    )


def run_import_task(
    job: ImportJob,
    processor: ImportProcessor,
    registry: ImportTaskRegistry,
    workspace: ImportWorkspaceCleaner,
    *,
    lease: ImportTaskLease | None = None,
    retry_delay_seconds: float | None = None,
) -> None:
    began = False
    cancellation: BaseException | None = None
    lease_token = lease.lease_token if lease is not None else None
    try:
        if lease is None:
            registry.begin(job.task_id, job.tenant_id, job.owner_subject)
        elif not _lease_matches_job(job, lease):
            raise ValueError("import task lease does not match its job")
        began = True
        receipt: ImportCommitReceipt | None = None
        fence: RegistryCommitFence | None = None
        events: Iterable[ImportStepEvent | ImportCommitEvent]
        if lease is not None and getattr(processor, "fenced_commits", False) is True:
            stream_fenced = getattr(processor, "stream_fenced", None)
            if not callable(stream_fenced):
                raise ValueError("fenced import processor has no fenced stream")
            fence = RegistryCommitFence(registry, lease)
            fence.assert_current()
            typed_stream = cast(
                Callable[
                    [ImportJob, CommitFence],
                    Iterable[ImportStepEvent | ImportCommitEvent],
                ],
                stream_fenced,
            )
            events = typed_stream(job, fence)
        else:
            events = processor.stream(job)

        for event in events:
            if receipt is not None:
                raise ValueError("import processor emitted data after its commit receipt")
            if isinstance(event, ImportCommitEvent):
                if not _valid_receipt(job, event.receipt):
                    raise ValueError("invalid import commit receipt")
                registry.start_step(
                    job.task_id,
                    job.tenant_id,
                    job.owner_subject,
                    "commit_index",
                    lease_token=lease_token,
                )
                registry.complete_step(
                    job.task_id,
                    job.tenant_id,
                    job.owner_subject,
                    "commit_index",
                    lease_token=lease_token,
                )
                receipt = event.receipt
                continue
            if not isinstance(event, ImportStepEvent) or event.step not in IMPORT_STEP_NAMES:
                raise ValueError("invalid import processor event")
            if event.step == "commit_index":
                raise ValueError("commit_index requires a typed commit receipt")
            registry.start_step(
                job.task_id,
                job.tenant_id,
                job.owner_subject,
                event.step,
                lease_token=lease_token,
            )
            registry.complete_step(
                job.task_id,
                job.tenant_id,
                job.owner_subject,
                event.step,
                lease_token=lease_token,
            )

        if receipt is None:
            raise ValueError("import processor did not prove an index commit")
        if fence is not None:
            fence.assert_current()
        registry.complete(
            job.task_id,
            job.tenant_id,
            job.owner_subject,
            imported_chunk_count=receipt.imported_chunk_count,
            commit_token=receipt.commit_token,
            lease_token=lease_token,
        )
    except BaseException as exc:
        if began:
            try:
                if lease is not None and retry_delay_seconds is not None and _is_retryable(exc):
                    registry.retry(
                        lease,
                        error_code=_classify_error(exc),
                        delay_seconds=retry_delay_seconds,
                    )
                else:
                    registry.fail(
                        job.task_id,
                        job.tenant_id,
                        job.owner_subject,
                        error_code=_classify_error(exc),
                        lease_token=lease_token,
                    )
            except Exception as registry_error:
                logger.bind(
                    task_id=job.task_id,
                    tenant_id=job.tenant_id,
                    error_type=type(registry_error).__name__,
                ).warning("Unable to persist failed import status")
        logger.bind(
            task_id=job.task_id,
            tenant_id=job.tenant_id,
            error_type=type(exc).__name__,
        ).error("Import task failed")
        if not isinstance(exc, Exception):
            cancellation = exc
    finally:
        if began:
            try:
                workspace.cleanup_task(job.task_id)
            except Exception as cleanup_error:
                logger.bind(
                    task_id=job.task_id,
                    tenant_id=job.tenant_id,
                    error_type=type(cleanup_error).__name__,
                ).warning("Import workspace cleanup failed")
    if cancellation is not None:
        raise cancellation


def run_import_batch(
    jobs: Sequence[ImportJob],
    processor: ImportProcessor,
    registry: ImportTaskRegistry,
    workspace: ImportWorkspaceCleaner,
) -> None:
    for index, job in enumerate(jobs):
        try:
            run_import_task(job, processor, registry, workspace)
        except BaseException:
            for pending in jobs[index + 1 :]:
                try:
                    registry.fail(
                        pending.task_id,
                        pending.tenant_id,
                        pending.owner_subject,
                        error_code=ErrorCode.INTERNAL_ERROR.value,
                    )
                except Exception as status_error:
                    logger.bind(
                        task_id=pending.task_id,
                        tenant_id=pending.tenant_id,
                        error_type=type(status_error).__name__,
                    ).warning("Unable to cancel pending import status")
                try:
                    workspace.cleanup_task(pending.task_id)
                except Exception as cleanup_error:
                    logger.bind(
                        task_id=pending.task_id,
                        tenant_id=pending.tenant_id,
                        error_type=type(cleanup_error).__name__,
                    ).warning("Pending import workspace cleanup failed")
            raise
