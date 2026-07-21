from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.errors import AppError, ErrorCode
from app.core.logger import logger
from app.infrastructure.minio_store import MinioStore, ObservedObject
from app.services.import_admission import (
    CleanupLease,
    ImportAdmissionRepository,
    ObjectStatus,
)
from app.services.import_workspace import ImportWorkspace


@dataclass(frozen=True, slots=True)
class CleanupCycleResult:
    claimed: int
    completed: int
    retry_pending: int


def _expected_metadata(lease: CleanupLease) -> dict[str, str]:
    return {
        "schema-version": "1",
        "document-id": uuid.UUID(lease.document_id).hex,
        "import-id": uuid.UUID(lease.task_id).hex,
        "sha256": lease.expected_sha256,
        "object-kind": lease.object_kind,
    }


def _matches_intent(lease: CleanupLease, observed: ObservedObject) -> bool:
    return (
        observed.stored.ref == lease.expected_ref
        and observed.size == lease.expected_size
        and dict(observed.metadata) == _expected_metadata(lease)
    )


def _stable_error_code(exc: BaseException) -> str:
    if isinstance(exc, AppError):
        return exc.code.value
    return ErrorCode.INTERNAL_ERROR.value


def _release(
    repository: ImportAdmissionRepository,
    lease: CleanupLease,
    exc: BaseException,
) -> None:
    try:
        repository.release_cleanup(lease, error_code=_stable_error_code(exc))
    except Exception as release_error:
        logger.bind(
            batch_id=lease.batch_id,
            task_id=lease.task_id,
            tenant_id=lease.tenant_id,
            error_type=type(release_error).__name__,
        ).warning("Import cleanup retry state could not be persisted")


def _cleanup_workspace(workspace: ImportWorkspace, lease: CleanupLease) -> None:
    try:
        workspace.cleanup_task(lease.task_id)
    except Exception as cleanup_error:
        logger.bind(
            batch_id=lease.batch_id,
            task_id=lease.task_id,
            tenant_id=lease.tenant_id,
            error_type=type(cleanup_error).__name__,
        ).warning("Import admission workspace cleanup failed")


def run_cleanup_cycle(
    repository: ImportAdmissionRepository,
    store: MinioStore,
    workspace: ImportWorkspace,
    *,
    worker_id: str,
    limit: int = 20,
    lease_seconds: int = 60,
) -> CleanupCycleResult:
    """Reconcile or precisely delete leased admission objects without losing retry evidence."""
    leases = repository.claim_cleanup(
        worker_id,
        limit=limit,
        lease_seconds=lease_seconds,
    )
    completed = 0
    retry_pending = 0
    for lease in leases:
        try:
            if lease.object_status is ObjectStatus.WRITE_UNKNOWN:
                observed = store.inspect_object(lease.tenant_id, lease.expected_ref)
                if observed is None:
                    repository.mark_no_object(lease)
                    _cleanup_workspace(workspace, lease)
                    completed += 1
                    continue
                if not _matches_intent(lease, observed):
                    raise AppError(
                        ErrorCode.RESOURCE_CONFLICT,
                        "Import object reconciliation conflict",
                        409,
                    )
                repository.record_stored(
                    lease.batch_id,
                    lease.object_id,
                    lease.tenant_id,
                    lease.owner_subject,
                    observed.stored,
                    expected_revision=lease.object_revision,
                    lease_token=lease.lease_token,
                )
                completed += 1
                continue

            if lease.object_status is not ObjectStatus.DELETE_PENDING:
                raise AppError(ErrorCode.RESOURCE_CONFLICT, "Import cleanup state conflict", 409)
            if lease.stored_object is None:
                raise AppError(ErrorCode.RESOURCE_CONFLICT, "Import cleanup state conflict", 409)
            store.delete_stored(lease.tenant_id, lease.stored_object)
            repository.mark_deleted(lease)
            _cleanup_workspace(workspace, lease)
            completed += 1
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            _release(repository, lease, exc)
            retry_pending += 1
            logger.bind(
                batch_id=lease.batch_id,
                task_id=lease.task_id,
                tenant_id=lease.tenant_id,
                error_type=type(exc).__name__,
            ).warning("Import admission cleanup attempt failed")
    return CleanupCycleResult(
        claimed=len(leases),
        completed=completed,
        retry_pending=retry_pending,
    )
