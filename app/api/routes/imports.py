from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Annotated, Any, BinaryIO, cast

import anyio
from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, UploadFile, status

from app.api.dependencies import (
    get_app_settings,
    get_import_admission_repository,
    get_import_coordinator,
    get_import_processor,
    get_import_registry,
    get_import_workspace,
    get_object_store,
)
from app.api.schemas.common import ErrorResponse
from app.api.schemas.imports import (
    ImportBatchAcceptedResponse,
    ImportDocumentResponse,
    ImportProgressResponse,
    ImportTaskResponse,
)
from app.core.errors import AppError, ErrorCode
from app.core.logger import logger
from app.core.security import Principal, require_roles
from app.core.settings import Settings
from app.infrastructure.minio_store import MinioStore, StoredObject, build_object_ref
from app.services.import_admission import (
    AdmissionBatchIntent,
    AdmissionObjectIntent,
    AdmissionObjectSnapshot,
    AdmissionStatus,
    ImportAdmissionRepository,
)
from app.services.import_cleanup import run_cleanup_cycle
from app.services.import_coordinator import ImportCoordinator
from app.services.import_registry import ImportTaskRegistry, ImportTaskSnapshot, NewImportTask
from app.services.import_runner import ImportJob, ImportProcessor, run_import_batch
from app.services.import_workspace import ImportWorkspace
from app.services.upload_service import StoredUpload, store_upload, validate_upload_count

router = APIRouter(prefix="/imports", tags=["imports"])
import_writer = require_roles("editor", "admin")
import_reader = require_roles("reader", "editor", "admin")
_ERROR_RESPONSE: dict[str, Any] = {"model": ErrorResponse}
_CREATE_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: _ERROR_RESPONSE for code in (401, 403, 409, 413, 415, 422, 500, 503)
}
_GET_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: _ERROR_RESPONSE for code in (401, 403, 404, 422, 500, 503)
}


class _DigestingReader:
    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self._digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._source.read(size)
        self.size += len(chunk)
        self._digest.update(chunk)
        return chunk

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


class _PersistedIntegrityFailure(Exception):
    def __init__(self, stored_object: StoredObject) -> None:
        super().__init__("persisted upload integrity failure")
        self.stored_object = stored_object


class _WriteLeaseHeartbeat:
    def __init__(
        self,
        *,
        admissions: ImportAdmissionRepository,
        batch_id: str,
        current: AdmissionObjectSnapshot,
        tenant_id: str,
        owner_subject: str,
        lease_seconds: int,
        heartbeat_seconds: float,
    ) -> None:
        if current.write_lease_token is None or current.write_lease_expires_at is None:
            raise AppError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "Import admission write lease is unavailable",
                503,
            )
        self._admissions = admissions
        self._batch_id = batch_id
        self._tenant_id = tenant_id
        self._owner_subject = owner_subject
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._stop = anyio.Event()
        self._logged_failure = False
        self.current = current

    @property
    def lease_token(self) -> str:
        token = self.current.write_lease_token
        if token is None:
            raise AppError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "Import admission write lease is unavailable",
                503,
            )
        return token

    def stop(self) -> None:
        self._stop.set()

    async def renew_now(self) -> None:
        self.current = await anyio.to_thread.run_sync(
            partial(
                self._admissions.renew_write_lease,
                self._batch_id,
                self.current.object_id,
                self._tenant_id,
                self._owner_subject,
                lease_token=self.lease_token,
                expected_revision=self.current.revision,
                lease_seconds=self._lease_seconds,
            )
        )

    async def run(self) -> None:
        while not self._stop.is_set():
            with anyio.move_on_after(self._heartbeat_seconds):
                await self._stop.wait()
            if self._stop.is_set():
                return
            try:
                await self.renew_now()
                self._logged_failure = False
            except Exception as renew_error:
                if not self._logged_failure:
                    logger.bind(
                        task_id=self.current.task_id,
                        tenant_id=self._tenant_id,
                        error_type=type(renew_error).__name__,
                    ).warning("Import object write lease renewal failed")
                    self._logged_failure = True
                retry_seconds = min(1.0, max(0.05, self._heartbeat_seconds / 4))
                with anyio.move_on_after(retry_seconds):
                    await self._stop.wait()


def _response(snapshot: ImportTaskSnapshot) -> ImportTaskResponse:
    return ImportTaskResponse(
        task_id=uuid.UUID(snapshot.task_id),
        status=snapshot.status,
        document=ImportDocumentResponse(
            document_id=uuid.UUID(snapshot.document_id),
            original_name=snapshot.original_name,
            content_type=snapshot.content_type,
            size=snapshot.size,
            sha256=snapshot.sha256,
        ),
        progress=ImportProgressResponse(
            current_step=snapshot.current_step,
            completed_steps=snapshot.completed_steps,
        ),
        imported_chunk_count=snapshot.imported_chunk_count,
        error_code=snapshot.error_code,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
    )


def _persist_upload(
    store: MinioStore,
    tenant_id: str,
    stored: StoredUpload,
    object_ref,
    metadata: dict[str, str],
) -> StoredObject:
    with stored.open_verified() as source:
        digesting = _DigestingReader(source)
        result = store.put_stream(
            tenant_id,
            object_ref,
            cast(BinaryIO, digesting),
            length=stored.size,
            content_type=stored.content_type,
            metadata=metadata,
        )
    if digesting.size != stored.size or digesting.sha256 != stored.sha256:
        raise _PersistedIntegrityFailure(result)
    return result


async def _persist_upload_with_heartbeat(
    heartbeat: _WriteLeaseHeartbeat,
    store: MinioStore,
    tenant_id: str,
    stored: StoredUpload,
    object_ref,
    metadata: dict[str, str],
) -> StoredObject:
    result: StoredObject | None = None
    failure: BaseException | None = None
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(heartbeat.run)
        try:
            result = await anyio.to_thread.run_sync(
                partial(
                    _persist_upload,
                    store,
                    tenant_id,
                    stored,
                    object_ref,
                    metadata,
                )
            )
        except BaseException as write_error:
            failure = write_error
        finally:
            heartbeat.stop()
    if failure is not None:
        raise failure
    if result is None:
        raise AppError(ErrorCode.INTERNAL_ERROR, "Object persistence failed", 500)
    return result


def _idempotency_key(value: str | None, settings: Settings) -> str:
    if value is None:
        if settings.import_runtime.require_idempotency_key:
            raise AppError(
                ErrorCode.REQUEST_INVALID,
                "Idempotency-Key header is required",
                422,
            )
        return str(uuid.uuid4())
    if (
        not 1 <= len(value) <= 256
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise AppError(ErrorCode.REQUEST_INVALID, "Invalid Idempotency-Key header", 422)
    return value


def _payload_hash(uploads: Sequence[StoredUpload]) -> str:
    payload = [
        {
            "ordinal": ordinal,
            "original_name": upload.original_name,
            "content_type": upload.content_type,
            "size": upload.size,
            "sha256": upload.sha256,
        }
        for ordinal, upload in enumerate(uploads)
    ]
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_error_code(exc: BaseException) -> str:
    if isinstance(exc, AppError):
        return exc.code.value
    return ErrorCode.INTERNAL_ERROR.value


def _public_persistence_failure(exc: BaseException) -> BaseException:
    if isinstance(exc, AppError) and exc.code is ErrorCode.DEPENDENCY_UNAVAILABLE:
        details = {"retryable": True} if exc.details.get("retryable") is True else None
        return AppError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "Object storage is unavailable",
            503,
            details,
        )
    return exc


async def _close_uploads(files: Sequence[UploadFile]) -> None:
    for upload in files:
        try:
            await upload.close()
        except BaseException as close_error:
            logger.bind(error_type=type(close_error).__name__).warning(
                "Upload stream close failed"
            )


async def _cleanup_local(
    workspace: ImportWorkspace,
    task_ids: Sequence[str],
    uploads: Sequence[StoredUpload],
    request_files: Sequence[UploadFile],
) -> None:
    for stored_upload in reversed(uploads):
        await anyio.to_thread.run_sync(stored_upload.delete_owned)
    for task_id in reversed(task_ids):
        try:
            await anyio.to_thread.run_sync(workspace.cleanup_task, task_id)
        except Exception as cleanup_error:
            logger.bind(
                task_id=task_id,
                error_type=type(cleanup_error).__name__,
            ).warning("Import workspace rollback failed")
    await _close_uploads(request_files)


async def _request_durable_rollback(
    *,
    admissions: ImportAdmissionRepository,
    store: MinioStore,
    workspace: ImportWorkspace,
    principal: Principal,
    batch_id: str,
) -> None:
    try:
        current = await anyio.to_thread.run_sync(
            admissions.get_owned,
            batch_id,
            principal.tenant_id,
            principal.subject,
        )
        if current.status is AdmissionStatus.ACCEPTED:
            return
        await anyio.to_thread.run_sync(
            partial(
                admissions.request_rollback,
                batch_id,
                principal.tenant_id,
                principal.subject,
                reason_code="ADMISSION_FAILED",
                expected_revision=current.revision,
            )
        )
    except Exception as rollback_error:
        logger.bind(
            batch_id=batch_id,
            tenant_id=principal.tenant_id,
            error_type=type(rollback_error).__name__,
        ).warning("Import admission rollback could not be persisted")
        return

    for cycle in range(2):
        try:
            result = await anyio.to_thread.run_sync(
                partial(
                    run_cleanup_cycle,
                    admissions,
                    store,
                    workspace,
                    worker_id=f"request-{uuid.uuid4()}",
                    limit=100,
                    lease_seconds=60,
                )
            )
        except Exception as cleanup_error:
            logger.bind(
                batch_id=batch_id,
                tenant_id=principal.tenant_id,
                error_type=type(cleanup_error).__name__,
            ).warning("Immediate import admission cleanup failed")
            return
        if result.claimed == 0:
            return
        if cycle == 1:
            return


async def _accepted_retry_response(
    admission,
    registry: ImportTaskRegistry,
) -> ImportBatchAcceptedResponse:
    ordered = tuple(sorted(admission.objects, key=lambda item: item.ordinal))
    snapshots: list[ImportTaskSnapshot] = []
    for item in ordered:
        snapshot = await anyio.to_thread.run_sync(
            registry.get_owned,
            item.task_id,
            admission.tenant_id,
            admission.owner_subject,
        )
        snapshots.append(snapshot)
    return ImportBatchAcceptedResponse(tasks=tuple(_response(item) for item in snapshots))


@router.post(
    "",
    response_model=ImportBatchAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_CREATE_RESPONSES,
    summary="Create document import tasks",
)
async def create_imports(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(...)],
    principal: Annotated[Principal, Depends(import_writer)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    store: Annotated[MinioStore, Depends(get_object_store)],
    registry: Annotated[ImportTaskRegistry, Depends(get_import_registry)],
    admissions: Annotated[
        ImportAdmissionRepository,
        Depends(get_import_admission_repository),
    ],
    coordinator: Annotated[ImportCoordinator, Depends(get_import_coordinator)],
    processor: Annotated[ImportProcessor, Depends(get_import_processor)],
    workspace: Annotated[ImportWorkspace, Depends(get_import_workspace)],
    idempotency_key_header: Annotated[
        str | None,
        Header(alias="Idempotency-Key"),
    ] = None,
) -> ImportBatchAcceptedResponse:
    validate_upload_count(files, settings.api.max_upload_files)
    request_key = _idempotency_key(idempotency_key_header, settings)
    batch_id = str(uuid.uuid4())
    task_ids: list[str] = []
    document_ids: list[str] = []
    local_dirs: list[Path] = []
    stored_uploads: list[StoredUpload] = []
    manifest_owned = False
    batch_size = 0

    try:
        for upload in files:
            task_id = str(uuid.uuid4())
            document_id = str(uuid.uuid4())
            task_ids.append(task_id)
            document_ids.append(document_id)
            local_dir = await anyio.to_thread.run_sync(workspace.prepare, task_id)
            local_dirs.append(local_dir)
            stored = await store_upload(upload, local_dir, settings.api.max_upload_bytes)
            stored_uploads.append(stored)
            batch_size += stored.size
            if batch_size > settings.api.max_upload_batch_bytes:
                raise AppError(
                    ErrorCode.FILE_TOO_LARGE,
                    "Upload batch exceeds configured size limit",
                    413,
                )

        intents: list[AdmissionObjectIntent] = []
        new_tasks: list[NewImportTask] = []
        jobs: list[ImportJob] = []
        for ordinal, (task_id, document_id, local_dir, stored) in enumerate(
            zip(task_ids, document_ids, local_dirs, stored_uploads, strict=True)
        ):
            task_uuid = uuid.UUID(task_id)
            document_uuid = uuid.UUID(document_id)
            suffix = Path(stored.stored_name).suffix
            object_ref = build_object_ref(
                principal.tenant_id,
                settings.minio.pdf_prefix,
                document_uuid.hex,
                "imports",
                task_uuid.hex,
                f"source{suffix}",
            )
            intents.append(
                AdmissionObjectIntent(
                    object_id=task_id,
                    task_id=task_id,
                    document_id=document_id,
                    ordinal=ordinal,
                    ref=object_ref,
                    original_name=stored.original_name,
                    content_type=stored.content_type,
                    size=stored.size,
                    sha256=stored.sha256,
                )
            )
            new_tasks.append(
                NewImportTask(
                    task_id=task_id,
                    document_id=document_id,
                    tenant_id=principal.tenant_id,
                    owner_subject=principal.subject,
                    source_ref=object_ref.uri,
                    original_name=stored.original_name,
                    content_type=stored.content_type,
                    size=stored.size,
                    sha256=stored.sha256,
                )
            )
            jobs.append(
                ImportJob(
                    task_id=task_id,
                    document_id=document_id,
                    tenant_id=principal.tenant_id,
                    owner_subject=principal.subject,
                    local_dir=local_dir,
                    local_file_path=stored.path,
                    source_ref=object_ref.uri,
                    original_name=stored.original_name,
                    content_type=stored.content_type,
                    size=stored.size,
                    sha256=stored.sha256,
                )
            )

        manifest = await anyio.to_thread.run_sync(
            admissions.prepare_batch,
            AdmissionBatchIntent(
                batch_id=batch_id,
                tenant_id=principal.tenant_id,
                owner_subject=principal.subject,
                idempotency_key=request_key,
                payload_hash=_payload_hash(stored_uploads),
                objects=tuple(intents),
            ),
        )
        manifest_owned = manifest.batch_id == batch_id
        if not manifest_owned:
            await _cleanup_local(workspace, task_ids, stored_uploads, files)
            if manifest.status is AdmissionStatus.ACCEPTED:
                return await _accepted_retry_response(manifest, registry)
            raise AppError(
                ErrorCode.RESOURCE_CONFLICT,
                "Import request is already in progress",
                409,
            )

        ordered_manifest_objects = tuple(sorted(manifest.objects, key=lambda entry: entry.ordinal))
        for item, stored in zip(ordered_manifest_objects, stored_uploads, strict=True):
            started = await anyio.to_thread.run_sync(
                partial(
                    admissions.mark_write_started,
                    batch_id,
                    item.object_id,
                    principal.tenant_id,
                    principal.subject,
                    expected_revision=item.revision,
                    writer_id=f"upload-{uuid.uuid4()}",
                    lease_seconds=settings.import_runtime.write_lease_seconds,
                )
            )
            heartbeat = _WriteLeaseHeartbeat(
                admissions=admissions,
                batch_id=batch_id,
                current=started,
                tenant_id=principal.tenant_id,
                owner_subject=principal.subject,
                lease_seconds=settings.import_runtime.write_lease_seconds,
                heartbeat_seconds=settings.import_runtime.heartbeat_seconds,
            )
            try:
                persisted = await _persist_upload_with_heartbeat(
                    heartbeat,
                    store,
                    principal.tenant_id,
                    stored,
                    item.ref,
                    {
                        "schema-version": "1",
                        "document-id": uuid.UUID(item.document_id).hex,
                        "import-id": uuid.UUID(item.task_id).hex,
                        "sha256": item.sha256,
                        "object-kind": item.object_kind,
                    },
                )
            except _PersistedIntegrityFailure as integrity_error:
                await anyio.to_thread.run_sync(
                    partial(
                        admissions.record_stored,
                        batch_id,
                        item.object_id,
                        principal.tenant_id,
                        principal.subject,
                        integrity_error.stored_object,
                        expected_revision=heartbeat.current.revision,
                        lease_token=heartbeat.lease_token,
                    )
                )
                raise AppError(
                    ErrorCode.REQUEST_INVALID,
                    "Stored upload changed before persistence",
                    422,
                ) from None
            except BaseException as write_error:
                try:
                    await heartbeat.renew_now()
                except Exception as renew_error:
                    logger.bind(
                        batch_id=batch_id,
                        task_id=item.task_id,
                        tenant_id=principal.tenant_id,
                        error_type=type(renew_error).__name__,
                    ).warning("Unknown import write lease could not be extended")
                try:
                    await anyio.to_thread.run_sync(
                        partial(
                            admissions.record_write_unknown,
                            batch_id,
                            item.object_id,
                            principal.tenant_id,
                            principal.subject,
                            error_code=_write_error_code(write_error),
                            expected_revision=heartbeat.current.revision,
                            lease_token=heartbeat.lease_token,
                        )
                    )
                except Exception as record_error:
                    logger.bind(
                        batch_id=batch_id,
                        task_id=item.task_id,
                        tenant_id=principal.tenant_id,
                        error_type=type(record_error).__name__,
                    ).warning("Unknown import object write could not be persisted")
                raise _public_persistence_failure(write_error) from None

            await anyio.to_thread.run_sync(
                partial(
                    admissions.record_stored,
                    batch_id,
                    item.object_id,
                    principal.tenant_id,
                    principal.subject,
                    persisted,
                    expected_revision=heartbeat.current.revision,
                    lease_token=heartbeat.lease_token,
                )
            )

        ready = await anyio.to_thread.run_sync(
            admissions.get_owned,
            batch_id,
            principal.tenant_id,
            principal.subject,
        )
        _, snapshots = await anyio.to_thread.run_sync(
            partial(
                coordinator.accept_and_enqueue,
                batch_id,
                principal.tenant_id,
                principal.subject,
                tuple(new_tasks),
                expected_revision=ready.revision,
            )
        )
        if coordinator.durable is True:
            await _cleanup_local(workspace, task_ids, stored_uploads, files)
        else:
            background_tasks.add_task(
                run_import_batch,
                tuple(jobs),
                processor,
                registry,
                workspace,
            )
        return ImportBatchAcceptedResponse(tasks=tuple(_response(item) for item in snapshots))
    except BaseException:
        with anyio.CancelScope(shield=True):
            if manifest_owned:
                await _request_durable_rollback(
                    admissions=admissions,
                    store=store,
                    workspace=workspace,
                    principal=principal,
                    batch_id=batch_id,
                )
            await _cleanup_local(workspace, task_ids, stored_uploads, files)
        raise


@router.get(
    "/{task_id}",
    response_model=ImportTaskResponse,
    responses=_GET_RESPONSES,
    summary="Get an owned import task",
)
async def get_import_task(
    task_id: uuid.UUID,
    principal: Annotated[Principal, Depends(import_reader)],
    registry: Annotated[ImportTaskRegistry, Depends(get_import_registry)],
) -> ImportTaskResponse:
    snapshot = registry.get_owned(str(task_id), principal.tenant_id, principal.subject)
    return _response(snapshot)
