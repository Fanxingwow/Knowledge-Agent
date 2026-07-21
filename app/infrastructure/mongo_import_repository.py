from __future__ import annotations

import copy
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any, TypeVar, cast

from pymongo.client_session import ClientSession

from app.core.errors import AppError, DependencyError, ErrorCode
from app.infrastructure.minio_store import ObjectRef, StoredObject, parse_object_ref
from app.infrastructure.mongo_chat_repository import MongoChatSchemaManager, MongoDatabase
from app.infrastructure.mongo_import_store import MongoDocument, MongoImportStore
from app.infrastructure.mongo_import_visibility import MongoCompletedImportVisibilityReader
from app.services.chat_history import ChatHistoryRepository
from app.services.import_admission import (
    AdmissionBatchIntent,
    AdmissionBatchSnapshot,
    AdmissionObjectIntent,
    AdmissionObjectSnapshot,
    AdmissionStatus,
    CleanupLease,
    ImportAdmissionRepository,
    ObjectStatus,
)
from app.services.import_coordinator import ImportCoordinator
from app.services.import_registry import (
    ImportTaskLease,
    ImportTaskRegistry,
    ImportTaskSnapshot,
    ImportTaskStatus,
    NewImportTask,
)

_SCHEMA_VERSION = 1
_ADMISSION_DOCUMENT = "import_admission"
_TASK_DOCUMENT = "import_task"
_OUTBOX_DOCUMENT = "import_outbox"
_OUTBOX_EVENT_TYPE = "import.execute"
_OUTBOX_AGGREGATE_TYPE = "import_task"
_UNAVAILABLE_MESSAGE = "MongoDB import repository is unavailable"
_WRITE_LEASE = "write"
_CLEANUP_LEASE = "cleanup"
_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_STEP_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

_T = TypeVar("_T")


def _resource_not_found(kind: str) -> AppError:
    message = "Import task not found" if kind == "task" else "Import admission not found"
    return AppError(ErrorCode.RESOURCE_NOT_FOUND, message, 404)


def _state_conflict(kind: str) -> AppError:
    message = "Import task state conflict" if kind == "task" else "Import admission state conflict"
    return AppError(ErrorCode.RESOURCE_CONFLICT, message, 409)


def _request_conflict() -> AppError:
    return AppError(ErrorCode.RESOURCE_CONFLICT, "Import admission request conflict", 409)


def _dependency_error() -> DependencyError:
    return DependencyError(_UNAVAILABLE_MESSAGE)


def _is_canonical_uuid(value: object) -> bool:
    try:
        parsed = uuid.UUID(cast(str, value))
    except (AttributeError, TypeError, ValueError):
        return False
    return isinstance(value, str) and str(parsed) == value


def _valid_text(value: object, *, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= maximum
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _valid_opaque(value: object, *, maximum: int = 1024) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= maximum
        and not any(ord(character) < 32 or ord(character) > 126 for character in value)
    )


def _valid_int(value: object, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _valid_datetime(value: object) -> bool:
    if not isinstance(value, datetime) or value.tzinfo is None:
        return False
    try:
        return value.utcoffset() is not None
    except Exception:
        return False


def _require_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError("invalid MongoDB import document")
    return dict(value)


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not _valid_datetime(value):
        raise ValueError("invalid MongoDB import document")
    return cast(datetime, value)


def _optional_text(value: object, *, maximum: int) -> str | None:
    if value is None:
        return None
    if not _valid_text(value, maximum=maximum):
        raise ValueError("invalid MongoDB import document")
    return cast(str, value)


def _optional_opaque(value: object) -> str | None:
    if value is None:
        return None
    if not _valid_opaque(value):
        raise ValueError("invalid MongoDB import document")
    return cast(str, value)


def _validate_error_code(error_code: str) -> None:
    if not isinstance(error_code, str) or not _ERROR_CODE.fullmatch(error_code):
        raise ValueError("invalid import error code")


def _validate_revision(revision: int | None, *, kind: str) -> None:
    if revision is not None and not _valid_int(revision):
        raise ValueError(f"invalid import {kind} revision")


def _validate_admission_intent(intent: AdmissionBatchIntent) -> None:
    if not isinstance(intent, AdmissionBatchIntent):
        raise ValueError("invalid import admission batch")
    if not _is_canonical_uuid(intent.batch_id):
        raise ValueError("invalid import admission identifier")
    if not isinstance(intent.tenant_id, str) or not _TENANT_ID.fullmatch(intent.tenant_id):
        raise ValueError("invalid import admission tenant")
    if not _valid_text(intent.owner_subject, maximum=256):
        raise ValueError("invalid import admission owner")
    if not _valid_text(intent.idempotency_key, maximum=256):
        raise ValueError("invalid import admission idempotency key")
    if not isinstance(intent.payload_hash, str) or not _SHA256.fullmatch(intent.payload_hash):
        raise ValueError("invalid import admission payload hash")
    if not intent.objects:
        raise ValueError("an import admission batch cannot be empty")

    object_ids: set[str] = set()
    task_ids: set[str] = set()
    document_ids: set[str] = set()
    ordinals: set[int] = set()
    refs: set[ObjectRef] = set()
    for item in intent.objects:
        if not isinstance(item, AdmissionObjectIntent):
            raise ValueError("invalid import admission object")
        if not _is_canonical_uuid(item.object_id):
            raise ValueError("invalid import admission object identifier")
        if not _is_canonical_uuid(item.task_id) or not _is_canonical_uuid(item.document_id):
            raise ValueError("invalid import admission task identifier")
        if not _valid_int(item.ordinal):
            raise ValueError("invalid import admission object ordinal")
        if not isinstance(item.ref, ObjectRef) or item.ref.tenant_id != intent.tenant_id:
            raise ValueError("invalid import admission object owner")
        if not _valid_text(item.original_name, maximum=255):
            raise ValueError("invalid import admission document name")
        if not _valid_text(item.content_type, maximum=128):
            raise ValueError("invalid import admission content type")
        if not _valid_int(item.size, minimum=1):
            raise ValueError("invalid import admission object size")
        if not isinstance(item.sha256, str) or not _SHA256.fullmatch(item.sha256):
            raise ValueError("invalid import admission object digest")
        if item.object_kind != "source":
            raise ValueError("invalid import admission object kind")
        if (
            item.object_id in object_ids
            or item.task_id in task_ids
            or item.document_id in document_ids
            or item.ordinal in ordinals
            or item.ref in refs
        ):
            raise _request_conflict()
        object_ids.add(item.object_id)
        task_ids.add(item.task_id)
        document_ids.add(item.document_id)
        ordinals.add(item.ordinal)
        refs.add(item.ref)
    if ordinals != set(range(len(intent.objects))):
        raise ValueError("invalid import admission object ordering")


def _validate_task(task: NewImportTask) -> None:
    if not isinstance(task, NewImportTask):
        raise ValueError("invalid import task")
    if not _is_canonical_uuid(task.task_id) or not _is_canonical_uuid(task.document_id):
        raise ValueError("invalid import task identifier")
    if not isinstance(task.tenant_id, str) or not _TENANT_ID.fullmatch(task.tenant_id):
        raise ValueError("invalid import task tenant")
    if not _valid_text(task.owner_subject, maximum=256):
        raise ValueError("invalid import task owner")
    if not _valid_text(task.original_name, maximum=255):
        raise ValueError("invalid import document name")
    if not _valid_text(task.content_type, maximum=128):
        raise ValueError("invalid import content type")
    if not _valid_int(task.size, minimum=1):
        raise ValueError("invalid import document size")
    if not isinstance(task.sha256, str) or not _SHA256.fullmatch(task.sha256):
        raise ValueError("invalid import document digest")
    try:
        ref = parse_object_ref(task.source_ref)
    except AppError:
        raise ValueError("invalid import source reference") from None
    if ref.tenant_id != task.tenant_id:
        raise ValueError("invalid import source owner")


def _validate_stored_object(stored_object: StoredObject) -> None:
    if not isinstance(stored_object, StoredObject):
        raise ValueError("invalid stored import object")
    if not isinstance(stored_object.ref, ObjectRef):
        raise ValueError("invalid stored import object")
    for value in (stored_object.etag, stored_object.version_id):
        if value is not None and not _valid_opaque(value):
            raise ValueError("invalid stored import object")


def _stored_document(stored_object: StoredObject | None) -> MongoDocument | None:
    if stored_object is None:
        return None
    _validate_stored_object(stored_object)
    return {
        "ref": stored_object.ref.uri,
        "etag": stored_object.etag,
        "version_id": stored_object.version_id,
    }


def _stored_from_document(value: object, *, expected_ref: ObjectRef) -> StoredObject | None:
    if value is None:
        return None
    document = _require_mapping(value)
    if set(document) != {"ref", "etag", "version_id"}:
        raise ValueError("invalid MongoDB import document")
    ref_value = document["ref"]
    if not isinstance(ref_value, str):
        raise ValueError("invalid MongoDB import document")
    ref = parse_object_ref(ref_value)
    if ref != expected_ref:
        raise ValueError("invalid MongoDB import document")
    stored = StoredObject(
        ref=ref,
        etag=_optional_opaque(document["etag"]),
        version_id=_optional_opaque(document["version_id"]),
    )
    _validate_stored_object(stored)
    return stored


def _admission_object_document(item: AdmissionObjectIntent, now: datetime) -> MongoDocument:
    return {
        "object_id": item.object_id,
        "task_id": item.task_id,
        "document_id": item.document_id,
        "ordinal": item.ordinal,
        "ref": item.ref.uri,
        "original_name": item.original_name,
        "content_type": item.content_type,
        "size": item.size,
        "sha256": item.sha256,
        "object_kind": item.object_kind,
        "state": ObjectStatus.INTENT_RECORDED.value,
        "stored_object": None,
        "revision": 0,
        "last_error_code": None,
        "cleanup_lease_owner": None,
        "cleanup_lease_token": None,
        "cleanup_lease_expires_at": None,
        "lease_kind": None,
        "cleanup_attempts": 0,
        "cleanup_next_attempt_at": None,
        "created_at": now,
        "updated_at": now,
    }


def _admission_document(intent: AdmissionBatchIntent, now: datetime) -> MongoDocument:
    return {
        "_id": intent.batch_id,
        "schema_version": _SCHEMA_VERSION,
        "document_type": _ADMISSION_DOCUMENT,
        "tenant_id": intent.tenant_id,
        "owner_subject": intent.owner_subject,
        "idempotency_key": intent.idempotency_key,
        "payload_hash": intent.payload_hash,
        "task_ids": [item.task_id for item in intent.objects],
        "state": AdmissionStatus.PREPARING.value,
        "objects": [_admission_object_document(item, now) for item in intent.objects],
        "rollback_requested": False,
        "rollback_reason_code": None,
        "revision": 0,
        "cleanup_next_attempt_at": None,
        "cleanup_lease_expires_at": None,
        "created_at": now,
        "updated_at": now,
        "expire_at": None,
    }


def _object_snapshot_from_document(value: object, *, tenant_id: str) -> AdmissionObjectSnapshot:
    document = _require_mapping(value)
    required = {
        "object_id",
        "task_id",
        "document_id",
        "ordinal",
        "ref",
        "original_name",
        "content_type",
        "size",
        "sha256",
        "object_kind",
        "state",
        "stored_object",
        "revision",
        "last_error_code",
        "cleanup_lease_owner",
        "cleanup_lease_token",
        "cleanup_lease_expires_at",
        "lease_kind",
        "cleanup_attempts",
        "cleanup_next_attempt_at",
        "created_at",
        "updated_at",
    }
    if not required.issubset(document):
        raise ValueError("invalid MongoDB import document")
    ref_value = document["ref"]
    if not isinstance(ref_value, str):
        raise ValueError("invalid MongoDB import document")
    ref = parse_object_ref(ref_value)
    if ref.tenant_id != tenant_id:
        raise ValueError("invalid MongoDB import document")
    try:
        status = ObjectStatus(document["state"])
    except (TypeError, ValueError):
        raise ValueError("invalid MongoDB import document") from None

    intent = AdmissionObjectIntent(
        object_id=document["object_id"],
        task_id=document["task_id"],
        document_id=document["document_id"],
        ordinal=document["ordinal"],
        ref=ref,
        original_name=document["original_name"],
        content_type=document["content_type"],
        size=document["size"],
        sha256=document["sha256"],
        object_kind=document["object_kind"],
    )
    if not _valid_int(document["revision"]) or not _valid_int(document["cleanup_attempts"]):
        raise ValueError("invalid MongoDB import document")
    last_error_code = _optional_text(document["last_error_code"], maximum=64)
    if last_error_code is not None and not _ERROR_CODE.fullmatch(last_error_code):
        raise ValueError("invalid MongoDB import document")
    lease_owner = _optional_text(document["cleanup_lease_owner"], maximum=128)
    lease_token = _optional_opaque(document["cleanup_lease_token"])
    lease_expires = _optional_datetime(document["cleanup_lease_expires_at"])
    if (lease_owner is None) != (lease_token is None) or (lease_token is None) != (lease_expires is None):
        raise ValueError("invalid MongoDB import document")
    lease_kind = document["lease_kind"]
    if lease_kind not in {None, _WRITE_LEASE, _CLEANUP_LEASE}:
        raise ValueError("invalid MongoDB import document")
    if (lease_kind is None) != (lease_token is None):
        raise ValueError("invalid MongoDB import document")
    created_at = document["created_at"]
    updated_at = document["updated_at"]
    if not _valid_datetime(created_at) or not _valid_datetime(updated_at):
        raise ValueError("invalid MongoDB import document")
    stored_object = _stored_from_document(document["stored_object"], expected_ref=ref)
    if status in {ObjectStatus.WRITE_CONFIRMED, ObjectStatus.DELETE_PENDING, ObjectStatus.DELETED}:
        if stored_object is None:
            raise ValueError("invalid MongoDB import document")
    elif status in {ObjectStatus.INTENT_RECORDED, ObjectStatus.WRITE_STARTED, ObjectStatus.WRITE_UNKNOWN}:
        if stored_object is not None:
            raise ValueError("invalid MongoDB import document")
    if lease_kind == _WRITE_LEASE and status not in {ObjectStatus.WRITE_STARTED, ObjectStatus.WRITE_UNKNOWN}:
        raise ValueError("invalid MongoDB import document")
    if lease_kind == _CLEANUP_LEASE and status not in {ObjectStatus.DELETE_PENDING, ObjectStatus.WRITE_UNKNOWN}:
        raise ValueError("invalid MongoDB import document")

    return AdmissionObjectSnapshot(
        object_id=intent.object_id,
        task_id=intent.task_id,
        document_id=intent.document_id,
        ordinal=intent.ordinal,
        ref=intent.ref,
        original_name=intent.original_name,
        content_type=intent.content_type,
        size=intent.size,
        sha256=intent.sha256,
        object_kind=intent.object_kind,
        status=status,
        stored_object=stored_object,
        revision=cast(int, document["revision"]),
        last_error_code=last_error_code,
        cleanup_attempts=cast(int, document["cleanup_attempts"]),
        cleanup_next_attempt_at=_optional_datetime(document["cleanup_next_attempt_at"]),
        created_at=cast(datetime, created_at),
        updated_at=cast(datetime, updated_at),
        write_lease_token=lease_token if lease_kind == _WRITE_LEASE else None,
        write_lease_expires_at=lease_expires if lease_kind == _WRITE_LEASE else None,
    )


def _admission_snapshot_from_document(value: object) -> AdmissionBatchSnapshot:
    document = _require_mapping(value)
    required = {
        "_id",
        "schema_version",
        "document_type",
        "tenant_id",
        "owner_subject",
        "idempotency_key",
        "payload_hash",
        "task_ids",
        "state",
        "objects",
        "rollback_requested",
        "rollback_reason_code",
        "revision",
        "created_at",
        "updated_at",
    }
    if not required.issubset(document):
        raise ValueError("invalid MongoDB import document")
    if document["schema_version"] != _SCHEMA_VERSION or document["document_type"] != _ADMISSION_DOCUMENT:
        raise ValueError("invalid MongoDB import document")
    if not isinstance(document["tenant_id"], str) or not _TENANT_ID.fullmatch(document["tenant_id"]):
        raise ValueError("invalid MongoDB import document")
    if not isinstance(document["objects"], list) or not document["objects"]:
        raise ValueError("invalid MongoDB import document")
    objects = tuple(
        sorted(
            (
                _object_snapshot_from_document(item, tenant_id=document["tenant_id"])
                for item in document["objects"]
            ),
            key=lambda item: item.ordinal,
        )
    )
    intent = AdmissionBatchIntent(
        batch_id=document["_id"],
        tenant_id=document["tenant_id"],
        owner_subject=document["owner_subject"],
        idempotency_key=document["idempotency_key"],
        payload_hash=document["payload_hash"],
        objects=tuple(
            AdmissionObjectIntent(
                object_id=item.object_id,
                task_id=item.task_id,
                document_id=item.document_id,
                ordinal=item.ordinal,
                ref=item.ref,
                original_name=item.original_name,
                content_type=item.content_type,
                size=item.size,
                sha256=item.sha256,
                object_kind=item.object_kind,
            )
            for item in objects
        ),
    )
    _validate_admission_intent(intent)
    task_ids = document["task_ids"]
    if not isinstance(task_ids, list) or task_ids != [item.task_id for item in objects]:
        raise ValueError("invalid MongoDB import document")
    try:
        status = AdmissionStatus(document["state"])
    except (TypeError, ValueError):
        raise ValueError("invalid MongoDB import document") from None
    if not isinstance(document["rollback_requested"], bool) or not _valid_int(document["revision"]):
        raise ValueError("invalid MongoDB import document")
    rollback_reason = _optional_text(document["rollback_reason_code"], maximum=64)
    if rollback_reason is not None and not _ERROR_CODE.fullmatch(rollback_reason):
        raise ValueError("invalid MongoDB import document")
    created_at = document["created_at"]
    updated_at = document["updated_at"]
    if not _valid_datetime(created_at) or not _valid_datetime(updated_at):
        raise ValueError("invalid MongoDB import document")
    rollback_requested = document["rollback_requested"]
    if status is AdmissionStatus.ACCEPTED and (
        rollback_requested
        or any(item.status is not ObjectStatus.WRITE_CONFIRMED for item in objects)
    ):
        raise ValueError("invalid MongoDB import document")
    if status is AdmissionStatus.ROLLED_BACK and (
        not rollback_requested
        or any(item.status not in {ObjectStatus.DELETED, ObjectStatus.NO_OBJECT} for item in objects)
    ):
        raise ValueError("invalid MongoDB import document")
    if status is AdmissionStatus.ROLLBACK_PENDING and not rollback_requested:
        raise ValueError("invalid MongoDB import document")

    return AdmissionBatchSnapshot(
        batch_id=intent.batch_id,
        tenant_id=intent.tenant_id,
        owner_subject=intent.owner_subject,
        idempotency_key=intent.idempotency_key,
        payload_hash=intent.payload_hash,
        status=status,
        objects=objects,
        rollback_requested=rollback_requested,
        rollback_reason_code=rollback_reason,
        revision=cast(int, document["revision"]),
        created_at=cast(datetime, created_at),
        updated_at=cast(datetime, updated_at),
    )


def _task_document(task: NewImportTask, now: datetime, *, admission_id: str | None) -> MongoDocument:
    _validate_task(task)
    return {
        "_id": task.task_id,
        "schema_version": _SCHEMA_VERSION,
        "document_type": _TASK_DOCUMENT,
        "admission_id": admission_id,
        "task_id": task.task_id,
        "document_id": task.document_id,
        "tenant_id": task.tenant_id,
        "owner_subject": task.owner_subject,
        "source_ref": task.source_ref,
        "original_name": task.original_name,
        "content_type": task.content_type,
        "size": task.size,
        "sha256": task.sha256,
        "status": ImportTaskStatus.QUEUED.value,
        "current_step": None,
        "completed_steps": [],
        "imported_chunk_count": None,
        "commit_token": None,
        "error_code": None,
        "attempt_count": 0,
        "available_at": now,
        "next_attempt_at": now,
        "lease_owner": None,
        "lease_token": None,
        "lease_expires_at": None,
        "revision": 0,
        "created_at": now,
        "updated_at": now,
        "expire_at": None,
    }


def _task_snapshot_from_document(value: object) -> ImportTaskSnapshot:
    document = _require_mapping(value)
    required = {
        "_id",
        "schema_version",
        "document_type",
        "admission_id",
        "task_id",
        "document_id",
        "tenant_id",
        "owner_subject",
        "source_ref",
        "original_name",
        "content_type",
        "size",
        "sha256",
        "status",
        "current_step",
        "completed_steps",
        "imported_chunk_count",
        "commit_token",
        "error_code",
        "attempt_count",
        "available_at",
        "next_attempt_at",
        "lease_owner",
        "lease_token",
        "lease_expires_at",
        "revision",
        "created_at",
        "updated_at",
    }
    if not required.issubset(document):
        raise ValueError("invalid MongoDB import document")
    if document["schema_version"] != _SCHEMA_VERSION or document["document_type"] != _TASK_DOCUMENT:
        raise ValueError("invalid MongoDB import document")
    if document["_id"] != document["task_id"]:
        raise ValueError("invalid MongoDB import document")
    task = NewImportTask(
        task_id=document["task_id"],
        document_id=document["document_id"],
        tenant_id=document["tenant_id"],
        owner_subject=document["owner_subject"],
        source_ref=document["source_ref"],
        original_name=document["original_name"],
        content_type=document["content_type"],
        size=document["size"],
        sha256=document["sha256"],
    )
    _validate_task(task)
    admission_id = document["admission_id"]
    if admission_id is not None and not _is_canonical_uuid(admission_id):
        raise ValueError("invalid MongoDB import document")
    try:
        status = ImportTaskStatus(document["status"])
    except (TypeError, ValueError):
        raise ValueError("invalid MongoDB import document") from None
    current_step = _optional_text(document["current_step"], maximum=64)
    if current_step is not None and not _STEP_NAME.fullmatch(current_step):
        raise ValueError("invalid MongoDB import document")
    completed_steps = document["completed_steps"]
    if (
        not isinstance(completed_steps, list)
        or any(not isinstance(item, str) or not _STEP_NAME.fullmatch(item) for item in completed_steps)
        or len(set(completed_steps)) != len(completed_steps)
    ):
        raise ValueError("invalid MongoDB import document")
    imported_count = document["imported_chunk_count"]
    if imported_count is not None and not _valid_int(imported_count):
        raise ValueError("invalid MongoDB import document")
    commit_token = _optional_opaque(document["commit_token"])
    error_code = _optional_text(document["error_code"], maximum=64)
    if error_code is not None and not _ERROR_CODE.fullmatch(error_code):
        raise ValueError("invalid MongoDB import document")
    if not _valid_int(document["attempt_count"]) or not _valid_int(document["revision"]):
        raise ValueError("invalid MongoDB import document")
    available_at = document["available_at"]
    if not _valid_datetime(available_at) or document["next_attempt_at"] != available_at:
        raise ValueError("invalid MongoDB import document")
    lease_owner = _optional_text(document["lease_owner"], maximum=128)
    lease_token = _optional_opaque(document["lease_token"])
    lease_expires = _optional_datetime(document["lease_expires_at"])
    if (lease_owner is None) != (lease_token is None) or (lease_token is None) != (lease_expires is None):
        raise ValueError("invalid MongoDB import document")
    created_at = document["created_at"]
    updated_at = document["updated_at"]
    if not _valid_datetime(created_at) or not _valid_datetime(updated_at):
        raise ValueError("invalid MongoDB import document")
    if status in {ImportTaskStatus.QUEUED, ImportTaskStatus.COMPLETED, ImportTaskStatus.FAILED}:
        if lease_token is not None:
            raise ValueError("invalid MongoDB import document")
    if status is ImportTaskStatus.COMPLETED and (imported_count is None or commit_token is None):
        raise ValueError("invalid MongoDB import document")
    if status is not ImportTaskStatus.COMPLETED and (imported_count is not None or commit_token is not None):
        raise ValueError("invalid MongoDB import document")

    return ImportTaskSnapshot(
        task_id=task.task_id,
        document_id=task.document_id,
        tenant_id=task.tenant_id,
        owner_subject=task.owner_subject,
        source_ref=task.source_ref,
        original_name=task.original_name,
        content_type=task.content_type,
        size=task.size,
        sha256=task.sha256,
        status=status,
        current_step=current_step,
        completed_steps=tuple(completed_steps),
        imported_chunk_count=cast(int | None, imported_count),
        commit_token=commit_token,
        error_code=error_code,
        attempt_count=cast(int, document["attempt_count"]),
        available_at=cast(datetime, available_at),
        created_at=cast(datetime, created_at),
        updated_at=cast(datetime, updated_at),
    )


def _outbox_document(task: NewImportTask, admission_id: str, now: datetime) -> MongoDocument:
    dedupe_key = f"import.execute:{task.task_id}:v1"
    return {
        "_id": dedupe_key,
        "schema_version": _SCHEMA_VERSION,
        "document_type": _OUTBOX_DOCUMENT,
        "dedupe_key": dedupe_key,
        "event_type": _OUTBOX_EVENT_TYPE,
        "aggregate_type": _OUTBOX_AGGREGATE_TYPE,
        "aggregate_id": task.task_id,
        "admission_id": admission_id,
        "tenant_id": task.tenant_id,
        "owner_subject": task.owner_subject,
        "payload": {
            "task_id": task.task_id,
            "document_id": task.document_id,
            "tenant_id": task.tenant_id,
        },
        "status": "pending",
        "available_at": now,
        "lease_owner": None,
        "lease_token": None,
        "lease_expires_at": None,
        "attempt_count": 0,
        "last_error_code": None,
        "claimed_at": None,
        "completed_at": None,
        "failed_at": None,
        "revision": 0,
        "created_at": now,
        "updated_at": now,
        "expire_at": None,
    }


def _validate_outbox_document(value: object, task: NewImportTask, admission_id: str) -> None:
    document = _require_mapping(value)
    dedupe_key = f"import.execute:{task.task_id}:v1"
    if (
        document.get("_id") != dedupe_key
        or document.get("schema_version") != _SCHEMA_VERSION
        or document.get("document_type") != _OUTBOX_DOCUMENT
        or document.get("dedupe_key") != dedupe_key
        or document.get("event_type") != _OUTBOX_EVENT_TYPE
        or document.get("aggregate_type") != _OUTBOX_AGGREGATE_TYPE
        or document.get("aggregate_id") != task.task_id
        or document.get("admission_id") != admission_id
        or document.get("tenant_id") != task.tenant_id
        or document.get("owner_subject") != task.owner_subject
        or document.get("payload")
        != {
            "task_id": task.task_id,
            "document_id": task.document_id,
            "tenant_id": task.tenant_id,
        }
    ):
        raise ValueError("invalid MongoDB import document")
    status = document.get("status")
    if status not in {"pending", "claimed", "retry", "completed", "failed"}:
        raise ValueError("invalid MongoDB import document")
    if not _valid_datetime(document.get("available_at")):
        raise ValueError("invalid MongoDB import document")
    if not _valid_int(document.get("attempt_count")) or not _valid_int(document.get("revision")):
        raise ValueError("invalid MongoDB import document")
    lease_owner = _optional_text(document.get("lease_owner"), maximum=128)
    lease_token = _optional_opaque(document.get("lease_token"))
    lease_expires_at = _optional_datetime(document.get("lease_expires_at"))
    if (lease_owner is None) != (lease_token is None) or (lease_token is None) != (lease_expires_at is None):
        raise ValueError("invalid MongoDB import document")
    if status == "claimed" and lease_token is None:
        raise ValueError("invalid MongoDB import document")
    if status != "claimed" and lease_token is not None:
        raise ValueError("invalid MongoDB import document")
    error_code = _optional_text(document.get("last_error_code"), maximum=64)
    if error_code is not None and not _ERROR_CODE.fullmatch(error_code):
        raise ValueError("invalid MongoDB import document")
    for key in ("claimed_at", "completed_at", "failed_at", "expire_at"):
        _optional_datetime(document.get(key))
    expire_at = document.get("expire_at")
    if status in {"completed", "failed"}:
        if expire_at is None:
            raise ValueError("invalid MongoDB import document")
    elif expire_at is not None:
        raise ValueError("invalid MongoDB import document")


def _object_index(document: MongoDocument, object_id: str) -> int:
    objects = document.get("objects")
    if not isinstance(objects, list):
        raise ValueError("invalid MongoDB import document")
    matches = [
        index
        for index, item in enumerate(objects)
        if isinstance(item, Mapping) and item.get("object_id") == object_id
    ]
    if len(matches) != 1:
        raise _resource_not_found("admission")
    return matches[0]


def _clear_cleanup_lease(item: MongoDocument) -> None:
    item["cleanup_lease_owner"] = None
    item["cleanup_lease_token"] = None
    item["cleanup_lease_expires_at"] = None
    item["lease_kind"] = None


def _clear_task_lease(document: MongoDocument) -> None:
    document["lease_owner"] = None
    document["lease_token"] = None
    document["lease_expires_at"] = None


def _touch_object(document: MongoDocument, item: MongoDocument, now: datetime) -> None:
    item["revision"] = cast(int, item["revision"]) + 1
    item["updated_at"] = now
    document["revision"] = cast(int, document["revision"]) + 1
    document["updated_at"] = now


def _refresh_cleanup_summary(document: MongoDocument) -> None:
    objects = cast(list[MongoDocument], document["objects"])
    next_attempts = [item["cleanup_next_attempt_at"] for item in objects if item["cleanup_next_attempt_at"] is not None]
    lease_expiries = [
        item["cleanup_lease_expires_at"]
        for item in objects
        if item["lease_kind"] == _CLEANUP_LEASE and item["cleanup_lease_expires_at"] is not None
    ]
    document["cleanup_next_attempt_at"] = min(next_attempts) if next_attempts else None
    document["cleanup_lease_expires_at"] = min(lease_expiries) if lease_expiries else None


def _refresh_rollback_status(document: MongoDocument) -> None:
    if not document["rollback_requested"] or document["state"] == AdmissionStatus.ROLLED_BACK.value:
        _refresh_cleanup_summary(document)
        return
    states = {ObjectStatus(item["state"]) for item in cast(list[MongoDocument], document["objects"])}
    if states.issubset({ObjectStatus.DELETED, ObjectStatus.NO_OBJECT}):
        document["state"] = AdmissionStatus.ROLLED_BACK.value
    elif ObjectStatus.WRITE_UNKNOWN in states:
        document["state"] = AdmissionStatus.RECONCILIATION_REQUIRED.value
    else:
        document["state"] = AdmissionStatus.ROLLBACK_PENDING.value
    _refresh_cleanup_summary(document)


def _touch_task(document: MongoDocument, now: datetime) -> None:
    document["revision"] = cast(int, document["revision"]) + 1
    document["updated_at"] = now


def _task_from_snapshot(snapshot: ImportTaskSnapshot) -> NewImportTask:
    return NewImportTask(
        task_id=snapshot.task_id,
        document_id=snapshot.document_id,
        tenant_id=snapshot.tenant_id,
        owner_subject=snapshot.owner_subject,
        source_ref=snapshot.source_ref,
        original_name=snapshot.original_name,
        content_type=snapshot.content_type,
        size=snapshot.size,
        sha256=snapshot.sha256,
    )


class MongoImportCoordinator:
    """Durable MongoDB implementation shared by admission, task, and outbox façades."""

    durable = True

    def __init__(
        self,
        store: MongoImportStore,
        *,
        clock: Callable[[], datetime] | None = None,
        cleanup_retry_base_seconds: float = 1.0,
        cleanup_retry_max_seconds: float = 300.0,
        terminal_retention_days: int = 30,
        visibility_batch_size: int = 500,
        chat_enabled: bool = False,
        chat_sessions_collection: str = "chat_sessions_v1",
        chat_messages_collection: str = "chat_messages_v1",
    ) -> None:
        if cleanup_retry_base_seconds <= 0 or cleanup_retry_max_seconds < cleanup_retry_base_seconds:
            raise ValueError("invalid cleanup retry window")
        if not _valid_int(terminal_retention_days, minimum=1) or terminal_retention_days > 3650:
            raise ValueError("invalid import terminal retention")
        if not isinstance(chat_enabled, bool):
            raise ValueError("invalid chat runtime flag")
        self._store = store
        self._clock = clock or (lambda: datetime.now(UTC))
        self._cleanup_retry_base_seconds = cleanup_retry_base_seconds
        self._cleanup_retry_max_seconds = cleanup_retry_max_seconds
        self._terminal_retention = timedelta(days=terminal_retention_days)
        self._close_lock = RLock()
        self._closed = False
        self.admissions: ImportAdmissionRepository = _MongoAdmissionFacade(self)
        self.tasks: ImportTaskRegistry = _MongoTaskFacade(self)
        self.completed_import_visibility = MongoCompletedImportVisibilityReader(
            store.tasks,
            max_batch_size=visibility_batch_size,
        )
        self.chat_history_repository: ChatHistoryRepository | None = None
        if chat_enabled:
            chat_schema = MongoChatSchemaManager(
                cast(MongoDatabase, store.database),
                sessions_collection=chat_sessions_collection,
                messages_collection=chat_messages_collection,
            )
            self.chat_history_repository = chat_schema.build_repository(
                store,
                clock=self._clock,
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise _dependency_error()

    def _now(self) -> datetime:
        value = self._clock()
        if not _valid_datetime(value):
            raise _dependency_error()
        normalized = value.astimezone(UTC)
        return normalized.replace(microsecond=(normalized.microsecond // 1000) * 1000)

    @staticmethod
    def _db_call(callback: Callable[[], _T]) -> _T:
        failed = False
        value: _T | None = None
        try:
            value = callback()
        except AppError as exc:
            if isinstance(exc, DependencyError):
                failed = True
            else:
                raise
        except Exception:
            failed = True
        if failed:
            raise _dependency_error() from None
        return cast(_T, value)

    def _find_one(
        self,
        collection: Any,
        query: Mapping[str, object],
        *,
        session: ClientSession | None = None,
    ) -> MongoDocument | None:
        self._ensure_open()
        value = self._db_call(lambda: collection.find_one(dict(query), session=session))
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise _dependency_error()
        return copy.deepcopy(dict(value))

    def _find_many(
        self,
        collection: Any,
        query: Mapping[str, object],
        *,
        session: ClientSession | None = None,
    ) -> list[MongoDocument]:
        self._ensure_open()
        values = self._db_call(lambda: list(collection.find(dict(query), session=session)))
        if not all(isinstance(item, Mapping) for item in values):
            raise _dependency_error()
        return [copy.deepcopy(dict(item)) for item in values]

    def _insert_one(
        self,
        collection: Any,
        document: MongoDocument,
        *,
        session: ClientSession | None = None,
    ) -> None:
        self._ensure_open()
        self._db_call(lambda: collection.insert_one(copy.deepcopy(document), session=session))

    def _insert_many(
        self,
        collection: Any,
        documents: Sequence[MongoDocument],
        *,
        session: ClientSession | None = None,
    ) -> None:
        self._ensure_open()
        copied = [copy.deepcopy(document) for document in documents]
        self._db_call(lambda: collection.insert_many(copied, ordered=True, session=session))

    def _replace_one(
        self,
        collection: Any,
        query: Mapping[str, object],
        document: MongoDocument,
        *,
        session: ClientSession | None = None,
    ) -> bool:
        self._ensure_open()
        result = self._db_call(
            lambda: collection.replace_one(
                dict(query),
                copy.deepcopy(document),
                upsert=False,
                session=session,
            )
        )
        matched_count = getattr(result, "matched_count", None)
        if not _valid_int(matched_count):
            raise _dependency_error()
        return matched_count == 1

    def _delete_many(
        self,
        collection: Any,
        query: Mapping[str, object],
        *,
        session: ClientSession | None = None,
    ) -> int:
        self._ensure_open()
        result = self._db_call(lambda: collection.delete_many(dict(query), session=session))
        deleted_count = getattr(result, "deleted_count", None)
        if not _valid_int(deleted_count):
            raise _dependency_error()
        return cast(int, deleted_count)

    @staticmethod
    def _admission_snapshot(document: MongoDocument) -> AdmissionBatchSnapshot:
        try:
            return _admission_snapshot_from_document(document)
        except Exception:
            raise _dependency_error() from None

    @staticmethod
    def _task_snapshot(document: MongoDocument) -> ImportTaskSnapshot:
        try:
            return _task_snapshot_from_document(document)
        except Exception:
            raise _dependency_error() from None

    def _admission_owned_document(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        session: ClientSession | None = None,
    ) -> tuple[MongoDocument, AdmissionBatchSnapshot]:
        document = self._find_one(
            self._store.admissions,
            {"_id": batch_id, "tenant_id": tenant_id, "owner_subject": owner_subject},
            session=session,
        )
        if document is None:
            raise _resource_not_found("admission")
        snapshot = self._admission_snapshot(document)
        if snapshot.tenant_id != tenant_id or snapshot.owner_subject != owner_subject:
            raise _dependency_error()
        return document, snapshot

    def _task_owned_document(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        session: ClientSession | None = None,
    ) -> tuple[MongoDocument, ImportTaskSnapshot]:
        document = self._find_one(
            self._store.tasks,
            {"_id": task_id, "tenant_id": tenant_id, "owner_subject": owner_subject},
            session=session,
        )
        if document is None:
            raise _resource_not_found("task")
        snapshot = self._task_snapshot(document)
        if snapshot.tenant_id != tenant_id or snapshot.owner_subject != owner_subject:
            raise _dependency_error()
        return document, snapshot

    def _admission_prepare(self, intent: AdmissionBatchIntent) -> AdmissionBatchSnapshot:
        _validate_admission_intent(intent)
        self._ensure_open()
        document = _admission_document(intent, self._now())
        insert_failed = False
        try:
            self._insert_one(self._store.admissions, document)
        except DependencyError:
            insert_failed = True
        if not insert_failed:
            return self._admission_snapshot(document)

        existing = self._find_one(
            self._store.admissions,
            {
                "tenant_id": intent.tenant_id,
                "owner_subject": intent.owner_subject,
                "idempotency_key": intent.idempotency_key,
            },
        )
        if existing is None:
            by_id = self._find_one(self._store.admissions, {"_id": intent.batch_id})
            if by_id is None:
                raise _dependency_error()
            raise _request_conflict()
        snapshot = self._admission_snapshot(existing)
        if snapshot.payload_hash != intent.payload_hash:
            raise _request_conflict()
        return snapshot

    def _admission_get_owned(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
    ) -> AdmissionBatchSnapshot:
        return self._admission_owned_document(batch_id, tenant_id, owner_subject)[1]

    def _admission_get_by_idempotency(
        self,
        tenant_id: str,
        owner_subject: str,
        idempotency_key: str,
    ) -> AdmissionBatchSnapshot:
        document = self._find_one(
            self._store.admissions,
            {
                "tenant_id": tenant_id,
                "owner_subject": owner_subject,
                "idempotency_key": idempotency_key,
            },
        )
        if document is None:
            raise _resource_not_found("admission")
        snapshot = self._admission_snapshot(document)
        if snapshot.tenant_id != tenant_id or snapshot.owner_subject != owner_subject:
            raise _dependency_error()
        return snapshot

    def _admission_get_by_task_id(self, task_id: str) -> AdmissionBatchSnapshot:
        document = self._find_one(self._store.admissions, {"task_ids": task_id})
        if document is None:
            raise _resource_not_found("admission")
        snapshot = self._admission_snapshot(document)
        if sum(item.task_id == task_id for item in snapshot.objects) != 1:
            raise _dependency_error()
        return snapshot

    @staticmethod
    def _object_snapshot(snapshot: AdmissionBatchSnapshot, object_id: str) -> AdmissionObjectSnapshot:
        matches = [item for item in snapshot.objects if item.object_id == object_id]
        if len(matches) != 1:
            raise _resource_not_found("admission")
        return matches[0]

    def _replace_admission(
        self,
        original: MongoDocument,
        updated: MongoDocument,
        *,
        session: ClientSession | None = None,
    ) -> bool:
        return self._replace_one(
            self._store.admissions,
            {"_id": original["_id"], "revision": original["revision"]},
            updated,
            session=session,
        )

    def _admission_mark_write_started(
        self,
        batch_id: str,
        object_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        expected_revision: int | None = None,
        writer_id: str | None = None,
        lease_seconds: int | None = None,
    ) -> AdmissionObjectSnapshot:
        _validate_revision(expected_revision, kind="admission")
        if (writer_id is None) is not (lease_seconds is None):
            raise ValueError("write lease owner and duration must be provided together")
        if writer_id is not None:
            if not _valid_text(writer_id, maximum=128):
                raise ValueError("invalid import writer")
            assert lease_seconds is not None
            if not _valid_int(lease_seconds, minimum=1) or lease_seconds > 3600:
                raise ValueError("invalid import write lease duration")
        original, snapshot = self._admission_owned_document(batch_id, tenant_id, owner_subject)
        current = self._object_snapshot(snapshot, object_id)
        if current.status is ObjectStatus.WRITE_STARTED:
            raw_item = cast(MongoDocument, original["objects"][_object_index(original, object_id)])
            if writer_id is None and raw_item["lease_kind"] is None:
                return current
            if (
                writer_id is not None
                and raw_item["lease_kind"] == _WRITE_LEASE
                and raw_item["cleanup_lease_owner"] == writer_id
            ):
                return current
            raise _state_conflict("admission")
        if expected_revision is not None and current.revision != expected_revision:
            raise _state_conflict("admission")
        if (
            snapshot.status is not AdmissionStatus.PREPARING
            or snapshot.rollback_requested
            or current.status is not ObjectStatus.INTENT_RECORDED
            or current.stored_object is not None
        ):
            raise _state_conflict("admission")

        updated = copy.deepcopy(original)
        item = cast(MongoDocument, updated["objects"][_object_index(updated, object_id)])
        now = self._now()
        item["state"] = ObjectStatus.WRITE_STARTED.value
        if writer_id is not None:
            item["cleanup_lease_owner"] = writer_id
            item["cleanup_lease_token"] = str(uuid.uuid4())
            item["cleanup_lease_expires_at"] = now + timedelta(seconds=cast(int, lease_seconds))
            item["lease_kind"] = _WRITE_LEASE
        _touch_object(updated, item, now)
        try:
            matched = self._replace_admission(original, updated)
        except DependencyError:
            recovered = self._admission_get_owned(batch_id, tenant_id, owner_subject)
            recovered_item = self._object_snapshot(recovered, object_id)
            if recovered_item.status is ObjectStatus.WRITE_STARTED:
                return recovered_item
            raise
        if not matched:
            recovered = self._admission_get_owned(batch_id, tenant_id, owner_subject)
            recovered_item = self._object_snapshot(recovered, object_id)
            if recovered_item.status is ObjectStatus.WRITE_STARTED:
                return recovered_item
            raise _state_conflict("admission")
        return self._object_snapshot(self._admission_snapshot(updated), object_id)

    def _admission_renew_write_lease(
        self,
        batch_id: str,
        object_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        lease_token: str,
        expected_revision: int,
        lease_seconds: int,
    ) -> AdmissionObjectSnapshot:
        if not _valid_text(lease_token, maximum=128):
            raise ValueError("invalid import write lease token")
        _validate_revision(expected_revision, kind="admission")
        if not _valid_int(lease_seconds, minimum=1) or lease_seconds > 3600:
            raise ValueError("invalid import write lease duration")
        original, snapshot = self._admission_owned_document(batch_id, tenant_id, owner_subject)
        current = self._object_snapshot(snapshot, object_id)
        if current.revision != expected_revision:
            raise _state_conflict("admission")
        index = _object_index(original, object_id)
        raw_item = cast(MongoDocument, original["objects"][index])
        now = self._now()
        if (
            raw_item["lease_kind"] != _WRITE_LEASE
            or raw_item["cleanup_lease_token"] != lease_token
            or not _valid_datetime(raw_item["cleanup_lease_expires_at"])
            or raw_item["cleanup_lease_expires_at"] <= now
            or current.stored_object is not None
            or current.status not in {ObjectStatus.WRITE_STARTED, ObjectStatus.WRITE_UNKNOWN}
        ):
            raise _state_conflict("admission")
        updated = copy.deepcopy(original)
        item = cast(MongoDocument, updated["objects"][index])
        expires_at = now + timedelta(seconds=lease_seconds)
        item["cleanup_lease_expires_at"] = expires_at
        _touch_object(updated, item, now)
        _refresh_cleanup_summary(updated)
        try:
            matched = self._replace_admission(original, updated)
        except DependencyError:
            recovered = self._object_snapshot(
                self._admission_get_owned(batch_id, tenant_id, owner_subject),
                object_id,
            )
            if (
                recovered.write_lease_token == lease_token
                and recovered.write_lease_expires_at == expires_at
            ):
                return recovered
            raise
        if not matched:
            raise _state_conflict("admission")
        return self._object_snapshot(self._admission_snapshot(updated), object_id)

    def _admission_record_stored(
        self,
        batch_id: str,
        object_id: str,
        tenant_id: str,
        owner_subject: str,
        stored_object: StoredObject,
        *,
        expected_revision: int | None = None,
        lease_token: str | None = None,
    ) -> AdmissionObjectSnapshot:
        _validate_stored_object(stored_object)
        _validate_revision(expected_revision, kind="admission")
        original, snapshot = self._admission_owned_document(batch_id, tenant_id, owner_subject)
        current = self._object_snapshot(snapshot, object_id)
        if stored_object.ref != current.ref or stored_object.ref.tenant_id != snapshot.tenant_id:
            raise _resource_not_found("admission")
        if current.stored_object is not None:
            if current.stored_object == stored_object and current.status in {
                ObjectStatus.WRITE_CONFIRMED,
                ObjectStatus.DELETE_PENDING,
            }:
                return current
            raise _state_conflict("admission")
        if expected_revision is not None and current.revision != expected_revision:
            raise _state_conflict("admission")
        if snapshot.status in {AdmissionStatus.ACCEPTED, AdmissionStatus.ROLLED_BACK}:
            raise _state_conflict("admission")
        if current.status not in {ObjectStatus.WRITE_STARTED, ObjectStatus.WRITE_UNKNOWN}:
            raise _state_conflict("admission")

        source_item = cast(MongoDocument, original["objects"][_object_index(original, object_id)])
        active_token = source_item["cleanup_lease_token"]
        if active_token is not None:
            if lease_token != active_token or expected_revision is None:
                raise _state_conflict("admission")
            lease_expires_at = source_item["cleanup_lease_expires_at"]
            if not _valid_datetime(lease_expires_at) or lease_expires_at <= self._now():
                raise _state_conflict("admission")
        elif lease_token is not None:
            raise _state_conflict("admission")

        updated = copy.deepcopy(original)
        item = cast(MongoDocument, updated["objects"][_object_index(updated, object_id)])
        now = self._now()
        item["stored_object"] = _stored_document(stored_object)
        item["last_error_code"] = None
        _clear_cleanup_lease(item)
        item["state"] = (
            ObjectStatus.DELETE_PENDING.value
            if snapshot.rollback_requested
            else ObjectStatus.WRITE_CONFIRMED.value
        )
        _touch_object(updated, item, now)
        if snapshot.rollback_requested:
            _refresh_rollback_status(updated)
        elif not any(
            candidate["state"] == ObjectStatus.WRITE_UNKNOWN.value
            for candidate in cast(list[MongoDocument], updated["objects"])
        ):
            updated["state"] = AdmissionStatus.PREPARING.value
            _refresh_cleanup_summary(updated)

        def target(value: AdmissionObjectSnapshot) -> bool:
            return value.stored_object == stored_object and value.status in {
                ObjectStatus.WRITE_CONFIRMED,
                ObjectStatus.DELETE_PENDING,
            }

        try:
            matched = self._replace_admission(original, updated)
        except DependencyError:
            recovered = self._object_snapshot(
                self._admission_get_owned(batch_id, tenant_id, owner_subject),
                object_id,
            )
            if target(recovered):
                return recovered
            raise
        if not matched:
            recovered = self._object_snapshot(
                self._admission_get_owned(batch_id, tenant_id, owner_subject),
                object_id,
            )
            if target(recovered):
                return recovered
            raise _state_conflict("admission")
        return self._object_snapshot(self._admission_snapshot(updated), object_id)

    def _admission_record_write_unknown(
        self,
        batch_id: str,
        object_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        error_code: str = ErrorCode.DEPENDENCY_UNAVAILABLE.value,
        expected_revision: int | None = None,
        lease_token: str | None = None,
    ) -> AdmissionObjectSnapshot:
        _validate_error_code(error_code)
        _validate_revision(expected_revision, kind="admission")
        original, snapshot = self._admission_owned_document(batch_id, tenant_id, owner_subject)
        current = self._object_snapshot(snapshot, object_id)
        if current.status is ObjectStatus.WRITE_UNKNOWN:
            return current
        if expected_revision is not None and current.revision != expected_revision:
            raise _state_conflict("admission")
        if (
            snapshot.status in {AdmissionStatus.ACCEPTED, AdmissionStatus.ROLLED_BACK}
            or snapshot.rollback_requested
            or current.status is not ObjectStatus.WRITE_STARTED
            or current.stored_object is not None
        ):
            raise _state_conflict("admission")
        raw_item = cast(MongoDocument, original["objects"][_object_index(original, object_id)])
        if raw_item["lease_kind"] == _WRITE_LEASE:
            if lease_token != raw_item["cleanup_lease_token"] or expected_revision is None:
                raise _state_conflict("admission")
            if (
                not _valid_datetime(raw_item["cleanup_lease_expires_at"])
                or raw_item["cleanup_lease_expires_at"] <= self._now()
            ):
                raise _state_conflict("admission")
        elif lease_token is not None or raw_item["lease_kind"] is not None:
            raise _state_conflict("admission")

        updated = copy.deepcopy(original)
        item = cast(MongoDocument, updated["objects"][_object_index(updated, object_id)])
        now = self._now()
        item["state"] = ObjectStatus.WRITE_UNKNOWN.value
        item["last_error_code"] = error_code
        updated["state"] = AdmissionStatus.RECONCILIATION_REQUIRED.value
        _touch_object(updated, item, now)

        try:
            matched = self._replace_admission(original, updated)
        except DependencyError:
            recovered = self._object_snapshot(
                self._admission_get_owned(batch_id, tenant_id, owner_subject),
                object_id,
            )
            if recovered.status is ObjectStatus.WRITE_UNKNOWN:
                return recovered
            raise
        if not matched:
            recovered = self._object_snapshot(
                self._admission_get_owned(batch_id, tenant_id, owner_subject),
                object_id,
            )
            if recovered.status is ObjectStatus.WRITE_UNKNOWN:
                return recovered
            raise _state_conflict("admission")
        return self._object_snapshot(self._admission_snapshot(updated), object_id)

    @staticmethod
    def _tasks_from_admission(snapshot: AdmissionBatchSnapshot) -> tuple[NewImportTask, ...]:
        return tuple(
            NewImportTask(
                task_id=item.task_id,
                document_id=item.document_id,
                tenant_id=snapshot.tenant_id,
                owner_subject=snapshot.owner_subject,
                source_ref=item.ref.uri,
                original_name=item.original_name,
                content_type=item.content_type,
                size=item.size,
                sha256=item.sha256,
            )
            for item in sorted(snapshot.objects, key=lambda candidate: candidate.ordinal)
        )

    def _admission_accept_batch(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot:
        snapshot = self._admission_get_owned(batch_id, tenant_id, owner_subject)
        accepted, _ = self.accept_and_enqueue(
            batch_id,
            tenant_id,
            owner_subject,
            self._tasks_from_admission(snapshot),
            expected_revision=expected_revision,
        )
        return accepted

    def _admission_request_rollback(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        reason_code: str = "ADMISSION_FAILED",
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot:
        _validate_error_code(reason_code)
        _validate_revision(expected_revision, kind="admission")
        original, snapshot = self._admission_owned_document(batch_id, tenant_id, owner_subject)
        if snapshot.status is AdmissionStatus.ACCEPTED:
            raise _state_conflict("admission")
        if snapshot.status is AdmissionStatus.ROLLED_BACK:
            return snapshot
        if snapshot.rollback_requested:
            if snapshot.rollback_reason_code == reason_code:
                return snapshot
            raise _state_conflict("admission")
        if expected_revision is not None and snapshot.revision != expected_revision:
            raise _state_conflict("admission")

        updated = copy.deepcopy(original)
        updated["rollback_requested"] = True
        updated["rollback_reason_code"] = reason_code
        now = self._now()
        for item in cast(list[MongoDocument], updated["objects"]):
            status = ObjectStatus(item["state"])
            next_status = status
            if status is ObjectStatus.INTENT_RECORDED:
                next_status = ObjectStatus.NO_OBJECT
            elif status is ObjectStatus.WRITE_STARTED:
                next_status = ObjectStatus.WRITE_UNKNOWN
            elif status is ObjectStatus.WRITE_CONFIRMED:
                next_status = ObjectStatus.DELETE_PENDING
            if next_status is not status:
                item["state"] = next_status.value
                if next_status is ObjectStatus.WRITE_UNKNOWN:
                    item["last_error_code"] = reason_code
                item["revision"] = cast(int, item["revision"]) + 1
                item["updated_at"] = now
        updated["revision"] = cast(int, updated["revision"]) + 1
        updated["updated_at"] = now
        _refresh_rollback_status(updated)

        try:
            matched = self._replace_admission(original, updated)
        except DependencyError:
            recovered = self._admission_get_owned(batch_id, tenant_id, owner_subject)
            if recovered.rollback_requested and recovered.rollback_reason_code == reason_code:
                return recovered
            raise
        if not matched:
            recovered = self._admission_get_owned(batch_id, tenant_id, owner_subject)
            if recovered.rollback_requested and recovered.rollback_reason_code == reason_code:
                return recovered
            raise _state_conflict("admission")
        return self._admission_snapshot(updated)

    @staticmethod
    def _cleanup_lease_from_document(
        batch: AdmissionBatchSnapshot,
        item: AdmissionObjectSnapshot,
        raw_item: MongoDocument,
    ) -> CleanupLease:
        token = raw_item["cleanup_lease_token"]
        expires_at = raw_item["cleanup_lease_expires_at"]
        if (
            raw_item["lease_kind"] != _CLEANUP_LEASE
            or not _valid_opaque(token)
            or not _valid_datetime(expires_at)
        ):
            raise _dependency_error()
        return CleanupLease(
            batch_id=batch.batch_id,
            object_id=item.object_id,
            tenant_id=batch.tenant_id,
            owner_subject=batch.owner_subject,
            expected_ref=item.ref,
            task_id=item.task_id,
            document_id=item.document_id,
            expected_size=item.size,
            expected_sha256=item.sha256,
            object_kind=item.object_kind,
            object_status=item.status,
            stored_object=item.stored_object,
            lease_token=cast(str, token),
            object_revision=item.revision,
            cleanup_attempt=item.cleanup_attempts + 1,
            expires_at=cast(datetime, expires_at),
        )

    def _admission_claim_cleanup(
        self,
        worker_id: str,
        *,
        limit: int,
        lease_seconds: int,
    ) -> tuple[CleanupLease, ...]:
        if not _valid_text(worker_id, maximum=128):
            raise ValueError("invalid cleanup worker")
        if not _valid_int(limit, minimum=1) or limit > 1000:
            raise ValueError("invalid cleanup claim limit")
        if not _valid_int(lease_seconds, minimum=1) or lease_seconds > 3600:
            raise ValueError("invalid cleanup lease duration")

        now = self._now()
        candidates = self._find_many(
            self._store.admissions,
            {
                "rollback_requested": True,
                "state": {
                    "$in": [
                        AdmissionStatus.ROLLBACK_PENDING.value,
                        AdmissionStatus.RECONCILIATION_REQUIRED.value,
                    ]
                },
            },
        )
        try:
            candidates.sort(key=lambda item: item["created_at"])
        except Exception:
            raise _dependency_error() from None

        claimed: list[CleanupLease] = []
        for original in candidates:
            self._admission_snapshot(original)
            updated = copy.deepcopy(original)
            changed_ids: list[str] = []
            for item in cast(list[MongoDocument], updated["objects"]):
                if len(claimed) + len(changed_ids) >= limit:
                    break
                status = ObjectStatus(item["state"])
                if status not in {ObjectStatus.DELETE_PENDING, ObjectStatus.WRITE_UNKNOWN}:
                    continue
                lease_expires_at = item["cleanup_lease_expires_at"]
                if lease_expires_at is not None and (
                    not _valid_datetime(lease_expires_at) or lease_expires_at > now
                ):
                    continue
                next_attempt_at = item["cleanup_next_attempt_at"]
                if next_attempt_at is not None and (
                    not _valid_datetime(next_attempt_at) or next_attempt_at > now
                ):
                    continue
                item["cleanup_lease_owner"] = worker_id
                item["cleanup_lease_token"] = str(uuid.uuid4())
                item["cleanup_lease_expires_at"] = now + timedelta(seconds=lease_seconds)
                item["lease_kind"] = _CLEANUP_LEASE
                _touch_object(updated, item, now)
                changed_ids.append(cast(str, item["object_id"]))
            if not changed_ids:
                continue
            _refresh_cleanup_summary(updated)
            if not self._replace_admission(original, updated):
                continue
            updated_snapshot = self._admission_snapshot(updated)
            for object_id in changed_ids:
                item_snapshot = self._object_snapshot(updated_snapshot, object_id)
                raw_item = cast(MongoDocument, updated["objects"][_object_index(updated, object_id)])
                claimed.append(self._cleanup_lease_from_document(updated_snapshot, item_snapshot, raw_item))
            if len(claimed) >= limit:
                break
        return tuple(claimed[:limit])

    def _leased_admission(
        self,
        lease: CleanupLease,
    ) -> tuple[MongoDocument, AdmissionBatchSnapshot, int, AdmissionObjectSnapshot]:
        if not isinstance(lease, CleanupLease):
            raise ValueError("invalid cleanup lease")
        document = self._find_one(self._store.admissions, {"_id": lease.batch_id})
        if document is None:
            raise _state_conflict("admission")
        snapshot = self._admission_snapshot(document)
        if snapshot.tenant_id != lease.tenant_id or snapshot.owner_subject != lease.owner_subject:
            raise _state_conflict("admission")
        try:
            index = _object_index(document, lease.object_id)
        except AppError:
            raise _state_conflict("admission") from None
        item = self._object_snapshot(snapshot, lease.object_id)
        raw_item = cast(MongoDocument, document["objects"][index])
        if (
            item.ref != lease.expected_ref
            or item.task_id != lease.task_id
            or item.document_id != lease.document_id
            or item.size != lease.expected_size
            or item.sha256 != lease.expected_sha256
            or item.object_kind != lease.object_kind
            or item.status is not lease.object_status
            or item.stored_object != lease.stored_object
            or raw_item["cleanup_lease_token"] != lease.lease_token
            or raw_item["lease_kind"] != _CLEANUP_LEASE
            or item.revision != lease.object_revision
            or raw_item["cleanup_lease_expires_at"] != lease.expires_at
            or lease.expires_at <= self._now()
            or lease.cleanup_attempt != item.cleanup_attempts + 1
        ):
            raise _state_conflict("admission")
        return document, snapshot, index, item

    def _admission_mark_cleanup(
        self,
        lease: CleanupLease,
        *,
        target: ObjectStatus,
    ) -> AdmissionObjectSnapshot:
        original, _, index, current = self._leased_admission(lease)
        if target is ObjectStatus.DELETED:
            if current.status is not ObjectStatus.DELETE_PENDING or current.stored_object is None:
                raise _state_conflict("admission")
        elif current.status not in {ObjectStatus.DELETE_PENDING, ObjectStatus.WRITE_UNKNOWN}:
            raise _state_conflict("admission")

        updated = copy.deepcopy(original)
        item = cast(MongoDocument, updated["objects"][index])
        now = self._now()
        item["state"] = target.value
        item["last_error_code"] = None
        item["cleanup_next_attempt_at"] = None
        _clear_cleanup_lease(item)
        _touch_object(updated, item, now)
        _refresh_rollback_status(updated)

        try:
            matched = self._replace_admission(original, updated)
        except DependencyError:
            recovered = self._object_snapshot(
                self._admission_get_owned(lease.batch_id, lease.tenant_id, lease.owner_subject),
                lease.object_id,
            )
            if recovered.status is target:
                return recovered
            raise
        if not matched:
            recovered = self._object_snapshot(
                self._admission_get_owned(lease.batch_id, lease.tenant_id, lease.owner_subject),
                lease.object_id,
            )
            if recovered.status is target:
                return recovered
            raise _state_conflict("admission")
        return self._object_snapshot(self._admission_snapshot(updated), lease.object_id)

    def _admission_mark_deleted(self, lease: CleanupLease) -> AdmissionObjectSnapshot:
        return self._admission_mark_cleanup(lease, target=ObjectStatus.DELETED)

    def _admission_mark_no_object(self, lease: CleanupLease) -> AdmissionObjectSnapshot:
        return self._admission_mark_cleanup(lease, target=ObjectStatus.NO_OBJECT)

    def _admission_release_cleanup(
        self,
        lease: CleanupLease,
        *,
        error_code: str | None = None,
    ) -> AdmissionObjectSnapshot:
        if error_code is not None:
            _validate_error_code(error_code)
        original, _, index, current = self._leased_admission(lease)
        updated = copy.deepcopy(original)
        item = cast(MongoDocument, updated["objects"][index])
        now = self._now()
        attempts = current.cleanup_attempts + 1
        exponent = min(attempts - 1, 30)
        delay = min(
            self._cleanup_retry_base_seconds * (2**exponent),
            self._cleanup_retry_max_seconds,
        )
        item["last_error_code"] = error_code
        item["cleanup_attempts"] = attempts
        item["cleanup_next_attempt_at"] = now + timedelta(seconds=delay)
        _clear_cleanup_lease(item)
        _touch_object(updated, item, now)
        _refresh_cleanup_summary(updated)

        def recovered_result() -> AdmissionObjectSnapshot:
            return self._object_snapshot(
                self._admission_get_owned(lease.batch_id, lease.tenant_id, lease.owner_subject),
                lease.object_id,
            )

        try:
            matched = self._replace_admission(original, updated)
        except DependencyError:
            recovered = recovered_result()
            if recovered.cleanup_attempts == attempts and recovered.last_error_code == error_code:
                return recovered
            raise
        if not matched:
            recovered = recovered_result()
            if recovered.cleanup_attempts == attempts and recovered.last_error_code == error_code:
                return recovered
            raise _state_conflict("admission")
        return self._object_snapshot(self._admission_snapshot(updated), lease.object_id)

    def _admission_finish_cleanup(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot:
        _validate_revision(expected_revision, kind="admission")
        original, snapshot = self._admission_owned_document(batch_id, tenant_id, owner_subject)
        if snapshot.status is AdmissionStatus.ROLLED_BACK:
            return snapshot
        if expected_revision is not None and snapshot.revision != expected_revision:
            raise _state_conflict("admission")
        if not snapshot.rollback_requested or any(
            item.status not in {ObjectStatus.DELETED, ObjectStatus.NO_OBJECT}
            for item in snapshot.objects
        ):
            raise _state_conflict("admission")
        updated = copy.deepcopy(original)
        updated["state"] = AdmissionStatus.ROLLED_BACK.value
        updated["revision"] = cast(int, updated["revision"]) + 1
        updated["updated_at"] = self._now()
        _refresh_cleanup_summary(updated)
        try:
            matched = self._replace_admission(original, updated)
        except DependencyError:
            recovered = self._admission_get_owned(batch_id, tenant_id, owner_subject)
            if recovered.status is AdmissionStatus.ROLLED_BACK:
                return recovered
            raise
        if not matched:
            recovered = self._admission_get_owned(batch_id, tenant_id, owner_subject)
            if recovered.status is AdmissionStatus.ROLLED_BACK:
                return recovered
            raise _state_conflict("admission")
        return self._admission_snapshot(updated)

    def _admission_recover_stale_preparing(
        self,
        older_than: datetime,
        *,
        limit: int,
    ) -> tuple[AdmissionBatchSnapshot, ...]:
        if not _valid_datetime(older_than):
            raise ValueError("invalid import admission recovery cutoff")
        if not _valid_int(limit, minimum=1) or limit > 1000:
            raise ValueError("invalid import admission recovery limit")
        cutoff = older_than.astimezone(UTC).replace(
            microsecond=(older_than.microsecond // 1000) * 1000
        )
        documents = self._find_many(
            self._store.admissions,
            {
                "rollback_requested": False,
                "state": {"$nin": [AdmissionStatus.ACCEPTED.value, AdmissionStatus.ROLLED_BACK.value]},
            },
        )
        try:
            documents.sort(key=lambda item: item["updated_at"])
        except Exception:
            raise _dependency_error() from None
        recovered: list[AdmissionBatchSnapshot] = []
        for document in documents:
            if len(recovered) == limit:
                break
            snapshot = self._admission_snapshot(document)
            if snapshot.updated_at > cutoff:
                continue
            if any(
                item.get("lease_kind") == _WRITE_LEASE
                and _valid_datetime(item.get("cleanup_lease_expires_at"))
                and item["cleanup_lease_expires_at"] > self._now()
                for item in cast(list[MongoDocument], document["objects"])
            ):
                continue
            try:
                recovered.append(
                    self._admission_request_rollback(
                        snapshot.batch_id,
                        snapshot.tenant_id,
                        snapshot.owner_subject,
                        reason_code="ADMISSION_RECOVERY",
                        expected_revision=snapshot.revision,
                    )
                )
            except AppError as exc:
                if exc.code is ErrorCode.RESOURCE_CONFLICT:
                    continue
                raise
        return tuple(recovered)

    def _with_transaction(self, callback: Callable[[ClientSession], _T]) -> _T:
        self._ensure_open()
        failed = False
        result: _T | None = None
        try:
            result = self._store.with_transaction(callback)
        except AppError as exc:
            if isinstance(exc, DependencyError):
                failed = True
            else:
                raise
        except Exception:
            failed = True
        if failed:
            raise _dependency_error() from None
        return cast(_T, result)

    @staticmethod
    def _same_task(snapshot: ImportTaskSnapshot, task: NewImportTask) -> bool:
        return _task_from_snapshot(snapshot) == task

    def _task_create_many(self, tasks: Sequence[NewImportTask]) -> tuple[ImportTaskSnapshot, ...]:
        batch = tuple(tasks)
        if not batch:
            raise ValueError("an import task batch cannot be empty")
        for task in batch:
            _validate_task(task)
        task_ids = [task.task_id for task in batch]
        if len(set(task_ids)) != len(task_ids):
            raise _state_conflict("task")
        now = self._now()
        documents = [_task_document(task, now, admission_id=None) for task in batch]

        def create(session: ClientSession) -> tuple[ImportTaskSnapshot, ...]:
            existing = self._find_many(
                self._store.tasks,
                {"_id": {"$in": task_ids}},
                session=session,
            )
            if existing:
                raise _state_conflict("task")
            self._insert_many(self._store.tasks, documents, session=session)
            return tuple(self._task_snapshot(document) for document in documents)

        try:
            return self._with_transaction(create)
        except DependencyError:
            recovered: list[ImportTaskSnapshot] = []
            for task in batch:
                document = self._find_one(self._store.tasks, {"_id": task.task_id})
                if document is None:
                    raise
                snapshot = self._task_snapshot(document)
                if not self._same_task(snapshot, task):
                    raise _state_conflict("task") from None
                recovered.append(snapshot)
            return tuple(recovered)

    def _task_get_owned(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
    ) -> ImportTaskSnapshot:
        return self._task_owned_document(task_id, tenant_id, owner_subject)[1]

    def _replace_task(
        self,
        original: MongoDocument,
        updated: MongoDocument,
        *,
        session: ClientSession | None = None,
    ) -> bool:
        return self._replace_one(
            self._store.tasks,
            {"_id": original["_id"], "revision": original["revision"]},
            updated,
            session=session,
        )

    def _task_outbox_document(
        self,
        task_document: MongoDocument,
        *,
        session: ClientSession | None = None,
    ) -> MongoDocument | None:
        admission_id = task_document.get("admission_id")
        if admission_id is None:
            return None
        if not _is_canonical_uuid(admission_id):
            raise _dependency_error()
        snapshot = self._task_snapshot(task_document)
        task = _task_from_snapshot(snapshot)
        outbox = self._find_one(
            self._store.outbox,
            {"_id": f"import.execute:{task.task_id}:v1"},
            session=session,
        )
        if outbox is None:
            raise _dependency_error()
        try:
            _validate_outbox_document(outbox, task, cast(str, admission_id))
        except Exception:
            raise _dependency_error() from None
        return outbox

    def _replace_outbox(
        self,
        original: MongoDocument,
        updated: MongoDocument,
        *,
        session: ClientSession | None = None,
    ) -> bool:
        return self._replace_one(
            self._store.outbox,
            {"_id": original["_id"], "revision": original["revision"]},
            updated,
            session=session,
        )

    @staticmethod
    def _claim_outbox(
        original: MongoDocument,
        *,
        worker_id: str,
        lease_token: str,
        attempt_count: int,
        expires_at: datetime,
        now: datetime,
    ) -> MongoDocument:
        if original["status"] not in {"pending", "retry", "claimed"}:
            raise _dependency_error()
        if original["status"] == "claimed":
            existing_expiry = original["lease_expires_at"]
            if not _valid_datetime(existing_expiry) or existing_expiry > now:
                raise _state_conflict("task")
        updated = copy.deepcopy(original)
        updated["status"] = "claimed"
        updated["lease_owner"] = worker_id
        updated["lease_token"] = lease_token
        updated["lease_expires_at"] = expires_at
        updated["attempt_count"] = attempt_count
        updated["claimed_at"] = now
        updated["last_error_code"] = None
        updated["available_at"] = now
        updated["revision"] = cast(int, updated["revision"]) + 1
        updated["updated_at"] = now
        return updated

    def _terminal_outbox(
        self,
        original: MongoDocument,
        *,
        status: str,
        error_code: str | None,
        now: datetime,
    ) -> MongoDocument:
        if status not in {"completed", "failed"}:
            raise ValueError("invalid outbox terminal status")
        updated = copy.deepcopy(original)
        updated["status"] = status
        updated["lease_owner"] = None
        updated["lease_token"] = None
        updated["lease_expires_at"] = None
        updated["last_error_code"] = error_code
        updated["completed_at"] = now if status == "completed" else None
        updated["failed_at"] = now if status == "failed" else None
        updated["expire_at"] = now + self._terminal_retention
        updated["revision"] = cast(int, updated["revision"]) + 1
        updated["updated_at"] = now
        return updated

    def _task_begin(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
    ) -> ImportTaskSnapshot:
        original, snapshot = self._task_owned_document(task_id, tenant_id, owner_subject)
        if snapshot.status is not ImportTaskStatus.QUEUED:
            raise _state_conflict("task")
        updated = copy.deepcopy(original)
        now = self._now()
        updated["status"] = ImportTaskStatus.PROCESSING.value
        updated["attempt_count"] = snapshot.attempt_count + 1
        _touch_task(updated, now)
        try:
            matched = self._replace_task(original, updated)
        except DependencyError:
            recovered = self._task_get_owned(task_id, tenant_id, owner_subject)
            if (
                recovered.status is ImportTaskStatus.PROCESSING
                and recovered.attempt_count == snapshot.attempt_count + 1
            ):
                return recovered
            raise
        if not matched:
            raise _state_conflict("task")
        return self._task_snapshot(updated)

    def _task_claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        max_attempts: int,
    ) -> ImportTaskLease | None:
        if not _valid_text(worker_id, maximum=128):
            raise ValueError("invalid import worker")
        if not _valid_int(lease_seconds, minimum=1) or lease_seconds > 3600:
            raise ValueError("invalid import task lease duration")
        if not _valid_int(max_attempts, minimum=1) or max_attempts > 100:
            raise ValueError("invalid import task attempt limit")

        now = self._now()
        candidates = self._find_many(
            self._store.tasks,
            {
                "status": {
                    "$in": [ImportTaskStatus.QUEUED.value, ImportTaskStatus.PROCESSING.value]
                }
            },
        )
        try:
            candidates.sort(key=lambda item: item["created_at"])
        except Exception:
            raise _dependency_error() from None
        for candidate in candidates:
            task_id = candidate.get("_id")
            if not isinstance(task_id, str):
                raise _dependency_error()

            def claim(
                session: ClientSession,
                selected_task_id: str = task_id,
            ) -> ImportTaskLease | None:
                original = self._find_one(
                    self._store.tasks,
                    {"_id": selected_task_id},
                    session=session,
                )
                if original is None:
                    return None
                snapshot = self._task_snapshot(original)
                eligible = (
                    snapshot.status is ImportTaskStatus.QUEUED and snapshot.available_at <= now
                ) or (
                    snapshot.status is ImportTaskStatus.PROCESSING
                    and original["lease_token"] is not None
                    and _valid_datetime(original["lease_expires_at"])
                    and original["lease_expires_at"] <= now
                )
                if not eligible:
                    return None
                outbox = self._task_outbox_document(original, session=session)
                updated = copy.deepcopy(original)
                if snapshot.attempt_count >= max_attempts:
                    updated["status"] = ImportTaskStatus.FAILED.value
                    updated["current_step"] = None
                    updated["error_code"] = ErrorCode.INTERNAL_ERROR.value
                    updated["expire_at"] = now + self._terminal_retention
                    _clear_task_lease(updated)
                    _touch_task(updated, now)
                    if not self._replace_task(original, updated, session=session):
                        raise _state_conflict("task")
                    if outbox is not None:
                        terminal = self._terminal_outbox(
                            outbox,
                            status="failed",
                            error_code=ErrorCode.INTERNAL_ERROR.value,
                            now=now,
                        )
                        if not self._replace_outbox(outbox, terminal, session=session):
                            raise _state_conflict("task")
                    return None

                token = str(uuid.uuid4())
                expires_at = now + timedelta(seconds=lease_seconds)
                updated["status"] = ImportTaskStatus.PROCESSING.value
                updated["current_step"] = None
                updated["attempt_count"] = snapshot.attempt_count + 1
                updated["lease_owner"] = worker_id
                updated["lease_token"] = token
                updated["lease_expires_at"] = expires_at
                _touch_task(updated, now)
                if not self._replace_task(original, updated, session=session):
                    raise _state_conflict("task")
                if outbox is not None:
                    claimed_outbox = self._claim_outbox(
                        outbox,
                        worker_id=worker_id,
                        lease_token=token,
                        attempt_count=snapshot.attempt_count + 1,
                        expires_at=expires_at,
                        now=now,
                    )
                    if not self._replace_outbox(outbox, claimed_outbox, session=session):
                        raise _state_conflict("task")
                claimed = self._task_snapshot(updated)
                return ImportTaskLease(
                    task=claimed,
                    worker_id=worker_id,
                    lease_token=token,
                    attempt_count=claimed.attempt_count,
                    expires_at=expires_at,
                )

            try:
                lease = self._with_transaction(claim)
            except AppError as exc:
                if exc.code is ErrorCode.RESOURCE_CONFLICT:
                    continue
                raise
            if lease is not None:
                return lease
        return None

    def _task_leased_document(
        self,
        lease: ImportTaskLease,
        *,
        session: ClientSession | None = None,
        require_exact_expiry: bool = True,
    ) -> tuple[MongoDocument, ImportTaskSnapshot]:
        if not isinstance(lease, ImportTaskLease):
            raise ValueError("invalid import task lease")
        original, snapshot = self._task_owned_document(
            lease.task.task_id,
            lease.task.tenant_id,
            lease.task.owner_subject,
            session=session,
        )
        current_expiry = original["lease_expires_at"]
        if (
            snapshot.status is not ImportTaskStatus.PROCESSING
            or original["lease_owner"] != lease.worker_id
            or original["lease_token"] != lease.lease_token
            or snapshot.attempt_count != lease.attempt_count
            or not _valid_datetime(current_expiry)
            or current_expiry <= self._now()
            or require_exact_expiry
            and current_expiry != lease.expires_at
        ):
            raise _state_conflict("task")
        return original, snapshot

    def _task_renew_lease(
        self,
        lease: ImportTaskLease,
        *,
        lease_seconds: int,
    ) -> ImportTaskLease:
        if not _valid_int(lease_seconds, minimum=1) or lease_seconds > 3600:
            raise ValueError("invalid import task lease duration")
        now = self._now()
        expires_at = now + timedelta(seconds=lease_seconds)

        def renew(session: ClientSession) -> ImportTaskLease:
            original, _ = self._task_leased_document(lease, session=session)
            outbox = self._task_outbox_document(original, session=session)
            updated = copy.deepcopy(original)
            updated["lease_expires_at"] = expires_at
            _touch_task(updated, now)
            if not self._replace_task(original, updated, session=session):
                raise _state_conflict("task")
            if outbox is not None:
                if outbox["status"] != "claimed" or outbox["lease_token"] != lease.lease_token:
                    raise _dependency_error()
                renewed_outbox = copy.deepcopy(outbox)
                renewed_outbox["lease_expires_at"] = expires_at
                renewed_outbox["revision"] = cast(int, renewed_outbox["revision"]) + 1
                renewed_outbox["updated_at"] = now
                if not self._replace_outbox(outbox, renewed_outbox, session=session):
                    raise _state_conflict("task")
            return ImportTaskLease(
                task=self._task_snapshot(updated),
                worker_id=lease.worker_id,
                lease_token=lease.lease_token,
                attempt_count=lease.attempt_count,
                expires_at=expires_at,
            )

        try:
            return self._with_transaction(renew)
        except DependencyError:
            task_document, snapshot = self._task_owned_document(
                lease.task.task_id,
                lease.task.tenant_id,
                lease.task.owner_subject,
            )
            if task_document["lease_token"] != lease.lease_token or task_document["lease_expires_at"] != expires_at:
                raise
            outbox = self._task_outbox_document(task_document)
            if outbox is not None and (
                outbox["status"] != "claimed"
                or outbox["lease_token"] != lease.lease_token
                or outbox["lease_expires_at"] != expires_at
            ):
                raise
            return ImportTaskLease(
                task=snapshot,
                worker_id=lease.worker_id,
                lease_token=lease.lease_token,
                attempt_count=lease.attempt_count,
                expires_at=expires_at,
            )

    def _task_assert_lease(self, lease: ImportTaskLease) -> ImportTaskLease:
        original, snapshot = self._task_leased_document(
            lease,
            require_exact_expiry=False,
        )
        expires_at = original["lease_expires_at"]
        if not _valid_datetime(expires_at):
            raise _state_conflict("task")
        outbox = self._task_outbox_document(original)
        if outbox is not None and (
            outbox["status"] != "claimed"
            or outbox["lease_token"] != lease.lease_token
            or outbox["lease_expires_at"] != expires_at
        ):
            raise _dependency_error()
        return ImportTaskLease(
            task=snapshot,
            worker_id=lease.worker_id,
            lease_token=lease.lease_token,
            attempt_count=lease.attempt_count,
            expires_at=expires_at,
        )

    def _task_retry(
        self,
        lease: ImportTaskLease,
        *,
        error_code: str,
        delay_seconds: float,
    ) -> ImportTaskSnapshot:
        _validate_error_code(error_code)
        if (
            isinstance(delay_seconds, bool)
            or not isinstance(delay_seconds, (int, float))
            or not 0 <= delay_seconds <= 86400
        ):
            raise ValueError("invalid import retry delay")
        now = self._now()
        available_at = now + timedelta(seconds=float(delay_seconds))

        def retry(session: ClientSession) -> ImportTaskSnapshot:
            original, _ = self._task_leased_document(
                lease,
                session=session,
                require_exact_expiry=False,
            )
            outbox = self._task_outbox_document(original, session=session)
            updated = copy.deepcopy(original)
            updated["status"] = ImportTaskStatus.QUEUED.value
            updated["current_step"] = None
            updated["error_code"] = error_code
            updated["available_at"] = available_at
            updated["next_attempt_at"] = available_at
            _clear_task_lease(updated)
            _touch_task(updated, now)
            if not self._replace_task(original, updated, session=session):
                raise _state_conflict("task")
            if outbox is not None:
                if outbox["status"] != "claimed" or outbox["lease_token"] != lease.lease_token:
                    raise _dependency_error()
                retry_outbox = copy.deepcopy(outbox)
                retry_outbox["status"] = "retry"
                retry_outbox["available_at"] = available_at
                retry_outbox["lease_owner"] = None
                retry_outbox["lease_token"] = None
                retry_outbox["lease_expires_at"] = None
                retry_outbox["last_error_code"] = error_code
                retry_outbox["revision"] = cast(int, retry_outbox["revision"]) + 1
                retry_outbox["updated_at"] = now
                if not self._replace_outbox(outbox, retry_outbox, session=session):
                    raise _state_conflict("task")
            return self._task_snapshot(updated)

        try:
            return self._with_transaction(retry)
        except DependencyError:
            task_document, snapshot = self._task_owned_document(
                lease.task.task_id,
                lease.task.tenant_id,
                lease.task.owner_subject,
            )
            if (
                snapshot.status is not ImportTaskStatus.QUEUED
                or snapshot.available_at != available_at
                or snapshot.error_code != error_code
            ):
                raise
            outbox = self._task_outbox_document(task_document)
            if outbox is not None and (
                outbox["status"] != "retry"
                or outbox["available_at"] != available_at
                or outbox["last_error_code"] != error_code
            ):
                raise
            return snapshot

    def _require_task_token(self, document: MongoDocument, lease_token: str | None) -> None:
        active_token = document["lease_token"]
        if active_token is None:
            if lease_token is not None:
                raise _state_conflict("task")
            return
        if (
            lease_token != active_token
            or not _valid_datetime(document["lease_expires_at"])
            or document["lease_expires_at"] <= self._now()
        ):
            raise _state_conflict("task")

    def _task_start_step(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        step: str,
        *,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot:
        if not isinstance(step, str) or not _STEP_NAME.fullmatch(step):
            raise ValueError("invalid import step")
        original, snapshot = self._task_owned_document(task_id, tenant_id, owner_subject)
        self._require_task_token(original, lease_token)
        if snapshot.status is not ImportTaskStatus.PROCESSING or snapshot.current_step is not None:
            raise _state_conflict("task")
        updated = copy.deepcopy(original)
        updated["current_step"] = step
        _touch_task(updated, self._now())
        try:
            matched = self._replace_task(original, updated)
        except DependencyError:
            recovered = self._task_get_owned(task_id, tenant_id, owner_subject)
            if recovered.status is ImportTaskStatus.PROCESSING and recovered.current_step == step:
                return recovered
            raise
        if not matched:
            raise _state_conflict("task")
        return self._task_snapshot(updated)

    def _task_complete_step(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        step: str,
        *,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot:
        original, snapshot = self._task_owned_document(task_id, tenant_id, owner_subject)
        self._require_task_token(original, lease_token)
        if snapshot.status is not ImportTaskStatus.PROCESSING or snapshot.current_step != step:
            raise _state_conflict("task")
        updated = copy.deepcopy(original)
        completed_steps = cast(list[str], updated["completed_steps"])
        if step not in completed_steps:
            completed_steps.append(step)
        updated["current_step"] = None
        _touch_task(updated, self._now())
        try:
            matched = self._replace_task(original, updated)
        except DependencyError:
            recovered = self._task_get_owned(task_id, tenant_id, owner_subject)
            if recovered.current_step is None and step in recovered.completed_steps:
                return recovered
            raise
        if not matched:
            raise _state_conflict("task")
        return self._task_snapshot(updated)

    def _task_complete(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        imported_chunk_count: int,
        commit_token: str,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot:
        if not _valid_int(imported_chunk_count):
            raise ValueError("invalid imported chunk count")
        if not _valid_opaque(commit_token, maximum=256) or " " in commit_token:
            raise ValueError("invalid import commit token")
        now = self._now()

        def complete(session: ClientSession) -> ImportTaskSnapshot:
            original, snapshot = self._task_owned_document(
                task_id,
                tenant_id,
                owner_subject,
                session=session,
            )
            outbox = self._task_outbox_document(original, session=session)
            if snapshot.status is ImportTaskStatus.COMPLETED:
                if (
                    snapshot.imported_chunk_count != imported_chunk_count
                    or snapshot.commit_token != commit_token
                ):
                    raise _state_conflict("task")
                if outbox is not None and outbox["status"] != "completed":
                    terminal = self._terminal_outbox(
                        outbox,
                        status="completed",
                        error_code=None,
                        now=now,
                    )
                    if not self._replace_outbox(outbox, terminal, session=session):
                        raise _state_conflict("task")
                return snapshot
            self._require_task_token(original, lease_token)
            if snapshot.status is not ImportTaskStatus.PROCESSING or snapshot.current_step is not None:
                raise _state_conflict("task")
            updated = copy.deepcopy(original)
            updated["status"] = ImportTaskStatus.COMPLETED.value
            updated["imported_chunk_count"] = imported_chunk_count
            updated["commit_token"] = commit_token
            updated["expire_at"] = now + self._terminal_retention
            _clear_task_lease(updated)
            _touch_task(updated, now)
            if not self._replace_task(original, updated, session=session):
                raise _state_conflict("task")
            if outbox is not None:
                terminal = self._terminal_outbox(
                    outbox,
                    status="completed",
                    error_code=None,
                    now=now,
                )
                if not self._replace_outbox(outbox, terminal, session=session):
                    raise _state_conflict("task")
            return self._task_snapshot(updated)

        try:
            return self._with_transaction(complete)
        except DependencyError:
            document, recovered = self._task_owned_document(task_id, tenant_id, owner_subject)
            if (
                recovered.status is ImportTaskStatus.COMPLETED
                and recovered.imported_chunk_count == imported_chunk_count
                and recovered.commit_token == commit_token
            ):
                outbox = self._task_outbox_document(document)
                if outbox is not None and outbox["status"] != "completed":
                    raise
                return recovered
            raise

    def _task_fail(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        error_code: str,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot:
        _validate_error_code(error_code)
        now = self._now()

        def fail(session: ClientSession) -> ImportTaskSnapshot:
            original, snapshot = self._task_owned_document(
                task_id,
                tenant_id,
                owner_subject,
                session=session,
            )
            outbox = self._task_outbox_document(original, session=session)
            if snapshot.status is ImportTaskStatus.FAILED:
                if outbox is not None and outbox["status"] != "failed":
                    terminal = self._terminal_outbox(
                        outbox,
                        status="failed",
                        error_code=snapshot.error_code,
                        now=now,
                    )
                    if not self._replace_outbox(outbox, terminal, session=session):
                        raise _state_conflict("task")
                return snapshot
            if snapshot.status is ImportTaskStatus.COMPLETED:
                raise _state_conflict("task")
            self._require_task_token(original, lease_token)
            updated = copy.deepcopy(original)
            updated["status"] = ImportTaskStatus.FAILED.value
            updated["current_step"] = None
            updated["error_code"] = error_code
            updated["expire_at"] = now + self._terminal_retention
            _clear_task_lease(updated)
            _touch_task(updated, now)
            if not self._replace_task(original, updated, session=session):
                raise _state_conflict("task")
            if outbox is not None:
                terminal = self._terminal_outbox(
                    outbox,
                    status="failed",
                    error_code=error_code,
                    now=now,
                )
                if not self._replace_outbox(outbox, terminal, session=session):
                    raise _state_conflict("task")
            return self._task_snapshot(updated)

        try:
            return self._with_transaction(fail)
        except DependencyError:
            document, recovered = self._task_owned_document(task_id, tenant_id, owner_subject)
            if recovered.status is ImportTaskStatus.FAILED:
                outbox = self._task_outbox_document(document)
                if outbox is not None and outbox["status"] != "failed":
                    raise
                return recovered
            raise

    def _task_rollback_queued(
        self,
        tenant_id: str,
        owner_subject: str,
        task_ids: Sequence[str],
    ) -> None:
        unique_ids = tuple(dict.fromkeys(task_ids))
        if not unique_ids:
            return

        def rollback(session: ClientSession) -> None:
            existing: list[MongoDocument] = []
            outbox_ids: list[str] = []
            for task_id in unique_ids:
                document = self._find_one(self._store.tasks, {"_id": task_id}, session=session)
                if document is None:
                    continue
                snapshot = self._task_snapshot(document)
                if snapshot.tenant_id != tenant_id or snapshot.owner_subject != owner_subject:
                    raise _resource_not_found("task")
                if snapshot.status is not ImportTaskStatus.QUEUED:
                    raise _state_conflict("task")
                outbox = self._task_outbox_document(document, session=session)
                if outbox is not None:
                    outbox_ids.append(cast(str, outbox["_id"]))
                existing.append(document)
            if existing:
                ids = [document["_id"] for document in existing]
                deleted = self._delete_many(
                    self._store.tasks,
                    {"_id": {"$in": ids}, "status": ImportTaskStatus.QUEUED.value},
                    session=session,
                )
                if deleted != len(ids):
                    raise _state_conflict("task")
                if outbox_ids:
                    deleted_outbox = self._delete_many(
                        self._store.outbox,
                        {"_id": {"$in": outbox_ids}},
                        session=session,
                    )
                    if deleted_outbox != len(outbox_ids):
                        raise _state_conflict("task")

        self._with_transaction(rollback)

    @staticmethod
    def _validate_coordinator_tasks(
        admission: AdmissionBatchSnapshot,
        tasks: Sequence[NewImportTask],
    ) -> tuple[NewImportTask, ...]:
        batch = tuple(tasks)
        ordered_objects = tuple(sorted(admission.objects, key=lambda item: item.ordinal))
        if len(batch) != len(ordered_objects):
            raise _state_conflict("admission")
        for task, item in zip(batch, ordered_objects, strict=True):
            _validate_task(task)
            if (
                task.task_id != item.task_id
                or task.document_id != item.document_id
                or task.tenant_id != admission.tenant_id
                or task.owner_subject != admission.owner_subject
                or task.source_ref != item.ref.uri
                or task.original_name != item.original_name
                or task.content_type != item.content_type
                or task.size != item.size
                or task.sha256 != item.sha256
            ):
                raise _state_conflict("admission")
        return batch

    def _accepted_result(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        tasks: Sequence[NewImportTask],
        *,
        session: ClientSession | None = None,
    ) -> tuple[AdmissionBatchSnapshot, tuple[ImportTaskSnapshot, ...]]:
        _, admission = self._admission_owned_document(
            batch_id,
            tenant_id,
            owner_subject,
            session=session,
        )
        batch = self._validate_coordinator_tasks(admission, tasks)
        if admission.status is not AdmissionStatus.ACCEPTED:
            raise _state_conflict("admission")
        snapshots: list[ImportTaskSnapshot] = []
        for task in batch:
            document = self._find_one(self._store.tasks, {"_id": task.task_id}, session=session)
            if document is None:
                raise _dependency_error()
            snapshot = self._task_snapshot(document)
            if document.get("admission_id") != batch_id or not self._same_task(snapshot, task):
                raise _dependency_error()
            outbox = self._find_one(
                self._store.outbox,
                {"_id": f"import.execute:{task.task_id}:v1"},
                session=session,
            )
            if outbox is None:
                raise _dependency_error()
            try:
                _validate_outbox_document(outbox, task, batch_id)
            except Exception:
                raise _dependency_error() from None
            snapshots.append(snapshot)
        return admission, tuple(snapshots)

    def accept_and_enqueue(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        tasks: Sequence[NewImportTask],
        *,
        expected_revision: int | None = None,
    ) -> tuple[AdmissionBatchSnapshot, tuple[ImportTaskSnapshot, ...]]:
        _validate_revision(expected_revision, kind="admission")
        requested = tuple(tasks)

        def accept(
            session: ClientSession,
        ) -> tuple[AdmissionBatchSnapshot, tuple[ImportTaskSnapshot, ...]]:
            original, admission = self._admission_owned_document(
                batch_id,
                tenant_id,
                owner_subject,
                session=session,
            )
            batch = self._validate_coordinator_tasks(admission, requested)
            if admission.status is AdmissionStatus.ACCEPTED:
                return self._accepted_result(
                    batch_id,
                    tenant_id,
                    owner_subject,
                    batch,
                    session=session,
                )
            if expected_revision is not None and admission.revision != expected_revision:
                raise _state_conflict("admission")
            if admission.status is AdmissionStatus.ROLLED_BACK or admission.rollback_requested:
                raise _state_conflict("admission")
            if not all(
                item.status is ObjectStatus.WRITE_CONFIRMED and item.stored_object is not None
                for item in admission.objects
            ):
                raise _state_conflict("admission")

            task_ids = [task.task_id for task in batch]
            dedupe_keys = [f"import.execute:{task_id}:v1" for task_id in task_ids]
            if self._find_many(self._store.tasks, {"_id": {"$in": task_ids}}, session=session):
                raise _state_conflict("admission")
            if self._find_many(self._store.outbox, {"_id": {"$in": dedupe_keys}}, session=session):
                raise _state_conflict("admission")

            now = self._now()
            task_documents = [_task_document(task, now, admission_id=batch_id) for task in batch]
            outbox_documents = [_outbox_document(task, batch_id, now) for task in batch]
            self._insert_many(self._store.tasks, task_documents, session=session)
            self._insert_many(self._store.outbox, outbox_documents, session=session)

            updated = copy.deepcopy(original)
            updated["state"] = AdmissionStatus.ACCEPTED.value
            updated["revision"] = cast(int, updated["revision"]) + 1
            updated["updated_at"] = now
            if not self._replace_admission(original, updated, session=session):
                raise _state_conflict("admission")
            return (
                self._admission_snapshot(updated),
                tuple(self._task_snapshot(document) for document in task_documents),
            )

        try:
            return self._with_transaction(accept)
        except DependencyError:
            try:
                return self._accepted_result(batch_id, tenant_id, owner_subject, requested)
            except (AppError, ValueError):
                raise _dependency_error() from None

    def probe(self) -> None:
        self._ensure_open()
        self._db_call(self._store.require_ready)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._store.close()


class _MongoAdmissionFacade:
    durable = True

    def __init__(self, coordinator: MongoImportCoordinator) -> None:
        self._coordinator = coordinator

    def prepare_batch(self, intent: AdmissionBatchIntent) -> AdmissionBatchSnapshot:
        return self._coordinator._admission_prepare(intent)

    def get_owned(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
    ) -> AdmissionBatchSnapshot:
        return self._coordinator._admission_get_owned(batch_id, tenant_id, owner_subject)

    def get_by_idempotency(
        self,
        tenant_id: str,
        owner_subject: str,
        idempotency_key: str,
    ) -> AdmissionBatchSnapshot:
        return self._coordinator._admission_get_by_idempotency(
            tenant_id,
            owner_subject,
            idempotency_key,
        )

    def get_by_task_id(self, task_id: str) -> AdmissionBatchSnapshot:
        return self._coordinator._admission_get_by_task_id(task_id)

    def record_stored(
        self,
        batch_id: str,
        object_id: str,
        tenant_id: str,
        owner_subject: str,
        stored_object: StoredObject,
        *,
        expected_revision: int | None = None,
        lease_token: str | None = None,
    ) -> AdmissionObjectSnapshot:
        return self._coordinator._admission_record_stored(
            batch_id,
            object_id,
            tenant_id,
            owner_subject,
            stored_object,
            expected_revision=expected_revision,
            lease_token=lease_token,
        )

    def mark_write_started(
        self,
        batch_id: str,
        object_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        expected_revision: int | None = None,
        writer_id: str | None = None,
        lease_seconds: int | None = None,
    ) -> AdmissionObjectSnapshot:
        return self._coordinator._admission_mark_write_started(
            batch_id,
            object_id,
            tenant_id,
            owner_subject,
            expected_revision=expected_revision,
            writer_id=writer_id,
            lease_seconds=lease_seconds,
        )

    def renew_write_lease(
        self,
        batch_id: str,
        object_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        lease_token: str,
        expected_revision: int,
        lease_seconds: int,
    ) -> AdmissionObjectSnapshot:
        return self._coordinator._admission_renew_write_lease(
            batch_id,
            object_id,
            tenant_id,
            owner_subject,
            lease_token=lease_token,
            expected_revision=expected_revision,
            lease_seconds=lease_seconds,
        )

    def record_write_unknown(
        self,
        batch_id: str,
        object_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        error_code: str = ErrorCode.DEPENDENCY_UNAVAILABLE.value,
        expected_revision: int | None = None,
        lease_token: str | None = None,
    ) -> AdmissionObjectSnapshot:
        return self._coordinator._admission_record_write_unknown(
            batch_id,
            object_id,
            tenant_id,
            owner_subject,
            error_code=error_code,
            expected_revision=expected_revision,
            lease_token=lease_token,
        )

    def accept_batch(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot:
        return self._coordinator._admission_accept_batch(
            batch_id,
            tenant_id,
            owner_subject,
            expected_revision=expected_revision,
        )

    def request_rollback(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        reason_code: str = "ADMISSION_FAILED",
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot:
        return self._coordinator._admission_request_rollback(
            batch_id,
            tenant_id,
            owner_subject,
            reason_code=reason_code,
            expected_revision=expected_revision,
        )

    def claim_cleanup(
        self,
        worker_id: str,
        *,
        limit: int,
        lease_seconds: int,
    ) -> tuple[CleanupLease, ...]:
        return self._coordinator._admission_claim_cleanup(
            worker_id,
            limit=limit,
            lease_seconds=lease_seconds,
        )

    def mark_deleted(self, lease: CleanupLease) -> AdmissionObjectSnapshot:
        return self._coordinator._admission_mark_deleted(lease)

    def mark_no_object(self, lease: CleanupLease) -> AdmissionObjectSnapshot:
        return self._coordinator._admission_mark_no_object(lease)

    def release_cleanup(
        self,
        lease: CleanupLease,
        *,
        error_code: str | None = None,
    ) -> AdmissionObjectSnapshot:
        return self._coordinator._admission_release_cleanup(lease, error_code=error_code)

    def finish_cleanup(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot:
        return self._coordinator._admission_finish_cleanup(
            batch_id,
            tenant_id,
            owner_subject,
            expected_revision=expected_revision,
        )

    def recover_stale_preparing(
        self,
        older_than: datetime,
        *,
        limit: int,
    ) -> tuple[AdmissionBatchSnapshot, ...]:
        return self._coordinator._admission_recover_stale_preparing(
            older_than,
            limit=limit,
        )

    def probe(self) -> None:
        self._coordinator.probe()

    def close(self) -> None:
        self._coordinator.close()


class _MongoTaskFacade:
    durable = True

    def __init__(self, coordinator: MongoImportCoordinator) -> None:
        self._coordinator = coordinator

    def create_many(self, tasks: Sequence[NewImportTask]) -> tuple[ImportTaskSnapshot, ...]:
        return self._coordinator._task_create_many(tasks)

    def get_owned(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
    ) -> ImportTaskSnapshot:
        return self._coordinator._task_get_owned(task_id, tenant_id, owner_subject)

    def begin(self, task_id: str, tenant_id: str, owner_subject: str) -> ImportTaskSnapshot:
        return self._coordinator._task_begin(task_id, tenant_id, owner_subject)

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        max_attempts: int,
    ) -> ImportTaskLease | None:
        return self._coordinator._task_claim_next(
            worker_id,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
        )

    def renew_lease(
        self,
        lease: ImportTaskLease,
        *,
        lease_seconds: int,
    ) -> ImportTaskLease:
        return self._coordinator._task_renew_lease(lease, lease_seconds=lease_seconds)

    def assert_lease(self, lease: ImportTaskLease) -> ImportTaskLease:
        return self._coordinator._task_assert_lease(lease)

    def retry(
        self,
        lease: ImportTaskLease,
        *,
        error_code: str,
        delay_seconds: float,
    ) -> ImportTaskSnapshot:
        return self._coordinator._task_retry(
            lease,
            error_code=error_code,
            delay_seconds=delay_seconds,
        )

    def start_step(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        step: str,
        *,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot:
        return self._coordinator._task_start_step(
            task_id,
            tenant_id,
            owner_subject,
            step,
            lease_token=lease_token,
        )

    def complete_step(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        step: str,
        *,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot:
        return self._coordinator._task_complete_step(
            task_id,
            tenant_id,
            owner_subject,
            step,
            lease_token=lease_token,
        )

    def complete(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        imported_chunk_count: int,
        commit_token: str,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot:
        return self._coordinator._task_complete(
            task_id,
            tenant_id,
            owner_subject,
            imported_chunk_count=imported_chunk_count,
            commit_token=commit_token,
            lease_token=lease_token,
        )

    def fail(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        error_code: str,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot:
        return self._coordinator._task_fail(
            task_id,
            tenant_id,
            owner_subject,
            error_code=error_code,
            lease_token=lease_token,
        )

    def rollback_queued(
        self,
        tenant_id: str,
        owner_subject: str,
        task_ids: Sequence[str],
    ) -> None:
        self._coordinator._task_rollback_queued(tenant_id, owner_subject, task_ids)

    def probe(self) -> None:
        self._coordinator.probe()

    def close(self) -> None:
        self._coordinator.close()


def _assert_protocols(
    coordinator: MongoImportCoordinator,
) -> tuple[ImportCoordinator, ImportAdmissionRepository, ImportTaskRegistry]:
    """Keep static protocol conformance visible to mypy without runtime side effects."""
    return coordinator, coordinator.admissions, coordinator.tasks
