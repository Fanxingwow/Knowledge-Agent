from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import RLock
from typing import Protocol

from app.core.errors import AppError, DependencyError, ErrorCode
from app.infrastructure.minio_store import ObjectRef, StoredObject

_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class AdmissionStatus(StrEnum):
    PREPARING = "preparing"
    ACCEPTED = "accepted"
    ROLLBACK_PENDING = "rollback_pending"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    ROLLED_BACK = "rolled_back"


class ObjectStatus(StrEnum):
    INTENT_RECORDED = "intent_recorded"
    WRITE_STARTED = "write_started"
    WRITE_CONFIRMED = "write_confirmed"
    WRITE_UNKNOWN = "write_unknown"
    DELETE_PENDING = "delete_pending"
    DELETED = "deleted"
    NO_OBJECT = "no_object"


class _LeaseKind(StrEnum):
    WRITE = "write"
    CLEANUP = "cleanup"


@dataclass(frozen=True, slots=True)
class AdmissionObjectIntent:
    object_id: str
    task_id: str
    document_id: str
    ordinal: int
    ref: ObjectRef
    original_name: str
    content_type: str
    size: int
    sha256: str
    object_kind: str = "source"


@dataclass(frozen=True, slots=True)
class AdmissionBatchIntent:
    batch_id: str
    tenant_id: str
    owner_subject: str
    idempotency_key: str
    payload_hash: str
    objects: tuple[AdmissionObjectIntent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.objects, tuple):
            object.__setattr__(self, "objects", tuple(self.objects))


@dataclass(frozen=True, slots=True)
class AdmissionObjectSnapshot:
    object_id: str
    task_id: str
    document_id: str
    ordinal: int
    ref: ObjectRef
    original_name: str
    content_type: str
    size: int
    sha256: str
    object_kind: str
    status: ObjectStatus
    stored_object: StoredObject | None
    revision: int
    last_error_code: str | None
    cleanup_attempts: int
    cleanup_next_attempt_at: datetime | None
    created_at: datetime
    updated_at: datetime
    write_lease_token: str | None = None
    write_lease_expires_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AdmissionBatchSnapshot:
    batch_id: str
    tenant_id: str
    owner_subject: str
    idempotency_key: str
    payload_hash: str
    status: AdmissionStatus
    objects: tuple[AdmissionObjectSnapshot, ...]
    rollback_requested: bool
    rollback_reason_code: str | None
    revision: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CleanupLease:
    batch_id: str
    object_id: str
    tenant_id: str
    owner_subject: str
    expected_ref: ObjectRef
    task_id: str
    document_id: str
    expected_size: int
    expected_sha256: str
    object_kind: str
    object_status: ObjectStatus
    stored_object: StoredObject | None
    lease_token: str
    object_revision: int
    cleanup_attempt: int
    expires_at: datetime


# Short aliases keep the public vocabulary convenient without creating duplicate types.
ObjectIntent = AdmissionObjectIntent
BatchIntent = AdmissionBatchIntent
ObjectSnapshot = AdmissionObjectSnapshot
BatchSnapshot = AdmissionBatchSnapshot


class ImportAdmissionRepository(Protocol):
    durable: bool

    def prepare_batch(self, intent: AdmissionBatchIntent) -> AdmissionBatchSnapshot: ...

    def get_owned(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
    ) -> AdmissionBatchSnapshot: ...

    def get_by_idempotency(
        self,
        tenant_id: str,
        owner_subject: str,
        idempotency_key: str,
    ) -> AdmissionBatchSnapshot: ...

    def get_by_task_id(self, task_id: str) -> AdmissionBatchSnapshot: ...

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
    ) -> AdmissionObjectSnapshot: ...

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
    ) -> AdmissionObjectSnapshot: ...

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
    ) -> AdmissionObjectSnapshot: ...

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
    ) -> AdmissionObjectSnapshot: ...

    def accept_batch(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot: ...

    def request_rollback(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        reason_code: str = "ADMISSION_FAILED",
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot: ...

    def claim_cleanup(
        self,
        worker_id: str,
        *,
        limit: int,
        lease_seconds: int,
    ) -> tuple[CleanupLease, ...]: ...

    def mark_deleted(self, lease: CleanupLease) -> AdmissionObjectSnapshot: ...

    def mark_no_object(self, lease: CleanupLease) -> AdmissionObjectSnapshot: ...

    def release_cleanup(
        self,
        lease: CleanupLease,
        *,
        error_code: str | None = None,
    ) -> AdmissionObjectSnapshot: ...

    def finish_cleanup(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot: ...

    def recover_stale_preparing(
        self,
        older_than: datetime,
        *,
        limit: int,
    ) -> tuple[AdmissionBatchSnapshot, ...]: ...

    def probe(self) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _ObjectRecord:
    intent: AdmissionObjectIntent
    status: ObjectStatus
    stored_object: StoredObject | None
    revision: int
    last_error_code: str | None
    lease_owner: str | None
    lease_token: str | None
    lease_expires_at: datetime | None
    lease_kind: _LeaseKind | None
    cleanup_attempts: int
    cleanup_next_attempt_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class _BatchRecord:
    intent: AdmissionBatchIntent
    status: AdmissionStatus
    objects: dict[str, _ObjectRecord]
    rollback_requested: bool
    rollback_reason_code: str | None
    revision: int
    created_at: datetime
    updated_at: datetime


def _resource_not_found() -> AppError:
    return AppError(ErrorCode.RESOURCE_NOT_FOUND, "Import admission not found", 404)


def _state_conflict() -> AppError:
    return AppError(ErrorCode.RESOURCE_CONFLICT, "Import admission state conflict", 409)


def _request_conflict() -> AppError:
    return AppError(ErrorCode.RESOURCE_CONFLICT, "Import admission request conflict", 409)


def _is_canonical_uuid(value: str) -> bool:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        return False
    return str(parsed) == value


def _valid_text(value: str, *, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= maximum
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _validate_error_code(error_code: str) -> None:
    if not isinstance(error_code, str) or not _ERROR_CODE.fullmatch(error_code):
        raise ValueError("invalid import admission error code")


def _validate_revision(revision: int | None) -> None:
    if revision is not None and (isinstance(revision, bool) or not isinstance(revision, int) or revision < 0):
        raise ValueError("invalid import admission revision")


def _validate_lease_seconds(lease_seconds: int) -> None:
    if (
        isinstance(lease_seconds, bool)
        or not isinstance(lease_seconds, int)
        or not 1 <= lease_seconds <= 86_400
    ):
        raise ValueError("invalid import admission lease duration")


def _validate_intent(intent: AdmissionBatchIntent) -> None:
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
        if isinstance(item.ordinal, bool) or not isinstance(item.ordinal, int) or item.ordinal < 0:
            raise ValueError("invalid import admission object ordinal")
        if not isinstance(item.ref, ObjectRef) or item.ref.tenant_id != intent.tenant_id:
            raise ValueError("invalid import admission object owner")
        if not _valid_text(item.original_name, maximum=255):
            raise ValueError("invalid import admission document name")
        if not _valid_text(item.content_type, maximum=128):
            raise ValueError("invalid import admission content type")
        if isinstance(item.size, bool) or not isinstance(item.size, int) or item.size <= 0:
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


def _validate_stored_object(stored_object: StoredObject) -> None:
    if not isinstance(stored_object, StoredObject):
        raise ValueError("invalid stored import object")
    for value in (stored_object.etag, stored_object.version_id):
        if value is not None and (
            not isinstance(value, str)
            or not value
            or len(value) > 1024
            or any(ord(character) < 32 or ord(character) > 126 for character in value)
        ):
            raise ValueError("invalid stored import object")


class InMemoryImportAdmissionRepository:
    """Thread-safe, non-durable admission manifest used by local development and tests."""

    durable = False

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        cleanup_retry_base_seconds: float = 1.0,
        cleanup_retry_max_seconds: float = 300.0,
    ) -> None:
        if cleanup_retry_base_seconds <= 0 or cleanup_retry_max_seconds < cleanup_retry_base_seconds:
            raise ValueError("invalid cleanup retry window")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._cleanup_retry_base_seconds = cleanup_retry_base_seconds
        self._cleanup_retry_max_seconds = cleanup_retry_max_seconds
        self._records: dict[str, _BatchRecord] = {}
        self._idempotency_index: dict[tuple[str, str, str], str] = {}
        self._lock = RLock()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise DependencyError("Import admission repository is unavailable")

    @staticmethod
    def _object_snapshot(record: _ObjectRecord) -> AdmissionObjectSnapshot:
        intent = record.intent
        write_lease = record.lease_kind is _LeaseKind.WRITE
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
            status=record.status,
            stored_object=record.stored_object,
            revision=record.revision,
            last_error_code=record.last_error_code,
            cleanup_attempts=record.cleanup_attempts,
            cleanup_next_attempt_at=record.cleanup_next_attempt_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
            write_lease_token=record.lease_token if write_lease else None,
            write_lease_expires_at=record.lease_expires_at if write_lease else None,
        )

    @classmethod
    def _snapshot(cls, record: _BatchRecord) -> AdmissionBatchSnapshot:
        intent = record.intent
        return AdmissionBatchSnapshot(
            batch_id=intent.batch_id,
            tenant_id=intent.tenant_id,
            owner_subject=intent.owner_subject,
            idempotency_key=intent.idempotency_key,
            payload_hash=intent.payload_hash,
            status=record.status,
            objects=tuple(cls._object_snapshot(item) for item in record.objects.values()),
            rollback_requested=record.rollback_requested,
            rollback_reason_code=record.rollback_reason_code,
            revision=record.revision,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def _owned(self, batch_id: str, tenant_id: str, owner_subject: str) -> _BatchRecord:
        record = self._records.get(batch_id)
        if (
            record is None
            or record.intent.tenant_id != tenant_id
            or record.intent.owner_subject != owner_subject
        ):
            raise _resource_not_found()
        return record

    @staticmethod
    def _object(record: _BatchRecord, object_id: str) -> _ObjectRecord:
        item = record.objects.get(object_id)
        if item is None:
            raise _resource_not_found()
        return item

    @staticmethod
    def _check_revision(actual: int, expected: int | None) -> None:
        _validate_revision(expected)
        if expected is not None and actual != expected:
            raise _state_conflict()

    @staticmethod
    def _clear_lease(item: _ObjectRecord) -> None:
        item.lease_owner = None
        item.lease_token = None
        item.lease_expires_at = None
        item.lease_kind = None

    @staticmethod
    def _write_lease_is_active(item: _ObjectRecord, now: datetime) -> bool:
        return (
            item.lease_kind is _LeaseKind.WRITE
            and item.lease_token is not None
            and item.lease_expires_at is not None
            and item.lease_expires_at > now
        )

    def _touch_object(self, batch: _BatchRecord, item: _ObjectRecord, now: datetime) -> None:
        item.revision += 1
        item.updated_at = now
        batch.revision += 1
        batch.updated_at = now

    @staticmethod
    def _refresh_rollback_status(batch: _BatchRecord) -> None:
        if not batch.rollback_requested or batch.status is AdmissionStatus.ROLLED_BACK:
            return
        if all(
            item.status in {ObjectStatus.DELETED, ObjectStatus.NO_OBJECT}
            for item in batch.objects.values()
        ):
            batch.status = AdmissionStatus.ROLLED_BACK
        elif any(item.status is ObjectStatus.WRITE_UNKNOWN for item in batch.objects.values()):
            batch.status = AdmissionStatus.RECONCILIATION_REQUIRED
        else:
            batch.status = AdmissionStatus.ROLLBACK_PENDING

    def prepare_batch(self, intent: AdmissionBatchIntent) -> AdmissionBatchSnapshot:
        _validate_intent(intent)
        key = (intent.tenant_id, intent.owner_subject, intent.idempotency_key)
        with self._lock:
            self._ensure_open()
            existing_id = self._idempotency_index.get(key)
            if existing_id is not None:
                existing = self._records[existing_id]
                if existing.intent.payload_hash != intent.payload_hash:
                    raise _request_conflict()
                return self._snapshot(existing)
            if intent.batch_id in self._records:
                raise _request_conflict()

            now = self._clock()
            objects = {
                item.object_id: _ObjectRecord(
                    intent=item,
                    status=ObjectStatus.INTENT_RECORDED,
                    stored_object=None,
                    revision=0,
                    last_error_code=None,
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                    lease_kind=None,
                    cleanup_attempts=0,
                    cleanup_next_attempt_at=None,
                    created_at=now,
                    updated_at=now,
                )
                for item in intent.objects
            }
            record = _BatchRecord(
                intent=intent,
                status=AdmissionStatus.PREPARING,
                objects=objects,
                rollback_requested=False,
                rollback_reason_code=None,
                revision=0,
                created_at=now,
                updated_at=now,
            )
            self._records[intent.batch_id] = record
            self._idempotency_index[key] = intent.batch_id
            return self._snapshot(record)

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
        if (writer_id is None) is not (lease_seconds is None):
            raise ValueError("write lease owner and duration must be provided together")
        if writer_id is not None:
            if not _valid_text(writer_id, maximum=128):
                raise ValueError("invalid import writer")
            assert lease_seconds is not None
            _validate_lease_seconds(lease_seconds)
        with self._lock:
            self._ensure_open()
            batch = self._owned(batch_id, tenant_id, owner_subject)
            item = self._object(batch, object_id)
            if item.status is ObjectStatus.WRITE_STARTED:
                if writer_id is None and item.lease_kind is None:
                    return self._object_snapshot(item)
                if (
                    writer_id is not None
                    and item.lease_kind is _LeaseKind.WRITE
                    and item.lease_owner == writer_id
                ):
                    return self._object_snapshot(item)
                raise _state_conflict()
            self._check_revision(item.revision, expected_revision)
            if batch.status is not AdmissionStatus.PREPARING or batch.rollback_requested:
                raise _state_conflict()
            if item.status is not ObjectStatus.INTENT_RECORDED or item.stored_object is not None:
                raise _state_conflict()
            now = self._clock()
            item.status = ObjectStatus.WRITE_STARTED
            if writer_id is not None:
                assert lease_seconds is not None
                item.lease_owner = writer_id
                item.lease_token = str(uuid.uuid4())
                item.lease_expires_at = now + timedelta(seconds=lease_seconds)
                item.lease_kind = _LeaseKind.WRITE
            self._touch_object(batch, item, now)
            return self._object_snapshot(item)

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
        if not _valid_text(lease_token, maximum=128):
            raise ValueError("invalid import write lease token")
        _validate_revision(expected_revision)
        _validate_lease_seconds(lease_seconds)
        with self._lock:
            self._ensure_open()
            batch = self._owned(batch_id, tenant_id, owner_subject)
            item = self._object(batch, object_id)
            self._check_revision(item.revision, expected_revision)
            now = self._clock()
            if (
                item.lease_kind is not _LeaseKind.WRITE
                or item.lease_token != lease_token
                or item.lease_expires_at is None
                or item.lease_expires_at <= now
                or item.stored_object is not None
                or item.status not in {ObjectStatus.WRITE_STARTED, ObjectStatus.WRITE_UNKNOWN}
            ):
                raise _state_conflict()
            item.lease_expires_at = now + timedelta(seconds=lease_seconds)
            self._touch_object(batch, item, now)
            return self._object_snapshot(item)

    def get_owned(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
    ) -> AdmissionBatchSnapshot:
        with self._lock:
            self._ensure_open()
            return self._snapshot(self._owned(batch_id, tenant_id, owner_subject))

    def get_by_idempotency(
        self,
        tenant_id: str,
        owner_subject: str,
        idempotency_key: str,
    ) -> AdmissionBatchSnapshot:
        with self._lock:
            self._ensure_open()
            batch_id = self._idempotency_index.get((tenant_id, owner_subject, idempotency_key))
            if batch_id is None:
                raise _resource_not_found()
            return self._snapshot(self._owned(batch_id, tenant_id, owner_subject))

    def get_by_task_id(self, task_id: str) -> AdmissionBatchSnapshot:
        with self._lock:
            self._ensure_open()
            for record in self._records.values():
                if any(item.intent.task_id == task_id for item in record.objects.values()):
                    return self._snapshot(record)
            raise _resource_not_found()

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
        _validate_stored_object(stored_object)
        with self._lock:
            self._ensure_open()
            batch = self._owned(batch_id, tenant_id, owner_subject)
            item = self._object(batch, object_id)
            if stored_object.ref != item.intent.ref or stored_object.ref.tenant_id != batch.intent.tenant_id:
                raise _resource_not_found()

            if item.stored_object is not None:
                if item.stored_object == stored_object and item.status in {
                    ObjectStatus.WRITE_CONFIRMED,
                    ObjectStatus.DELETE_PENDING,
                }:
                    return self._object_snapshot(item)
                raise _state_conflict()
            self._check_revision(item.revision, expected_revision)
            if batch.status in {AdmissionStatus.ACCEPTED, AdmissionStatus.ROLLED_BACK}:
                raise _state_conflict()
            if item.status not in {ObjectStatus.WRITE_STARTED, ObjectStatus.WRITE_UNKNOWN}:
                raise _state_conflict()

            if item.lease_kind is not None:
                if lease_token != item.lease_token or expected_revision is None:
                    raise _state_conflict()
                expires_at = item.lease_expires_at
                if expires_at is None or expires_at <= self._clock():
                    raise _state_conflict()
            elif lease_token is not None:
                raise _state_conflict()

            now = self._clock()
            item.stored_object = stored_object
            item.last_error_code = None
            self._clear_lease(item)
            item.status = ObjectStatus.DELETE_PENDING if batch.rollback_requested else ObjectStatus.WRITE_CONFIRMED
            self._touch_object(batch, item, now)
            if batch.rollback_requested:
                self._refresh_rollback_status(batch)
            elif not any(candidate.status is ObjectStatus.WRITE_UNKNOWN for candidate in batch.objects.values()):
                batch.status = AdmissionStatus.PREPARING
            return self._object_snapshot(item)

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
        _validate_error_code(error_code)
        with self._lock:
            self._ensure_open()
            batch = self._owned(batch_id, tenant_id, owner_subject)
            item = self._object(batch, object_id)
            if item.status is ObjectStatus.WRITE_UNKNOWN:
                return self._object_snapshot(item)
            self._check_revision(item.revision, expected_revision)
            if batch.status in {AdmissionStatus.ACCEPTED, AdmissionStatus.ROLLED_BACK} or batch.rollback_requested:
                raise _state_conflict()
            if item.status is not ObjectStatus.WRITE_STARTED or item.stored_object is not None:
                raise _state_conflict()
            if item.lease_kind is _LeaseKind.WRITE:
                if lease_token != item.lease_token or expected_revision is None:
                    raise _state_conflict()
                if item.lease_expires_at is None or item.lease_expires_at <= self._clock():
                    raise _state_conflict()
            elif lease_token is not None:
                raise _state_conflict()

            now = self._clock()
            item.status = ObjectStatus.WRITE_UNKNOWN
            item.last_error_code = error_code
            batch.status = AdmissionStatus.RECONCILIATION_REQUIRED
            self._touch_object(batch, item, now)
            return self._object_snapshot(item)

    def accept_batch(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot:
        with self._lock:
            self._ensure_open()
            batch = self._owned(batch_id, tenant_id, owner_subject)
            if batch.status is AdmissionStatus.ACCEPTED:
                return self._snapshot(batch)
            self._check_revision(batch.revision, expected_revision)
            if batch.status is AdmissionStatus.ROLLED_BACK or batch.rollback_requested:
                raise _state_conflict()
            if not all(
                item.status is ObjectStatus.WRITE_CONFIRMED and item.stored_object is not None
                for item in batch.objects.values()
            ):
                raise _state_conflict()

            now = self._clock()
            batch.status = AdmissionStatus.ACCEPTED
            batch.revision += 1
            batch.updated_at = now
            return self._snapshot(batch)

    def request_rollback(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        reason_code: str = "ADMISSION_FAILED",
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot:
        _validate_error_code(reason_code)
        with self._lock:
            self._ensure_open()
            batch = self._owned(batch_id, tenant_id, owner_subject)
            if batch.status is AdmissionStatus.ACCEPTED:
                raise _state_conflict()
            if batch.status is AdmissionStatus.ROLLED_BACK:
                return self._snapshot(batch)
            if batch.rollback_requested:
                if batch.rollback_reason_code == reason_code:
                    return self._snapshot(batch)
                raise _state_conflict()
            self._check_revision(batch.revision, expected_revision)

            now = self._clock()
            batch.rollback_requested = True
            batch.rollback_reason_code = reason_code
            for item in batch.objects.values():
                next_status = item.status
                if item.status is ObjectStatus.INTENT_RECORDED:
                    next_status = ObjectStatus.NO_OBJECT
                elif item.status is ObjectStatus.WRITE_STARTED:
                    next_status = ObjectStatus.WRITE_UNKNOWN
                elif item.status is ObjectStatus.WRITE_CONFIRMED:
                    next_status = ObjectStatus.DELETE_PENDING
                if next_status is not item.status:
                    item.status = next_status
                    if next_status is ObjectStatus.WRITE_UNKNOWN:
                        item.last_error_code = reason_code
                    item.revision += 1
                    item.updated_at = now
            batch.revision += 1
            batch.updated_at = now
            self._refresh_rollback_status(batch)
            return self._snapshot(batch)

    def claim_cleanup(
        self,
        worker_id: str,
        *,
        limit: int,
        lease_seconds: int,
    ) -> tuple[CleanupLease, ...]:
        if not _valid_text(worker_id, maximum=128):
            raise ValueError("invalid cleanup worker")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0 or limit > 1000:
            raise ValueError("invalid cleanup claim limit")
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or lease_seconds <= 0
            or lease_seconds > 3600
        ):
            raise ValueError("invalid cleanup lease duration")

        with self._lock:
            self._ensure_open()
            now = self._clock()
            expires_at = now + timedelta(seconds=lease_seconds)
            claimed: list[CleanupLease] = []
            for batch in self._records.values():
                if not batch.rollback_requested or batch.status not in {
                    AdmissionStatus.ROLLBACK_PENDING,
                    AdmissionStatus.RECONCILIATION_REQUIRED,
                }:
                    continue
                for item in batch.objects.values():
                    if item.status not in {ObjectStatus.DELETE_PENDING, ObjectStatus.WRITE_UNKNOWN}:
                        continue
                    lease_is_active = (
                        item.lease_token is not None
                        and item.lease_expires_at is not None
                        and item.lease_expires_at > now
                    )
                    if lease_is_active:
                        continue
                    if (
                        item.cleanup_next_attempt_at is not None
                        and item.cleanup_next_attempt_at > now
                    ):
                        continue

                    item.lease_owner = worker_id
                    item.lease_token = str(uuid.uuid4())
                    item.lease_expires_at = expires_at
                    item.lease_kind = _LeaseKind.CLEANUP
                    self._touch_object(batch, item, now)
                    claimed.append(
                        CleanupLease(
                            batch_id=batch.intent.batch_id,
                            object_id=item.intent.object_id,
                            tenant_id=batch.intent.tenant_id,
                            owner_subject=batch.intent.owner_subject,
                            expected_ref=item.intent.ref,
                            task_id=item.intent.task_id,
                            document_id=item.intent.document_id,
                            expected_size=item.intent.size,
                            expected_sha256=item.intent.sha256,
                            object_kind=item.intent.object_kind,
                            object_status=item.status,
                            stored_object=item.stored_object,
                            lease_token=item.lease_token,
                            object_revision=item.revision,
                            cleanup_attempt=item.cleanup_attempts + 1,
                            expires_at=expires_at,
                        )
                    )
                    if len(claimed) == limit:
                        return tuple(claimed)
            return tuple(claimed)

    def _leased(self, lease: CleanupLease) -> tuple[_BatchRecord, _ObjectRecord]:
        if not isinstance(lease, CleanupLease):
            raise ValueError("invalid cleanup lease")
        batch = self._records.get(lease.batch_id)
        if (
            batch is None
            or batch.intent.tenant_id != lease.tenant_id
            or batch.intent.owner_subject != lease.owner_subject
        ):
            raise _state_conflict()
        item = batch.objects.get(lease.object_id)
        if (
            item is None
            or item.intent.ref != lease.expected_ref
            or item.status is not lease.object_status
            or item.stored_object != lease.stored_object
            or item.lease_token != lease.lease_token
            or item.revision != lease.object_revision
            or item.lease_expires_at != lease.expires_at
            or item.lease_expires_at is None
            or item.lease_expires_at <= self._clock()
            or item.lease_kind is not _LeaseKind.CLEANUP
            or lease.cleanup_attempt != item.cleanup_attempts + 1
        ):
            raise _state_conflict()
        return batch, item

    def mark_deleted(self, lease: CleanupLease) -> AdmissionObjectSnapshot:
        with self._lock:
            self._ensure_open()
            batch, item = self._leased(lease)
            if item.status is not ObjectStatus.DELETE_PENDING or item.stored_object is None:
                raise _state_conflict()
            now = self._clock()
            item.status = ObjectStatus.DELETED
            item.last_error_code = None
            self._clear_lease(item)
            item.cleanup_next_attempt_at = None
            self._touch_object(batch, item, now)
            self._refresh_rollback_status(batch)
            return self._object_snapshot(item)

    def mark_no_object(self, lease: CleanupLease) -> AdmissionObjectSnapshot:
        with self._lock:
            self._ensure_open()
            batch, item = self._leased(lease)
            if item.status not in {ObjectStatus.DELETE_PENDING, ObjectStatus.WRITE_UNKNOWN}:
                raise _state_conflict()
            now = self._clock()
            item.status = ObjectStatus.NO_OBJECT
            item.last_error_code = None
            self._clear_lease(item)
            item.cleanup_next_attempt_at = None
            self._touch_object(batch, item, now)
            self._refresh_rollback_status(batch)
            return self._object_snapshot(item)

    def release_cleanup(
        self,
        lease: CleanupLease,
        *,
        error_code: str | None = None,
    ) -> AdmissionObjectSnapshot:
        if error_code is not None:
            _validate_error_code(error_code)
        with self._lock:
            self._ensure_open()
            batch, item = self._leased(lease)
            now = self._clock()
            item.last_error_code = error_code
            item.cleanup_attempts += 1
            delay = min(
                self._cleanup_retry_base_seconds * (2 ** (item.cleanup_attempts - 1)),
                self._cleanup_retry_max_seconds,
            )
            item.cleanup_next_attempt_at = now + timedelta(seconds=delay)
            self._clear_lease(item)
            self._touch_object(batch, item, now)
            return self._object_snapshot(item)

    def finish_cleanup(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        expected_revision: int | None = None,
    ) -> AdmissionBatchSnapshot:
        with self._lock:
            self._ensure_open()
            batch = self._owned(batch_id, tenant_id, owner_subject)
            if batch.status is AdmissionStatus.ROLLED_BACK:
                return self._snapshot(batch)
            self._check_revision(batch.revision, expected_revision)
            if not batch.rollback_requested or not all(
                item.status in {ObjectStatus.DELETED, ObjectStatus.NO_OBJECT}
                for item in batch.objects.values()
            ):
                raise _state_conflict()

            now = self._clock()
            batch.status = AdmissionStatus.ROLLED_BACK
            batch.revision += 1
            batch.updated_at = now
            return self._snapshot(batch)

    def recover_stale_preparing(
        self,
        older_than: datetime,
        *,
        limit: int,
    ) -> tuple[AdmissionBatchSnapshot, ...]:
        if (
            not isinstance(older_than, datetime)
            or older_than.tzinfo is None
            or older_than.utcoffset() is None
        ):
            raise ValueError("invalid import admission recovery cutoff")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("invalid import admission recovery limit")
        with self._lock:
            self._ensure_open()
            candidates = sorted(self._records.values(), key=lambda item: item.updated_at)
            recovered: list[AdmissionBatchSnapshot] = []
            for batch in candidates:
                if len(recovered) == limit:
                    break
                if (
                    batch.rollback_requested
                    or batch.status in {AdmissionStatus.ACCEPTED, AdmissionStatus.ROLLED_BACK}
                    or batch.updated_at > older_than
                    or any(
                        self._write_lease_is_active(item, self._clock())
                        for item in batch.objects.values()
                    )
                ):
                    continue
                recovered.append(
                    self.request_rollback(
                        batch.intent.batch_id,
                        batch.intent.tenant_id,
                        batch.intent.owner_subject,
                        reason_code="ADMISSION_RECOVERY",
                        expected_revision=batch.revision,
                    )
                )
            return tuple(recovered)

    def probe(self) -> None:
        with self._lock:
            self._ensure_open()

    def close(self) -> None:
        with self._lock:
            self._closed = True
