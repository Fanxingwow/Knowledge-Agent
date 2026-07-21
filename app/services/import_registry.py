from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import RLock
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.infrastructure.minio_store import parse_object_ref

_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STEP_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class ImportTaskStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class NewImportTask:
    task_id: str
    document_id: str
    tenant_id: str
    owner_subject: str
    source_ref: str
    original_name: str
    content_type: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ImportTaskSnapshot:
    task_id: str
    document_id: str
    tenant_id: str
    owner_subject: str
    source_ref: str
    original_name: str
    content_type: str
    size: int
    sha256: str
    status: ImportTaskStatus
    current_step: str | None
    completed_steps: tuple[str, ...]
    imported_chunk_count: int | None
    commit_token: str | None
    error_code: str | None
    attempt_count: int
    available_at: datetime
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ImportTaskLease:
    task: ImportTaskSnapshot
    worker_id: str
    lease_token: str
    attempt_count: int
    expires_at: datetime


class ImportTaskRegistry(Protocol):
    durable: bool

    def create_many(self, tasks: Sequence[NewImportTask]) -> tuple[ImportTaskSnapshot, ...]: ...

    def get_owned(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
    ) -> ImportTaskSnapshot: ...

    def begin(self, task_id: str, tenant_id: str, owner_subject: str) -> ImportTaskSnapshot: ...

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        max_attempts: int,
    ) -> ImportTaskLease | None: ...

    def renew_lease(
        self,
        lease: ImportTaskLease,
        *,
        lease_seconds: int,
    ) -> ImportTaskLease: ...

    def assert_lease(self, lease: ImportTaskLease) -> ImportTaskLease: ...

    def retry(
        self,
        lease: ImportTaskLease,
        *,
        error_code: str,
        delay_seconds: float,
    ) -> ImportTaskSnapshot: ...

    def start_step(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        step: str,
        *,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot: ...

    def complete_step(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        step: str,
        *,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot: ...

    def complete(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        imported_chunk_count: int,
        commit_token: str,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot: ...

    def fail(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        error_code: str,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot: ...

    def rollback_queued(
        self,
        tenant_id: str,
        owner_subject: str,
        task_ids: Sequence[str],
    ) -> None: ...


@dataclass(slots=True)
class _ImportTaskRecord:
    source: NewImportTask
    status: ImportTaskStatus
    current_step: str | None
    completed_steps: list[str]
    imported_chunk_count: int | None
    commit_token: str | None
    error_code: str | None
    attempt_count: int
    available_at: datetime
    lease_owner: str | None
    lease_token: str | None
    lease_expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


def _resource_not_found() -> AppError:
    return AppError(ErrorCode.RESOURCE_NOT_FOUND, "Import task not found", 404)


def _state_conflict() -> AppError:
    return AppError(ErrorCode.RESOURCE_CONFLICT, "Import task state conflict", 409)


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


def _validate_new_task(task: NewImportTask) -> None:
    if not _is_canonical_uuid(task.task_id) or not _is_canonical_uuid(task.document_id):
        raise ValueError("invalid import task identifier")
    if not _TENANT_ID.fullmatch(task.tenant_id):
        raise ValueError("invalid import task tenant")
    if not _valid_text(task.owner_subject, maximum=256):
        raise ValueError("invalid import task owner")
    if not _valid_text(task.original_name, maximum=255):
        raise ValueError("invalid import document name")
    if not _valid_text(task.content_type, maximum=128):
        raise ValueError("invalid import content type")
    if isinstance(task.size, bool) or not isinstance(task.size, int) or task.size <= 0:
        raise ValueError("invalid import document size")
    if not _SHA256.fullmatch(task.sha256):
        raise ValueError("invalid import document digest")
    parsed_ref = parse_object_ref(task.source_ref)
    if parsed_ref.tenant_id != task.tenant_id:
        raise ValueError("invalid import source owner")


class InMemoryImportTaskRegistry:
    """Thread-safe development registry behind a durable-repository compatible contract."""

    durable = False

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._records: dict[str, _ImportTaskRecord] = {}
        self._lock = RLock()

    @staticmethod
    def _snapshot(record: _ImportTaskRecord) -> ImportTaskSnapshot:
        task = record.source
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
            status=record.status,
            current_step=record.current_step,
            completed_steps=tuple(record.completed_steps),
            imported_chunk_count=record.imported_chunk_count,
            commit_token=record.commit_token,
            error_code=record.error_code,
            attempt_count=record.attempt_count,
            available_at=record.available_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def _owned(self, task_id: str, tenant_id: str, owner_subject: str) -> _ImportTaskRecord:
        record = self._records.get(task_id)
        if (
            record is None
            or record.source.tenant_id != tenant_id
            or record.source.owner_subject != owner_subject
        ):
            raise _resource_not_found()
        return record

    def _require_lease(self, record: _ImportTaskRecord, lease_token: str | None) -> None:
        if record.lease_token is None:
            if lease_token is not None:
                raise _state_conflict()
            return
        if (
            lease_token != record.lease_token
            or record.lease_expires_at is None
            or record.lease_expires_at <= self._clock()
        ):
            raise _state_conflict()

    @staticmethod
    def _clear_lease(record: _ImportTaskRecord) -> None:
        record.lease_owner = None
        record.lease_token = None
        record.lease_expires_at = None

    def create_many(self, tasks: Sequence[NewImportTask]) -> tuple[ImportTaskSnapshot, ...]:
        batch = tuple(tasks)
        if not batch:
            raise ValueError("an import task batch cannot be empty")
        for task in batch:
            _validate_new_task(task)
        task_ids = [task.task_id for task in batch]
        if len(set(task_ids)) != len(task_ids):
            raise _state_conflict()

        with self._lock:
            if any(task_id in self._records for task_id in task_ids):
                raise _state_conflict()
            now = self._clock()
            records = [
                _ImportTaskRecord(
                    source=task,
                    status=ImportTaskStatus.QUEUED,
                    current_step=None,
                    completed_steps=[],
                    imported_chunk_count=None,
                    commit_token=None,
                    error_code=None,
                    attempt_count=0,
                    available_at=now,
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                    created_at=now,
                    updated_at=now,
                )
                for task in batch
            ]
            self._records.update({record.source.task_id: record for record in records})
            return tuple(self._snapshot(record) for record in records)

    def get_owned(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
    ) -> ImportTaskSnapshot:
        with self._lock:
            return self._snapshot(self._owned(task_id, tenant_id, owner_subject))

    def begin(self, task_id: str, tenant_id: str, owner_subject: str) -> ImportTaskSnapshot:
        with self._lock:
            record = self._owned(task_id, tenant_id, owner_subject)
            if record.status is not ImportTaskStatus.QUEUED:
                raise _state_conflict()
            record.status = ImportTaskStatus.PROCESSING
            record.attempt_count += 1
            record.updated_at = self._clock()
            return self._snapshot(record)

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        max_attempts: int,
    ) -> ImportTaskLease | None:
        if not _valid_text(worker_id, maximum=128):
            raise ValueError("invalid import worker")
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or not 1 <= lease_seconds <= 3600
        ):
            raise ValueError("invalid import task lease duration")
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or not 1 <= max_attempts <= 100
        ):
            raise ValueError("invalid import task attempt limit")

        with self._lock:
            now = self._clock()
            candidates = sorted(self._records.values(), key=lambda item: item.created_at)
            for record in candidates:
                eligible = (
                    record.status is ImportTaskStatus.QUEUED and record.available_at <= now
                ) or (
                    record.status is ImportTaskStatus.PROCESSING
                    and record.lease_token is not None
                    and record.lease_expires_at is not None
                    and record.lease_expires_at <= now
                )
                if not eligible:
                    continue
                if record.attempt_count >= max_attempts:
                    record.status = ImportTaskStatus.FAILED
                    record.current_step = None
                    record.error_code = ErrorCode.INTERNAL_ERROR.value
                    self._clear_lease(record)
                    record.updated_at = now
                    continue
                token = str(uuid.uuid4())
                expires_at = now + timedelta(seconds=lease_seconds)
                record.status = ImportTaskStatus.PROCESSING
                record.current_step = None
                record.attempt_count += 1
                record.lease_owner = worker_id
                record.lease_token = token
                record.lease_expires_at = expires_at
                record.updated_at = now
                snapshot = self._snapshot(record)
                return ImportTaskLease(
                    task=snapshot,
                    worker_id=worker_id,
                    lease_token=token,
                    attempt_count=record.attempt_count,
                    expires_at=expires_at,
                )
            return None

    def renew_lease(
        self,
        lease: ImportTaskLease,
        *,
        lease_seconds: int,
    ) -> ImportTaskLease:
        if not isinstance(lease, ImportTaskLease):
            raise ValueError("invalid import task lease")
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or not 1 <= lease_seconds <= 3600
        ):
            raise ValueError("invalid import task lease duration")
        with self._lock:
            record = self._owned(
                lease.task.task_id,
                lease.task.tenant_id,
                lease.task.owner_subject,
            )
            if (
                record.status is not ImportTaskStatus.PROCESSING
                or record.lease_owner != lease.worker_id
                or record.lease_token != lease.lease_token
                or record.attempt_count != lease.attempt_count
                or record.lease_expires_at != lease.expires_at
            ):
                raise _state_conflict()
            self._require_lease(record, lease.lease_token)
            expires_at = self._clock() + timedelta(seconds=lease_seconds)
            record.lease_expires_at = expires_at
            record.updated_at = self._clock()
            return ImportTaskLease(
                task=self._snapshot(record),
                worker_id=lease.worker_id,
                lease_token=lease.lease_token,
                attempt_count=lease.attempt_count,
                expires_at=expires_at,
            )

    def assert_lease(self, lease: ImportTaskLease) -> ImportTaskLease:
        """Prove a lease is still current without requiring its old expiry value."""
        if not isinstance(lease, ImportTaskLease):
            raise ValueError("invalid import task lease")
        with self._lock:
            record = self._owned(
                lease.task.task_id,
                lease.task.tenant_id,
                lease.task.owner_subject,
            )
            if (
                record.status is not ImportTaskStatus.PROCESSING
                or record.lease_owner != lease.worker_id
                or record.lease_token != lease.lease_token
                or record.attempt_count != lease.attempt_count
            ):
                raise _state_conflict()
            self._require_lease(record, lease.lease_token)
            expires_at = record.lease_expires_at
            if expires_at is None:
                raise _state_conflict()
            return ImportTaskLease(
                task=self._snapshot(record),
                worker_id=lease.worker_id,
                lease_token=lease.lease_token,
                attempt_count=lease.attempt_count,
                expires_at=expires_at,
            )

    def retry(
        self,
        lease: ImportTaskLease,
        *,
        error_code: str,
        delay_seconds: float,
    ) -> ImportTaskSnapshot:
        if not isinstance(lease, ImportTaskLease):
            raise ValueError("invalid import task lease")
        if not isinstance(error_code, str) or not _ERROR_CODE.fullmatch(error_code):
            raise ValueError("invalid import error code")
        if (
            isinstance(delay_seconds, bool)
            or not isinstance(delay_seconds, (int, float))
            or not 0 <= delay_seconds <= 86400
        ):
            raise ValueError("invalid import retry delay")
        with self._lock:
            record = self._owned(
                lease.task.task_id,
                lease.task.tenant_id,
                lease.task.owner_subject,
            )
            if record.status is not ImportTaskStatus.PROCESSING:
                raise _state_conflict()
            self._require_lease(record, lease.lease_token)
            if record.attempt_count != lease.attempt_count:
                raise _state_conflict()
            now = self._clock()
            record.status = ImportTaskStatus.QUEUED
            record.current_step = None
            record.error_code = error_code
            record.available_at = now + timedelta(seconds=float(delay_seconds))
            self._clear_lease(record)
            record.updated_at = now
            return self._snapshot(record)

    def start_step(
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
        with self._lock:
            record = self._owned(task_id, tenant_id, owner_subject)
            self._require_lease(record, lease_token)
            if record.status is not ImportTaskStatus.PROCESSING or record.current_step is not None:
                raise _state_conflict()
            record.current_step = step
            record.updated_at = self._clock()
            return self._snapshot(record)

    def complete_step(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        step: str,
        *,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot:
        with self._lock:
            record = self._owned(task_id, tenant_id, owner_subject)
            self._require_lease(record, lease_token)
            if record.status is not ImportTaskStatus.PROCESSING or record.current_step != step:
                raise _state_conflict()
            if step not in record.completed_steps:
                record.completed_steps.append(step)
            record.current_step = None
            record.updated_at = self._clock()
            return self._snapshot(record)

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
        if (
            isinstance(imported_chunk_count, bool)
            or not isinstance(imported_chunk_count, int)
            or imported_chunk_count < 0
        ):
            raise ValueError("invalid imported chunk count")
        if (
            not isinstance(commit_token, str)
            or not 1 <= len(commit_token) <= 256
            or any(ord(character) < 33 or ord(character) > 126 for character in commit_token)
        ):
            raise ValueError("invalid import commit token")
        with self._lock:
            record = self._owned(task_id, tenant_id, owner_subject)
            self._require_lease(record, lease_token)
            if record.status is not ImportTaskStatus.PROCESSING or record.current_step is not None:
                raise _state_conflict()
            record.status = ImportTaskStatus.COMPLETED
            record.imported_chunk_count = imported_chunk_count
            record.commit_token = commit_token
            self._clear_lease(record)
            record.updated_at = self._clock()
            return self._snapshot(record)

    def fail(
        self,
        task_id: str,
        tenant_id: str,
        owner_subject: str,
        *,
        error_code: str,
        lease_token: str | None = None,
    ) -> ImportTaskSnapshot:
        if not isinstance(error_code, str) or not _ERROR_CODE.fullmatch(error_code):
            raise ValueError("invalid import error code")
        with self._lock:
            record = self._owned(task_id, tenant_id, owner_subject)
            if record.status is ImportTaskStatus.FAILED:
                return self._snapshot(record)
            if record.status is ImportTaskStatus.COMPLETED:
                raise _state_conflict()
            self._require_lease(record, lease_token)
            record.status = ImportTaskStatus.FAILED
            record.current_step = None
            record.error_code = error_code
            self._clear_lease(record)
            record.updated_at = self._clock()
            return self._snapshot(record)

    def rollback_queued(
        self,
        tenant_id: str,
        owner_subject: str,
        task_ids: Sequence[str],
    ) -> None:
        unique_ids = tuple(dict.fromkeys(task_ids))
        with self._lock:
            records: list[_ImportTaskRecord] = []
            for task_id in unique_ids:
                record = self._records.get(task_id)
                if record is None:
                    continue
                if (
                    record.source.tenant_id != tenant_id
                    or record.source.owner_subject != owner_subject
                ):
                    raise _resource_not_found()
                if record.status is not ImportTaskStatus.QUEUED:
                    raise _state_conflict()
                records.append(record)
            for record in records:
                del self._records[record.source.task_id]
