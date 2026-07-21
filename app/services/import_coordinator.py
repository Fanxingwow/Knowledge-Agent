from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.services.import_admission import (
    AdmissionBatchSnapshot,
    AdmissionStatus,
    ImportAdmissionRepository,
    InMemoryImportAdmissionRepository,
)
from app.services.import_registry import (
    ImportTaskRegistry,
    ImportTaskSnapshot,
    InMemoryImportTaskRegistry,
    NewImportTask,
)


class ImportCoordinator(Protocol):
    durable: bool
    admissions: ImportAdmissionRepository
    tasks: ImportTaskRegistry

    def accept_and_enqueue(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        tasks: Sequence[NewImportTask],
        *,
        expected_revision: int | None = None,
    ) -> tuple[AdmissionBatchSnapshot, tuple[ImportTaskSnapshot, ...]]: ...

    def probe(self) -> None: ...

    def close(self) -> None: ...


def _state_conflict() -> AppError:
    return AppError(ErrorCode.RESOURCE_CONFLICT, "Import admission state conflict", 409)


class InMemoryImportCoordinator:
    """Single-process coordinator mirroring the durable transaction contract for tests/dev."""

    durable = False

    def __init__(
        self,
        *,
        admissions: ImportAdmissionRepository | None = None,
        tasks: ImportTaskRegistry | None = None,
    ) -> None:
        self.admissions = admissions or InMemoryImportAdmissionRepository()
        self.tasks = tasks or InMemoryImportTaskRegistry()
        self._lock = RLock()
        self._closed = False

    @staticmethod
    def _validate_tasks(
        admission: AdmissionBatchSnapshot,
        tasks: Sequence[NewImportTask],
    ) -> tuple[NewImportTask, ...]:
        batch = tuple(tasks)
        ordered_objects = tuple(sorted(admission.objects, key=lambda item: item.ordinal))
        if len(batch) != len(ordered_objects):
            raise _state_conflict()
        for task, item in zip(batch, ordered_objects, strict=True):
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
                raise _state_conflict()
        return batch

    def accept_and_enqueue(
        self,
        batch_id: str,
        tenant_id: str,
        owner_subject: str,
        tasks: Sequence[NewImportTask],
        *,
        expected_revision: int | None = None,
    ) -> tuple[AdmissionBatchSnapshot, tuple[ImportTaskSnapshot, ...]]:
        with self._lock:
            if self._closed:
                self.probe()
            admission = self.admissions.get_owned(batch_id, tenant_id, owner_subject)
            batch = self._validate_tasks(admission, tasks)
            if admission.status is AdmissionStatus.ACCEPTED:
                snapshots = tuple(
                    self.tasks.get_owned(task.task_id, tenant_id, owner_subject)
                    for task in batch
                )
                return admission, snapshots

            snapshots = self.tasks.create_many(batch)
            try:
                accepted = self.admissions.accept_batch(
                    batch_id,
                    tenant_id,
                    owner_subject,
                    expected_revision=expected_revision,
                )
            except BaseException:
                self.tasks.rollback_queued(
                    tenant_id,
                    owner_subject,
                    [task.task_id for task in batch],
                )
                raise
            return accepted, snapshots

    def probe(self) -> None:
        with self._lock:
            if self._closed:
                self.admissions.probe()
            self.admissions.probe()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.admissions.close()
            close_tasks = getattr(self.tasks, "close", None)
            if callable(close_tasks):
                close_tasks()
