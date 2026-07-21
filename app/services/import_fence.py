from __future__ import annotations

import hashlib
from threading import RLock
from typing import Protocol, Self

from app.services.import_registry import ImportTaskLease, ImportTaskRegistry


class CommitFence(Protocol):
    """Read-only capability proving that one import attempt still owns its lease."""

    @property
    def task_id(self) -> str: ...

    @property
    def document_id(self) -> str: ...

    @property
    def tenant_id(self) -> str: ...

    @property
    def import_version(self) -> str: ...

    @property
    def attempt_count(self) -> int: ...

    @property
    def fence_id(self) -> str: ...

    def assert_current(self) -> None: ...


class RegistryCommitFence:
    """Bridge a durable task lease to side-effecting import dependencies.

    The raw lease token never leaves this object.  Dependencies receive only a
    stable digest for audit correlation and must call ``assert_current`` before
    and after every external mutation.
    """

    def __init__(self, registry: ImportTaskRegistry, lease: ImportTaskLease) -> None:
        if not isinstance(lease, ImportTaskLease):
            raise ValueError("invalid import task lease")
        self._registry = registry
        self._lease = lease
        self._lock = RLock()
        self._task_id = lease.task.task_id
        self._document_id = lease.task.document_id
        self._tenant_id = lease.task.tenant_id
        self._attempt_count = lease.attempt_count
        material = "\0".join(
            (
                self._tenant_id,
                self._document_id,
                self._task_id,
                str(self._attempt_count),
                lease.lease_token,
            )
        ).encode("utf-8")
        self._fence_id = hashlib.sha256(material).hexdigest()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def document_id(self) -> str:
        return self._document_id

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def import_version(self) -> str:
        return self._task_id

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    @property
    def fence_id(self) -> str:
        return self._fence_id

    def assert_current(self) -> None:
        with self._lock:
            refreshed = self._registry.assert_lease(self._lease)
            if (
                refreshed.task.task_id != self._task_id
                or refreshed.task.document_id != self._document_id
                or refreshed.task.tenant_id != self._tenant_id
                or refreshed.attempt_count != self._attempt_count
                or refreshed.lease_token != self._lease.lease_token
            ):
                raise RuntimeError("import task fence identity changed")
            self._lease = refreshed

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        del memo
        return self
