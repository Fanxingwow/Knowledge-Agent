from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from app.core.errors import DependencyError
from app.services.tenant_vector_search import CompletedImportVersion

_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_UNAVAILABLE_MESSAGE = "Knowledge visibility registry is unavailable"


class MongoTaskCollection(Protocol):
    def find(
        self,
        filter: Mapping[str, object],
        projection: Mapping[str, int],
    ) -> Iterable[Mapping[str, object]]: ...


def _canonical_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid import version")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("invalid import version") from None
    if str(parsed) != value:
        raise ValueError("invalid import version")
    return value


class MongoCompletedImportVisibilityReader:
    """Read completed import tasks as the sole vector activation manifest."""

    def __init__(
        self,
        tasks: MongoTaskCollection,
        *,
        max_batch_size: int = 500,
    ) -> None:
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size < 1
        ):
            raise ValueError("invalid visibility batch size")
        self._tasks = tasks
        self._max_batch_size = max_batch_size

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @staticmethod
    def _unavailable() -> DependencyError:
        return DependencyError(_UNAVAILABLE_MESSAGE)

    def get_completed_versions(
        self,
        tenant_id: str,
        import_versions: Sequence[str],
    ) -> Mapping[str, CompletedImportVersion]:
        if not isinstance(tenant_id, str) or _TENANT_ID.fullmatch(tenant_id) is None:
            raise ValueError("invalid tenant identity")
        if (
            isinstance(import_versions, (str, bytes, bytearray))
            or not isinstance(import_versions, Sequence)
        ):
            raise ValueError("invalid import versions")
        unique_versions: list[str] = []
        seen: set[str] = set()
        for value in import_versions:
            version = _canonical_uuid(value)
            if version not in seen:
                unique_versions.append(version)
                seen.add(version)
        if len(unique_versions) > self._max_batch_size:
            raise ValueError("too many import versions")
        if not unique_versions:
            return {}

        query: dict[str, object] = {
            "_id": {"$in": unique_versions},
            "tenant_id": tenant_id,
            "status": "completed",
        }
        projection = {
            "_id": 1,
            "tenant_id": 1,
            "document_id": 1,
            "status": 1,
            "commit_token": 1,
        }
        try:
            documents = tuple(self._tasks.find(query, projection))
        except Exception:
            raise self._unavailable() from None
        if len(documents) > len(unique_versions):
            raise self._unavailable()

        completed: dict[str, CompletedImportVersion] = {}
        try:
            for document in documents:
                if not isinstance(document, Mapping):
                    raise ValueError("invalid visibility record")
                task_id = document.get("_id")
                record_tenant = document.get("tenant_id")
                document_id = document.get("document_id")
                commit_token = document.get("commit_token")
                if (
                    not isinstance(task_id, str)
                    or task_id not in seen
                    or not isinstance(record_tenant, str)
                    or record_tenant != tenant_id
                    or not isinstance(document_id, str)
                    or not isinstance(commit_token, str)
                    or document.get("status") != "completed"
                ):
                    raise ValueError("out-of-scope visibility record")
                activation = CompletedImportVersion(
                    task_id=task_id,
                    tenant_id=record_tenant,
                    document_id=document_id,
                    commit_token=commit_token,
                )
                if activation.task_id in completed:
                    raise ValueError("duplicate visibility record")
                completed[activation.task_id] = activation
        except (KeyError, TypeError, ValueError):
            raise self._unavailable() from None
        return completed
