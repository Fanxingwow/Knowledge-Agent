from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import cast

from app.core.errors import DependencyError, ErrorCode
from app.infrastructure.minio_store import MinioStore, ObservedObject, StoredObject, build_object_ref
from app.services.import_admission import (
    AdmissionBatchIntent,
    AdmissionObjectIntent,
    AdmissionStatus,
    InMemoryImportAdmissionRepository,
    ObjectStatus,
)
from app.services.import_cleanup import run_cleanup_cycle
from app.services.import_workspace import ImportWorkspace


class Store:
    def __init__(self) -> None:
        self.observed: ObservedObject | None = None
        self.failure: Exception | None = None
        self.deleted: list[StoredObject] = []

    def inspect_object(self, tenant_id: str, ref):
        assert tenant_id == ref.tenant_id
        if self.failure:
            raise self.failure
        return self.observed

    def delete_stored(self, tenant_id: str, stored: StoredObject) -> None:
        assert tenant_id == stored.ref.tenant_id
        if self.failure:
            raise self.failure
        self.deleted.append(stored)


def prepared(tmp_path: Path):
    repository = InMemoryImportAdmissionRepository()
    workspace = ImportWorkspace(tmp_path / "imports")
    task_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    body = b"hello"
    ref = build_object_ref(
        "tenant-a",
        "documents",
        uuid.UUID(document_id).hex,
        "imports",
        uuid.UUID(task_id).hex,
        "source.md",
    )
    item = AdmissionObjectIntent(
        object_id=task_id,
        task_id=task_id,
        document_id=document_id,
        ordinal=0,
        ref=ref,
        original_name="guide.md",
        content_type="text/markdown",
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )
    intent = AdmissionBatchIntent(
        batch_id=str(uuid.uuid4()),
        tenant_id="tenant-a",
        owner_subject="user-a",
        idempotency_key="request-1",
        payload_hash="a" * 64,
        objects=(item,),
    )
    repository.prepare_batch(intent)
    repository.mark_write_started(
        intent.batch_id,
        task_id,
        intent.tenant_id,
        intent.owner_subject,
    )
    task_dir = workspace.prepare(task_id)
    (task_dir / "source.md").write_bytes(body)
    return repository, workspace, intent, item


def observation(item: AdmissionObjectIntent) -> ObservedObject:
    stored = StoredObject(ref=item.ref, etag="etag-1", version_id="version-1")
    return ObservedObject(
        stored=stored,
        size=item.size,
        metadata={
            "schema-version": "1",
            "document-id": uuid.UUID(item.document_id).hex,
            "import-id": uuid.UUID(item.task_id).hex,
            "sha256": item.sha256,
            "object-kind": "source",
        },
    )


def test_unknown_absent_object_is_finalized_without_delete(tmp_path: Path) -> None:
    repository, workspace, intent, item = prepared(tmp_path)
    repository.record_write_unknown(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
    )
    repository.request_rollback(intent.batch_id, intent.tenant_id, intent.owner_subject)
    store = Store()

    result = run_cleanup_cycle(
        repository,
        cast(MinioStore, store),
        workspace,
        worker_id="cleanup-1",
    )

    assert result.completed == 1
    assert store.deleted == []
    assert repository.get_owned(intent.batch_id, "tenant-a", "user-a").status is (
        AdmissionStatus.ROLLED_BACK
    )
    assert not any(path.is_file() for path in workspace.root.rglob("*"))


def test_unknown_matching_object_is_reconciled_then_precisely_deleted(tmp_path: Path) -> None:
    repository, workspace, intent, item = prepared(tmp_path)
    repository.record_write_unknown(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
    )
    repository.request_rollback(intent.batch_id, intent.tenant_id, intent.owner_subject)
    store = Store()
    store.observed = observation(item)

    first = run_cleanup_cycle(
        repository,
        cast(MinioStore, store),
        workspace,
        worker_id="reconcile",
    )
    second = run_cleanup_cycle(
        repository,
        cast(MinioStore, store),
        workspace,
        worker_id="delete",
    )

    assert first.completed == second.completed == 1
    assert store.deleted == [observation(item).stored]
    final = repository.get_owned(intent.batch_id, "tenant-a", "user-a")
    assert final.status is AdmissionStatus.ROLLED_BACK
    assert final.objects[0].status is ObjectStatus.DELETED


def test_mismatched_observation_is_not_deleted_and_retains_retry_state(tmp_path: Path) -> None:
    repository, workspace, intent, item = prepared(tmp_path)
    repository.record_write_unknown(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
    )
    repository.request_rollback(intent.batch_id, intent.tenant_id, intent.owner_subject)
    store = Store()
    good = observation(item)
    store.observed = ObservedObject(good.stored, good.size, {**good.metadata, "sha256": "f" * 64})

    result = run_cleanup_cycle(
        repository,
        cast(MinioStore, store),
        workspace,
        worker_id="reconcile",
    )

    assert result.retry_pending == 1
    assert store.deleted == []
    current = repository.get_owned(intent.batch_id, "tenant-a", "user-a")
    assert current.status is AdmissionStatus.RECONCILIATION_REQUIRED
    assert current.objects[0].last_error_code == ErrorCode.RESOURCE_CONFLICT.value


def test_delete_failure_keeps_exact_version_for_retry(tmp_path: Path) -> None:
    repository, workspace, intent, item = prepared(tmp_path)
    persisted = observation(item).stored
    repository.record_stored(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        persisted,
    )
    repository.request_rollback(intent.batch_id, intent.tenant_id, intent.owner_subject)
    store = Store()
    store.failure = DependencyError("endpoint-secret")

    result = run_cleanup_cycle(
        repository,
        cast(MinioStore, store),
        workspace,
        worker_id="cleanup",
    )

    assert result.retry_pending == 1
    current = repository.get_owned(intent.batch_id, "tenant-a", "user-a")
    assert current.objects[0].stored_object == persisted
    assert current.objects[0].cleanup_attempts == 1
    assert current.objects[0].last_error_code == ErrorCode.DEPENDENCY_UNAVAILABLE.value
