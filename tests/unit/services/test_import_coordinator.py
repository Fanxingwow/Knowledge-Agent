from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from app.core.errors import AppError, DependencyError, ErrorCode
from app.infrastructure.minio_store import StoredObject, build_object_ref
from app.services.import_admission import (
    AdmissionBatchIntent,
    AdmissionObjectIntent,
    AdmissionStatus,
)
from app.services.import_coordinator import InMemoryImportCoordinator
from app.services.import_registry import ImportTaskStatus, NewImportTask


def prepared_runtime() -> tuple[
    InMemoryImportCoordinator,
    AdmissionBatchIntent,
    tuple[NewImportTask, ...],
]:
    coordinator = InMemoryImportCoordinator()
    batch_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
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
        size=5,
        sha256="a" * 64,
    )
    intent = AdmissionBatchIntent(
        batch_id=batch_id,
        tenant_id="tenant-a",
        owner_subject="user-a",
        idempotency_key="request-1",
        payload_hash="b" * 64,
        objects=(item,),
    )
    task = NewImportTask(
        task_id=task_id,
        document_id=document_id,
        tenant_id="tenant-a",
        owner_subject="user-a",
        source_ref=ref.uri,
        original_name="guide.md",
        content_type="text/markdown",
        size=5,
        sha256="a" * 64,
    )
    coordinator.admissions.prepare_batch(intent)
    started = coordinator.admissions.mark_write_started(
        batch_id,
        task_id,
        "tenant-a",
        "user-a",
        expected_revision=0,
    )
    coordinator.admissions.record_stored(
        batch_id,
        task_id,
        "tenant-a",
        "user-a",
        StoredObject(ref=ref, etag="etag-1", version_id="version-1"),
        expected_revision=started.revision,
    )
    return coordinator, intent, (task,)


def test_accept_and_enqueue_commits_both_views_and_is_idempotent() -> None:
    coordinator, intent, tasks = prepared_runtime()
    before = coordinator.admissions.get_owned(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
    )

    admission, snapshots = coordinator.accept_and_enqueue(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        tasks,
        expected_revision=before.revision,
    )
    retried_admission, retried_tasks = coordinator.accept_and_enqueue(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        tasks,
        expected_revision=before.revision,
    )

    assert admission.status is AdmissionStatus.ACCEPTED
    assert snapshots[0].status is ImportTaskStatus.QUEUED
    assert retried_admission == admission
    assert retried_tasks == snapshots


def test_mismatched_task_cannot_be_committed() -> None:
    coordinator, intent, tasks = prepared_runtime()
    forged = replace(tasks[0], tenant_id="tenant-b")

    with pytest.raises(AppError) as captured:
        coordinator.accept_and_enqueue(
            intent.batch_id,
            intent.tenant_id,
            intent.owner_subject,
            (forged,),
        )

    assert captured.value.code is ErrorCode.RESOURCE_CONFLICT
    with pytest.raises(AppError) as missing:
        coordinator.tasks.get_owned(tasks[0].task_id, "tenant-a", "user-a")
    assert missing.value.code is ErrorCode.RESOURCE_NOT_FOUND


def test_closed_coordinator_fails_health_without_secret_detail() -> None:
    coordinator = InMemoryImportCoordinator()
    coordinator.close()
    coordinator.close()

    with pytest.raises(DependencyError) as captured:
        coordinator.probe()

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert "secret" not in str(captured.value).lower()
