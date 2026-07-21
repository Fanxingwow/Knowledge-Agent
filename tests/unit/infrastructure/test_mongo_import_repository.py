from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

import app.infrastructure.mongo_import_repository as repository_module
from app.core.errors import AppError, DependencyError, ErrorCode
from app.infrastructure.minio_store import StoredObject, build_object_ref
from app.infrastructure.mongo_import_repository import MongoImportCoordinator
from app.infrastructure.mongo_import_store import MongoImportStore
from app.services.import_admission import (
    AdmissionBatchIntent,
    AdmissionObjectIntent,
    AdmissionStatus,
    ObjectStatus,
)
from app.services.import_registry import ImportTaskStatus, NewImportTask


@dataclass
class MutableClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _matches(document: dict[str, Any], query: dict[str, object]) -> bool:
    for key, condition in query.items():
        value = document.get(key)
        if isinstance(condition, dict):
            if "$in" in condition:
                candidates = condition["$in"]
                if not isinstance(candidates, list):
                    return False
                if isinstance(value, list):
                    if not any(item in candidates for item in value):
                        return False
                elif value not in candidates:
                    return False
            elif "$nin" in condition:
                candidates = condition["$nin"]
                if not isinstance(candidates, list) or value in candidates:
                    return False
            else:
                return False
        elif isinstance(value, list):
            if condition not in value:
                return False
        elif value != condition:
            return False
    return True


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.documents: dict[str, dict[str, Any]] = {}
        self.fail_inserts = False
        self.fail_replaces = False

    def find_one(
        self,
        query: dict[str, object],
        *,
        session: object | None = None,
    ) -> dict[str, Any] | None:
        del session
        for document in self.documents.values():
            if _matches(document, query):
                return copy.deepcopy(document)
        return None

    def find(
        self,
        query: dict[str, object],
        *,
        session: object | None = None,
    ) -> list[dict[str, Any]]:
        del session
        return [
            copy.deepcopy(document)
            for document in self.documents.values()
            if _matches(document, query)
        ]

    def _check_unique(self, document: dict[str, Any]) -> None:
        identifier = document["_id"]
        if identifier in self.documents:
            raise RuntimeError("duplicate Mongo key")
        if self.name == "admissions":
            key = (
                document["tenant_id"],
                document["owner_subject"],
                document["idempotency_key"],
            )
            for existing in self.documents.values():
                if key == (
                    existing["tenant_id"],
                    existing["owner_subject"],
                    existing["idempotency_key"],
                ):
                    raise RuntimeError("duplicate Mongo key")

    def insert_one(
        self,
        document: dict[str, Any],
        *,
        session: object | None = None,
    ) -> object:
        del session
        if self.fail_inserts:
            raise RuntimeError("injected Mongo write failure")
        self._check_unique(document)
        self.documents[document["_id"]] = copy.deepcopy(document)
        return SimpleNamespace(inserted_id=document["_id"])

    def insert_many(
        self,
        documents: list[dict[str, Any]],
        *,
        ordered: bool,
        session: object | None = None,
    ) -> object:
        del ordered, session
        if self.fail_inserts:
            raise RuntimeError("injected Mongo write failure")
        for document in documents:
            self._check_unique(document)
        for document in documents:
            self.documents[document["_id"]] = copy.deepcopy(document)
        return SimpleNamespace(inserted_ids=[document["_id"] for document in documents])

    def replace_one(
        self,
        query: dict[str, object],
        document: dict[str, Any],
        *,
        upsert: bool,
        session: object | None = None,
    ) -> object:
        del upsert, session
        if self.fail_replaces:
            raise RuntimeError("injected Mongo replace failure")
        for identifier, existing in self.documents.items():
            if _matches(existing, query):
                self.documents[identifier] = copy.deepcopy(document)
                return SimpleNamespace(matched_count=1)
        return SimpleNamespace(matched_count=0)

    def delete_many(
        self,
        query: dict[str, object],
        *,
        session: object | None = None,
    ) -> object:
        del session
        identifiers = [
            identifier
            for identifier, document in self.documents.items()
            if _matches(document, query)
        ]
        for identifier in identifiers:
            del self.documents[identifier]
        return SimpleNamespace(deleted_count=len(identifiers))


class FakeMongoImportStore:
    durable = True

    def __init__(self) -> None:
        self.admissions = FakeCollection("admissions")
        self.tasks = FakeCollection("tasks")
        self.outbox = FakeCollection("outbox")
        self.database = object()
        self.fail_after_commit = False
        self.ready = True
        self.ready_calls = 0
        self.closed = False

    def with_transaction(self, callback: Any) -> Any:
        snapshots = (
            copy.deepcopy(self.admissions.documents),
            copy.deepcopy(self.tasks.documents),
            copy.deepcopy(self.outbox.documents),
        )
        try:
            result = callback(SimpleNamespace())
        except BaseException:
            self.admissions.documents, self.tasks.documents, self.outbox.documents = snapshots
            raise
        if self.fail_after_commit:
            self.fail_after_commit = False
            raise DependencyError("unknown commit result")
        return result

    def require_ready(self) -> None:
        self.ready_calls += 1
        if not self.ready or self.closed:
            raise DependencyError("fake endpoint secret")

    def close(self) -> None:
        self.closed = True


def new_intent(*, object_count: int = 2) -> AdmissionBatchIntent:
    tenant_id = "tenant-a"
    objects: list[AdmissionObjectIntent] = []
    for ordinal in range(object_count):
        task_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())
        objects.append(
            AdmissionObjectIntent(
                object_id=task_id,
                task_id=task_id,
                document_id=document_id,
                ordinal=ordinal,
                ref=build_object_ref(
                    tenant_id,
                    "documents",
                    uuid.UUID(document_id).hex,
                    "imports",
                    uuid.UUID(task_id).hex,
                    "source.md",
                ),
                original_name=f"source-{ordinal}.md",
                content_type="text/markdown",
                size=ordinal + 1,
                sha256=f"{ordinal + 1:x}" * 64,
            )
        )
    return AdmissionBatchIntent(
        batch_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        owner_subject="user-a",
        idempotency_key="request-1",
        payload_hash="f" * 64,
        objects=tuple(objects),
    )


def tasks_for(intent: AdmissionBatchIntent) -> tuple[NewImportTask, ...]:
    return tuple(
        NewImportTask(
            task_id=item.task_id,
            document_id=item.document_id,
            tenant_id=intent.tenant_id,
            owner_subject=intent.owner_subject,
            source_ref=item.ref.uri,
            original_name=item.original_name,
            content_type=item.content_type,
            size=item.size,
            sha256=item.sha256,
        )
        for item in intent.objects
    )


def coordinator_and_store(
    *,
    clock: MutableClock | None = None,
) -> tuple[MongoImportCoordinator, FakeMongoImportStore]:
    store = FakeMongoImportStore()
    coordinator = MongoImportCoordinator(
        cast(MongoImportStore, store),
        clock=clock,
        cleanup_retry_base_seconds=2,
        cleanup_retry_max_seconds=30,
    )
    return coordinator, store


def test_chat_repository_is_disabled_by_default_without_touching_shared_database() -> None:
    coordinator, store = coordinator_and_store()

    assert coordinator.chat_history_repository is None
    assert coordinator.completed_import_visibility.max_batch_size == 500
    assert store.closed is False


def test_optional_chat_repository_uses_shared_database_transaction_runner_and_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMongoImportStore()
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    repository_marker = object()
    observed: dict[str, object] = {}

    class RecordingSchemaManager:
        def __init__(
            self,
            database: object,
            *,
            sessions_collection: str,
            messages_collection: str,
        ) -> None:
            observed["database"] = database
            observed["sessions_collection"] = sessions_collection
            observed["messages_collection"] = messages_collection

        def build_repository(
            self,
            transaction_runner: object,
            *,
            clock: object,
        ) -> object:
            observed["transaction_runner"] = transaction_runner
            observed["clock"] = clock
            return repository_marker

    monkeypatch.setattr(
        repository_module,
        "MongoChatSchemaManager",
        RecordingSchemaManager,
    )

    coordinator = MongoImportCoordinator(
        cast(MongoImportStore, store),
        clock=clock,
        visibility_batch_size=73,
        chat_enabled=True,
        chat_sessions_collection="tenant_chat_sessions_v2",
        chat_messages_collection="tenant_chat_messages_v2",
    )

    assert coordinator.chat_history_repository is repository_marker
    assert coordinator.completed_import_visibility.max_batch_size == 73
    assert observed == {
        "database": store.database,
        "sessions_collection": "tenant_chat_sessions_v2",
        "messages_collection": "tenant_chat_messages_v2",
        "transaction_runner": store,
        "clock": clock,
    }
    coordinator.probe()
    assert store.ready_calls == 1
    coordinator.close()
    assert store.closed is True


def test_chat_schema_failure_leaves_shared_store_open_for_caller_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeMongoImportStore()

    class FailingSchemaManager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def build_repository(self, *_args: object, **_kwargs: object) -> object:
            raise DependencyError("sanitized chat schema failure")

    monkeypatch.setattr(
        repository_module,
        "MongoChatSchemaManager",
        FailingSchemaManager,
    )

    with pytest.raises(DependencyError, match="sanitized chat schema failure"):
        MongoImportCoordinator(
            cast(MongoImportStore, store),
            chat_enabled=True,
        )

    assert store.closed is False
    store.close()
    assert store.closed is True


def prepare_confirmed(
    coordinator: MongoImportCoordinator,
    intent: AdmissionBatchIntent,
) -> None:
    coordinator.admissions.prepare_batch(intent)
    for ordinal, item in enumerate(intent.objects):
        coordinator.admissions.mark_write_started(
            intent.batch_id,
            item.object_id,
            intent.tenant_id,
            intent.owner_subject,
        )
        coordinator.admissions.record_stored(
            intent.batch_id,
            item.object_id,
            intent.tenant_id,
            intent.owner_subject,
            StoredObject(
                ref=item.ref,
                etag=f"etag-{ordinal}",
                version_id=f"version-{ordinal}",
            ),
        )


def assert_conflict(error: AppError) -> None:
    assert error.code is ErrorCode.RESOURCE_CONFLICT
    assert error.status_code == 409


def test_accept_transaction_is_all_or_none() -> None:
    coordinator, store = coordinator_and_store()
    intent = new_intent()
    prepare_confirmed(coordinator, intent)
    store.outbox.fail_inserts = True

    with pytest.raises(DependencyError) as captured:
        coordinator.accept_and_enqueue(
            intent.batch_id,
            intent.tenant_id,
            intent.owner_subject,
            tasks_for(intent),
        )

    assert str(captured.value) == "MongoDB import repository is unavailable"
    assert coordinator.admissions.get_owned(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
    ).status is AdmissionStatus.PREPARING
    assert store.tasks.documents == {}
    assert store.outbox.documents == {}


def test_unknown_commit_is_recovered_idempotently_with_deterministic_outbox() -> None:
    coordinator, store = coordinator_and_store()
    intent = new_intent()
    prepare_confirmed(coordinator, intent)
    before = coordinator.admissions.get_owned(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
    )
    store.fail_after_commit = True

    accepted, task_snapshots = coordinator.accept_and_enqueue(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        tasks_for(intent),
        expected_revision=before.revision,
    )
    retried = coordinator.accept_and_enqueue(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        tasks_for(intent),
        expected_revision=before.revision,
    )

    assert accepted.status is AdmissionStatus.ACCEPTED
    assert retried == (accepted, task_snapshots)
    assert set(store.outbox.documents) == {
        f"import.execute:{item.task_id}:v1" for item in intent.objects
    }
    assert len(store.tasks.documents) == len(intent.objects)


def test_cross_tenant_and_owner_reads_are_indistinguishable_from_missing() -> None:
    coordinator, _ = coordinator_and_store()
    intent = new_intent(object_count=1)
    prepare_confirmed(coordinator, intent)
    coordinator.accept_and_enqueue(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        tasks_for(intent),
    )

    with pytest.raises(AppError) as admission_error:
        coordinator.admissions.get_owned(intent.batch_id, "tenant-b", intent.owner_subject)
    with pytest.raises(AppError) as task_error:
        coordinator.tasks.get_owned(intent.objects[0].task_id, intent.tenant_id, "user-b")

    assert admission_error.value.code is ErrorCode.RESOURCE_NOT_FOUND
    assert str(admission_error.value) == "Import admission not found"
    assert task_error.value.code is ErrorCode.RESOURCE_NOT_FOUND
    assert str(task_error.value) == "Import task not found"


def test_task_lease_token_fences_stale_worker() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    coordinator, _ = coordinator_and_store(clock=clock)
    intent = new_intent(object_count=1)
    prepare_confirmed(coordinator, intent)
    coordinator.accept_and_enqueue(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        tasks_for(intent),
    )
    task = intent.objects[0]
    first = coordinator.tasks.claim_next("worker-a", lease_seconds=10, max_attempts=3)
    assert first is not None

    with pytest.raises(AppError) as wrong_token:
        coordinator.tasks.start_step(
            task.task_id,
            intent.tenant_id,
            intent.owner_subject,
            "parse_document",
            lease_token=str(uuid.uuid4()),
        )
    assert_conflict(wrong_token.value)

    clock.advance(10)
    second = coordinator.tasks.claim_next("worker-b", lease_seconds=30, max_attempts=3)
    assert second is not None
    assert second.lease_token != first.lease_token
    with pytest.raises(AppError) as stale:
        coordinator.tasks.start_step(
            task.task_id,
            intent.tenant_id,
            intent.owner_subject,
            "parse_document",
            lease_token=first.lease_token,
        )
    assert_conflict(stale.value)

    coordinator.tasks.start_step(
        task.task_id,
        intent.tenant_id,
        intent.owner_subject,
        "parse_document",
        lease_token=second.lease_token,
    )
    coordinator.tasks.complete_step(
        task.task_id,
        intent.tenant_id,
        intent.owner_subject,
        "parse_document",
        lease_token=second.lease_token,
    )
    completed = coordinator.tasks.complete(
        task.task_id,
        intent.tenant_id,
        intent.owner_subject,
        imported_chunk_count=2,
        commit_token="index-revision-2",  # noqa: S106 - opaque index revision
        lease_token=second.lease_token,
    )
    assert completed.status is ImportTaskStatus.COMPLETED


def test_retry_accepts_same_attempt_after_heartbeat_extended_the_lease() -> None:
    clock = MutableClock(datetime(2026, 7, 16, tzinfo=UTC))
    coordinator, store = coordinator_and_store(clock=clock)
    intent = new_intent()
    prepare_confirmed(coordinator, intent)
    coordinator.accept_and_enqueue(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        tasks_for(intent),
    )

    claimed = coordinator.tasks.claim_next("worker-a", lease_seconds=10, max_attempts=3)
    assert claimed is not None
    renewed = coordinator.tasks.renew_lease(claimed, lease_seconds=30)
    assert renewed.expires_at > claimed.expires_at
    asserted = coordinator.tasks.assert_lease(claimed)
    assert asserted.expires_at == renewed.expires_at
    assert asserted.lease_token == claimed.lease_token
    clock.advance(11)

    retried = coordinator.tasks.retry(
        claimed,
        error_code=ErrorCode.DEPENDENCY_UNAVAILABLE.value,
        delay_seconds=5,
    )

    assert retried.status is ImportTaskStatus.QUEUED
    outbox = next(iter(store.outbox.documents.values()))
    assert outbox["status"] == "retry"
    assert outbox["lease_token"] is None
    with pytest.raises(AppError) as stale:
        coordinator.tasks.assert_lease(claimed)
    assert stale.value.code is ErrorCode.RESOURCE_CONFLICT


def test_task_claim_retry_and_completion_drain_pending_outbox_with_terminal_ttl() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    coordinator, store = coordinator_and_store(clock=clock)
    intent = new_intent(object_count=1)
    prepare_confirmed(coordinator, intent)
    coordinator.accept_and_enqueue(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        tasks_for(intent),
    )
    task_id = intent.objects[0].task_id
    outbox_id = f"import.execute:{task_id}:v1"
    assert store.outbox.documents[outbox_id]["status"] == "pending"

    first = coordinator.tasks.claim_next("worker-a", lease_seconds=30, max_attempts=3)
    assert first is not None
    assert store.outbox.documents[outbox_id]["status"] == "claimed"
    assert store.outbox.documents[outbox_id]["lease_token"] == first.lease_token

    retried = coordinator.tasks.retry(
        first,
        error_code=ErrorCode.DEPENDENCY_UNAVAILABLE.value,
        delay_seconds=5,
    )
    assert retried.status is ImportTaskStatus.QUEUED
    assert store.outbox.documents[outbox_id]["status"] == "retry"
    assert store.outbox.documents[outbox_id]["last_error_code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value

    clock.advance(5)
    second = coordinator.tasks.claim_next("worker-b", lease_seconds=30, max_attempts=3)
    assert second is not None
    completed = coordinator.tasks.complete(
        task_id,
        intent.tenant_id,
        intent.owner_subject,
        imported_chunk_count=1,
        commit_token="index-revision-1",  # noqa: S106 - opaque index revision
        lease_token=second.lease_token,
    )

    assert completed.status is ImportTaskStatus.COMPLETED
    terminal_expiry = clock.value + timedelta(days=30)
    assert store.outbox.documents[outbox_id]["status"] == "completed"
    assert store.outbox.documents[outbox_id]["expire_at"] == terminal_expiry
    assert store.tasks.documents[task_id]["expire_at"] == terminal_expiry
    assert not any(document["status"] == "pending" for document in store.outbox.documents.values())


def test_outbox_claim_failure_rolls_back_task_claim_without_losing_work() -> None:
    coordinator, store = coordinator_and_store()
    intent = new_intent(object_count=1)
    prepare_confirmed(coordinator, intent)
    coordinator.accept_and_enqueue(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        tasks_for(intent),
    )
    task_id = intent.objects[0].task_id
    outbox_id = f"import.execute:{task_id}:v1"
    store.outbox.fail_replaces = True

    with pytest.raises(DependencyError):
        coordinator.tasks.claim_next("worker-a", lease_seconds=30, max_attempts=3)

    assert store.tasks.documents[task_id]["status"] == ImportTaskStatus.QUEUED.value
    assert store.outbox.documents[outbox_id]["status"] == "pending"
    store.outbox.fail_replaces = False
    lease = coordinator.tasks.claim_next("worker-b", lease_seconds=30, max_attempts=3)
    assert lease is not None
    assert store.outbox.documents[outbox_id]["status"] == "claimed"


def test_cleanup_lease_binds_the_exact_stored_version_and_retries_with_backoff() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    coordinator, _ = coordinator_and_store(clock=clock)
    intent = new_intent(object_count=1)
    item = intent.objects[0]
    coordinator.admissions.prepare_batch(intent)
    coordinator.admissions.mark_write_started(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
    )
    exact = StoredObject(ref=item.ref, etag="etag", version_id="exact-version")
    coordinator.admissions.record_stored(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        exact,
    )
    coordinator.admissions.request_rollback(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
    )
    lease = coordinator.admissions.claim_cleanup("cleanup-a", limit=1, lease_seconds=30)[0]
    assert lease.stored_object == exact

    tampered = replace(
        lease,
        stored_object=StoredObject(ref=item.ref, etag="etag", version_id="other-version"),
    )
    with pytest.raises(AppError) as rejected:
        coordinator.admissions.mark_deleted(tampered)
    assert_conflict(rejected.value)

    released = coordinator.admissions.release_cleanup(
        lease,
        error_code=ErrorCode.DEPENDENCY_UNAVAILABLE.value,
    )
    assert released.cleanup_attempts == 1
    assert coordinator.admissions.claim_cleanup("cleanup-b", limit=1, lease_seconds=30) == ()
    clock.advance(2)
    replacement = coordinator.admissions.claim_cleanup("cleanup-b", limit=1, lease_seconds=30)[0]
    assert replacement.stored_object == exact
    deleted = coordinator.admissions.mark_deleted(replacement)
    assert deleted.status is ObjectStatus.DELETED
    assert coordinator.admissions.get_owned(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
    ).status is AdmissionStatus.ROLLED_BACK


def test_write_lease_is_persisted_renewed_and_fences_write_completion() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    coordinator, _ = coordinator_and_store(clock=clock)
    intent = new_intent(object_count=1)
    item = intent.objects[0]
    coordinator.admissions.prepare_batch(intent)

    started = coordinator.admissions.mark_write_started(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        writer_id="api-writer-a",
        lease_seconds=10,
    )
    assert started.write_lease_token is not None
    assert started.write_lease_expires_at == clock.value + timedelta(seconds=10)

    renewed = coordinator.admissions.renew_write_lease(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        lease_token=started.write_lease_token,
        expected_revision=started.revision,
        lease_seconds=30,
    )
    assert renewed.write_lease_token == started.write_lease_token
    assert renewed.write_lease_expires_at == clock.value + timedelta(seconds=30)

    with pytest.raises(AppError) as stale_writer:
        coordinator.admissions.record_write_unknown(
            intent.batch_id,
            item.object_id,
            intent.tenant_id,
            intent.owner_subject,
            expected_revision=renewed.revision,
            lease_token=str(uuid.uuid4()),
        )
    assert_conflict(stale_writer.value)

    unknown = coordinator.admissions.record_write_unknown(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=renewed.revision,
        lease_token=renewed.write_lease_token,
    )
    assert unknown.status is ObjectStatus.WRITE_UNKNOWN
    assert unknown.write_lease_token == renewed.write_lease_token

    confirmed = coordinator.admissions.record_stored(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        StoredObject(ref=item.ref, etag="etag", version_id="write-version"),
        expected_revision=unknown.revision,
        lease_token=unknown.write_lease_token,
    )
    assert confirmed.status is ObjectStatus.WRITE_CONFIRMED
    assert confirmed.write_lease_token is None
    assert confirmed.write_lease_expires_at is None


def test_active_write_lease_blocks_cleanup_and_stale_recovery_until_expiry() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    coordinator, _ = coordinator_and_store(clock=clock)
    intent = new_intent(object_count=1)
    item = intent.objects[0]
    coordinator.admissions.prepare_batch(intent)
    started = coordinator.admissions.mark_write_started(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        writer_id="api-writer-a",
        lease_seconds=10,
    )

    assert coordinator.admissions.recover_stale_preparing(clock.value, limit=10) == ()
    rollback = coordinator.admissions.request_rollback(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
    )
    assert rollback.status is AdmissionStatus.RECONCILIATION_REQUIRED
    assert coordinator.admissions.claim_cleanup("cleanup", limit=1, lease_seconds=30) == ()

    clock.advance(10)
    cleanup = coordinator.admissions.claim_cleanup("cleanup", limit=1, lease_seconds=30)[0]
    assert cleanup.lease_token != started.write_lease_token
    assert cleanup.object_status is ObjectStatus.WRITE_UNKNOWN

    second = replace(
        new_intent(object_count=1),
        idempotency_key="request-2",
        payload_hash="e" * 64,
    )
    coordinator.admissions.prepare_batch(second)
    coordinator.admissions.mark_write_started(
        second.batch_id,
        second.objects[0].object_id,
        second.tenant_id,
        second.owner_subject,
        writer_id="api-writer-b",
        lease_seconds=10,
    )
    assert coordinator.admissions.recover_stale_preparing(clock.value, limit=10) == ()
    clock.advance(10)
    recovered = coordinator.admissions.recover_stale_preparing(clock.value, limit=10)
    assert tuple(snapshot.batch_id for snapshot in recovered) == (second.batch_id,)
    assert recovered[0].rollback_reason_code == "ADMISSION_RECOVERY"


def test_corrupt_schema_and_closed_probe_return_sanitized_dependency_errors() -> None:
    coordinator, store = coordinator_and_store()
    intent = new_intent(object_count=1)
    coordinator.admissions.prepare_batch(intent)
    store.admissions.documents[intent.batch_id]["schema_version"] = 999

    with pytest.raises(DependencyError) as corrupt:
        coordinator.admissions.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject)
    assert str(corrupt.value) == "MongoDB import repository is unavailable"

    coordinator.close()
    coordinator.close()
    with pytest.raises(DependencyError) as closed:
        coordinator.probe()
    assert str(closed.value) == "MongoDB import repository is unavailable"
