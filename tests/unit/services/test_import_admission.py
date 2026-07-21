from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import UTC, datetime, timedelta

import pytest

from app.core.errors import AppError, DependencyError, ErrorCode
from app.infrastructure.minio_store import ObjectRef, StoredObject, build_object_ref
from app.services.import_admission import (
    AdmissionBatchIntent,
    AdmissionObjectIntent,
    AdmissionStatus,
    CleanupLease,
    InMemoryImportAdmissionRepository,
    ObjectStatus,
)


@dataclass
class MutableClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def new_intent(
    *,
    batch_id: str | None = None,
    tenant_id: str = "tenant-a",
    owner_subject: str = "user-a",
    idempotency_key: str = "request-1",
    payload_hash: str = "f" * 64,
    object_count: int = 2,
) -> AdmissionBatchIntent:
    resolved_batch_id = batch_id or str(uuid.uuid4())
    objects: list[AdmissionObjectIntent] = []
    for index in range(object_count):
        task_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())
        objects.append(
            AdmissionObjectIntent(
                object_id=task_id,
                task_id=task_id,
                document_id=document_id,
                ordinal=index,
                ref=build_object_ref(
                    tenant_id,
                    "documents",
                    uuid.UUID(document_id).hex,
                    "imports",
                    uuid.UUID(task_id).hex,
                    "source.md",
                ),
                original_name=f"source-{index}.md",
                content_type="text/markdown",
                size=index + 1,
                sha256=f"{index + 1:x}" * 64,
            )
        )
    return AdmissionBatchIntent(
        batch_id=resolved_batch_id,
        tenant_id=tenant_id,
        owner_subject=owner_subject,
        idempotency_key=idempotency_key,
        payload_hash=payload_hash,
        objects=tuple(objects),
    )


def stored(item: AdmissionObjectIntent, *, version_id: str = "version-1") -> StoredObject:
    return StoredObject(ref=item.ref, etag="etag-1", version_id=version_id)


def start_write(
    repository: InMemoryImportAdmissionRepository,
    intent: AdmissionBatchIntent,
    item: AdmissionObjectIntent,
) -> None:
    current = repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject)
    object_state = next(candidate for candidate in current.objects if candidate.object_id == item.object_id)
    repository.mark_write_started(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=object_state.revision,
    )


def assert_conflict(call: object) -> None:
    assert isinstance(call, AppError)
    assert call.code is ErrorCode.RESOURCE_CONFLICT
    assert call.status_code == 409


def test_prepare_is_atomic_and_returns_deeply_immutable_snapshots() -> None:
    repository = InMemoryImportAdmissionRepository()
    intent = new_intent()

    snapshot = repository.prepare_batch(intent)

    assert snapshot.status is AdmissionStatus.PREPARING
    assert snapshot.revision == 0
    assert snapshot.objects[0].status is ObjectStatus.INTENT_RECORDED
    assert isinstance(snapshot.objects, tuple)
    with pytest.raises(FrozenInstanceError):
        snapshot.status = AdmissionStatus.ACCEPTED  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.objects[0].status = ObjectStatus.DELETED  # type: ignore[misc]


def test_prepare_is_idempotent_per_owner_and_rejects_a_different_payload() -> None:
    repository = InMemoryImportAdmissionRepository()
    intent = new_intent()
    first = repository.prepare_batch(intent)

    retry = repository.prepare_batch(replace(intent, batch_id=str(uuid.uuid4())))

    assert retry == first
    with pytest.raises(AppError) as captured:
        repository.prepare_batch(
            replace(intent, batch_id=str(uuid.uuid4()), payload_hash="e" * 64)
        )
    assert_conflict(captured.value)
    assert str(captured.value) == "Import admission request conflict"
    assert intent.idempotency_key not in str(captured.value)

    other_owner = new_intent(owner_subject="user-b", idempotency_key=intent.idempotency_key)
    other_tenant = new_intent(tenant_id="tenant-b", idempotency_key=intent.idempotency_key)
    assert repository.prepare_batch(other_owner).batch_id == other_owner.batch_id
    assert repository.prepare_batch(other_tenant).batch_id == other_tenant.batch_id


@pytest.mark.parametrize(
    ("tenant_id", "owner_subject"),
    [("tenant-b", "user-a"), ("tenant-a", "user-b")],
)
def test_get_owned_hides_cross_tenant_and_cross_owner_records(
    tenant_id: str,
    owner_subject: str,
) -> None:
    repository = InMemoryImportAdmissionRepository()
    intent = new_intent()
    repository.prepare_batch(intent)

    with pytest.raises(AppError) as captured:
        repository.get_owned(intent.batch_id, tenant_id, owner_subject)

    assert captured.value.code is ErrorCode.RESOURCE_NOT_FOUND
    assert str(captured.value) == "Import admission not found"


def test_record_stored_requires_the_exact_intended_ref_and_tenant() -> None:
    repository = InMemoryImportAdmissionRepository()
    intent = new_intent(object_count=1)
    item = intent.objects[0]
    repository.prepare_batch(intent)
    start_write(repository, intent, item)
    wrong_ref = ObjectRef(tenant_id="tenant-b", relative_key=item.ref.relative_key)

    with pytest.raises(AppError) as captured:
        repository.record_stored(
            intent.batch_id,
            item.object_id,
            intent.tenant_id,
            intent.owner_subject,
            StoredObject(ref=wrong_ref, etag="etag", version_id="wrong-version"),
        )

    assert captured.value.code is ErrorCode.RESOURCE_NOT_FOUND
    current = repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject)
    assert current.objects[0].status is ObjectStatus.WRITE_STARTED
    assert current.objects[0].stored_object is None

    result = repository.record_stored(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        stored(item),
        expected_revision=1,
    )
    assert result.status is ObjectStatus.WRITE_CONFIRMED
    assert result.stored_object == stored(item)
    assert repository.record_stored(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        stored(item),
        expected_revision=1,
    ) == result


def test_accept_requires_every_exact_write_confirmation_and_is_terminal() -> None:
    repository = InMemoryImportAdmissionRepository()
    intent = new_intent()
    repository.prepare_batch(intent)
    start_write(repository, intent, intent.objects[0])
    repository.record_stored(
        intent.batch_id,
        intent.objects[0].object_id,
        intent.tenant_id,
        intent.owner_subject,
        stored(intent.objects[0]),
    )

    with pytest.raises(AppError) as incomplete:
        repository.accept_batch(intent.batch_id, intent.tenant_id, intent.owner_subject)
    assert_conflict(incomplete.value)

    start_write(repository, intent, intent.objects[1])
    repository.record_stored(
        intent.batch_id,
        intent.objects[1].object_id,
        intent.tenant_id,
        intent.owner_subject,
        stored(intent.objects[1], version_id="version-2"),
    )
    before_accept = repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject)
    accepted = repository.accept_batch(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=before_accept.revision,
    )

    assert accepted.status is AdmissionStatus.ACCEPTED
    assert repository.accept_batch(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=before_accept.revision,
    ) == accepted
    with pytest.raises(AppError) as rollback:
        repository.request_rollback(intent.batch_id, intent.tenant_id, intent.owner_subject)
    assert_conflict(rollback.value)


def test_unknown_write_can_be_reconciled_then_accepted() -> None:
    repository = InMemoryImportAdmissionRepository()
    intent = new_intent(object_count=1)
    item = intent.objects[0]
    repository.prepare_batch(intent)
    start_write(repository, intent, item)

    unknown = repository.record_write_unknown(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=1,
    )

    assert unknown.status is ObjectStatus.WRITE_UNKNOWN
    assert repository.record_write_unknown(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=0,
    ) == unknown
    assert repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject).status is (
        AdmissionStatus.RECONCILIATION_REQUIRED
    )
    with pytest.raises(AppError):
        repository.accept_batch(intent.batch_id, intent.tenant_id, intent.owner_subject)

    confirmed = repository.record_stored(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        stored(item),
        expected_revision=unknown.revision,
    )
    assert confirmed.status is ObjectStatus.WRITE_CONFIRMED
    assert repository.accept_batch(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
    ).status is AdmissionStatus.ACCEPTED


def test_rollback_persists_before_exact_cleanup_and_keeps_the_manifest() -> None:
    repository = InMemoryImportAdmissionRepository()
    intent = new_intent(object_count=3)
    repository.prepare_batch(intent)
    start_write(repository, intent, intent.objects[0])
    repository.record_stored(
        intent.batch_id,
        intent.objects[0].object_id,
        intent.tenant_id,
        intent.owner_subject,
        stored(intent.objects[0]),
    )
    start_write(repository, intent, intent.objects[1])
    repository.record_write_unknown(
        intent.batch_id,
        intent.objects[1].object_id,
        intent.tenant_id,
        intent.owner_subject,
    )

    before_rollback = repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject)
    rollback = repository.request_rollback(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        reason_code="UPLOAD_FAILED",
        expected_revision=before_rollback.revision,
    )

    assert rollback.status is AdmissionStatus.RECONCILIATION_REQUIRED
    assert rollback.rollback_requested is True
    assert rollback.rollback_reason_code == "UPLOAD_FAILED"
    assert tuple(item.status for item in rollback.objects) == (
        ObjectStatus.DELETE_PENDING,
        ObjectStatus.WRITE_UNKNOWN,
        ObjectStatus.NO_OBJECT,
    )
    assert repository.request_rollback(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        reason_code="UPLOAD_FAILED",
        expected_revision=before_rollback.revision,
    ) == rollback

    leases = repository.claim_cleanup("cleanup-1", limit=3, lease_seconds=30)
    assert len(leases) == 2
    delete_lease = next(lease for lease in leases if lease.object_status is ObjectStatus.DELETE_PENDING)
    unknown_leases = tuple(lease for lease in leases if lease.object_status is ObjectStatus.WRITE_UNKNOWN)
    assert delete_lease.stored_object == stored(intent.objects[0])
    repository.mark_deleted(delete_lease)
    for unknown_lease in unknown_leases:
        repository.mark_no_object(unknown_lease)

    pending = repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject)
    assert pending.status is AdmissionStatus.ROLLED_BACK
    rolled_back = repository.finish_cleanup(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=pending.revision,
    )
    assert rolled_back.status is AdmissionStatus.ROLLED_BACK
    assert repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject) == rolled_back
    assert repository.finish_cleanup(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=pending.revision,
    ) == rolled_back


def test_unknown_object_found_during_rollback_becomes_exact_delete_pending() -> None:
    repository = InMemoryImportAdmissionRepository()
    intent = new_intent(object_count=1)
    item = intent.objects[0]
    repository.prepare_batch(intent)
    start_write(repository, intent, item)
    repository.record_write_unknown(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
    )
    repository.request_rollback(intent.batch_id, intent.tenant_id, intent.owner_subject)
    lease = repository.claim_cleanup("reconciler", limit=1, lease_seconds=30)[0]

    reconciled = repository.record_stored(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        stored(item, version_id="reconciled-version"),
        expected_revision=lease.object_revision,
        lease_token=lease.lease_token,
    )

    assert reconciled.status is ObjectStatus.DELETE_PENDING
    assert reconciled.stored_object == stored(item, version_id="reconciled-version")
    current = repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject)
    assert current.status is AdmissionStatus.ROLLBACK_PENDING


def test_cleanup_claim_is_thread_safe_and_release_requires_fresh_cas_lease() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    repository = InMemoryImportAdmissionRepository(clock=clock)
    intent = new_intent(object_count=1)
    item = intent.objects[0]
    repository.prepare_batch(intent)
    start_write(repository, intent, item)
    repository.record_stored(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        stored(item),
    )
    repository.request_rollback(intent.batch_id, intent.tenant_id, intent.owner_subject)

    def claim(index: int) -> tuple[CleanupLease, ...]:
        return repository.claim_cleanup(f"worker-{index}", limit=1, lease_seconds=30)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(claim, range(8)))
    claimed = [lease for result in results for lease in result]
    assert len(claimed) == 1

    lease = claimed[0]
    released = repository.release_cleanup(
        lease,
        error_code=ErrorCode.DEPENDENCY_UNAVAILABLE.value,
    )
    assert released.status is ObjectStatus.DELETE_PENDING
    assert released.last_error_code == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    with pytest.raises(AppError) as stale:
        repository.mark_deleted(lease)
    assert_conflict(stale.value)

    assert repository.claim_cleanup("too-soon", limit=1, lease_seconds=30) == ()
    clock.advance(1)
    replacement = repository.claim_cleanup("replacement", limit=1, lease_seconds=30)[0]
    assert replacement.lease_token != lease.lease_token
    assert repository.mark_deleted(replacement).status is ObjectStatus.DELETED


def test_expired_or_tampered_cleanup_lease_cannot_mutate_state() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    repository = InMemoryImportAdmissionRepository(clock=clock)
    intent = new_intent(object_count=1)
    item = intent.objects[0]
    repository.prepare_batch(intent)
    start_write(repository, intent, item)
    repository.record_stored(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        stored(item),
    )
    repository.request_rollback(intent.batch_id, intent.tenant_id, intent.owner_subject)
    lease = repository.claim_cleanup("cleanup", limit=1, lease_seconds=10)[0]

    tampered = replace(lease, expected_ref=build_object_ref(intent.tenant_id, "unrelated", "object"))
    with pytest.raises(AppError) as rejected:
        repository.mark_deleted(tampered)
    assert_conflict(rejected.value)

    clock.advance(10)
    with pytest.raises(AppError) as expired:
        repository.mark_deleted(lease)
    assert_conflict(expired.value)

    replacement = repository.claim_cleanup("replacement", limit=1, lease_seconds=10)[0]
    assert replacement.object_revision > lease.object_revision
    assert repository.mark_deleted(replacement).status is ObjectStatus.DELETED


def test_finish_cleanup_rejects_pending_work_and_terminal_state_cannot_regress() -> None:
    repository = InMemoryImportAdmissionRepository()
    intent = new_intent(object_count=1)
    repository.prepare_batch(intent)
    start_write(repository, intent, intent.objects[0])
    repository.record_stored(
        intent.batch_id,
        intent.objects[0].object_id,
        intent.tenant_id,
        intent.owner_subject,
        stored(intent.objects[0]),
    )
    repository.request_rollback(intent.batch_id, intent.tenant_id, intent.owner_subject)

    with pytest.raises(AppError) as pending:
        repository.finish_cleanup(intent.batch_id, intent.tenant_id, intent.owner_subject)
    assert_conflict(pending.value)

    lease = repository.claim_cleanup("cleanup", limit=1, lease_seconds=30)[0]
    repository.mark_deleted(lease)
    rolled_back = repository.finish_cleanup(intent.batch_id, intent.tenant_id, intent.owner_subject)

    with pytest.raises(AppError) as accept:
        repository.accept_batch(intent.batch_id, intent.tenant_id, intent.owner_subject)
    assert_conflict(accept.value)
    with pytest.raises(AppError) as write:
        repository.record_stored(
            intent.batch_id,
            intent.objects[0].object_id,
            intent.tenant_id,
            intent.owner_subject,
            stored(intent.objects[0]),
        )
    assert_conflict(write.value)
    assert repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject) == rolled_back


def test_probe_and_close_are_idempotent_and_use_a_sanitized_dependency_error() -> None:
    repository = InMemoryImportAdmissionRepository()
    repository.probe()

    repository.close()
    repository.close()

    with pytest.raises(DependencyError) as captured:
        repository.probe()
    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert str(captured.value) == "Import admission repository is unavailable"


def test_stale_preparing_batch_is_recovered_into_persistent_cleanup_state() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    repository = InMemoryImportAdmissionRepository(clock=clock)
    intent = new_intent(object_count=2)
    repository.prepare_batch(intent)
    start_write(repository, intent, intent.objects[0])
    clock.advance(120)

    recovered = repository.recover_stale_preparing(
        clock.value - timedelta(seconds=60),
        limit=10,
    )

    assert len(recovered) == 1
    assert recovered[0].rollback_requested is True
    assert recovered[0].rollback_reason_code == "ADMISSION_RECOVERY"
    assert recovered[0].status is AdmissionStatus.RECONCILIATION_REQUIRED
    assert tuple(item.status for item in recovered[0].objects) == (
        ObjectStatus.WRITE_UNKNOWN,
        ObjectStatus.NO_OBJECT,
    )
    assert repository.recover_stale_preparing(clock.value, limit=10) == ()


def test_active_write_lease_heartbeat_blocks_stale_recovery_and_uses_fenced_cas() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    repository = InMemoryImportAdmissionRepository(clock=clock)
    intent = new_intent(object_count=1)
    prepared = repository.prepare_batch(intent)
    item = prepared.objects[0]
    started = repository.mark_write_started(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=item.revision,
        writer_id="request-writer-1",
        lease_seconds=10,
    )
    assert started.write_lease_token is not None
    assert started.write_lease_expires_at == clock.value + timedelta(seconds=10)

    clock.advance(6)
    renewed = repository.renew_write_lease(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        lease_token=started.write_lease_token,
        expected_revision=started.revision,
        lease_seconds=10,
    )
    assert renewed.write_lease_token == started.write_lease_token
    assert renewed.write_lease_expires_at == clock.value + timedelta(seconds=10)
    assert repository.recover_stale_preparing(clock.value + timedelta(days=1), limit=10) == ()

    with pytest.raises(AppError) as stale_heartbeat:
        repository.renew_write_lease(
            intent.batch_id,
            item.object_id,
            intent.tenant_id,
            intent.owner_subject,
            lease_token=started.write_lease_token,
            expected_revision=started.revision,
            lease_seconds=10,
        )
    assert_conflict(stale_heartbeat.value)
    with pytest.raises(AppError) as wrong_tenant:
        repository.renew_write_lease(
            intent.batch_id,
            item.object_id,
            "tenant-b",
            intent.owner_subject,
            lease_token=started.write_lease_token,
            expected_revision=renewed.revision,
            lease_seconds=10,
        )
    assert wrong_tenant.value.code is ErrorCode.RESOURCE_NOT_FOUND


def test_expired_crashed_write_lease_recovers_but_unknown_result_keeps_barrier() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    repository = InMemoryImportAdmissionRepository(clock=clock)
    intent = new_intent(object_count=1)
    prepared = repository.prepare_batch(intent)
    item = prepared.objects[0]
    started = repository.mark_write_started(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=item.revision,
        writer_id="request-writer-1",
        lease_seconds=10,
    )
    assert started.write_lease_token is not None
    unknown = repository.record_write_unknown(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=started.revision,
        lease_token=started.write_lease_token,
    )
    current = repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject)
    rolled_back = repository.request_rollback(
        intent.batch_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=current.revision,
    )

    assert unknown.write_lease_token == started.write_lease_token
    assert rolled_back.status is AdmissionStatus.RECONCILIATION_REQUIRED
    assert repository.claim_cleanup("too-early", limit=1, lease_seconds=30) == ()

    clock.advance(11)
    lease = repository.claim_cleanup("after-expiry", limit=1, lease_seconds=30)[0]
    tombstone = repository.mark_no_object(lease)
    terminal = repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject)
    assert tombstone.status is ObjectStatus.NO_OBJECT
    assert terminal.status is AdmissionStatus.ROLLED_BACK


def test_crashed_write_is_not_recovered_until_its_lease_expires() -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    repository = InMemoryImportAdmissionRepository(clock=clock)
    intent = new_intent(object_count=1)
    prepared = repository.prepare_batch(intent)
    item = prepared.objects[0]
    repository.mark_write_started(
        intent.batch_id,
        item.object_id,
        intent.tenant_id,
        intent.owner_subject,
        expected_revision=item.revision,
        writer_id="crashed-request",
        lease_seconds=10,
    )

    assert repository.recover_stale_preparing(clock.value + timedelta(days=1), limit=10) == ()
    clock.advance(11)
    recovered = repository.recover_stale_preparing(clock.value, limit=10)

    assert len(recovered) == 1
    assert recovered[0].status is AdmissionStatus.RECONCILIATION_REQUIRED
    assert recovered[0].objects[0].status is ObjectStatus.WRITE_UNKNOWN


@pytest.mark.parametrize(
    "change",
    [
        {"batch_id": "not-a-uuid"},
        {"owner_subject": ""},
        {"idempotency_key": "bad\nkey"},
        {"payload_hash": "not-a-digest"},
        {"objects": ()},
    ],
)
def test_invalid_batch_intents_are_rejected_before_mutation(change: dict[str, object]) -> None:
    repository = InMemoryImportAdmissionRepository()
    intent = new_intent()

    with pytest.raises((AppError, ValueError)):
        repository.prepare_batch(replace(intent, **change))  # type: ignore[arg-type]

    with pytest.raises(AppError) as missing:
        repository.get_owned(intent.batch_id, intent.tenant_id, intent.owner_subject)
    assert missing.value.code is ErrorCode.RESOURCE_NOT_FOUND
