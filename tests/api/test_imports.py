from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from threading import Event, Timer
from typing import Any, BinaryIO, cast

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.core.errors import DependencyError, ErrorCode
from app.core.security import Principal, get_current_principal
from app.core.settings import (
    ApiSettings,
    AppEnvironment,
    AuthSettings,
    ImportRuntimeSettings,
    LoggingSettings,
    MinioSettings,
    Settings,
)
from app.infrastructure.minio_store import MinioStore, ObjectRef, ObservedObject, StoredObject
from app.main import create_app
from app.services.import_admission import AdmissionStatus, ObjectStatus
from app.services.import_coordinator import InMemoryImportCoordinator
from app.services.import_registry import InMemoryImportTaskRegistry
from app.services.import_runner import (
    ImportCommitEvent,
    ImportCommitReceipt,
    ImportJob,
    ImportStepEvent,
)


class FakeStore:
    def __init__(
        self,
        *,
        ready: bool = True,
        fail_on_put: int | None = None,
        fail_delete: bool = False,
    ) -> None:
        self.ready = ready
        self.fail_on_put = fail_on_put
        self.fail_delete = fail_delete
        self.before_put: Callable[[ObjectRef], None] | None = None
        self.put_count = 0
        self.put_calls: list[tuple[ObjectRef, dict[str, str] | None]] = []
        self.delete_calls: list[tuple[str, str | None]] = []
        self.objects: dict[str, tuple[bytes, StoredObject]] = {}
        self.closed = False

    def require_ready(self) -> None:
        if not self.ready:
            raise DependencyError("sensitive-storage-detail")

    def close(self) -> None:
        self.closed = True

    def put_stream(
        self,
        tenant_id: str,
        ref: ObjectRef,
        data: BinaryIO,
        *,
        length: int,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> StoredObject:
        del content_type
        assert tenant_id == ref.tenant_id
        if self.before_put is not None:
            self.before_put(ref)
        self.put_count += 1
        if self.fail_on_put == self.put_count:
            raise DependencyError("https://secret.invalid/?token=must-not-leak")
        chunks: list[bytes] = []
        while chunk := data.read(1024):
            chunks.append(chunk)
        body = b"".join(chunks)
        assert len(body) == length
        stored = StoredObject(ref=ref, etag=f"etag-{self.put_count}", version_id=f"v-{self.put_count}")
        self.put_calls.append((ref, dict(metadata) if metadata is not None else None))
        self.objects[ref.uri] = (body, stored)
        return stored

    def delete(self, tenant_id: str, ref: ObjectRef | str) -> None:
        resolved = ref if isinstance(ref, ObjectRef) else next(
            stored.ref for _, stored in self.objects.values() if stored.ref.uri == ref
        )
        assert resolved.tenant_id == tenant_id
        self.delete_calls.append((resolved.uri, None))
        self.objects.pop(resolved.uri, None)

    def delete_stored(self, tenant_id: str, stored: StoredObject) -> None:
        assert stored.ref.tenant_id == tenant_id
        if self.fail_delete:
            raise DependencyError("delete-secret")
        self.delete_calls.append((stored.ref.uri, stored.version_id))
        current = self.objects.get(stored.ref.uri)
        if current is not None and current[1].version_id == stored.version_id:
            del self.objects[stored.ref.uri]

    def inspect_object(self, tenant_id: str, ref: ObjectRef | str) -> ObservedObject | None:
        resolved = ref if isinstance(ref, ObjectRef) else next(
            (stored.ref for _, stored in self.objects.values() if stored.ref.uri == ref),
            None,
        )
        if resolved is None:
            return None
        assert resolved.tenant_id == tenant_id
        current = self.objects.get(resolved.uri)
        if current is None:
            return None
        body, stored = current
        metadata = next(
            (metadata for called_ref, metadata in self.put_calls if called_ref == resolved),
            None,
        )
        return ObservedObject(stored=stored, size=len(body), metadata=metadata or {})

    @contextmanager
    def open_stored(self, tenant_id: str, stored: StoredObject):
        assert tenant_id == stored.ref.tenant_id
        current = self.objects.get(stored.ref.uri)
        if current is None or current[1].version_id != stored.version_id:
            raise DependencyError("object-secret")
        yield BytesIO(current[0])


class SlowStore(FakeStore):
    def __init__(self) -> None:
        super().__init__()
        self.put_started = Event()
        self.allow_put_to_finish = Event()

    def put_stream(
        self,
        tenant_id: str,
        ref: ObjectRef,
        data: BinaryIO,
        *,
        length: int,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> StoredObject:
        self.put_started.set()
        if not self.allow_put_to_finish.wait(timeout=10):
            raise TimeoutError("test upload was not released")
        return super().put_stream(
            tenant_id,
            ref,
            data,
            length=length,
            content_type=content_type,
            metadata=metadata,
        )


class Processor:
    idempotent_commits = False

    def __init__(self, *, failure: Exception | None = None, prove_commit: bool = True) -> None:
        self.failure = failure
        self.prove_commit = prove_commit
        self.healthy = True
        self.jobs: list[ImportJob] = []

    def probe(self) -> None:
        if not self.healthy:
            raise DependencyError("processor-endpoint-secret")

    def stream(self, job: ImportJob) -> Iterable[ImportStepEvent]:
        self.jobs.append(job)
        if self.failure is not None:
            raise self.failure
        yield ImportStepEvent("parse_document")
        if self.prove_commit:
            yield ImportCommitEvent(
                ImportCommitReceipt(
                    task_id=job.task_id,
                    document_id=job.document_id,
                    tenant_id=job.tenant_id,
                    imported_chunk_count=2,
                    commit_token=f"milvus-{job.task_id}",
                )
            )


class BlockingProcessor(Processor):
    def __init__(self, *, block_close: bool = False) -> None:
        super().__init__()
        self.stream_started = Event()
        self.release_stream = Event()
        self.block_close = block_close
        self.close_started = Event()
        self.release_close = Event()

    def stream(self, job: ImportJob) -> Iterable[ImportStepEvent]:
        self.jobs.append(job)
        self.stream_started.set()
        if not self.release_stream.wait(timeout=5):
            raise TimeoutError("test processor stream was not released")
        yield ImportStepEvent("parse_document")
        yield ImportCommitEvent(
            ImportCommitReceipt(
                task_id=job.task_id,
                document_id=job.document_id,
                tenant_id=job.tenant_id,
                imported_chunk_count=2,
                commit_token=f"milvus-{job.task_id}",
            )
        )

    def close(self) -> None:
        self.close_started.set()
        if self.block_close and not self.release_close.wait(timeout=5):
            raise TimeoutError("test processor close was not released")


class FailingRegistry(InMemoryImportTaskRegistry):
    def create_many(self, tasks):
        del tasks
        raise RuntimeError("POSTGRES_DSN=must-not-leak")


def make_settings(
    tmp_path: Path,
    *,
    max_files: int = 5,
    max_bytes: int = 1024 * 1024,
    max_batch_bytes: int = 500 * 1024 * 1024,
    environment: AppEnvironment = AppEnvironment.TEST,
    require_idempotency_key: bool = False,
    worker_enabled: bool = False,
    repository: str = "memory",
    lease_seconds: int = 120,
    heartbeat_seconds: float = 30.0,
    shutdown_grace_seconds: float = 30.0,
) -> Settings:
    return Settings.model_construct(
        api=ApiSettings.model_construct(
            environment=environment,
            import_enabled=True,
            project_root=tmp_path,
            import_work_root=Path("import-work"),
            cors_origins_raw="https://console.example.com",
            max_upload_files=max_files,
            max_upload_bytes=max_bytes,
            max_upload_batch_bytes=max_batch_bytes,
        ),
        import_runtime=ImportRuntimeSettings.model_construct(
            repository=repository,
            require_idempotency_key=require_idempotency_key,
            worker_enabled=worker_enabled,
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            shutdown_grace_seconds=shutdown_grace_seconds,
            write_lease_seconds=lease_seconds,
        ),
        auth=AuthSettings.model_construct(
            mode="disabled",
            issuer=None,
            audience=None,
            jwks_url=None,
            public_key=None,
            algorithms_raw="RS256",
            dev_tenant_id="tenant-a",
        ),
        logging=LoggingSettings.model_construct(
            console_enabled=False,
            console_level="INFO",
            file_enabled=False,
            file_level="INFO",
            retention="7 days",
            json_output=False,
            diagnose=False,
        ),
        minio=MinioSettings.model_construct(
            pdf_prefix="documents",
            image_prefix="images",
            readiness_interval_seconds=10.0,
            readiness_probe_timeout_seconds=0.2,
            readiness_ttl_seconds=30.0,
        ),
    )


def build_test_app(
    tmp_path: Path,
    *,
    store: FakeStore | None = None,
    processor: Processor | None = None,
    registry: InMemoryImportTaskRegistry | None = None,
    include_processor: bool = True,
    max_files: int = 5,
    max_bytes: int = 1024 * 1024,
    max_batch_bytes: int = 500 * 1024 * 1024,
    require_idempotency_key: bool = False,
    worker_enabled: bool = False,
    repository: str = "memory",
    coordinator: InMemoryImportCoordinator | None = None,
    lease_seconds: int = 120,
    heartbeat_seconds: float = 30.0,
    shutdown_grace_seconds: float = 30.0,
):
    resolved_store = store or FakeStore()
    resolved_processor = processor or Processor()
    resolved_registry = registry or InMemoryImportTaskRegistry()
    if coordinator is not None:
        resolved_registry = cast(InMemoryImportTaskRegistry, coordinator.tasks)
    app = create_app(
        settings=make_settings(
            tmp_path,
            max_files=max_files,
            max_bytes=max_bytes,
            max_batch_bytes=max_batch_bytes,
            require_idempotency_key=require_idempotency_key,
            worker_enabled=worker_enabled,
            repository=repository,
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            shutdown_grace_seconds=shutdown_grace_seconds,
        ),
        minio_builder=lambda _: cast(MinioStore, resolved_store),
        import_registry_factory=lambda: resolved_registry,
        import_coordinator_factory=(lambda: coordinator) if coordinator is not None else None,
        import_processor_builder=(lambda _: resolved_processor) if include_processor else None,
    )
    return app, resolved_store, resolved_processor, resolved_registry


def test_mongo_repository_is_built_during_lifespan_from_runtime_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mongo_store = object()
    store_builds: list[Settings] = []
    coordinator_builds: list[tuple[object, float, float, int]] = []

    def build_mongo(settings: Settings) -> object:
        store_builds.append(settings)
        return mongo_store

    def build_coordinator(
        store: object,
        *,
        cleanup_retry_base_seconds: float,
        cleanup_retry_max_seconds: float,
        terminal_retention_days: int,
    ) -> InMemoryImportCoordinator:
        coordinator_builds.append(
            (
                store,
                cleanup_retry_base_seconds,
                cleanup_retry_max_seconds,
                terminal_retention_days,
            )
        )
        return InMemoryImportCoordinator()

    monkeypatch.setattr(main_module, "build_mongo_import_store", build_mongo)
    monkeypatch.setattr(main_module, "MongoImportCoordinator", build_coordinator)
    settings = make_settings(tmp_path, repository="mongo")
    app = create_app(
        settings=settings,
        minio_builder=lambda _: cast(MinioStore, FakeStore()),
        import_processor_builder=lambda _: Processor(),
    )

    assert store_builds == []
    assert coordinator_builds == []
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200

    assert store_builds == [settings]
    assert coordinator_builds == [
        (
            mongo_store,
            settings.import_runtime.retry_base_seconds,
            settings.import_runtime.retry_max_seconds,
            settings.mongo.terminal_retention_days,
        )
    ]


def markdown_file(name: str = "guide.md", body: bytes = b"# Guide\n") -> dict[str, Any]:
    return {"files": (name, body, "text/markdown")}


def test_create_import_is_tenant_owned_private_and_eventually_completed(tmp_path: Path) -> None:
    app, store, processor, _ = build_test_app(tmp_path)

    with TestClient(app) as client:
        created = client.post("/api/v1/imports", files=markdown_file("../../private-guide.md"))
        assert created.status_code == 202
        payload = created.json()
        assert len(payload["tasks"]) == 1
        accepted = payload["tasks"][0]
        assert accepted["status"] == "queued"
        assert accepted["document"]["original_name"] == "private-guide.md"
        assert "kb-object" not in created.text
        assert "knowledge-private" not in created.text
        assert "import-work" not in created.text

        task_id = accepted["task_id"]
        status_response = client.get(f"/api/v1/imports/{task_id}")

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"
    assert status_response.json()["imported_chunk_count"] == 2
    assert len(processor.jobs) == 1
    assert processor.jobs[0].tenant_id == "tenant-a"
    assert processor.jobs[0].owner_subject == "local-dev"
    assert len(store.objects) == 1
    object_ref, metadata = store.put_calls[0]
    assert "private-guide" not in object_ref.relative_key
    assert object_ref.relative_key.endswith("/source.md")
    assert metadata == {
        "schema-version": "1",
        "document-id": uuid.UUID(accepted["document"]["document_id"]).hex,
        "import-id": uuid.UUID(task_id).hex,
        "sha256": accepted["document"]["sha256"],
        "object-kind": "source",
    }
    assert not any(path.is_file() for path in (tmp_path / "import-work").rglob("*"))


def test_manifest_and_write_started_are_persisted_before_minio_put(tmp_path: Path) -> None:
    store = FakeStore()
    app, _, _, _ = build_test_app(tmp_path, store=store)
    idempotency_key = "manifest-before-put"

    def assert_manifest_precedes_put(ref: ObjectRef) -> None:
        manifest = app.state.import_admissions.get_by_idempotency(
            "tenant-a",
            "local-dev",
            idempotency_key,
        )
        assert manifest.status is AdmissionStatus.PREPARING
        item = next(candidate for candidate in manifest.objects if candidate.ref == ref)
        assert item.status is ObjectStatus.WRITE_STARTED

    store.before_put = assert_manifest_precedes_put
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/imports",
            files=markdown_file(),
            headers={"Idempotency-Key": idempotency_key},
        )

    assert response.status_code == 202
    assert store.put_count == 1


def test_same_idempotency_key_and_payload_returns_original_tasks(tmp_path: Path) -> None:
    app, store, processor, _ = build_test_app(tmp_path)
    headers = {"Idempotency-Key": "retry-the-same-request"}

    with TestClient(app) as client:
        first = client.post("/api/v1/imports", files=markdown_file(), headers=headers)
        second = client.post("/api/v1/imports", files=markdown_file(), headers=headers)

    assert first.status_code == second.status_code == 202
    assert first.json()["tasks"][0]["task_id"] == second.json()["tasks"][0]["task_id"]
    assert first.json()["tasks"][0]["document"]["document_id"] == (
        second.json()["tasks"][0]["document"]["document_id"]
    )
    assert store.put_count == 1
    assert len(processor.jobs) == 1


def test_durable_coordinator_uses_worker_instead_of_response_background_task(
    tmp_path: Path,
) -> None:
    coordinator = InMemoryImportCoordinator()
    coordinator.durable = True
    app, store, processor, _ = build_test_app(
        tmp_path,
        coordinator=coordinator,
        worker_enabled=True,
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/imports",
            files=markdown_file(),
            headers={"Idempotency-Key": "durable-worker-delivery"},
        )
        task_id = created.json()["tasks"][0]["task_id"]
        deadline = time.monotonic() + 3
        current = client.get(f"/api/v1/imports/{task_id}")
        while current.json()["status"] != "completed" and time.monotonic() < deadline:
            time.sleep(0.05)
            current = client.get(f"/api/v1/imports/{task_id}")
        ready = client.get("/health/ready")

    assert created.status_code == 202
    assert current.json()["status"] == "completed"
    assert ready.status_code == 200
    assert ready.json()["checks"]["import_worker"] == "ok"
    assert len(processor.jobs) == 1
    assert len(store.objects) == 1
    assert not any(path.is_file() for path in (tmp_path / "import-work").rglob("*"))


def test_shutdown_grace_allows_active_import_to_finish(tmp_path: Path) -> None:
    coordinator = InMemoryImportCoordinator()
    coordinator.durable = True
    processor = BlockingProcessor()
    app, _, _, _ = build_test_app(
        tmp_path,
        coordinator=coordinator,
        processor=processor,
        worker_enabled=True,
        shutdown_grace_seconds=0.5,
    )
    release_timer = Timer(0.05, processor.release_stream.set)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/imports",
            files=markdown_file(),
            headers={"Idempotency-Key": "graceful-shutdown-active-import"},
        )
        task_id = created.json()["tasks"][0]["task_id"]
        assert processor.stream_started.wait(timeout=1)
        release_timer.start()

    release_timer.join(timeout=1)
    snapshot = coordinator.tasks.get_owned(task_id, "tenant-a", "local-dev")
    assert created.status_code == 202
    assert snapshot.status.value == "completed"
    assert app.state.import_shutting_down is True
    assert app.state.import_ready is False
    assert processor.close_started.is_set()


def test_shutdown_timeout_abandons_hanging_processor_and_close(tmp_path: Path) -> None:
    coordinator = InMemoryImportCoordinator()
    coordinator.durable = True
    processor = BlockingProcessor(block_close=True)
    app, _, _, _ = build_test_app(
        tmp_path,
        coordinator=coordinator,
        processor=processor,
        worker_enabled=True,
        shutdown_grace_seconds=0.1,
    )
    client = TestClient(app)
    client.__enter__()
    executor = ThreadPoolExecutor(max_workers=1)
    shutdown_future = None
    try:
        created = client.post(
            "/api/v1/imports",
            files=markdown_file(),
            headers={"Idempotency-Key": "hanging-processor-shutdown"},
        )
        assert created.status_code == 202
        assert processor.stream_started.wait(timeout=1)

        started_at = time.monotonic()
        shutdown_future = executor.submit(client.__exit__, None, None, None)
        deadline = time.monotonic() + 1
        while not app.state.import_shutting_down and time.monotonic() < deadline:
            time.sleep(0.01)
        assert app.state.import_shutting_down is True
        assert app.state.import_ready is False
        assert app.state.import_ready_until == 0.0
        shutdown_future.result(timeout=1)
        assert time.monotonic() - started_at < 0.8
        assert processor.close_started.is_set()
    finally:
        processor.release_stream.set()
        processor.release_close.set()
        if shutdown_future is None:
            client.__exit__(None, None, None)
        executor.shutdown(wait=True)


def test_same_idempotency_key_with_different_payload_is_conflict(tmp_path: Path) -> None:
    app, store, processor, _ = build_test_app(tmp_path)
    headers = {"Idempotency-Key": "request-key-reuse"}

    with TestClient(app) as client:
        first = client.post(
            "/api/v1/imports",
            files=markdown_file(body=b"first"),
            headers=headers,
        )
        second = client.post(
            "/api/v1/imports",
            files=markdown_file(body=b"second"),
            headers=headers,
        )

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["code"] == ErrorCode.RESOURCE_CONFLICT.value
    assert store.put_count == 1
    assert len(processor.jobs) == 1


def test_required_idempotency_key_fails_before_workspace_and_storage(tmp_path: Path) -> None:
    app, store, processor, _ = build_test_app(tmp_path, require_idempotency_key=True)

    with TestClient(app) as client:
        response = client.post("/api/v1/imports", files=markdown_file())

    assert response.status_code == 422
    assert response.json()["code"] == ErrorCode.REQUEST_INVALID.value
    assert store.put_count == 0
    assert processor.jobs == []
    assert not any(path.is_file() for path in (tmp_path / "import-work").rglob("*"))


def test_cleanup_failure_keeps_manifest_and_exact_version_for_retry(tmp_path: Path) -> None:
    store = FakeStore(fail_delete=True)
    app, _, processor, _ = build_test_app(
        tmp_path,
        store=store,
        registry=FailingRegistry(),
    )
    key = "cleanup-retry-evidence"

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/imports",
            files=markdown_file(),
            headers={"Idempotency-Key": key},
        )
        manifest = app.state.import_admissions.get_by_idempotency(
            "tenant-a",
            "local-dev",
            key,
        )
    assert response.status_code == 500
    assert manifest.status is AdmissionStatus.ROLLBACK_PENDING
    assert manifest.objects[0].status is ObjectStatus.DELETE_PENDING
    assert manifest.objects[0].stored_object is not None
    assert manifest.objects[0].stored_object.version_id == "v-1"
    assert manifest.objects[0].cleanup_attempts == 1
    assert store.objects
    assert processor.jobs == []


def test_unknown_put_result_retains_reconciliation_evidence_until_write_lease_expires(
    tmp_path: Path,
) -> None:
    store = FakeStore(fail_on_put=1)
    app, _, processor, _ = build_test_app(
        tmp_path,
        store=store,
        lease_seconds=1,
        heartbeat_seconds=0.1,
    )
    key = "unknown-write-tombstone"

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/imports",
            files=markdown_file(),
            headers={"Idempotency-Key": key},
        )
        manifest = app.state.import_admissions.get_by_idempotency(
            "tenant-a",
            "local-dev",
            key,
        )
        premature_leases = app.state.import_admissions.claim_cleanup(
            "premature-cleanup",
            limit=1,
            lease_seconds=60,
        )

    assert response.status_code == 503
    assert manifest.status is AdmissionStatus.RECONCILIATION_REQUIRED
    assert manifest.objects[0].status is ObjectStatus.WRITE_UNKNOWN
    assert manifest.objects[0].write_lease_token is not None
    assert manifest.objects[0].write_lease_expires_at is not None
    assert premature_leases == ()
    assert store.objects == {}
    assert processor.jobs == []


def test_slow_put_heartbeat_fences_concurrent_stale_recovery(tmp_path: Path) -> None:
    store = SlowStore()
    app, _, processor, _ = build_test_app(
        tmp_path,
        store=store,
        lease_seconds=1,
        heartbeat_seconds=0.1,
    )
    key = "slow-put-write-fence"

    with TestClient(app, raise_server_exceptions=False) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            response_future = pool.submit(
                client.post,
                "/api/v1/imports",
                files=markdown_file(),
                headers={"Idempotency-Key": key},
            )
            assert store.put_started.wait(timeout=5)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                recovered = app.state.import_admissions.recover_stale_preparing(
                    datetime.now(UTC) + timedelta(days=1),
                    limit=10,
                )
                assert recovered == ()
                time.sleep(0.05)
            manifest_during_put = app.state.import_admissions.get_by_idempotency(
                "tenant-a",
                "local-dev",
                key,
            )
            assert manifest_during_put.objects[0].status is ObjectStatus.WRITE_STARTED
            assert manifest_during_put.objects[0].write_lease_expires_at is not None
            assert manifest_during_put.objects[0].revision > 1
            store.allow_put_to_finish.set()
            response = response_future.result(timeout=5)

        completed = app.state.import_admissions.get_by_idempotency(
            "tenant-a",
            "local-dev",
            key,
        )

    assert response.status_code == 202
    assert completed.status is AdmissionStatus.ACCEPTED
    assert completed.objects[0].status is ObjectStatus.WRITE_CONFIRMED
    assert completed.objects[0].write_lease_token is None
    assert store.delete_calls == []
    assert len(processor.jobs) == 1


def test_reader_cannot_create_import(tmp_path: Path) -> None:
    app, store, processor, _ = build_test_app(tmp_path)
    app.dependency_overrides[get_current_principal] = lambda: Principal(
        "reader-a", "tenant-a", frozenset({"reader"})
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/imports", files=markdown_file())

    assert response.status_code == 403
    assert response.json()["code"] == ErrorCode.PERMISSION_DENIED.value
    assert store.put_count == 0
    assert processor.jobs == []


def test_missing_processor_fails_before_admission(tmp_path: Path) -> None:
    app, store, _, _ = build_test_app(tmp_path, include_processor=False)

    with pytest.raises(DependencyError, match="Import runtime is not configured"):
        with TestClient(app):
            pass

    assert store.put_count == 0


def test_omitted_processor_builder_uses_enterprise_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    processor = Processor()
    registry = InMemoryImportTaskRegistry()
    builds: list[Settings] = []

    def build_default(settings: Settings) -> Processor:
        builds.append(settings)
        return processor

    monkeypatch.setattr(
        main_module,
        "build_enterprise_import_processor",
        build_default,
    )
    settings = make_settings(tmp_path)
    app = create_app(
        settings=settings,
        minio_builder=lambda _: cast(MinioStore, store),
        import_registry_factory=lambda: registry,
    )

    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200

    assert builds == [settings]


def test_unready_object_store_fails_before_admission(tmp_path: Path) -> None:
    app, store, processor, _ = build_test_app(tmp_path, store=FakeStore(ready=False))

    with TestClient(app) as client:
        response = client.post("/api/v1/imports", files=markdown_file())

    assert response.status_code == 503
    assert response.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert "sensitive-storage-detail" not in response.text
    assert store.put_count == 0
    assert processor.jobs == []


def test_import_runtime_failure_degrades_readiness_without_leaking_details(tmp_path: Path) -> None:
    app, _, processor, _ = build_test_app(tmp_path)

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["checks"] == {
            "minio": "ok",
            "import_registry": "ok",
            "import_processor": "ok",
            "import_workspace": "ok",
        }
        processor.healthy = False
        time.sleep(1.2)
        degraded = client.get("/health/ready")
        rejected = client.post("/api/v1/imports", files=markdown_file())

    assert degraded.status_code == 503
    assert degraded.json()["checks"]["import_processor"] == "unavailable"
    assert "processor-endpoint-secret" not in degraded.text
    assert rejected.status_code == 503
    assert rejected.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value


def test_same_tenant_other_user_and_other_tenant_cannot_read_task(tmp_path: Path) -> None:
    app, _, _, _ = build_test_app(tmp_path)
    app.dependency_overrides[get_current_principal] = lambda: Principal(
        "user-a", "tenant-a", frozenset({"editor"})
    )

    with TestClient(app) as client:
        created = client.post("/api/v1/imports", files=markdown_file())
        task_id = created.json()["tasks"][0]["task_id"]

        for principal in (
            Principal("user-b", "tenant-a", frozenset({"reader"})),
            Principal("user-a", "tenant-b", frozenset({"reader"})),
        ):
            app.dependency_overrides[get_current_principal] = lambda principal=principal: principal
            hidden = client.get(f"/api/v1/imports/{task_id}")
            assert hidden.status_code == 404
            assert hidden.json()["code"] == ErrorCode.RESOURCE_NOT_FOUND.value


def test_unknown_task_does_not_create_registry_state(tmp_path: Path) -> None:
    app, _, _, _ = build_test_app(tmp_path)

    with TestClient(app) as client:
        first = client.get(f"/api/v1/imports/{uuid.uuid4()}")
        second = client.get(f"/api/v1/imports/{uuid.uuid4()}")

    assert first.status_code == second.status_code == 404
    assert first.json()["code"] == second.json()["code"] == ErrorCode.RESOURCE_NOT_FOUND.value


def test_invalid_document_is_rejected_and_workspace_is_cleaned(tmp_path: Path) -> None:
    app, store, processor, _ = build_test_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/imports",
            files={"files": ("fake.pdf", b"not-a-pdf", "application/pdf")},
        )

    assert response.status_code == 415
    assert response.json()["code"] == ErrorCode.DOCUMENT_TYPE_UNSUPPORTED.value
    assert store.put_count == 0
    assert processor.jobs == []
    assert not any(path.is_file() for path in (tmp_path / "import-work").rglob("*"))


def test_second_storage_failure_rolls_back_exact_first_version(tmp_path: Path) -> None:
    store = FakeStore(fail_on_put=2)
    app, _, processor, registry = build_test_app(tmp_path, store=store)
    files = [
        ("files", ("one.md", b"one", "text/markdown")),
        ("files", ("two.md", b"two", "text/markdown")),
    ]

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/imports", files=files)

    assert response.status_code == 503
    assert response.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert "must-not-leak" not in response.text
    assert store.objects == {}
    assert any(version == "v-1" for _, version in store.delete_calls)
    assert processor.jobs == []
    assert registry._records == {}
    assert not any(path.is_file() for path in (tmp_path / "import-work").rglob("*"))


def test_registry_failure_rolls_back_persisted_source(tmp_path: Path) -> None:
    store = FakeStore()
    app, _, processor, _ = build_test_app(
        tmp_path,
        store=store,
        registry=FailingRegistry(),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/imports", files=markdown_file())

    assert response.status_code == 500
    assert response.json()["code"] == ErrorCode.INTERNAL_ERROR.value
    assert "POSTGRES_DSN" not in response.text
    assert store.objects == {}
    assert processor.jobs == []


def test_processor_failure_is_async_sanitized_and_keeps_private_source(tmp_path: Path) -> None:
    processor = Processor(failure=RuntimeError("OPENAI_API_KEY=must-not-leak"))
    app, store, _, _ = build_test_app(tmp_path, processor=processor)

    with TestClient(app) as client:
        created = client.post("/api/v1/imports", files=markdown_file())
        task_id = created.json()["tasks"][0]["task_id"]
        task = client.get(f"/api/v1/imports/{task_id}")

    assert created.status_code == 202
    assert task.status_code == 200
    assert task.json()["status"] == "failed"
    assert task.json()["error_code"] == ErrorCode.INTERNAL_ERROR.value
    assert "must-not-leak" not in task.text
    assert len(store.objects) == 1
    assert not any(path.is_file() for path in (tmp_path / "import-work").rglob("*"))


def test_too_many_files_are_rejected_before_workspace_creation(tmp_path: Path) -> None:
    app, store, processor, _ = build_test_app(tmp_path, max_files=1)
    files = [
        ("files", ("one.md", b"one", "text/markdown")),
        ("files", ("two.md", b"two", "text/markdown")),
    ]

    with TestClient(app) as client:
        response = client.post("/api/v1/imports", files=files)

    assert response.status_code == 422
    assert response.json()["code"] == ErrorCode.TOO_MANY_FILES.value
    assert store.put_count == 0
    assert processor.jobs == []


def test_legacy_import_routes_are_not_mounted(tmp_path: Path) -> None:
    app, _, _, _ = build_test_app(tmp_path)

    with TestClient(app) as client:
        assert client.post("/upload", files=markdown_file()).status_code == 404
        assert client.get(f"/status/{uuid.uuid4()}").status_code == 404


def test_authentication_rejects_request_before_multipart_body_parsing(tmp_path: Path) -> None:
    store = FakeStore()
    processor = Processor()
    registry = InMemoryImportTaskRegistry()
    settings = make_settings(tmp_path).model_copy(
        update={
            "auth": AuthSettings.model_construct(
                mode="jwt",
                issuer="https://issuer.example",
                audience="kb-api",
                jwks_url="https://issuer.example/.well-known/jwks.json",
                public_key=None,
                algorithms_raw="RS256",
                dev_tenant_id="tenant-a",
            )
        }
    )
    app = create_app(
        settings=settings,
        minio_builder=lambda _: cast(MinioStore, store),
        import_registry_factory=lambda: registry,
        import_processor_builder=lambda _: processor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/imports",
            files={"files": ("huge.md", b"x" * (2 * 1024 * 1024), "text/markdown")},
        )

    assert response.status_code == 401
    assert response.json()["code"] == ErrorCode.AUTHENTICATION_REQUIRED.value
    assert store.put_count == 0
    assert processor.jobs == []


def test_chunked_body_limit_runs_before_multipart_parser(tmp_path: Path) -> None:
    app, store, processor, _ = build_test_app(
        tmp_path,
        max_files=1,
        max_bytes=16,
        max_batch_bytes=16,
    )
    oversized = 2 * 1024 * 1024

    def chunks():
        yield (
            b"--upload-boundary\r\n"
            b'Content-Disposition: form-data; name="files"; filename="large.md"\r\n'
            b"Content-Type: text/markdown\r\n\r\n"
        )
        for _ in range(oversized // 65536):
            yield b"x" * 65536
        yield b"\r\n--upload-boundary--\r\n"

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/imports",
            content=chunks(),
            headers={
                "Content-Type": "multipart/form-data; boundary=upload-boundary",
                "X-Request-ID": "body-limit-test",
            },
        )

    assert response.status_code == 413
    assert response.json()["code"] == ErrorCode.FILE_TOO_LARGE.value
    assert response.headers["X-Request-ID"] == "body-limit-test"
    assert store.put_count == 0
    assert processor.jobs == []


def test_production_rejects_non_durable_import_runtime(tmp_path: Path) -> None:
    store = FakeStore()
    processor = Processor()
    settings = make_settings(tmp_path, environment=AppEnvironment.PRODUCTION).model_copy(
        update={
            "auth": AuthSettings.model_construct(
                mode="jwt",
                issuer="https://issuer.example",
                audience="kb-api",
                jwks_url="https://issuer.example/.well-known/jwks.json",
                public_key=None,
                algorithms_raw="RS256",
                dev_tenant_id="tenant-a",
            )
        }
    )
    app = create_app(
        settings=settings,
        minio_builder=lambda _: cast(MinioStore, store),
        import_processor_builder=lambda _: processor,
    )

    with pytest.raises(DependencyError, match="Import runtime is not configured"):
        with TestClient(app):
            pass


def test_openapi_declares_security_and_stable_error_contracts(tmp_path: Path) -> None:
    app, _, _, _ = build_test_app(tmp_path)

    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    create_operation = schema["paths"]["/api/v1/imports"]["post"]
    get_operation = schema["paths"]["/api/v1/imports/{task_id}"]["get"]
    assert create_operation["security"] == [{"HTTPBearer": []}]
    assert set(create_operation["responses"]) >= {"202", "401", "403", "409", "413", "415", "422", "503"}
    assert set(get_operation["responses"]) >= {"200", "401", "403", "404", "422", "503"}
