from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from threading import Event
from typing import cast

import pytest
from fastapi.testclient import TestClient

from app.core.errors import DependencyError, ErrorCode
from app.core.settings import (
    ApiSettings,
    AppEnvironment,
    AuthSettings,
    ImportRuntimeSettings,
    KnowledgeSearchSettings,
    LoggingSettings,
    MinioSettings,
    MongoSettings,
    Settings,
)
from app.infrastructure.minio_store import MinioStore
from app.main import create_app
from app.services.import_coordinator import ImportCoordinator
from app.services.knowledge_search import KnowledgeSearchService
from app.services.knowledge_search_runtime import ManagedKnowledgeSearchRuntime
from app.services.tenant_vector_search import (
    CompletedImportVersion,
    HybridVectorQuery,
    TenantVectorSearch,
    VectorCandidate,
)


class Store:
    def __init__(self) -> None:
        self.closed = False

    def require_ready(self) -> None:
        return

    def close(self) -> None:
        self.closed = True


class Visibility:
    def get_completed_versions(
        self,
        tenant_id: str,
        import_versions: Sequence[str],
    ) -> Mapping[str, CompletedImportVersion]:
        del tenant_id, import_versions
        return {}


class Coordinator:
    durable = True

    def __init__(self) -> None:
        self.completed_import_visibility = Visibility()
        self.probe_calls = 0
        self.closed = False

    def probe(self) -> None:
        self.probe_calls += 1

    def close(self) -> None:
        self.closed = True


class Embedding:
    def embed_query(self, query_text: str) -> HybridVectorQuery:
        del query_text
        return HybridVectorQuery((0.1, 0.2), {1: 0.3})


class Source:
    def search_candidates(
        self,
        query: HybridVectorQuery,
        *,
        tenant_id: str,
        limit: int,
    ) -> Sequence[VectorCandidate]:
        del query, tenant_id, limit
        return ()


class SearchRuntime:
    tenant_safe = True
    authoritative_visibility = True

    def __init__(self) -> None:
        self.embedding_backend = cast(object, None)
        self.milvus_runtime = cast(object, None)
        self.service = KnowledgeSearchService(
            Embedding(),
            TenantVectorSearch(
                Source(),
                Visibility(),
                max_candidates=20,
                max_results=5,
            ),
            max_top_k=5,
        )
        self.ready = True
        self.probe_calls = 0
        self.probed = Event()
        self.closed = False

    def probe(self) -> None:
        self.probe_calls += 1
        self.probed.set()
        if not self.ready:
            raise RuntimeError("secret dependency detail")

    def close(self) -> None:
        self.closed = True


def settings() -> Settings:
    return Settings.model_construct(
        api=ApiSettings.model_construct(
            environment=AppEnvironment.TEST,
            import_enabled=False,
            cors_origins_raw="https://console.example.com",
        ),
        import_runtime=ImportRuntimeSettings.model_construct(
            repository="mongo",
            worker_enabled=False,
            poll_interval_seconds=0.05,
            shutdown_grace_seconds=1.0,
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
            readiness_interval_seconds=0.05,
            readiness_probe_timeout_seconds=0.2,
            readiness_ttl_seconds=0.5,
        ),
        search=KnowledgeSearchSettings.model_construct(
            enabled=True,
            timeout_seconds=1.0,
            max_top_k=5,
            max_candidates=20,
            initial_overfetch_factor=2,
            growth_factor=2,
            dense_weight=0.8,
            sparse_weight=0.2,
            readiness_interval_seconds=0.05,
            readiness_probe_timeout_seconds=0.2,
            readiness_ttl_seconds=0.5,
        ),
        mongo=MongoSettings.model_construct(
            url="mongodb://mongo.invalid:27017",
            health_timeout_seconds=0.2,
        ),
    )


def build_test_app(
    runtime: SearchRuntime,
    coordinator: Coordinator,
):
    store = Store()
    build_calls: list[tuple[object, object, object]] = []

    def build_search(
        app_settings: Settings,
        visibility: object,
        visibility_probe: object,
    ) -> ManagedKnowledgeSearchRuntime:
        build_calls.append((app_settings, visibility, visibility_probe))
        return cast(ManagedKnowledgeSearchRuntime, runtime)

    app = create_app(
        settings=settings(),
        minio_builder=lambda _: cast(MinioStore, store),
        import_coordinator_factory=lambda: cast(ImportCoordinator, coordinator),
        knowledge_search_runtime_builder=build_search,
    )
    return app, store, build_calls


def test_search_runtime_is_lazy_mounted_ready_and_closed() -> None:
    runtime = SearchRuntime()
    coordinator = Coordinator()
    app, store, build_calls = build_test_app(runtime, coordinator)

    assert build_calls == []
    with TestClient(app) as client:
        response = client.post("/api/v1/search", json={"query": "guide", "top_k": 2})
        readiness = client.get("/health/ready")
        schema = client.get("/openapi.json").json()

    assert response.status_code == 200
    assert response.json() == {"hits": []}
    assert readiness.status_code == 200
    assert readiness.json()["checks"] == {
        "minio": "ok",
        "knowledge_search": "ok",
    }
    assert "/api/v1/search" in schema["paths"]
    assert len(build_calls) == 1
    assert build_calls[0][0] is app.state.settings
    assert build_calls[0][1] is coordinator.completed_import_visibility
    assert callable(build_calls[0][2])
    assert runtime.probe_calls >= 1
    assert runtime.closed is True
    assert coordinator.closed is True
    assert store.closed is True
    assert app.state.knowledge_search is None


def test_failed_search_probe_withdraws_service_and_background_probe_recovers() -> None:
    runtime = SearchRuntime()
    runtime.ready = False
    coordinator = Coordinator()
    app, _, _ = build_test_app(runtime, coordinator)

    with TestClient(app) as client:
        unavailable = client.post(
            "/api/v1/search",
            json={"query": "guide", "top_k": 2},
        )
        assert unavailable.status_code == 503
        assert unavailable.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value
        assert "secret dependency detail" not in unavailable.text
        assert client.get("/health/ready").status_code == 503

        runtime.ready = True
        initial_calls = runtime.probe_calls
        deadline = time.monotonic() + 1.0
        while runtime.probe_calls == initial_calls and time.monotonic() < deadline:
            time.sleep(0.01)

        recovered = client.post(
            "/api/v1/search",
            json={"query": "guide", "top_k": 2},
        )
        assert recovered.status_code == 200
        assert client.get("/health/ready").status_code == 200


def test_stale_search_readiness_fails_closed_at_route_boundary() -> None:
    app, _, _ = build_test_app(SearchRuntime(), Coordinator())

    with TestClient(app) as client:
        assert client.post(
            "/api/v1/search",
            json={"query": "guide", "top_k": 1},
        ).status_code == 200
        app.state.knowledge_search_ready_until = time.monotonic() - 1
        response = client.post(
            "/api/v1/search",
            json={"query": "guide", "top_k": 1},
        )

    assert response.status_code == 503
    assert response.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value


def test_explicitly_missing_search_runtime_builder_aborts_startup_safely() -> None:
    app = create_app(
        settings=settings(),
        minio_builder=lambda _: cast(MinioStore, Store()),
        import_coordinator_factory=lambda: cast(ImportCoordinator, Coordinator()),
        knowledge_search_runtime_builder=None,
    )

    with pytest.raises(DependencyError, match="Knowledge search runtime is not configured"):
        with TestClient(app):
            pass
