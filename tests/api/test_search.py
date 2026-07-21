from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.search import router as search_router
from app.core.errors import ErrorCode, register_exception_handlers
from app.core.security import Principal, get_current_principal
from app.services.knowledge_search import KnowledgeSearchService
from app.services.tenant_vector_search import (
    CompletedImportVersion,
    HybridVectorQuery,
    TenantVectorSearch,
    VectorCandidate,
)

_DOCUMENT_ID = "00000000-0000-4000-8000-000000000001"
_IMPORT_VERSION = "10000000-0000-4000-8000-000000000001"
_CHUNK_ID = "b" * 64
_COMMIT_PROOF = "milvus-v1.api-proof"


def candidate(*, tenant_id: str = "tenant-a") -> VectorCandidate:
    content = "published answer context"
    return VectorCandidate(
        chunk_id=_CHUNK_ID,
        tenant_id=tenant_id,
        document_id=_DOCUMENT_ID,
        import_version=_IMPORT_VERSION,
        commit_token=_COMMIT_PROOF,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        score=0.75,
        content=content,
        title="Title",
        parent_title="Parent",
        file_title="File",
        item_name="Item",
    )


class Embedding:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.queries: list[str] = []

    def embed_query(self, query_text: str) -> HybridVectorQuery:
        self.queries.append(query_text)
        if self.failure is not None:
            raise self.failure
        return HybridVectorQuery((0.1, 0.2), {1: 0.3})


class Source:
    def __init__(self, items: Sequence[VectorCandidate]) -> None:
        self.items = tuple(items)
        self.calls: list[tuple[str, int]] = []

    def search_candidates(
        self,
        query: HybridVectorQuery,
        *,
        tenant_id: str,
        limit: int,
    ) -> Sequence[VectorCandidate]:
        self.calls.append((tenant_id, limit))
        return self.items[:limit]


class Visibility:
    def __init__(self, items: Sequence[VectorCandidate]) -> None:
        self.items = {item.import_version: item for item in items}

    def get_completed_versions(
        self,
        tenant_id: str,
        import_versions: Sequence[str],
    ) -> Mapping[str, CompletedImportVersion]:
        return {
            version: CompletedImportVersion(
                task_id=version,
                tenant_id=item.tenant_id,
                document_id=item.document_id,
                commit_token=item.commit_token,
            )
            for version in import_versions
            if (item := self.items.get(version)) is not None
        }


def build_app(
    *,
    principal: Principal | None = None,
    items: Sequence[VectorCandidate] = (),
    embedding: Embedding | None = None,
    install_service: bool = True,
) -> tuple[FastAPI, Embedding, Source]:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(search_router, prefix="/api/v1")
    current = principal or Principal("reader-a", "tenant-a", frozenset({"reader"}))
    app.dependency_overrides[get_current_principal] = lambda: current
    backend = embedding or Embedding()
    source = Source(items)
    if install_service:
        app.state.knowledge_search = KnowledgeSearchService(
            backend,
            TenantVectorSearch(
                source,
                Visibility(items),
                initial_overfetch_factor=1,
                max_candidates=20,
                max_results=20,
            ),
        )
    return app, backend, source


def test_search_uses_only_principal_tenant_and_returns_public_contract() -> None:
    item = candidate()
    app, embedding, source = build_app(items=(item,))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search",
            json={"query": "product guide", "top_k": 1},
        )

    assert response.status_code == 200
    assert embedding.queries == ["product guide"]
    assert source.calls == [("tenant-a", 1)]
    payload = response.json()
    assert payload == {
        "hits": [
            {
                "citation_id": payload["hits"][0]["citation_id"],
                "content": "published answer context",
                "metadata": {
                    "title": "Title",
                    "parent_title": "Parent",
                    "file_title": "File",
                    "item_name": "Item",
                },
                "score": 0.75,
            }
        ]
    }
    assert payload["hits"][0]["citation_id"].startswith("cit_")
    assert _CHUNK_ID not in response.text
    for secret_field in (
        "tenant_id",
        "document_id",
        "import_version",
        "commit_token",
        "dense_vector",
        "sparse_vector",
        "chunk_id",
    ):
        assert secret_field not in response.text


@pytest.mark.parametrize("injected_field", ["tenant_id", "tenant", "filter", "expr"])
def test_body_cannot_inject_tenant_or_backend_filter(injected_field: str) -> None:
    app, embedding, source = build_app(items=(candidate(),))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search",
            json={
                "query": "product guide",
                "top_k": 1,
                injected_field: 'tenant_id == "tenant-b"',
            },
        )

    assert response.status_code == 422
    assert response.json()["code"] == ErrorCode.REQUEST_INVALID.value
    assert response.json()["message"] == "Request validation failed"
    assert response.json()["details"]["errors"][0]["location"] == ["body"]
    assert embedding.queries == []
    assert source.calls == []


def test_even_a_leaking_source_cannot_return_foreign_tenant_content() -> None:
    app, _, source = build_app(items=(candidate(tenant_id="tenant-b"),))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search",
            json={"query": "product guide", "top_k": 1},
        )

    assert response.status_code == 200
    assert response.json() == {"hits": []}
    assert source.calls
    assert all(tenant_id == "tenant-a" for tenant_id, _ in source.calls)
    assert max(limit for _, limit in source.calls) <= 2
    assert "published answer context" not in response.text


@pytest.mark.parametrize("missing_value", [None, object()])
def test_missing_or_invalid_runtime_service_fails_closed(missing_value: object) -> None:
    app, _, source = build_app(install_service=False)
    if missing_value is not None:
        app.state.knowledge_search = missing_value

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search",
            json={"query": "product guide", "top_k": 1},
        )

    assert response.status_code == 503
    assert response.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert response.json()["message"] == "Knowledge search is unavailable"
    assert response.json()["details"] == {}
    assert source.calls == []


def test_embedding_dependency_failure_is_sanitized_at_api_boundary() -> None:
    leaked_detail = "embedding-key=must-not-leak"
    app, _, source = build_app(embedding=Embedding(failure=RuntimeError(leaked_detail)))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search",
            json={"query": "product guide", "top_k": 1},
        )

    assert response.status_code == 503
    assert response.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert response.json()["message"] == "Knowledge search is unavailable"
    assert response.json()["details"] == {}
    assert leaked_detail not in response.text
    assert source.calls == []


def test_reader_editor_and_admin_roles_are_allowed() -> None:
    for role in ("reader", "editor", "admin"):
        app, _, _ = build_app(
            principal=Principal(f"subject-{role}", "tenant-a", frozenset({role}))
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/search",
                json={"query": "product guide", "top_k": 1},
            )
        assert response.status_code == 200


def test_unapproved_role_is_rejected_before_search() -> None:
    app, embedding, source = build_app(
        principal=Principal("auditor", "tenant-a", frozenset({"auditor"}))
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/search",
            json={"query": "product guide", "top_k": 1},
        )

    assert response.status_code == 403
    assert response.json()["code"] == ErrorCode.PERMISSION_DENIED.value
    assert embedding.queries == []
    assert source.calls == []


def test_openapi_declares_bearer_security_and_non_sensitive_contracts() -> None:
    app, _, _ = build_app()

    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/api/v1/search"]["post"]
    assert operation["security"] == [{"HTTPBearer": []}]
    assert set(operation["responses"]) >= {"200", "401", "403", "422", "500", "503"}
    request_reference = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    request_name = request_reference.rsplit("/", 1)[-1]
    request_schema = schema["components"]["schemas"][request_name]
    assert set(request_schema["properties"]) == {"query", "top_k"}
    assert request_schema["additionalProperties"] is False

    response_reference = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    response_name = response_reference.rsplit("/", 1)[-1]
    response_schemas = {
        name: component
        for name, component in schema["components"]["schemas"].items()
        if name.startswith("KnowledgeSearch") and name != request_name
    }
    assert response_name in response_schemas
    response_contract = json.dumps(response_schemas, sort_keys=True)
    for internal_field in (
        "tenant_id",
        "filter",
        "chunk_id",
        "document_id",
        "import_version",
        "commit_token",
        "dense_vector",
        "sparse_vector",
    ):
        assert internal_field not in response_contract
