from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pytest

from app.infrastructure.milvus_tenant_vector_source import (
    MilvusCandidateContractError,
    MilvusTenantVectorSource,
    MilvusTenantVectorUnavailable,
)
from app.services.tenant_vector_search import HybridVectorQuery

_DOCUMENT_ID = "00000000-0000-4000-8000-000000000001"
_IMPORT_VERSION = "10000000-0000-4000-8000-000000000001"
_CHUNK_ID = "1" * 64
_COMMIT_PROOF = "milvus-v1.commit-proof"
_QUERY = HybridVectorQuery(dense_vector=(0.1, 0.2), sparse_vector={3: 0.4})


def hit(**overrides: object) -> dict[str, object]:
    entity: dict[str, object] = {
        "chunk_id": _CHUNK_ID,
        "tenant_id": "tenant-a",
        "document_id": _DOCUMENT_ID,
        "import_version": _IMPORT_VERSION,
        "commit_token": _COMMIT_PROOF,
        "content_sha256": hashlib.sha256(b"published content").hexdigest(),
        "content": "published content",
        "title": "title",
        "parent_title": "parent",
        "file_title": "file",
        "item_name": "item",
    }
    entity.update(overrides)
    return {"id": entity["chunk_id"], "distance": 0.91, "entity": entity}


class FakeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def hybrid_search(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class Factories:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.ranker_weights: list[tuple[float, float]] = []

    def request(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return dict(kwargs)

    def ranker(self, dense_weight: float, sparse_weight: float) -> object:
        self.ranker_weights.append((dense_weight, sparse_weight))
        return (dense_weight, sparse_weight)


def source(client: FakeClient, factories: Factories | None = None) -> MilvusTenantVectorSource:
    resolved = factories or Factories()
    return MilvusTenantVectorSource(
        client,
        collection_name="knowledge_chunks",
        request_factory=resolved.request,
        ranker_factory=resolved.ranker,
    )


def test_builds_server_owned_tenant_requests_and_requires_publication_fields() -> None:
    client = FakeClient([[hit()]])
    factories = Factories()
    adapter = source(client, factories)

    candidates = adapter.search_candidates(_QUERY, tenant_id="tenant-a", limit=5)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.tenant_id == "tenant-a"
    assert candidate.document_id == _DOCUMENT_ID
    assert candidate.import_version == _IMPORT_VERSION
    assert candidate.commit_token == _COMMIT_PROOF
    assert factories.ranker_weights == [(0.8, 0.2)]
    assert len(factories.requests) == 2
    assert {request["anns_field"] for request in factories.requests} == {
        "dense_vector",
        "sparse_vector",
    }
    assert all(request["expr"] == 'tenant_id == "tenant-a"' for request in factories.requests)
    assert all(request["limit"] == 5 for request in factories.requests)
    call = client.calls[0]
    assert call["collection_name"] == "knowledge_chunks"
    assert call["consistency_level"] == "Strong"
    assert call["limit"] == 5
    output_fields = call["output_fields"]
    assert isinstance(output_fields, list)
    assert set(output_fields) >= {
        "tenant_id",
        "document_id",
        "import_version",
        "commit_token",
        "content_sha256",
    }
    assert "filter" not in call


def test_default_pymilvus_request_and_ranker_factories_are_compatible() -> None:
    client = FakeClient([[]])
    adapter = MilvusTenantVectorSource(client, collection_name="knowledge_chunks")

    assert adapter.search_candidates(_QUERY, tenant_id="tenant-a", limit=2) == ()
    call = client.calls[0]
    requests = call["reqs"]
    assert isinstance(requests, list)
    assert type(requests[0]).__name__ == "AnnSearchRequest"
    assert type(call["ranker"]).__name__ == "WeightedRanker"


def test_adapter_preserves_foreign_candidate_for_authoritative_post_filter() -> None:
    client = FakeClient([[hit(tenant_id="tenant-b")]])
    adapter = source(client)

    candidates = adapter.search_candidates(_QUERY, tenant_id="tenant-a", limit=1)

    assert candidates[0].tenant_id == "tenant-b"


@pytest.mark.parametrize(
    "missing_field",
    ["document_id", "import_version", "commit_token", "content_sha256"],
)
def test_missing_publication_evidence_fails_closed(missing_field: str) -> None:
    entity = hit()["entity"]
    assert isinstance(entity, dict)
    entity.pop(missing_field)
    client = FakeClient([[{"id": _CHUNK_ID, "distance": 0.9, "entity": entity}]])

    with pytest.raises(MilvusCandidateContractError) as raised:
        source(client).search_candidates(_QUERY, tenant_id="tenant-a", limit=1)

    assert str(raised.value) == "Milvus candidate response is invalid"


def test_primary_id_must_match_entity_chunk_id() -> None:
    response_hit = hit()
    response_hit["id"] = "2" * 64

    with pytest.raises(MilvusCandidateContractError):
        source(FakeClient([[response_hit]])).search_candidates(
            _QUERY,
            tenant_id="tenant-a",
            limit=1,
        )


def test_content_tampering_with_reused_digest_fails_closed() -> None:
    response_hit = hit(content="tampered content")

    with pytest.raises(MilvusCandidateContractError) as raised:
        source(FakeClient([[response_hit]])).search_candidates(
            _QUERY,
            tenant_id="tenant-a",
            limit=1,
        )

    assert str(raised.value) == "Milvus candidate response is invalid"


def test_primary_id_is_used_when_milvus_omits_it_from_entity() -> None:
    response_hit = hit()
    entity = response_hit["entity"]
    assert isinstance(entity, dict)
    entity.pop("chunk_id")

    candidates = source(FakeClient([[response_hit]])).search_candidates(
        _QUERY,
        tenant_id="tenant-a",
        limit=1,
    )

    assert candidates[0].chunk_id == _CHUNK_ID


@pytest.mark.parametrize("tenant_id", ["", "tenant a", "tenant\nother", '../tenant"'])
def test_invalid_tenant_never_reaches_milvus(tenant_id: str) -> None:
    client = FakeClient([])

    with pytest.raises(ValueError, match="tenant"):
        source(client).search_candidates(_QUERY, tenant_id=tenant_id, limit=1)

    assert client.calls == []


def test_client_errors_are_sanitized() -> None:
    leaked_detail = "https://user:credential@private-milvus"  # pragma: allowlist secret

    class FailingClient(FakeClient):
        def hybrid_search(self, **kwargs: object) -> object:
            raise RuntimeError(leaked_detail)

    with pytest.raises(MilvusTenantVectorUnavailable) as raised:
        source(FailingClient([])).search_candidates(
            _QUERY,
            tenant_id="tenant-a",
            limit=1,
        )

    assert str(raised.value) == "Milvus vector search is unavailable"
    assert leaked_detail not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    "response",
    [None, [], [[], []], "invalid", [[hit(), hit()]]],
)
def test_invalid_or_oversized_response_fails_closed(response: object) -> None:
    with pytest.raises(MilvusCandidateContractError):
        source(FakeClient(response)).search_candidates(
            _QUERY,
            tenant_id="tenant-a",
            limit=1,
        )


def test_output_field_sequence_is_not_caller_mutable() -> None:
    client = FakeClient([[hit()]])
    adapter = source(client)

    adapter.search_candidates(_QUERY, tenant_id="tenant-a", limit=1)
    output_fields = client.calls[0]["output_fields"]
    assert isinstance(output_fields, list)
    output_fields.clear()
    adapter.search_candidates(_QUERY, tenant_id="tenant-a", limit=1)

    next_output_fields = client.calls[1]["output_fields"]
    assert isinstance(next_output_fields, list)
    assert "commit_token" in next_output_fields


def test_response_entity_must_be_a_mapping() -> None:
    invalid_hit: Mapping[str, object] = {"id": _CHUNK_ID, "distance": 0.5, "entity": []}

    with pytest.raises(MilvusCandidateContractError):
        source(FakeClient([[invalid_hit]])).search_candidates(
            _QUERY,
            tenant_id="tenant-a",
            limit=1,
        )
