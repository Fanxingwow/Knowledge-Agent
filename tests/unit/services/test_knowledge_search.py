from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Mapping, Sequence
from threading import Event

import pytest

from app.core.errors import AppError, DependencyError, ErrorCode
from app.services.knowledge_search import (
    MAX_QUERY_CHARACTERS,
    MAX_QUERY_UTF8_BYTES,
    MAX_SEARCH_TOP_K,
    KnowledgeSearchService,
)
from app.services.tenant_vector_search import (
    CompletedImportVersion,
    HybridVectorQuery,
    TenantVectorSearch,
    VectorCandidate,
)

_DOCUMENT_ID = "00000000-0000-4000-8000-000000000001"
_IMPORT_VERSION = "10000000-0000-4000-8000-000000000001"
_CHUNK_ID = "a" * 64
_COMMIT_PROOF = "milvus-v1.safe-proof"
_QUERY_VECTOR = HybridVectorQuery((0.1, 0.2), {1: 0.3})


def candidate(*, tenant_id: str = "tenant-a") -> VectorCandidate:
    content = "tenant-owned knowledge"
    return VectorCandidate(
        chunk_id=_CHUNK_ID,
        tenant_id=tenant_id,
        document_id=_DOCUMENT_ID,
        import_version=_IMPORT_VERSION,
        commit_token=_COMMIT_PROOF,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        score=0.875,
        content=content,
        title="Section",
        parent_title="Chapter",
        file_title="Guide",
        item_name="Product",
    )


class RecordingEmbedding:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query_text: str) -> HybridVectorQuery:
        self.queries.append(query_text)
        return _QUERY_VECTOR


class Source:
    def __init__(self, candidates: Sequence[VectorCandidate]) -> None:
        self.candidates = tuple(candidates)
        self.calls: list[tuple[HybridVectorQuery, str, int]] = []

    def search_candidates(
        self,
        query: HybridVectorQuery,
        *,
        tenant_id: str,
        limit: int,
    ) -> Sequence[VectorCandidate]:
        self.calls.append((query, tenant_id, limit))
        return self.candidates[:limit]


class Visibility:
    def __init__(self, visible: Mapping[str, CompletedImportVersion]) -> None:
        self.visible = dict(visible)

    def get_completed_versions(
        self,
        tenant_id: str,
        import_versions: Sequence[str],
    ) -> Mapping[str, CompletedImportVersion]:
        return {
            version: self.visible[version]
            for version in import_versions
            if version in self.visible
        }


def service_for(
    items: Sequence[VectorCandidate] = (),
    *,
    embedding: RecordingEmbedding | None = None,
    timeout_seconds: float = 1.0,
    max_top_k: int = MAX_SEARCH_TOP_K,
    max_concurrent_requests: int = 16,
) -> tuple[KnowledgeSearchService, RecordingEmbedding, Source]:
    backend = embedding or RecordingEmbedding()
    source = Source(items)
    visible = {
        item.import_version: CompletedImportVersion(
            task_id=item.import_version,
            tenant_id=item.tenant_id,
            document_id=item.document_id,
            commit_token=item.commit_token,
        )
        for item in items
    }
    vector_search = TenantVectorSearch(
        source,
        Visibility(visible),
        initial_overfetch_factor=1,
        max_candidates=MAX_SEARCH_TOP_K,
        max_results=MAX_SEARCH_TOP_K,
    )
    return (
        KnowledgeSearchService(
            backend,
            vector_search,
            timeout_seconds=timeout_seconds,
            max_top_k=max_top_k,
            max_concurrent_requests=max_concurrent_requests,
        ),
        backend,
        source,
    )


@pytest.mark.asyncio
async def test_embeds_validated_text_and_applies_authenticated_tenant_scope() -> None:
    item = candidate()
    service, embedding, source = service_for((item,))

    result = await service.search(
        tenant_id="tenant-a",
        query_text="  cafe\u0301 商品说明  ",
        top_k=1,
    )

    assert embedding.queries == ["café 商品说明"]
    assert source.calls == [(_QUERY_VECTOR, "tenant-a", 1)]
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.citation_id.startswith("cit_")
    assert hit.citation_id != item.chunk_id
    assert hit.content == item.content
    assert hit.metadata.file_title == "Guide"
    assert hit.score == item.score
    for internal_name in (
        "chunk_id",
        "tenant_id",
        "document_id",
        "import_version",
        "commit_token",
        "dense_vector",
        "sparse_vector",
    ):
        assert not hasattr(hit, internal_name)


@pytest.mark.asyncio
async def test_foreign_candidates_are_filtered_by_tenant_vector_search() -> None:
    foreign = candidate(tenant_id="tenant-b")
    service, _, source = service_for((foreign,))

    result = await service.search(
        tenant_id="tenant-a",
        query_text="find knowledge",
        top_k=1,
    )

    assert result.hits == ()
    assert source.calls[0][1] == "tenant-a"


@pytest.mark.parametrize(
    "query_text",
    [
        "",
        "   ",
        "line\nbreak",
        "tab\tquery",
        "hidden\u202econtrol",
        "line\u2028separator",
        "private\ue000value",
        "\ud800",
        "a" * (MAX_QUERY_CHARACTERS + 1),
        "知" * (MAX_QUERY_UTF8_BYTES // 3 + 1),
    ],
)
@pytest.mark.asyncio
async def test_query_character_and_utf8_limits_fail_before_embedding(query_text: str) -> None:
    service, embedding, source = service_for()

    with pytest.raises(AppError) as raised:
        await service.search(tenant_id="tenant-a", query_text=query_text, top_k=1)

    assert raised.value.code is ErrorCode.REQUEST_INVALID
    assert raised.value.message == "Knowledge search request is invalid"
    assert raised.value.details == {}
    assert embedding.queries == []
    assert source.calls == []


@pytest.mark.parametrize("top_k", [True, 0, -1, MAX_SEARCH_TOP_K + 1])
@pytest.mark.asyncio
async def test_top_k_is_strictly_bounded(top_k: object) -> None:
    service, embedding, _ = service_for()

    with pytest.raises(AppError) as raised:
        await service.search(
            tenant_id="tenant-a",
            query_text="valid query",
            top_k=top_k,  # type: ignore[arg-type]
        )

    assert raised.value.code is ErrorCode.REQUEST_INVALID
    assert embedding.queries == []


@pytest.mark.asyncio
async def test_instance_level_top_k_limit_is_enforced() -> None:
    service, embedding, _ = service_for(max_top_k=3)

    with pytest.raises(AppError, match="Knowledge search request is invalid"):
        await service.search(tenant_id="tenant-a", query_text="valid query", top_k=4)

    assert embedding.queries == []


@pytest.mark.asyncio
async def test_embedding_failure_is_sanitized() -> None:
    leaked_detail = "https://embedding.internal/?token=must-not-leak"

    class FailingEmbedding(RecordingEmbedding):
        def embed_query(self, query_text: str) -> HybridVectorQuery:
            raise RuntimeError(leaked_detail)

    service, _, _ = service_for(embedding=FailingEmbedding())

    with pytest.raises(DependencyError) as raised:
        await service.search(tenant_id="tenant-a", query_text="valid query", top_k=1)

    rendered = f"{raised.value!r} {raised.value!s} {raised.value.details!r}"
    assert raised.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert raised.value.message == "Knowledge search is unavailable"
    assert raised.value.details == {}
    assert raised.value.__cause__ is None
    assert leaked_detail not in rendered


@pytest.mark.asyncio
async def test_invalid_embedding_contract_fails_closed() -> None:
    class InvalidEmbedding(RecordingEmbedding):
        def embed_query(self, query_text: str) -> HybridVectorQuery:
            return object()  # type: ignore[return-value]

    service, _, _ = service_for(embedding=InvalidEmbedding())

    with pytest.raises(DependencyError, match="Knowledge search is unavailable"):
        await service.search(tenant_id="tenant-a", query_text="valid query", top_k=1)


@pytest.mark.asyncio
async def test_total_operation_timeout_is_sanitized() -> None:
    class SlowEmbedding(RecordingEmbedding):
        def embed_query(self, query_text: str) -> HybridVectorQuery:
            time.sleep(0.05)
            return super().embed_query(query_text)

    service, _, _ = service_for(
        embedding=SlowEmbedding(),
        timeout_seconds=0.01,
    )

    with pytest.raises(DependencyError) as raised:
        await service.search(tenant_id="tenant-a", query_text="valid query", top_k=1)

    assert raised.value.message == "Knowledge search is unavailable"
    assert raised.value.details == {}
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_concurrency_bulkhead_bounds_abandoned_embedding_work() -> None:
    class BlockingEmbedding(RecordingEmbedding):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()

        def embed_query(self, query_text: str) -> HybridVectorQuery:
            self.queries.append(query_text)
            self.started.set()
            self.release.wait(timeout=1)
            return _QUERY_VECTOR

    embedding = BlockingEmbedding()
    service, _, _ = service_for(
        embedding=embedding,
        timeout_seconds=0.05,
        max_concurrent_requests=1,
    )
    first = asyncio.create_task(
        service.search(tenant_id="tenant-a", query_text="first", top_k=1)
    )
    assert await asyncio.to_thread(embedding.started.wait, 1)

    with pytest.raises(DependencyError, match="Knowledge search is unavailable"):
        await service.search(tenant_id="tenant-a", query_text="second", top_k=1)
    with pytest.raises(DependencyError, match="Knowledge search is unavailable"):
        await first

    assert embedding.queries == ["first"]
    embedding.release.set()


@pytest.mark.parametrize("tenant_id", ["", "../tenant", "tenant b", "tenant\nother"])
@pytest.mark.asyncio
async def test_invalid_tenant_fails_before_embedding(tenant_id: str) -> None:
    service, embedding, _ = service_for()

    with pytest.raises(AppError) as raised:
        await service.search(tenant_id=tenant_id, query_text="valid query", top_k=1)

    assert raised.value.code is ErrorCode.REQUEST_INVALID
    assert embedding.queries == []


@pytest.mark.parametrize(
    ("timeout_seconds", "max_top_k"),
    [
        (True, 1),
        (0.0, 1),
        (float("inf"), 1),
        (1.0, True),
        (1.0, 0),
        (1.0, MAX_SEARCH_TOP_K + 1),
    ],
)
def test_constructor_rejects_unsafe_resource_limits(
    timeout_seconds: object,
    max_top_k: object,
) -> None:
    source = Source(())
    vector_search = TenantVectorSearch(source, Visibility({}))

    with pytest.raises(ValueError):
        KnowledgeSearchService(
            RecordingEmbedding(),
            vector_search,
            timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
            max_top_k=max_top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [True, 0, 257])
def test_constructor_rejects_unsafe_concurrency_limit(value: object) -> None:
    vector_search = TenantVectorSearch(Source(()), Visibility({}))

    with pytest.raises(ValueError, match="concurrency"):
        KnowledgeSearchService(
            RecordingEmbedding(),
            vector_search,
            max_concurrent_requests=value,  # type: ignore[arg-type]
        )
