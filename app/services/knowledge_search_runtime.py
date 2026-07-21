from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Protocol

from app.core.errors import DependencyError
from app.core.logger import logger
from app.core.settings import Settings
from app.import_process.agent.nodes.node_hybrid_embedding import HybridEmbeddingBackend
from app.infrastructure.milvus_tenant_vector_source import MilvusTenantVectorSource
from app.infrastructure.query_embedding_adapter import HybridQueryEmbeddingAdapter
from app.services.import_processor_factory import (
    ManagedMilvusRuntime,
    build_default_milvus_runtime,
    build_local_embedding_backend,
)
from app.services.knowledge_search import KnowledgeSearchService
from app.services.tenant_vector_search import (
    CompletedImportVisibilityReader,
    TenantVectorSearch,
)

_UNAVAILABLE_MESSAGE = "Knowledge search runtime is unavailable"


class SearchMilvusRuntime(ManagedMilvusRuntime, Protocol):
    def hybrid_search(
        self,
        *,
        collection_name: str,
        reqs: list[object],
        ranker: object,
        limit: int,
        output_fields: list[str],
        consistency_level: str,
    ) -> object: ...


EmbeddingBackendBuilder = Callable[[Settings], HybridEmbeddingBackend]
MilvusRuntimeBuilder = Callable[[Settings, int], SearchMilvusRuntime]
VisibilityProbe = Callable[[], None]


class ManagedKnowledgeSearchRuntime(Protocol):
    tenant_safe: bool
    authoritative_visibility: bool
    service: KnowledgeSearchService
    embedding_backend: HybridEmbeddingBackend
    milvus_runtime: SearchMilvusRuntime

    def probe(self) -> None: ...

    def close(self) -> None: ...


class KnowledgeSearchRuntime:
    """Own the shared embedding/Milvus search plane and its public service.

    The visibility reader is authoritative; the Milvus tenant expression is
    only an optimization. Construction is side-effect free and ``probe`` owns
    all model/schema/dependency initialization.
    """

    tenant_safe = True
    authoritative_visibility = True

    def __init__(
        self,
        *,
        service: KnowledgeSearchService,
        embedding_backend: HybridEmbeddingBackend,
        milvus_runtime: SearchMilvusRuntime,
        visibility_probe: VisibilityProbe,
    ) -> None:
        if not isinstance(service, KnowledgeSearchService):
            raise ValueError("invalid knowledge search service")
        if not callable(visibility_probe):
            raise ValueError("invalid knowledge visibility probe")
        self.service = service
        self.embedding_backend = embedding_backend
        self.milvus_runtime = milvus_runtime
        self._visibility_probe = visibility_probe
        self._lock = RLock()
        self._closed = False

    def probe(self) -> None:
        with self._lock:
            if self._closed:
                raise DependencyError(_UNAVAILABLE_MESSAGE)
            try:
                self._visibility_probe()
                self.embedding_backend.probe()
                if self.milvus_runtime.schema_ready:
                    self.milvus_runtime.probe()
                else:
                    self.milvus_runtime.ensure_schema()
            except Exception:
                raise DependencyError(_UNAVAILABLE_MESSAGE) from None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        failed = False
        for component in (self.milvus_runtime, self.embedding_backend):
            try:
                component.close()
            except Exception:
                failed = True
        if failed:
            raise DependencyError(_UNAVAILABLE_MESSAGE)


def build_knowledge_search_runtime(
    settings: Settings,
    visibility: CompletedImportVisibilityReader,
    visibility_probe: VisibilityProbe,
    *,
    embedding_backend_builder: EmbeddingBackendBuilder = build_local_embedding_backend,
    milvus_runtime_builder: MilvusRuntimeBuilder = build_default_milvus_runtime,
) -> KnowledgeSearchRuntime:
    """Compose a lazy, tenant-safe search runtime without dependency I/O."""

    if not isinstance(settings, Settings) or not settings.search.enabled:
        raise ValueError("knowledge search settings are required")
    if not callable(getattr(visibility, "get_completed_versions", None)):
        raise ValueError("authoritative knowledge visibility is required")
    if not callable(visibility_probe):
        raise ValueError("knowledge visibility probe is required")
    settings.validate_capabilities("mongo", "milvus")

    backend: HybridEmbeddingBackend | None = None
    runtime: SearchMilvusRuntime | None = None
    try:
        backend = embedding_backend_builder(settings)
        query_embedding = HybridQueryEmbeddingAdapter(backend)
        runtime = milvus_runtime_builder(settings, query_embedding.dense_dimension)
        source = MilvusTenantVectorSource(
            runtime,
            collection_name=settings.milvus.chunks_collection,
            dense_weight=settings.search.dense_weight,
            sparse_weight=settings.search.sparse_weight,
            max_candidates=settings.search.max_candidates,
        )
        vector_search = TenantVectorSearch(
            source,
            visibility,
            initial_overfetch_factor=settings.search.initial_overfetch_factor,
            growth_factor=settings.search.growth_factor,
            max_candidates=settings.search.max_candidates,
            max_results=settings.search.max_top_k,
        )
        service = KnowledgeSearchService(
            query_embedding,
            vector_search,
            timeout_seconds=settings.search.timeout_seconds,
            max_top_k=settings.search.max_top_k,
            max_concurrent_requests=settings.search.max_concurrent_requests,
        )
        return KnowledgeSearchRuntime(
            service=service,
            embedding_backend=backend,
            milvus_runtime=runtime,
            visibility_probe=visibility_probe,
        )
    except BaseException:
        for component in (runtime, backend):
            closer = getattr(component, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as close_error:
                    logger.bind(error_type=type(close_error).__name__).warning(
                        "Partial knowledge search dependency cleanup failed"
                    )
        raise


__all__ = [
    "KnowledgeSearchRuntime",
    "ManagedKnowledgeSearchRuntime",
    "build_knowledge_search_runtime",
]
