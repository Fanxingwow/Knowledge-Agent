from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pytest

from app.core.settings import (
    EmbeddingSettings,
    KnowledgeSearchSettings,
    MilvusSettings,
    MongoSettings,
    Settings,
)
from app.import_process.agent.nodes.node_hybrid_embedding import HybridEmbedding
from app.services.knowledge_search_runtime import build_knowledge_search_runtime
from app.services.tenant_vector_search import CompletedImportVersion


class FakeBackend:
    dense_dimension = 2

    def __init__(self, closed: list[str]) -> None:
        self.probe_calls = 0
        self.closed = closed

    def embed(self, texts: Sequence[str]) -> Sequence[HybridEmbedding]:
        return tuple(HybridEmbedding((0.1, 0.2), {1: 0.3}) for _ in texts)

    def probe(self) -> None:
        self.probe_calls += 1

    def close(self) -> None:
        self.closed.append("embedding")


class FakeMilvusRuntime:
    def __init__(self, closed: list[str]) -> None:
        self.schema_ready = False
        self.ensure_calls = 0
        self.probe_calls = 0
        self.closed = closed

    def ensure_schema(self) -> object:
        self.ensure_calls += 1
        self.schema_ready = True
        return object()

    def probe(self) -> None:
        self.probe_calls += 1

    def query(self, **_kwargs: object) -> tuple[Mapping[str, object], ...]:
        return ()

    def hybrid_search(self, **_kwargs: object) -> object:
        return ((),)

    def upsert(self, **_kwargs: object) -> object:
        return object()

    def delete(self, **_kwargs: object) -> object:
        return object()

    def close(self) -> None:
        self.closed.append("milvus")


class Visibility:
    def get_completed_versions(
        self,
        tenant_id: str,
        import_versions: Sequence[str],
    ) -> Mapping[str, CompletedImportVersion]:
        del tenant_id, import_versions
        return {}


def settings() -> Settings:
    return Settings.model_construct(
        search=KnowledgeSearchSettings.model_construct(
            enabled=True,
            timeout_seconds=2.0,
            max_top_k=5,
            max_candidates=20,
            initial_overfetch_factor=2,
            growth_factor=2,
            dense_weight=0.8,
            sparse_weight=0.2,
            readiness_interval_seconds=1.0,
            readiness_probe_timeout_seconds=2.0,
            readiness_ttl_seconds=3.0,
        ),
        embedding=EmbeddingSettings.model_construct(
            model_path="C:/models/bge-m3",
            dense_dimension=2,
        ),
        milvus=MilvusSettings.model_construct(
            url="http://milvus.invalid:19530",
            chunks_collection="kb_chunks_v2",
        ),
        mongo=MongoSettings.model_construct(url="mongodb://mongo.invalid:27017"),
    )


def test_runtime_is_lazy_probes_visibility_and_owns_shared_resources() -> None:
    closed: list[str] = []
    backend = FakeBackend(closed)
    milvus = FakeMilvusRuntime(closed)
    visibility_probes: list[str] = []

    runtime = build_knowledge_search_runtime(
        settings(),
        Visibility(),
        lambda: visibility_probes.append("mongo"),
        embedding_backend_builder=lambda _: backend,
        milvus_runtime_builder=lambda _, dimension: (
            cast(object, milvus)
            if dimension == 2
            else pytest.fail("wrong embedding dimension")
        ),  # type: ignore[arg-type]
    )

    assert visibility_probes == []
    assert backend.probe_calls == milvus.ensure_calls == milvus.probe_calls == 0
    assert runtime.tenant_safe is True
    assert runtime.authoritative_visibility is True

    runtime.probe()
    runtime.probe()

    assert visibility_probes == ["mongo", "mongo"]
    assert backend.probe_calls == 2
    assert milvus.ensure_calls == 1
    assert milvus.probe_calls == 1
    runtime.close()
    runtime.close()
    assert closed == ["milvus", "embedding"]


def test_factory_closes_partial_backend_when_milvus_construction_fails() -> None:
    closed: list[str] = []
    backend = FakeBackend(closed)

    with pytest.raises(RuntimeError, match="construction failed"):
        build_knowledge_search_runtime(
            settings(),
            Visibility(),
            lambda: None,
            embedding_backend_builder=lambda _: backend,
            milvus_runtime_builder=lambda _settings, _dimension: (_ for _ in ()).throw(
                RuntimeError("construction failed")
            ),
        )

    assert closed == ["embedding"]
