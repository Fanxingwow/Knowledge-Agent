from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from app.core.logger import logger
from app.core.settings import MineruSettings, Settings
from app.import_process.agent.main_graph import ImportGraphNodes, build_import_graph
from app.import_process.agent.nodes.node_commit_index import MilvusCommitNode
from app.import_process.agent.nodes.node_document_parse import build_document_parse_node
from app.import_process.agent.nodes.node_hybrid_embedding import (
    HybridEmbeddingBackend,
    HybridEmbeddingNode,
    HybridEmbeddingPolicy,
    LazyLocalBgeM3Backend,
)
from app.import_process.agent.nodes.node_markdown_images import MarkdownImageTextNode
from app.import_process.agent.nodes.node_markdown_splitter import (
    MarkdownSplitPolicy,
    MarkdownSplitterNode,
)
from app.import_process.agent.nodes.node_safe_item_name import SafeItemNameNode
from app.import_process.agent.nodes.node_validated_entry import ValidatedEntryNode
from app.import_process.agent.state import ImportGraphState
from app.infrastructure.milvus_import_index import MilvusImportIndex
from app.infrastructure.milvus_runtime import build_milvus_runtime_from_settings
from app.services.graph_import_processor import GraphImportProcessor


class ManagedParseNode(Protocol):
    def __call__(self, state: ImportGraphState) -> Mapping[str, object]: ...

    def probe(self) -> None: ...

    def close(self) -> None: ...


class ManagedMilvusRuntime(Protocol):
    @property
    def schema_ready(self) -> bool: ...

    def ensure_schema(self) -> object: ...

    def probe(self) -> None: ...

    def query(
        self,
        *,
        collection_name: str,
        filter: str,
        output_fields: list[str],
        limit: int,
        consistency_level: str,
    ) -> Sequence[Mapping[str, object]]: ...

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

    def upsert(
        self,
        *,
        collection_name: str,
        data: list[dict[str, object]],
    ) -> object: ...

    def delete(self, *, collection_name: str, filter: str) -> object: ...

    def close(self) -> None: ...


ParseNodeBuilder = Callable[[MineruSettings], ManagedParseNode]
EmbeddingBackendBuilder = Callable[[Settings], HybridEmbeddingBackend]
MilvusRuntimeBuilder = Callable[[Settings, int], ManagedMilvusRuntime]


def build_local_embedding_backend(settings: Settings) -> HybridEmbeddingBackend:
    model_path = settings.embedding.model_path
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("BGE_M3_PATH is required for the enterprise import pipeline")
    return LazyLocalBgeM3Backend(
        Path(model_path),
        dense_dimension=settings.embedding.dense_dimension,
        device=settings.embedding.device,
        use_fp16=settings.embedding.fp16,
        model_batch_size=settings.embedding.model_batch_size,
        max_tokens=settings.embedding.max_tokens,
    )


def build_default_milvus_runtime(
    settings: Settings,
    dense_dimension: int,
) -> ManagedMilvusRuntime:
    milvus = settings.milvus
    return build_milvus_runtime_from_settings(
        settings,
        dense_dimension=dense_dimension,
        allow_create=milvus.allow_schema_create,
        token=milvus.token,
        user=milvus.username,
        password=milvus.password,
        database=milvus.database,
        timeout_seconds=milvus.timeout_seconds,
        close_wait_seconds=settings.import_runtime.shutdown_grace_seconds,
        max_query_limit=max(
            milvus.max_chunks_per_import + 1,
            settings.search.max_candidates,
        ),
    )


def _validate_pipeline_settings(settings: Settings) -> None:
    settings.validate_capabilities("milvus", "mineru")
    model_path = settings.embedding.model_path
    if not isinstance(model_path, str) or not model_path or not Path(model_path).is_absolute():
        raise ValueError("BGE_M3_PATH must be an absolute local model directory")
    if settings.mineru.max_request_bytes < settings.api.max_upload_bytes:
        raise ValueError("MINERU_MAX_REQUEST_BYTES must cover API_MAX_UPLOAD_BYTES")


def _close_partial(closers: Sequence[Callable[[], None]]) -> None:
    for closer in reversed(closers):
        try:
            closer()
        except Exception as close_error:
            logger.bind(error_type=type(close_error).__name__).warning(
                "Partial import processor dependency cleanup failed"
            )


def build_enterprise_import_processor(
    settings: Settings,
    *,
    parse_node_builder: ParseNodeBuilder = build_document_parse_node,
    embedding_backend_builder: EmbeddingBackendBuilder = build_local_embedding_backend,
    milvus_runtime_builder: MilvusRuntimeBuilder = build_default_milvus_runtime,
    manage_embedding_backend: bool = True,
    manage_milvus_runtime: bool = True,
) -> GraphImportProcessor:
    """Compose the production-safe import graph without dependency I/O."""

    if not isinstance(settings, Settings):
        raise ValueError("application settings are required")
    if not isinstance(manage_embedding_backend, bool) or not isinstance(
        manage_milvus_runtime,
        bool,
    ):
        raise ValueError("import dependency ownership policy is invalid")
    _validate_pipeline_settings(settings)
    closers: list[Callable[[], None]] = []
    try:
        parse_node = parse_node_builder(settings.mineru)
        closers.append(parse_node.close)
        backend = embedding_backend_builder(settings)
        embedding_node = HybridEmbeddingNode(
            backend,
            policy=HybridEmbeddingPolicy(
                batch_size=min(
                    settings.embedding.model_batch_size,
                    settings.milvus.max_chunks_per_import,
                ),
                max_chunks=settings.milvus.max_chunks_per_import,
            ),
        )
        if manage_embedding_backend:
            closers.append(embedding_node.close)
        runtime = milvus_runtime_builder(settings, backend.dense_dimension)
        if manage_milvus_runtime:
            closers.append(runtime.close)
        index = MilvusImportIndex(
            runtime,
            collection_name=settings.milvus.chunks_collection,
            pipeline_version=settings.milvus.pipeline_version,
            batch_size=settings.milvus.write_batch_size,
            max_chunks=settings.milvus.max_chunks_per_import,
        )

        def require_milvus_ready() -> None:
            if runtime.schema_ready:
                runtime.probe()
            else:
                runtime.ensure_schema()

        graph = build_import_graph(
            ImportGraphNodes(
                entry=ValidatedEntryNode(
                    max_source_bytes=settings.api.max_upload_bytes,
                ),
                parse_document=parse_node,
                extract_images=MarkdownImageTextNode(),
                split_document=MarkdownSplitterNode(
                    MarkdownSplitPolicy(
                        max_chunks=settings.milvus.max_chunks_per_import,
                    )
                ),
                recognize_item=SafeItemNameNode(),
                embed_chunks=embedding_node,
                commit_index=MilvusCommitNode(index),
            ),
            probes=(parse_node.probe, embedding_node.probe, require_milvus_ready),
            closers=tuple(closers),
            idempotent_commits=True,
            fenced_commits=True,
        )
        processor = GraphImportProcessor(graph)
    except BaseException:
        _close_partial(closers)
        raise
    return processor


__all__ = [
    "ManagedMilvusRuntime",
    "build_default_milvus_runtime",
    "build_enterprise_import_processor",
    "build_local_embedding_backend",
]
