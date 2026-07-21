from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.core.settings import (
    ApiSettings,
    EmbeddingSettings,
    ImportRuntimeSettings,
    MilvusSettings,
    MineruSettings,
    Settings,
)
from app.import_process.agent.nodes.node_hybrid_embedding import HybridEmbedding
from app.import_process.agent.state import ImportGraphState
from app.services.import_fence import CommitFence
from app.services.import_processor_factory import build_enterprise_import_processor
from app.services.import_runner import ImportCommitEvent, ImportJob, ImportStepEvent

_TASK_ID = "11111111-1111-4111-8111-111111111111"
_DOCUMENT_ID = "22222222-2222-4222-8222-222222222222"


class FakeParseNode:
    def __init__(self, closed: list[str]) -> None:
        self.probe_calls = 0
        self.closed = closed

    def __call__(self, state: ImportGraphState) -> Mapping[str, object]:
        assert state["source_kind"] == "markdown"
        return {
            "md_content": "# Guide\n\nThis is enterprise-safe content.",
            "md_path": "source.md",
        }

    def probe(self) -> None:
        self.probe_calls += 1

    def close(self) -> None:
        self.closed.append("parse")


class FakeEmbeddingBackend:
    dense_dimension = 2

    def __init__(self, closed: list[str]) -> None:
        self.probe_calls = 0
        self.closed = closed
        self.embed_calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> Sequence[HybridEmbedding]:
        copied = tuple(texts)
        self.embed_calls.append(copied)
        return tuple(
            HybridEmbedding(dense=(0.1, 0.2), sparse={1: 0.5})
            for _ in copied
        )

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
        self.records: dict[str, dict[str, object]] = {}

    def ensure_schema(self) -> object:
        self.ensure_calls += 1
        self.schema_ready = True
        return object()

    def probe(self) -> None:
        self.probe_calls += 1
        if not self.schema_ready:
            raise RuntimeError("schema is not ready")

    def query(
        self,
        *,
        collection_name: str,
        filter: str,
        output_fields: list[str],
        limit: int,
        consistency_level: str,
    ) -> Sequence[Mapping[str, object]]:
        assert collection_name == "kb_chunks_v2"
        assert filter
        assert consistency_level == "Strong"
        return tuple(
            {field: record[field] for field in output_fields}
            for record in self.records.values()
        )[:limit]

    def upsert(
        self,
        *,
        collection_name: str,
        data: list[dict[str, object]],
    ) -> object:
        assert collection_name == "kb_chunks_v2"
        for record in data:
            self.records[str(record["chunk_id"])] = dict(record)
        return {"upsert_count": len(data)}

    def delete(self, *, collection_name: str, filter: str) -> object:
        raise AssertionError((collection_name, filter))

    def close(self) -> None:
        self.closed.append("milvus")


class FakeFence:
    task_id = _TASK_ID
    document_id = _DOCUMENT_ID
    tenant_id = "tenant-a"
    import_version = _TASK_ID
    attempt_count = 1
    fence_id = "fence-1"

    def __init__(self) -> None:
        self.assertions = 0

    def assert_current(self) -> None:
        self.assertions += 1

    def __deepcopy__(self, memo: dict[int, object]) -> FakeFence:
        del memo
        return self


def _settings(tmp_path: Path) -> Settings:
    model_path = tmp_path / "models" / "bge-m3"
    return Settings.model_construct(
        api=ApiSettings.model_construct(
            max_upload_bytes=1024 * 1024,
        ),
        import_runtime=ImportRuntimeSettings.model_construct(
            shutdown_grace_seconds=1.0,
        ),
        embedding=EmbeddingSettings.model_construct(
            model_path=str(model_path),
            model_id="BAAI/bge-m3",
            device="cpu",
            fp16=False,
            dense_dimension=2,
            model_batch_size=4,
            max_tokens=512,
        ),
        milvus=MilvusSettings.model_construct(
            url="https://milvus.example",
            token=None,
            username=None,
            password=None,
            database=None,
            chunks_collection="kb_chunks_v2",
            item_name_collection="kb_item_names_v2",
            entity_name_collection=None,
            pipeline_version="bge-m3.v1",
            allow_schema_create=False,
            write_batch_size=4,
            max_chunks_per_import=100,
            timeout_seconds=1.0,
        ),
        mineru=MineruSettings.model_construct(
            base_url="https://mineru.example",
            api_key=SecretStr("test-token"),
            max_request_bytes=1024 * 1024,
        ),
    )


def _job(tmp_path: Path) -> ImportJob:
    task_hex = _TASK_ID.replace("-", "")
    workspace = tmp_path / "work" / task_hex[:2] / task_hex
    workspace.mkdir(parents=True)
    (workspace / ".import-workspace-owner-v1").mkdir()
    source = workspace / "source.md"
    body = b"source-body"
    source.write_bytes(body)
    return ImportJob(
        task_id=_TASK_ID,
        document_id=_DOCUMENT_ID,
        tenant_id="tenant-a",
        owner_subject="owner-a",
        local_dir=workspace,
        local_file_path=source,
        source_ref="minio://tenant-a/documents/source?versionId=v1",
        original_name="Guide.md",
        content_type="text/markdown",
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )


def _processor(tmp_path: Path):
    closed: list[str] = []
    parse = FakeParseNode(closed)
    backend = FakeEmbeddingBackend(closed)
    runtime = FakeMilvusRuntime(closed)
    processor = build_enterprise_import_processor(
        _settings(tmp_path),
        parse_node_builder=lambda _: parse,
        embedding_backend_builder=lambda _: backend,
        milvus_runtime_builder=lambda _, dimension: (
            runtime if dimension == 2 else pytest.fail("wrong dimension")
        ),
    )
    return processor, parse, backend, runtime, closed


def test_factory_is_lazy_and_manages_probe_and_close_order(tmp_path: Path) -> None:
    processor, parse, backend, runtime, closed = _processor(tmp_path)

    assert processor.idempotent_commits is True
    assert processor.fenced_commits is True
    assert parse.probe_calls == backend.probe_calls == runtime.ensure_calls == 0

    processor.probe()
    processor.probe()

    assert parse.probe_calls == 2
    assert backend.probe_calls == 2
    assert runtime.ensure_calls == 1
    assert runtime.probe_calls == 1
    processor.close()
    assert closed == ["milvus", "embedding", "parse"]


def test_factory_can_borrow_shared_embedding_and_milvus_dependencies(tmp_path: Path) -> None:
    closed: list[str] = []
    parse = FakeParseNode(closed)
    backend = FakeEmbeddingBackend(closed)
    runtime = FakeMilvusRuntime(closed)
    processor = build_enterprise_import_processor(
        _settings(tmp_path),
        parse_node_builder=lambda _: parse,
        embedding_backend_builder=lambda _: backend,
        milvus_runtime_builder=lambda _settings, _dimension: runtime,
        manage_embedding_backend=False,
        manage_milvus_runtime=False,
    )

    processor.close()

    assert closed == ["parse"]


def test_factory_graph_commits_markdown_with_fence_and_fixed_schema(tmp_path: Path) -> None:
    processor, _, backend, runtime, _ = _processor(tmp_path)
    processor.probe()
    fence: CommitFence = FakeFence()

    events = list(processor.stream_fenced(_job(tmp_path), fence))

    assert [
        event.step for event in events if isinstance(event, ImportStepEvent)
    ] == [
        "parse_document",
        "extract_images",
        "split_document",
        "recognize_item",
        "embed_chunks",
    ]
    commits = [event for event in events if isinstance(event, ImportCommitEvent)]
    assert len(commits) == 1
    assert commits[0].receipt.task_id == _TASK_ID
    assert commits[0].receipt.document_id == _DOCUMENT_ID
    assert commits[0].receipt.tenant_id == "tenant-a"
    assert commits[0].receipt.imported_chunk_count >= 1
    assert commits[0].receipt.commit_token.startswith("milvus-v1.")
    assert backend.embed_calls
    assert runtime.records
    assert all(record["tenant_id"] == "tenant-a" for record in runtime.records.values())
    assert all(
        record["commit_token"] == commits[0].receipt.commit_token
        for record in runtime.records.values()
    )


def test_factory_rejects_relative_model_path_before_building_dependencies(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.embedding.model_path = "relative/model"
    calls: list[str] = []

    with pytest.raises(ValueError, match="absolute"):
        build_enterprise_import_processor(
            settings,
            parse_node_builder=lambda _: calls.append("parse"),  # type: ignore[arg-type,return-value]
        )

    assert calls == []
