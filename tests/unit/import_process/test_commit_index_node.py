from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from app.core.errors import AppError, DependencyError, ErrorCode
from app.import_process.agent.nodes.node_commit_index import MilvusCommitNode
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.infrastructure.milvus_import_index import (
    CommitFenceError,
    ImportVersionConflict,
    IndexWriteUncertain,
    MilvusChunk,
    MilvusCommitReceipt,
)
from app.services.import_fence import CommitFence

_TASK_ID = "11111111-1111-4111-8111-111111111111"
_DOCUMENT_ID = "22222222-2222-4222-8222-222222222222"


@dataclass(slots=True)
class FakeFence:
    task_id: str = _TASK_ID
    document_id: str = _DOCUMENT_ID
    tenant_id: str = "tenant-a"
    import_version: str = _TASK_ID
    attempt_count: int = 1
    fence_id: str = "fence-1"

    def assert_current(self) -> None:
        return


class FakeIndex:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    def stage(
        self,
        *,
        tenant_id: str,
        document_id: str,
        import_version: str,
        chunks: Sequence[MilvusChunk],
        fence: CommitFence,
    ) -> MilvusCommitReceipt:
        kwargs: dict[str, object] = {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "import_version": import_version,
            "chunks": chunks,
            "fence": fence,
        }
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return MilvusCommitReceipt(
            task_id=_TASK_ID,
            document_id=_DOCUMENT_ID,
            tenant_id="tenant-a",
            import_version=_TASK_ID,
            pipeline_version="bge-m3.v1",
            imported_chunk_count=len(chunks),
            manifest_sha256="a" * 64,
            commit_token="milvus-v1." + "b" * 64,
            fence_id="fence-1",
            attempt_count=1,
        )


def _state(*, fence: FakeFence | None = None) -> ImportGraphState:
    state = create_default_state(
        task_id=_TASK_ID,
        document_id=_DOCUMENT_ID,
        tenant_id="tenant-a",
        owner_subject="owner-a",
        workspace_dir="C:/safe/workspace",
        source_path="C:/safe/workspace/source.md",
        source_ref="tenants/tenant-a/imports/version",
        original_name="source.md",
        content_type="text/markdown",
        size=10,
        sha256="a" * 64,
        source_kind="markdown",
        is_md_read_enabled=True,
        is_pdf_read_enabled=False,
        local_dir="C:/safe/workspace",
        local_file_path="C:/safe/workspace/source.md",
        file_title="Guide",
        item_name="Product",
        chunks=[
            {
                "content": "first",
                "title": "Section",
                "dense_vector": [0.1, 0.2],
                "sparse_vector": {1: 0.5},
            }
        ],
    )
    if fence is not None:
        state["commit_fence"] = fence
    return state


def test_commit_node_maps_fixed_chunk_fields_and_returns_graph_receipt() -> None:
    index = FakeIndex()
    fence = FakeFence()

    update = MilvusCommitNode(index)(_state(fence=fence))

    assert len(index.calls) == 1
    call = index.calls[0]
    assert call["tenant_id"] == "tenant-a"
    assert call["document_id"] == _DOCUMENT_ID
    assert call["import_version"] == _TASK_ID
    assert call["fence"] is fence
    chunks = call["chunks"]
    assert isinstance(chunks, tuple)
    assert chunks == (
        MilvusChunk(
            content="first",
            dense_vector=(0.1, 0.2),
            sparse_vector={1: 0.5},
            title="Section",
            file_title="Guide",
            item_name="Product",
        ),
    )
    assert update == {
        "commit_receipt": {
            "task_id": _TASK_ID,
            "document_id": _DOCUMENT_ID,
            "tenant_id": "tenant-a",
            "imported_chunk_count": 1,
            "commit_token": "milvus-v1." + "b" * 64,
        }
    }


def test_commit_node_requires_fence_before_index_access() -> None:
    index = FakeIndex()

    with pytest.raises(ValueError, match="durable commit fence"):
        MilvusCommitNode(index)(_state())

    assert index.calls == []


@pytest.mark.parametrize(
    ("failure", "expected_type", "expected_code"),
    [
        (CommitFenceError("secret lease"), AppError, ErrorCode.RESOURCE_CONFLICT),
        (IndexWriteUncertain("secret endpoint"), DependencyError, ErrorCode.DEPENDENCY_UNAVAILABLE),
        (ImportVersionConflict("secret payload"), AppError, ErrorCode.INTERNAL_ERROR),
    ],
)
def test_commit_node_sanitizes_index_failures(
    failure: Exception,
    expected_type: type[AppError],
    expected_code: ErrorCode,
) -> None:
    index = FakeIndex(failure=failure)

    with pytest.raises(expected_type) as captured:
        MilvusCommitNode(index)(_state(fence=FakeFence()))

    assert captured.value.code is expected_code
    assert "secret" not in str(captured.value)
