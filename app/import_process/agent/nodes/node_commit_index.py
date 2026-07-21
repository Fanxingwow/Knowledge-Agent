from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from app.core.errors import AppError, DependencyError, ErrorCode
from app.import_process.agent.state import ImportGraphState
from app.infrastructure.milvus_import_index import (
    CommitFenceError,
    ImportVersionConflict,
    IndexVerificationError,
    IndexWriteUncertain,
    MilvusChunk,
    MilvusCommitReceipt,
)
from app.services.import_fence import CommitFence


class MilvusImportWriter(Protocol):
    def stage(
        self,
        *,
        tenant_id: str,
        document_id: str,
        import_version: str,
        chunks: Sequence[MilvusChunk],
        fence: CommitFence,
    ) -> MilvusCommitReceipt: ...


def _metadata(chunk: Mapping[str, object], state: ImportGraphState, field: str) -> object:
    value = chunk.get(field)
    if value is None and field in {"file_title", "item_name"}:
        value = state.get(field, "")
    return "" if value is None else value


def _milvus_chunks(state: ImportGraphState) -> tuple[MilvusChunk, ...]:
    raw_chunks = state.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("import graph produced no indexable chunks")
    chunks: list[MilvusChunk] = []
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, Mapping):
            raise ValueError("import graph produced an invalid chunk")
        content = raw_chunk.get("content")
        dense = raw_chunk.get("dense_vector")
        sparse = raw_chunk.get("sparse_vector")
        if (
            not isinstance(content, str)
            or isinstance(dense, (str, bytes, bytearray))
            or not isinstance(dense, Sequence)
            or not isinstance(sparse, Mapping)
        ):
            raise ValueError("import graph produced an invalid chunk")
        chunks.append(
            MilvusChunk(
                content=content,
                dense_vector=cast(Sequence[float], tuple(dense)),
                sparse_vector=cast(Mapping[int, float], dict(sparse)),
                title=cast(str, _metadata(raw_chunk, state, "title")),
                parent_title=cast(str, _metadata(raw_chunk, state, "parent_title")),
                file_title=cast(str, _metadata(raw_chunk, state, "file_title")),
                item_name=cast(str, _metadata(raw_chunk, state, "item_name")),
            )
        )
    return tuple(chunks)


class MilvusCommitNode:
    """Commit graph chunks through the tenant/version scoped staging writer."""

    idempotent_commits = True
    fenced_commits = True

    def __init__(self, index: MilvusImportWriter) -> None:
        self._index = index

    def __call__(self, state: ImportGraphState) -> Mapping[str, object]:
        fence = state.get("commit_fence")
        if fence is None:
            raise ValueError("index commit requires a durable commit fence")
        chunks = _milvus_chunks(state)
        try:
            receipt = self._index.stage(
                tenant_id=state["tenant_id"],
                document_id=state["document_id"],
                import_version=state["task_id"],
                chunks=chunks,
                fence=fence,
            )
        except CommitFenceError:
            raise AppError(
                ErrorCode.RESOURCE_CONFLICT,
                "Import commit ownership changed",
                409,
            ) from None
        except IndexWriteUncertain:
            raise DependencyError("Import index is temporarily unavailable") from None
        except (ImportVersionConflict, IndexVerificationError):
            raise AppError(
                ErrorCode.INTERNAL_ERROR,
                "Import index verification failed",
                500,
            ) from None
        return {"commit_receipt": _graph_receipt(receipt)}


def _graph_receipt(receipt: MilvusCommitReceipt) -> dict[str, object]:
    return {
        "task_id": receipt.task_id,
        "document_id": receipt.document_id,
        "tenant_id": receipt.tenant_id,
        "imported_chunk_count": receipt.imported_chunk_count,
        "commit_token": receipt.commit_token,
    }
