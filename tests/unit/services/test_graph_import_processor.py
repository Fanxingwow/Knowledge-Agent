from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from app.core.errors import AppError, DependencyError, ErrorCode
from app.import_process.agent.main_graph import (
    COMMIT_NODE,
    EMBED_NODE,
    ENTRY_NODE,
    IMAGES_NODE,
    PARSE_NODE,
    RECOGNIZE_NODE,
    SPLIT_NODE,
    GraphNodeUpdate,
    ImportGraphNodes,
    build_import_graph,
)
from app.import_process.agent.state import ImportGraphState
from app.services.graph_import_processor import GraphImportProcessor
from app.services.import_fence import CommitFence
from app.services.import_runner import ImportCommitEvent, ImportJob, ImportStepEvent


def _job(tmp_path: Path, *, markdown: bool = False, original_name: str | None = None) -> ImportJob:
    task_hex = "11111111111141118111111111111111"
    workspace = tmp_path / "server-workspace" / task_hex[:2] / task_hex
    workspace.mkdir(parents=True)
    (workspace / ".import-workspace-owner-v1").mkdir()
    suffix = ".md" if markdown else ".pdf"
    source = workspace / f"source{suffix}"
    source.write_bytes(b"safe-source")
    return ImportJob(
        task_id="11111111-1111-4111-8111-111111111111",
        document_id="22222222-2222-4222-8222-222222222222",
        tenant_id="tenant-a",
        owner_subject="user-a",
        local_dir=workspace,
        local_file_path=source,
        source_ref="tenants/tenant-a/imports/object/version",
        original_name=original_name or f"manual{suffix}",
        content_type="text/markdown" if markdown else "application/pdf",
        size=len(b"safe-source"),
        sha256="a" * 64,
    )


def _processor(captured: list[tuple[str, str, str]]) -> GraphImportProcessor:
    def ordinary(state: ImportGraphState) -> dict[str, object]:
        captured.append((state["tenant_id"], state["document_id"], state["task_id"]))
        return {}

    def commit(state: ImportGraphState) -> dict[str, object]:
        captured.append((state["tenant_id"], state["document_id"], state["task_id"]))
        return {
            "commit_receipt": {
                "task_id": state["task_id"],
                "document_id": state["document_id"],
                "tenant_id": state["tenant_id"],
                "imported_chunk_count": 4,
                "commit_token": "milvus-commit-proof",
            }
        }

    graph = build_import_graph(
        ImportGraphNodes(
            entry=ordinary,
            parse_document=ordinary,
            extract_images=ordinary,
            split_document=ordinary,
            recognize_item=ordinary,
            embed_chunks=ordinary,
            commit_index=commit,
        )
    )
    return GraphImportProcessor(graph)


@pytest.mark.parametrize(
    ("markdown", "expected_steps"),
    [
        (
            False,
            [
                "parse_document",
                "extract_images",
                "split_document",
                "recognize_item",
                "embed_chunks",
            ],
        ),
        (
            True,
            [
                "parse_document",
                "extract_images",
                "split_document",
                "recognize_item",
                "embed_chunks",
            ],
        ),
    ],
)
def test_processor_streams_standard_steps_and_typed_commit(
    tmp_path: Path,
    markdown: bool,
    expected_steps: list[str],
) -> None:
    captured: list[tuple[str, str, str]] = []
    processor = _processor(captured)

    events = list(processor.stream(_job(tmp_path, markdown=markdown)))

    assert [event.step for event in events if isinstance(event, ImportStepEvent)] == expected_steps
    commit_events = [event for event in events if isinstance(event, ImportCommitEvent)]
    assert len(commit_events) == 1
    assert commit_events[0].receipt.tenant_id == "tenant-a"
    assert commit_events[0].receipt.document_id == "22222222-2222-4222-8222-222222222222"
    assert commit_events[0].receipt.task_id == "11111111-1111-4111-8111-111111111111"
    assert commit_events[0].receipt.imported_chunk_count == 4
    assert all(
        identity
        == (
            "tenant-a",
            "22222222-2222-4222-8222-222222222222",
            "11111111-1111-4111-8111-111111111111",
        )
        for identity in captured
    )


def test_processor_does_not_claim_unproven_idempotent_commits(tmp_path: Path) -> None:
    processor = _processor([])

    assert processor.idempotent_commits is False
    assert isinstance(list(processor.stream(_job(tmp_path)))[-1], ImportCommitEvent)


class _ProbeFailureGraph:
    def probe(self) -> None:
        raise RuntimeError(
            "mongodb://root:secret@internal:27017/admin"  # pragma: allowlist secret
        )

    def stream(self, state: ImportGraphState) -> Iterable[GraphNodeUpdate]:
        raise AssertionError("not used")


def test_probe_failure_is_sanitized() -> None:
    processor = GraphImportProcessor(_ProbeFailureGraph())

    with pytest.raises(DependencyError) as captured:
        processor.probe()

    assert captured.value.message == "Import graph dependencies are unavailable"
    assert captured.value.details == {}
    assert "secret" not in str(captured.value)
    assert captured.value.__cause__ is None


class _CloseFailureGraph:
    def probe(self) -> None:
        return

    def stream(self, state: ImportGraphState) -> Iterable[GraphNodeUpdate]:
        raise AssertionError(state)

    def close(self) -> None:
        raise RuntimeError("https://secret.invalid/token")


def test_close_failure_is_sanitized() -> None:
    processor = GraphImportProcessor(_CloseFailureGraph())

    with pytest.raises(DependencyError) as captured:
        processor.close()

    assert captured.value.message == "Import graph dependencies could not be closed"
    assert "secret" not in str(captured.value)


def test_processor_rejects_original_name_with_client_path(tmp_path: Path) -> None:
    processor = _processor([])

    with pytest.raises(AppError) as captured:
        list(processor.stream(_job(tmp_path, original_name="../../private.pdf")))

    assert captured.value.code is ErrorCode.REQUEST_INVALID


def test_processor_rejects_source_outside_server_workspace(tmp_path: Path) -> None:
    processor = _processor([])
    job = _job(tmp_path)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"safe-source")
    unsafe_job = ImportJob(
        task_id=job.task_id,
        document_id=job.document_id,
        tenant_id=job.tenant_id,
        owner_subject=job.owner_subject,
        local_dir=job.local_dir,
        local_file_path=outside,
        source_ref=job.source_ref,
        original_name=job.original_name,
        content_type=job.content_type,
        size=job.size,
        sha256=job.sha256,
    )

    with pytest.raises(AppError) as captured:
        list(processor.stream(unsafe_job))

    assert captured.value.code is ErrorCode.REQUEST_INVALID


def test_processor_requires_server_workspace_ownership_marker(tmp_path: Path) -> None:
    processor = _processor([])
    job = _job(tmp_path)
    (job.local_dir / ".import-workspace-owner-v1").rmdir()

    with pytest.raises(AppError) as captured:
        list(processor.stream(job))

    assert captured.value.code is ErrorCode.REQUEST_INVALID


class _IncompleteGraph:
    def probe(self) -> None:
        return

    def stream(self, state: ImportGraphState) -> Iterable[GraphNodeUpdate]:
        yield GraphNodeUpdate(ENTRY_NODE, {})
        yield GraphNodeUpdate(PARSE_NODE, {})
        yield GraphNodeUpdate(IMAGES_NODE, {})
        yield GraphNodeUpdate(SPLIT_NODE, {})
        yield GraphNodeUpdate(RECOGNIZE_NODE, {})
        yield GraphNodeUpdate(EMBED_NODE, {})


def test_processor_fails_closed_without_commit_receipt(tmp_path: Path) -> None:
    processor = GraphImportProcessor(_IncompleteGraph())

    with pytest.raises(ValueError, match="before a proven commit"):
        list(processor.stream(_job(tmp_path)))


class _OutOfOrderGraph:
    def probe(self) -> None:
        return

    def stream(self, state: ImportGraphState) -> Iterable[GraphNodeUpdate]:
        yield GraphNodeUpdate(ENTRY_NODE, {})
        yield GraphNodeUpdate(SPLIT_NODE, {})
        yield GraphNodeUpdate(COMMIT_NODE, {})


def test_processor_rejects_out_of_order_graph_updates(tmp_path: Path) -> None:
    processor = GraphImportProcessor(_OutOfOrderGraph())

    with pytest.raises(ValueError, match="out of order"):
        list(processor.stream(_job(tmp_path)))


class _WrongTenantReceiptGraph:
    def probe(self) -> None:
        return

    def stream(self, state: ImportGraphState) -> Iterable[GraphNodeUpdate]:
        for node in (
            ENTRY_NODE,
            PARSE_NODE,
            IMAGES_NODE,
            SPLIT_NODE,
            RECOGNIZE_NODE,
            EMBED_NODE,
        ):
            yield GraphNodeUpdate(node, {})
        yield GraphNodeUpdate(
            COMMIT_NODE,
            {
                "commit_receipt": {
                    "task_id": state["task_id"],
                    "document_id": state["document_id"],
                    "tenant_id": "tenant-b",
                    "imported_chunk_count": 1,
                    "commit_token": "cross-tenant-receipt",
                }
            },
        )


def test_processor_revalidates_commit_identity_at_adapter_boundary(tmp_path: Path) -> None:
    processor = GraphImportProcessor(_WrongTenantReceiptGraph())

    with pytest.raises(ValueError, match="does not match the job"):
        list(processor.stream(_job(tmp_path)))


class _Fence:
    task_id = "11111111-1111-4111-8111-111111111111"
    document_id = "22222222-2222-4222-8222-222222222222"
    import_version = task_id
    attempt_count = 1
    fence_id = "fence-1"

    def __init__(self, *, tenant_id: str = "tenant-a") -> None:
        self.tenant_id = tenant_id
        self.assertions = 0

    def assert_current(self) -> None:
        self.assertions += 1

    def __deepcopy__(self, memo: dict[int, object]) -> _Fence:
        del memo
        return self


def test_fenced_graph_requires_and_propagates_durable_commit_fence(tmp_path: Path) -> None:
    observed_fences: list[CommitFence] = []

    def ordinary(state: ImportGraphState) -> dict[str, object]:
        return {}

    def commit(state: ImportGraphState) -> dict[str, object]:
        observed_fences.append(state["commit_fence"])
        return {
            "commit_receipt": {
                "task_id": state["task_id"],
                "document_id": state["document_id"],
                "tenant_id": state["tenant_id"],
                "imported_chunk_count": 1,
                "commit_token": "milvus-v1." + "a" * 64,
            }
        }

    graph = build_import_graph(
        ImportGraphNodes(
            entry=ordinary,
            parse_document=ordinary,
            extract_images=ordinary,
            split_document=ordinary,
            recognize_item=ordinary,
            embed_chunks=ordinary,
            commit_index=commit,
        ),
        idempotent_commits=True,
        fenced_commits=True,
    )
    processor = GraphImportProcessor(graph)
    fence = _Fence()
    job = _job(tmp_path)

    assert processor.idempotent_commits is True
    assert processor.fenced_commits is True
    with pytest.raises(ValueError, match="requires a durable commit fence"):
        list(processor.stream(job))

    events = list(processor.stream_fenced(job, fence))

    assert isinstance(events[-1], ImportCommitEvent)
    assert observed_fences == [fence]
    assert fence.assertions >= 1


def test_fenced_graph_rejects_mismatched_fence_before_graph_execution(tmp_path: Path) -> None:
    calls: list[str] = []

    def node(state: ImportGraphState) -> dict[str, object]:
        del state
        calls.append("called")
        return {}

    graph = build_import_graph(
        ImportGraphNodes(
            entry=node,
            parse_document=node,
            extract_images=node,
            split_document=node,
            recognize_item=node,
            embed_chunks=node,
            commit_index=node,
        ),
        idempotent_commits=True,
        fenced_commits=True,
    )
    processor = GraphImportProcessor(graph)
    wrong = _Fence(tenant_id="tenant-b")

    with pytest.raises(ValueError, match="does not match"):
        list(processor.stream_fenced(_job(tmp_path), wrong))

    assert calls == []
