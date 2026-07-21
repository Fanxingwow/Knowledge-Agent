from __future__ import annotations

from collections.abc import Mapping

import pytest

from app.import_process.agent import main_graph
from app.import_process.agent.main_graph import (
    COMMIT_NODE,
    EMBED_NODE,
    ENTRY_NODE,
    IMAGES_NODE,
    PARSE_NODE,
    RECOGNIZE_NODE,
    SPLIT_NODE,
    ImportGraphContractError,
    ImportGraphNode,
    ImportGraphNodes,
    build_import_graph,
)
from app.import_process.agent.state import ImportGraphState, create_default_state


def _state(*, pdf: bool = True) -> ImportGraphState:
    source = "C:/server/work/source.pdf" if pdf else "C:/server/work/source.md"
    return create_default_state(
        task_id="11111111-1111-4111-8111-111111111111",
        document_id="22222222-2222-4222-8222-222222222222",
        tenant_id="tenant-a",
        owner_subject="user-a",
        workspace_dir="C:/server/work",
        source_path=source,
        local_dir="C:/server/work",
        local_file_path=source,
        source_ref="tenants/tenant-a/source",
        original_name="source.pdf" if pdf else "source.md",
        content_type="application/pdf" if pdf else "text/markdown",
        size=10,
        sha256="a" * 64,
        source_kind="pdf" if pdf else "markdown",
        is_pdf_read_enabled=pdf,
        is_md_read_enabled=not pdf,
    )


def _nodes(
    calls: list[tuple[str, str, str, str]],
    *,
    replacement: Mapping[str, object] | None = None,
    replacement_node: str = "",
) -> ImportGraphNodes:
    def node(name: str) -> ImportGraphNode:
        def execute(state: ImportGraphState) -> Mapping[str, object]:
            calls.append(
                (name, state["tenant_id"], state["document_id"], state["task_id"])
            )
            if name == replacement_node:
                return dict(replacement or {})
            if name == COMMIT_NODE:
                return {
                    "commit_receipt": {
                        "task_id": state["task_id"],
                        "document_id": state["document_id"],
                        "tenant_id": state["tenant_id"],
                        "imported_chunk_count": 3,
                        "commit_token": "milvus-proof-1",
                    }
                }
            return {}

        return execute

    return ImportGraphNodes(
        entry=node(ENTRY_NODE),
        parse_document=node(PARSE_NODE),
        extract_images=node(IMAGES_NODE),
        split_document=node(SPLIT_NODE),
        recognize_item=node(RECOGNIZE_NODE),
        embed_chunks=node(EMBED_NODE),
        commit_index=node(COMMIT_NODE),
    )


def test_importing_main_graph_does_not_create_a_global_graph() -> None:
    assert not hasattr(main_graph, "workflow")
    assert not hasattr(main_graph, "kb_import_app")


@pytest.mark.parametrize(
    ("pdf", "expected"),
    [
        (
            True,
            [
                ENTRY_NODE,
                PARSE_NODE,
                IMAGES_NODE,
                SPLIT_NODE,
                RECOGNIZE_NODE,
                EMBED_NODE,
                COMMIT_NODE,
            ],
        ),
        (
            False,
            [
                ENTRY_NODE,
                PARSE_NODE,
                IMAGES_NODE,
                SPLIT_NODE,
                RECOGNIZE_NODE,
                EMBED_NODE,
                COMMIT_NODE,
            ],
        ),
    ],
)
def test_factory_routes_injected_nodes_and_carries_job_identity(
    pdf: bool,
    expected: list[str],
) -> None:
    calls: list[tuple[str, str, str, str]] = []
    graph = build_import_graph(_nodes(calls))

    updates = list(graph.stream(_state(pdf=pdf)))

    assert [update.node for update in updates] == expected
    assert [call[0] for call in calls] == expected
    assert all(
        call[1:] == (
            "tenant-a",
            "22222222-2222-4222-8222-222222222222",
            "11111111-1111-4111-8111-111111111111",
        )
        for call in calls
    )


def test_graph_accepts_tenants_allowed_by_the_auth_contract() -> None:
    calls: list[tuple[str, str, str, str]] = []
    state = _state()
    state["tenant_id"] = "_tenant-a"

    updates = list(build_import_graph(_nodes(calls)).stream(state))

    assert updates[-1].node == COMMIT_NODE
    assert all(call[1] == "_tenant-a" for call in calls)


def test_factory_construction_does_not_run_nodes_or_probes() -> None:
    calls: list[tuple[str, str, str, str]] = []
    probe_calls: list[str] = []
    graph = build_import_graph(_nodes(calls), probes=[lambda: probe_calls.append("probe")])

    assert calls == []
    assert probe_calls == []

    graph.probe()
    assert probe_calls == ["probe"]


def test_graph_close_runs_all_closers_in_reverse_and_sanitizes_failure() -> None:
    calls: list[str] = []

    def failing() -> None:
        calls.append("first")
        raise RuntimeError("secret dependency endpoint")

    graph = build_import_graph(
        _nodes([]),
        closers=[failing, lambda: calls.append("second")],
    )

    with pytest.raises(RuntimeError, match="dependency close failed") as captured:
        graph.close()

    assert calls == ["second", "first"]
    assert "secret" not in str(captured.value)


def test_node_cannot_replace_tenant_identity() -> None:
    calls: list[tuple[str, str, str, str]] = []
    graph = build_import_graph(
        _nodes(
            calls,
            replacement={"tenant_id": "tenant-b"},
            replacement_node=SPLIT_NODE,
        )
    )

    with pytest.raises(ImportGraphContractError, match="immutable field tenant_id"):
        list(graph.stream(_state()))


def test_only_commit_node_may_emit_a_receipt() -> None:
    calls: list[tuple[str, str, str, str]] = []
    graph = build_import_graph(
        _nodes(
            calls,
            replacement={
                "commit_receipt": {
                    "task_id": "11111111-1111-4111-8111-111111111111",
                    "document_id": "22222222-2222-4222-8222-222222222222",
                    "tenant_id": "tenant-a",
                    "imported_chunk_count": 3,
                    "commit_token": "not-yet-committed",
                }
            },
            replacement_node=EMBED_NODE,
        )
    )

    with pytest.raises(ImportGraphContractError, match="only the commit node"):
        list(graph.stream(_state()))


def test_commit_receipt_must_match_all_job_identity_fields() -> None:
    calls: list[tuple[str, str, str, str]] = []
    graph = build_import_graph(
        _nodes(
            calls,
            replacement={
                "commit_receipt": {
                    "task_id": "11111111-1111-4111-8111-111111111111",
                    "document_id": "22222222-2222-4222-8222-222222222222",
                    "tenant_id": "tenant-b",
                    "imported_chunk_count": 3,
                    "commit_token": "wrong-tenant",
                }
            },
            replacement_node=COMMIT_NODE,
        )
    )

    with pytest.raises(ImportGraphContractError, match="identity does not match"):
        list(graph.stream(_state()))


def test_factory_rejects_ambiguous_source_reader_state() -> None:
    calls: list[tuple[str, str, str, str]] = []
    state = _state()
    state["is_md_read_enabled"] = True

    with pytest.raises(ImportGraphContractError, match="exactly one source reader"):
        list(build_import_graph(_nodes(calls)).stream(state))
