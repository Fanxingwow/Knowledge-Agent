from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from langgraph.graph import END, StateGraph

from app.import_process.agent.state import GraphCommitReceipt, ImportGraphState

ENTRY_NODE = "entry"
PARSE_NODE = "parse_document"
IMAGES_NODE = "extract_images"
SPLIT_NODE = "split_document"
RECOGNIZE_NODE = "recognize_item"
EMBED_NODE = "embed_chunks"
COMMIT_NODE = "commit_index"

IMPORT_GRAPH_NODES = frozenset(
    {
        ENTRY_NODE,
        PARSE_NODE,
        IMAGES_NODE,
        SPLIT_NODE,
        RECOGNIZE_NODE,
        EMBED_NODE,
        COMMIT_NODE,
    }
)

_IDENTITY_FIELDS = ("task_id", "document_id", "tenant_id")
_SOURCE_FIELDS = ("workspace_dir", "source_path", "local_dir", "local_file_path")
_CAPABILITY_FIELDS = ("commit_fence",)
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class ImportGraphContractError(ValueError):
    """Raised when an injected node violates the import graph contract."""


class ImportGraphNode(Protocol):
    def __call__(self, state: ImportGraphState) -> Mapping[str, object]: ...


class DependencyProbe(Protocol):
    def __call__(self) -> None: ...


class _CompiledGraph(Protocol):
    def stream(
        self,
        state: ImportGraphState,
        *,
        stream_mode: str,
    ) -> Iterable[Mapping[str, object]]: ...


@dataclass(frozen=True, slots=True)
class ImportGraphNodes:
    """All side-effecting graph behavior, supplied explicitly by composition."""

    entry: ImportGraphNode
    parse_document: ImportGraphNode
    extract_images: ImportGraphNode
    split_document: ImportGraphNode
    recognize_item: ImportGraphNode
    embed_chunks: ImportGraphNode
    commit_index: ImportGraphNode


@dataclass(frozen=True, slots=True)
class GraphNodeUpdate:
    node: str
    values: Mapping[str, object]


def _require_identifier(state: Mapping[str, object], field: str) -> str:
    value = state.get(field)
    pattern = _TENANT_ID if field == "tenant_id" else _SAFE_IDENTIFIER
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ImportGraphContractError(f"invalid {field}")
    return value


def _validate_initial_state(state: Mapping[str, object]) -> None:
    for field in _IDENTITY_FIELDS:
        _require_identifier(state, field)

    workspace = state.get("workspace_dir")
    source = state.get("source_path")
    if not isinstance(workspace, str) or not workspace:
        raise ImportGraphContractError("invalid workspace_dir")
    if not isinstance(source, str) or not source:
        raise ImportGraphContractError("invalid source_path")
    if state.get("local_dir") != workspace or state.get("local_file_path") != source:
        raise ImportGraphContractError("source aliases do not match the server workspace")

    pdf_enabled = state.get("is_pdf_read_enabled") is True
    markdown_enabled = state.get("is_md_read_enabled") is True
    if pdf_enabled == markdown_enabled:
        raise ImportGraphContractError("exactly one source reader must be enabled")


def _validate_immutable_fields(
    before: Mapping[str, object],
    update: Mapping[str, object],
) -> None:
    for field in (*_IDENTITY_FIELDS, *_SOURCE_FIELDS, *_CAPABILITY_FIELDS):
        if field in update and update[field] != before.get(field):
            raise ImportGraphContractError(f"node attempted to replace immutable field {field}")


def _validate_commit_receipt(
    state: Mapping[str, object],
    receipt: object,
) -> GraphCommitReceipt:
    if not isinstance(receipt, Mapping):
        raise ImportGraphContractError("commit node did not return a typed receipt")

    expected_keys = {
        "task_id",
        "document_id",
        "tenant_id",
        "imported_chunk_count",
        "commit_token",
    }
    if set(receipt) != expected_keys:
        raise ImportGraphContractError("commit receipt shape is invalid")
    for field in _IDENTITY_FIELDS:
        if receipt.get(field) != state.get(field):
            raise ImportGraphContractError("commit receipt identity does not match the job")

    count = receipt.get("imported_chunk_count")
    token = receipt.get("commit_token")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ImportGraphContractError("commit receipt count is invalid")
    if not isinstance(token, str) or not token or len(token) > 256 or not token.isascii() or not token.isprintable():
        raise ImportGraphContractError("commit receipt token is invalid")
    return cast(GraphCommitReceipt, dict(receipt))


def _guard_node(name: str, node: ImportGraphNode) -> ImportGraphNode:
    def guarded(state: ImportGraphState) -> Mapping[str, object]:
        _validate_initial_state(state)
        isolated_state = copy.deepcopy(state)
        result = node(isolated_state)
        if not isinstance(result, Mapping):
            raise ImportGraphContractError(f"{name} returned an invalid update")
        update = dict(result)
        _validate_immutable_fields(state, update)
        if name == COMMIT_NODE:
            update["commit_receipt"] = _validate_commit_receipt(
                state,
                update.get("commit_receipt"),
            )
        elif "commit_receipt" in update:
            raise ImportGraphContractError("only the commit node may return a receipt")
        return update

    return guarded


class ImportGraphRuntime:
    """Small typed boundary around a compiled LangGraph instance."""

    def __init__(
        self,
        compiled: _CompiledGraph,
        *,
        probes: Sequence[DependencyProbe] = (),
        closers: Sequence[Callable[[], None]] = (),
        idempotent_commits: bool = False,
        fenced_commits: bool = False,
    ) -> None:
        if fenced_commits and not idempotent_commits:
            raise ValueError("fenced commits require idempotent commits")
        self._compiled = compiled
        self._probes = tuple(probes)
        self._closers = tuple(closers)
        self.idempotent_commits = idempotent_commits
        self.fenced_commits = fenced_commits

    def probe(self) -> None:
        for probe in self._probes:
            probe()

    def close(self) -> None:
        failed = False
        for closer in reversed(self._closers):
            try:
                closer()
            except Exception:
                failed = True
        if failed:
            raise RuntimeError("Import graph dependency close failed") from None

    def stream(self, state: ImportGraphState) -> Iterable[GraphNodeUpdate]:
        _validate_initial_state(state)
        for raw_update in self._compiled.stream(
            copy.deepcopy(state),
            stream_mode="updates",
        ):
            if not isinstance(raw_update, Mapping) or len(raw_update) != 1:
                raise ImportGraphContractError("graph emitted an invalid node update")
            node, values = next(iter(raw_update.items()))
            if node not in IMPORT_GRAPH_NODES:
                raise ImportGraphContractError("graph emitted an unknown node update")
            if values is None:
                normalized_values: Mapping[str, object] = {}
            elif isinstance(values, Mapping):
                normalized_values = values
            else:
                raise ImportGraphContractError("graph emitted an invalid node update")
            _validate_immutable_fields(state, normalized_values)
            yield GraphNodeUpdate(node=node, values=dict(normalized_values))


def build_import_graph(
    nodes: ImportGraphNodes,
    *,
    probes: Sequence[DependencyProbe] = (),
    closers: Sequence[Callable[[], None]] = (),
    idempotent_commits: bool = False,
    fenced_commits: bool = False,
) -> ImportGraphRuntime:
    """Build an import graph without connecting to any external dependency.

    The factory deliberately has no implicit legacy-node fallback.  Existing
    migration nodes write to shared MinIO/Milvus resources and cannot prove a
    tenant-scoped idempotent commit, so production composition must inject
    hardened implementations explicitly.
    """

    workflow = StateGraph(ImportGraphState)
    workflow.add_node(ENTRY_NODE, _guard_node(ENTRY_NODE, nodes.entry))
    workflow.add_node(PARSE_NODE, _guard_node(PARSE_NODE, nodes.parse_document))
    workflow.add_node(IMAGES_NODE, _guard_node(IMAGES_NODE, nodes.extract_images))
    workflow.add_node(SPLIT_NODE, _guard_node(SPLIT_NODE, nodes.split_document))
    workflow.add_node(RECOGNIZE_NODE, _guard_node(RECOGNIZE_NODE, nodes.recognize_item))
    workflow.add_node(EMBED_NODE, _guard_node(EMBED_NODE, nodes.embed_chunks))
    workflow.add_node(COMMIT_NODE, _guard_node(COMMIT_NODE, nodes.commit_index))

    workflow.set_entry_point(ENTRY_NODE)
    # Markdown also crosses the hardened parser boundary so strict UTF-8/NUL
    # checks happen before any downstream node consumes caller-controlled text.
    workflow.add_edge(ENTRY_NODE, PARSE_NODE)
    workflow.add_edge(PARSE_NODE, IMAGES_NODE)
    workflow.add_edge(IMAGES_NODE, SPLIT_NODE)
    workflow.add_edge(SPLIT_NODE, RECOGNIZE_NODE)
    workflow.add_edge(RECOGNIZE_NODE, EMBED_NODE)
    workflow.add_edge(EMBED_NODE, COMMIT_NODE)
    workflow.add_edge(COMMIT_NODE, END)

    compiled = cast(_CompiledGraph, cast(Any, workflow.compile()))
    return ImportGraphRuntime(
        compiled,
        probes=probes,
        closers=closers,
        idempotent_commits=idempotent_commits,
        fenced_commits=fenced_commits,
    )
