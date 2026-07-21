from __future__ import annotations

import copy
from typing import Any, NotRequired, Required, TypedDict, cast

from app.services.import_fence import CommitFence


class GraphCommitReceipt(TypedDict):
    """Proof returned by an index writer after a durable commit.

    ``commit_token`` is intentionally opaque.  A graph adapter validates only the
    receipt shape and job identity; the concrete index writer remains responsible
    for proving that the token represents an idempotent, fenced commit.
    """

    task_id: str
    document_id: str
    tenant_id: str
    imported_chunk_count: int
    commit_token: str


class ImportGraphState(TypedDict, total=False):
    """Tenant-aware state shared by import graph nodes.

    Identity and source fields are immutable for the lifetime of one invocation.
    The graph factory enforces that invariant around every injected node.

    ``local_dir`` and ``local_file_path`` are compatibility aliases for the
    existing migration nodes.  They are populated only from the worker-created
    server workspace; callers must never put a client supplied path in them.
    """

    task_id: Required[str]
    document_id: Required[str]
    tenant_id: Required[str]
    owner_subject: Required[str]
    workspace_dir: Required[str]
    source_path: Required[str]
    source_ref: Required[str]
    original_name: Required[str]
    content_type: Required[str]
    size: Required[int]
    sha256: Required[str]

    source_kind: Required[str]
    is_md_read_enabled: Required[bool]
    is_pdf_read_enabled: Required[bool]

    local_dir: Required[str]
    local_file_path: Required[str]
    file_title: Required[str]
    pdf_path: Required[str]
    md_path: Required[str]
    split_path: Required[str]
    embeddings_path: Required[str]

    is_normal_split_enabled: Required[bool]
    is_silicon_flow_api_enabled: Required[bool]
    is_advanced_split_enabled: Required[bool]
    is_vllm_enabled: Required[bool]

    md_content: Required[str]
    chunks: Required[list[dict[str, Any]]]
    item_name: Required[str]
    embeddings_content: Required[list[dict[str, Any]]]

    commit_fence: NotRequired[CommitFence]
    commit_receipt: NotRequired[GraphCommitReceipt]


graph_default_state: ImportGraphState = {
    "task_id": "",
    "document_id": "",
    "tenant_id": "",
    "owner_subject": "",
    "workspace_dir": "",
    "source_path": "",
    "source_ref": "",
    "original_name": "",
    "content_type": "",
    "size": 0,
    "sha256": "",
    "source_kind": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": [],
}


def create_default_state(**overrides: Any) -> ImportGraphState:
    """Return an isolated graph state with optional migration-node overrides."""

    return cast(
        ImportGraphState,
        {**copy.deepcopy(graph_default_state), **overrides},
    )


def get_default_state() -> ImportGraphState:
    """Return an isolated copy so graph invocations never share mutable values."""

    return copy.deepcopy(graph_default_state)
