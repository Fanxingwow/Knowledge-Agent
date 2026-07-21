from __future__ import annotations

import ntpath
import re
import uuid
from collections.abc import Mapping

from app.import_process.agent.nodes.safe_node_contracts import (
    SafeNodeContractError,
    normalize_unicode,
    require_metadata,
    require_safe_text,
)
from app.import_process.agent.state import ImportGraphState

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_MARKDOWN_TYPES = frozenset({"text/markdown", "text/plain"})


def _canonical_uuid(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise SafeNodeContractError(f"invalid {field}")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise SafeNodeContractError(f"invalid {field}") from None
    if str(parsed) != value:
        raise SafeNodeContractError(f"invalid {field}")
    return value


class ValidatedEntryNode:
    """Validate immutable import metadata and deliberately return no update.

    Filesystem ownership and content integrity remain the parser's responsibility.
    This node performs no I/O and cannot make caller-controlled data authoritative.
    """

    def __init__(self, *, max_source_bytes: int = 100 * 1024 * 1024) -> None:
        if isinstance(max_source_bytes, bool) or not isinstance(max_source_bytes, int) or max_source_bytes <= 0:
            raise ValueError("invalid entry source limit")
        self._max_source_bytes = max_source_bytes

    def __call__(self, state: ImportGraphState) -> Mapping[str, object]:
        _canonical_uuid(state.get("task_id"), field="task identity")
        _canonical_uuid(state.get("document_id"), field="document identity")

        tenant_id = state.get("tenant_id")
        if not isinstance(tenant_id, str) or _TENANT_ID.fullmatch(tenant_id) is None:
            raise SafeNodeContractError("invalid tenant identity")
        require_safe_text(
            state.get("owner_subject"),
            field="owner identity",
            allow_empty=False,
            multiline=False,
            max_characters=512,
            max_bytes=2_048,
        )
        require_safe_text(
            state.get("source_ref"),
            field="source reference",
            allow_empty=False,
            multiline=False,
            max_characters=2_048,
            max_bytes=8_192,
        )

        workspace = state.get("workspace_dir")
        source = state.get("source_path")
        if (
            not isinstance(workspace, str)
            or not workspace
            or not isinstance(source, str)
            or not source
            or state.get("local_dir") != workspace
            or state.get("local_file_path") != source
        ):
            raise SafeNodeContractError("invalid server source aliases")

        original_name = require_metadata(
            state.get("original_name"),
            field="original name",
            allow_empty=False,
            max_characters=255,
            max_bytes=1_024,
        )
        if ntpath.basename(original_name) != original_name:
            raise SafeNodeContractError("invalid original name")
        suffix = ntpath.splitext(original_name)[1].lower()
        file_title = require_metadata(
            state.get("file_title"),
            field="file title",
            allow_empty=False,
        )
        if file_title != normalize_unicode(ntpath.splitext(original_name)[0]):
            raise SafeNodeContractError("invalid file title")

        kind = state.get("source_kind")
        content_type = state.get("content_type")
        pdf = kind == "pdf"
        markdown = kind == "markdown"
        if (
            (pdf == markdown)
            or state.get("is_pdf_read_enabled") is not pdf
            or state.get("is_md_read_enabled") is not markdown
            or (pdf and (suffix != ".pdf" or content_type != "application/pdf"))
            or (markdown and (suffix not in _MARKDOWN_SUFFIXES or content_type not in _MARKDOWN_TYPES))
            or state.get("pdf_path") != (source if pdf else "")
            or state.get("md_path") != (source if markdown else "")
        ):
            raise SafeNodeContractError("invalid source type metadata")

        size = state.get("size")
        digest = state.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size > self._max_source_bytes
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise SafeNodeContractError("invalid source integrity metadata")

        if (
            state.get("md_content") != ""
            or state.get("chunks") != []
            or state.get("item_name") != ""
            or state.get("embeddings_content") != []
            or "commit_receipt" in state
        ):
            raise SafeNodeContractError("entry state contains derived data")
        return {}


__all__ = ["ValidatedEntryNode"]
