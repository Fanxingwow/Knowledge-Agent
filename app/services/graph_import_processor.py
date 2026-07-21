from __future__ import annotations

import ntpath
import re
import stat
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

from app.core.errors import AppError, DependencyError, ErrorCode
from app.core.logger import logger
from app.import_process.agent.main_graph import (
    COMMIT_NODE,
    EMBED_NODE,
    ENTRY_NODE,
    IMAGES_NODE,
    PARSE_NODE,
    RECOGNIZE_NODE,
    SPLIT_NODE,
    GraphNodeUpdate,
)
from app.import_process.agent.state import GraphCommitReceipt, ImportGraphState, create_default_state
from app.services.import_fence import CommitFence
from app.services.import_runner import (
    ImportCommitEvent,
    ImportCommitReceipt,
    ImportJob,
    ImportStepEvent,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_TOKEN = re.compile(r"^[!-~]{1,256}$")
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
_MARKDOWN_CONTENT_TYPES = frozenset({"text/markdown", "text/plain"})
_WORKSPACE_OWNER_MARKER = ".import-workspace-owner-v1"


class ImportGraphExecution(Protocol):
    def probe(self) -> None: ...

    def stream(self, state: ImportGraphState) -> Iterable[GraphNodeUpdate]: ...


def _is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _invalid_source() -> AppError:
    return AppError(ErrorCode.REQUEST_INVALID, "Import source is invalid", 422)


def _safe_original_name(value: str) -> str:
    if (
        not value
        or len(value) > 255
        or ntpath.basename(value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise _invalid_source()
    return value


def _source_kind(original_name: str, content_type: str) -> str:
    suffix = Path(original_name).suffix.lower()
    normalized_type = content_type.lower()
    if suffix == ".pdf" and normalized_type == "application/pdf":
        return "pdf"
    if suffix in _MARKDOWN_SUFFIXES and normalized_type in _MARKDOWN_CONTENT_TYPES:
        return "markdown"
    raise AppError(
        ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
        "Import source type is unsupported",
        415,
    )


def _server_owned_source(job: ImportJob, *, suffix: str) -> tuple[Path, Path]:
    try:
        parsed_task_id = uuid.UUID(job.task_id)
        if str(parsed_task_id) != job.task_id:
            raise ValueError("non-canonical task identifier")
        workspace_candidate = Path(job.local_dir)
        source_candidate = Path(job.local_file_path)
        if _is_link_or_reparse(workspace_candidate) or not workspace_candidate.is_dir():
            raise OSError("unsafe workspace")
        if (
            workspace_candidate.name != parsed_task_id.hex
            or workspace_candidate.parent.name != parsed_task_id.hex[:2]
        ):
            raise OSError("workspace does not belong to task")
        marker = workspace_candidate / _WORKSPACE_OWNER_MARKER
        if (
            _is_link_or_reparse(marker)
            or not marker.is_dir()
            or next(marker.iterdir(), None) is not None
        ):
            raise OSError("workspace ownership is not proven")
        if _is_link_or_reparse(source_candidate) or not source_candidate.is_file():
            raise OSError("unsafe source")
        workspace = workspace_candidate.resolve(strict=True)
        source = source_candidate.resolve(strict=True)
        if (
            source.parent != workspace
            or not source.is_relative_to(workspace)
            or source.name != f"source{suffix}"
        ):
            raise OSError("source is outside its workspace")
        source_info = source.stat()
    except (OSError, RuntimeError, ValueError):
        raise _invalid_source() from None
    if source_info.st_size != job.size:
        raise _invalid_source()
    return workspace, source


def _initial_state(job: ImportJob) -> ImportGraphState:
    original_name = _safe_original_name(job.original_name)
    kind = _source_kind(original_name, job.content_type)
    suffix = Path(original_name).suffix.lower()
    if (
        isinstance(job.size, bool)
        or not isinstance(job.size, int)
        or job.size <= 0
        or _SHA256.fullmatch(job.sha256) is None
        or not job.owner_subject
        or not job.source_ref
    ):
        raise _invalid_source()
    workspace, source = _server_owned_source(job, suffix=suffix)
    source_text = str(source)
    workspace_text = str(workspace)
    return create_default_state(
        task_id=job.task_id,
        document_id=job.document_id,
        tenant_id=job.tenant_id,
        owner_subject=job.owner_subject,
        workspace_dir=workspace_text,
        source_path=source_text,
        source_ref=job.source_ref,
        original_name=original_name,
        content_type=job.content_type.lower(),
        size=job.size,
        sha256=job.sha256,
        source_kind=kind,
        local_dir=workspace_text,
        local_file_path=source_text,
        file_title=Path(original_name).stem,
        is_pdf_read_enabled=kind == "pdf",
        is_md_read_enabled=kind == "markdown",
        pdf_path=source_text if kind == "pdf" else "",
        md_path=source_text if kind == "markdown" else "",
    )


def _receipt_from_update(job: ImportJob, update: GraphNodeUpdate) -> ImportCommitReceipt:
    receipt = update.values.get("commit_receipt")
    expected_keys = {
        "task_id",
        "document_id",
        "tenant_id",
        "imported_chunk_count",
        "commit_token",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ValueError("import graph did not provide a commit receipt")
    typed_receipt = cast(GraphCommitReceipt, receipt)
    if (
        typed_receipt["task_id"] != job.task_id
        or typed_receipt["document_id"] != job.document_id
        or typed_receipt["tenant_id"] != job.tenant_id
    ):
        raise ValueError("import graph commit receipt does not match the job")
    count = typed_receipt["imported_chunk_count"]
    token = typed_receipt["commit_token"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or not isinstance(token, str)
        or _COMMIT_TOKEN.fullmatch(token) is None
    ):
        raise ValueError("import graph commit receipt is invalid")
    return ImportCommitReceipt(
        task_id=typed_receipt["task_id"],
        document_id=typed_receipt["document_id"],
        tenant_id=typed_receipt["tenant_id"],
        imported_chunk_count=typed_receipt["imported_chunk_count"],
        commit_token=typed_receipt["commit_token"],
    )


class GraphImportProcessor:
    """Adapt a tenant-aware import graph to the durable worker contract.

    Capabilities are inherited only from an explicitly composed graph runtime.
    Legacy graphs therefore remain fail-closed, while a graph backed by the
    tenant-scoped staging writer can require a durable lease fence end to end.
    """

    def __init__(self, graph: ImportGraphExecution) -> None:
        self._graph = graph
        self.idempotent_commits = getattr(graph, "idempotent_commits", False) is True
        self.fenced_commits = (
            self.idempotent_commits
            and getattr(graph, "fenced_commits", False) is True
        )

    def probe(self) -> None:
        try:
            self._graph.probe()
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).warning(
                "Import graph dependency probe failed"
            )
            raise DependencyError("Import graph dependencies are unavailable") from None

    def close(self) -> None:
        close = getattr(self._graph, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            raise DependencyError("Import graph dependencies could not be closed") from None

    def _stream_state(
        self,
        job: ImportJob,
        state: ImportGraphState,
    ) -> Iterable[ImportStepEvent | ImportCommitEvent]:
        expected = [
            ENTRY_NODE,
            PARSE_NODE,
            IMAGES_NODE,
            SPLIT_NODE,
            RECOGNIZE_NODE,
            EMBED_NODE,
            COMMIT_NODE,
        ]
        step_names = {
            PARSE_NODE: "parse_document",
            IMAGES_NODE: "extract_images",
            SPLIT_NODE: "split_document",
            RECOGNIZE_NODE: "recognize_item",
            EMBED_NODE: "embed_chunks",
        }

        observed: list[str] = []
        for update in self._graph.stream(state):
            observed.append(update.node)
            if observed != expected[: len(observed)]:
                raise ValueError("import graph emitted nodes out of order")
            if update.node == ENTRY_NODE:
                continue
            if update.node == COMMIT_NODE:
                yield ImportCommitEvent(receipt=_receipt_from_update(job, update))
                continue
            step = step_names.get(update.node)
            if step is None:
                raise ValueError("import graph emitted an unknown step")
            yield ImportStepEvent(step=step)

        if observed != expected:
            raise ValueError("import graph terminated before a proven commit")

    def stream(self, job: ImportJob) -> Iterable[ImportStepEvent | ImportCommitEvent]:
        if self.fenced_commits:
            raise ValueError("fenced import graph requires a durable commit fence")
        yield from self._stream_state(job, _initial_state(job))

    def stream_fenced(
        self,
        job: ImportJob,
        fence: CommitFence,
    ) -> Iterable[ImportStepEvent | ImportCommitEvent]:
        if not self.idempotent_commits or not self.fenced_commits:
            raise ValueError("import graph does not support fenced commits")
        try:
            identity_matches = (
                fence.task_id == job.task_id
                and fence.document_id == job.document_id
                and fence.tenant_id == job.tenant_id
                and fence.import_version == job.task_id
            )
        except Exception:
            identity_matches = False
        if not identity_matches:
            raise ValueError("import commit fence does not match the job")
        fence.assert_current()
        state = _initial_state(job)
        state["commit_fence"] = fence
        yield from self._stream_state(job, state)
