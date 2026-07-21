from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from app.core.errors import AppError, ErrorCode
from app.core.logger import logger
from app.core.settings import MineruSettings
from app.import_process.agent.state import ImportGraphState
from app.infrastructure.mineru_client import (
    MineruParseArtifact,
    MineruTransport,
    build_mineru_client,
)

_WORKSPACE_OWNER_MARKER = ".import-workspace-owner-v1"
_PDF_CONTENT_TYPE = "application/pdf"
_MARKDOWN_CONTENT_TYPES = frozenset({"text/markdown", "text/plain"})
_READ_CHUNK_BYTES = 1024 * 1024


class PdfDocumentParser(Protocol):
    @property
    def max_request_bytes(self) -> int: ...

    def parse_pdf(self, source: Path, *, source_sha256: str) -> MineruParseArtifact: ...

    def probe(self) -> None: ...

    def close(self) -> None: ...


def _invalid_source() -> AppError:
    return AppError(ErrorCode.REQUEST_INVALID, "Import source is invalid", 422)


def _unsafe_workspace() -> AppError:
    return AppError(ErrorCode.INTERNAL_ERROR, "Import workspace is invalid", 500)


def _is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _require_text(state: Mapping[str, object], field: str) -> str:
    value = state.get(field)
    if not isinstance(value, str) or not value:
        raise _invalid_source()
    return value


def _server_source(state: Mapping[str, object]) -> tuple[Path, Path, str]:
    kind = _require_text(state, "source_kind")
    if kind not in {"pdf", "markdown"}:
        raise _invalid_source()
    expected_name = "source.pdf" if kind == "pdf" else "source.md"
    content_type = _require_text(state, "content_type").lower()
    if (
        (kind == "pdf" and content_type != _PDF_CONTENT_TYPE)
        or (kind == "markdown" and content_type not in _MARKDOWN_CONTENT_TYPES)
        or state.get("is_pdf_read_enabled") is not (kind == "pdf")
        or state.get("is_md_read_enabled") is not (kind == "markdown")
    ):
        raise _invalid_source()

    try:
        task_id = _require_text(state, "task_id")
        parsed_task_id = uuid.UUID(task_id)
        if str(parsed_task_id) != task_id:
            raise ValueError("non-canonical task identifier")
        workspace_candidate = Path(_require_text(state, "workspace_dir"))
        source_candidate = Path(_require_text(state, "source_path"))
        if _is_link_or_reparse(workspace_candidate) or not workspace_candidate.is_dir():
            raise OSError("unsafe workspace")
        if (
            workspace_candidate.name != parsed_task_id.hex
            or workspace_candidate.parent.name != parsed_task_id.hex[:2]
        ):
            raise OSError("unowned workspace")
        marker = workspace_candidate / _WORKSPACE_OWNER_MARKER
        if (
            _is_link_or_reparse(marker)
            or not marker.is_dir()
            or next(marker.iterdir(), None) is not None
        ):
            raise OSError("workspace marker is invalid")
        if _is_link_or_reparse(source_candidate) or not source_candidate.is_file():
            raise OSError("unsafe source")
        workspace = workspace_candidate.resolve(strict=True)
        source = source_candidate.resolve(strict=True)
        if (
            source.parent != workspace
            or not source.is_relative_to(workspace)
            or source.name != expected_name
        ):
            raise OSError("source escaped workspace")
    except (OSError, RuntimeError, ValueError):
        raise _invalid_source() from None
    return workspace, source, kind


def _read_and_digest(
    source: Path,
    *,
    maximum: int,
    retain_content: bool,
) -> tuple[bytes, int, str]:
    digest = hashlib.sha256()
    content = bytearray()
    total = 0
    try:
        with source.open("rb") as stream:
            while chunk := stream.read(_READ_CHUNK_BYTES):
                total += len(chunk)
                if total > maximum:
                    raise _invalid_source()
                digest.update(chunk)
                if retain_content:
                    content.extend(chunk)
    except OSError:
        raise _invalid_source() from None
    if total <= 0:
        raise _invalid_source()
    return bytes(content), total, digest.hexdigest()


def _validate_source_metadata(
    state: Mapping[str, object],
    source: Path,
    *,
    maximum: int,
    retain_content: bool,
) -> tuple[bytes, str]:
    content, total, digest = _read_and_digest(
        source,
        maximum=maximum,
        retain_content=retain_content,
    )
    expected_size = state.get("size")
    expected_digest = state.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size != total
        or not isinstance(expected_digest, str)
        or expected_digest != digest
    ):
        raise _invalid_source()
    return content, digest


def _write_private(path: Path, content: bytes) -> None:
    descriptor: int | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise _unsafe_workspace() from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _remove_temporary(path: Path) -> bool:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        logger.bind(error_type=type(exc).__name__).warning(
            "Document parser temporary workspace cleanup failed"
        )
        return False
    return True


class DocumentParseNode:
    """Parse one fixed, worker-owned source without returning sensitive paths."""

    def __init__(self, parser: PdfDocumentParser) -> None:
        self._parser = parser

    def probe(self) -> None:
        self._parser.probe()

    def close(self) -> None:
        self._parser.close()

    def __call__(self, state: ImportGraphState) -> Mapping[str, object]:
        workspace, source, kind = _server_source(state)
        if kind == "markdown":
            raw_content, _ = _validate_source_metadata(
                state,
                source,
                maximum=self._parser.max_request_bytes,
                retain_content=True,
            )
            try:
                markdown = raw_content.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                raise _invalid_source() from None
            if "\x00" in markdown:
                raise _invalid_source()
            return {"md_content": markdown, "md_path": "source.md"}

        _, source_sha256 = _validate_source_metadata(
            state,
            source,
            maximum=self._parser.max_request_bytes,
            retain_content=False,
        )
        temporary = Path(tempfile.mkdtemp(prefix=".mineru-", dir=workspace))
        try:
            artifact = self._parser.parse_pdf(source, source_sha256=source_sha256)
            markdown_bytes = artifact.markdown.encode("utf-8")
            temporary_artifact = temporary / "parsed.md"
            _write_private(temporary_artifact, markdown_bytes)
            target = workspace / "parsed.md"
            try:
                if target.exists() and (
                    _is_link_or_reparse(target) or not target.is_file()
                ):
                    raise _unsafe_workspace()
                os.replace(temporary_artifact, target)
            except OSError:
                raise _unsafe_workspace() from None
            return {"md_content": artifact.markdown, "md_path": "parsed.md"}
        finally:
            cleaned = _remove_temporary(temporary)
            if not cleaned and sys.exc_info()[0] is None:
                raise _unsafe_workspace()


def build_document_parse_node(
    settings: MineruSettings,
    *,
    transport: MineruTransport | None = None,
) -> DocumentParseNode:
    """Build an injectable graph node without probing MinerU or touching disk."""

    return DocumentParseNode(build_mineru_client(settings, transport=transport))


__all__ = ["DocumentParseNode", "PdfDocumentParser", "build_document_parse_node"]
