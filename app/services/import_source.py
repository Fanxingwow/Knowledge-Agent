from __future__ import annotations

import hashlib
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import BinaryIO, Protocol

from app.core.errors import AppError, ErrorCode
from app.core.logger import logger
from app.infrastructure.minio_store import StoredObject
from app.services.import_workspace import ImportWorkspace
from app.services.upload_service import ALLOWED_SUFFIXES, CHUNK_SIZE


class ImportObjectReader(Protocol):
    def open_stored(
        self,
        tenant_id: str,
        stored: StoredObject,
    ) -> AbstractContextManager[BinaryIO]: ...


def _source_suffix(original_name: str) -> str:
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise AppError(
            ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
            "Import source type is unsupported",
            415,
        )
    return suffix


def materialize_import_source(
    store: ImportObjectReader,
    workspace: ImportWorkspace,
    *,
    task_id: str,
    tenant_id: str,
    source: StoredObject,
    original_name: str,
    expected_size: int,
    expected_sha256: str,
) -> tuple[Path, Path]:
    """Download one admitted object version into a new, server-owned task workspace."""
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise AppError(ErrorCode.INTERNAL_ERROR, "Import source metadata is invalid", 500)

    workspace.cleanup_task(task_id)
    task_dir = workspace.prepare(task_id)
    target = task_dir / f"source{_source_suffix(original_name)}"
    descriptor: int | None = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(target, flags, 0o600)
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = None
            with store.open_stored(tenant_id, source) as remote:
                while chunk := remote.read(CHUNK_SIZE):
                    if not isinstance(chunk, bytes):
                        raise AppError(
                            ErrorCode.DEPENDENCY_UNAVAILABLE,
                            "Object storage is unavailable",
                            503,
                        )
                    total += len(chunk)
                    if total > expected_size:
                        raise AppError(
                            ErrorCode.REQUEST_INVALID,
                            "Import source integrity check failed",
                            422,
                        )
                    digest.update(chunk)
                    destination.write(chunk)
        if total != expected_size or digest.hexdigest() != expected_sha256:
            raise AppError(
                ErrorCode.REQUEST_INVALID,
                "Import source integrity check failed",
                422,
            )
        return task_dir, target
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            workspace.cleanup_task(task_id)
        except Exception as cleanup_error:
            logger.bind(
                task_id=task_id,
                tenant_id=tenant_id,
                error_type=type(cleanup_error).__name__,
            ).warning("Import source workspace cleanup failed")
        raise
