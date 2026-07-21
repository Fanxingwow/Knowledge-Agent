from __future__ import annotations

import codecs
import hashlib
import ntpath
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import anyio
from starlette.datastructures import UploadFile

from app.core.errors import AppError, ErrorCode

CHUNK_SIZE = 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".md", ".markdown"}
ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf"},
    ".md": {"text/markdown", "text/plain"},
    ".markdown": {"text/markdown", "text/plain"},
}
MARKDOWN_SUFFIXES = {".md", ".markdown"}
ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
FileIdentity = tuple[int, int]


@dataclass(frozen=True, slots=True)
class StoredUpload:
    original_name: str
    stored_name: str
    path: Path
    content_type: str
    size: int
    sha256: str
    task_root: Path
    identity: FileIdentity

    def open_verified(self) -> BinaryIO:
        """Open the exact file created by ``store_upload`` without trusting its path again."""
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except OSError:
            raise AppError(
                ErrorCode.INTERNAL_ERROR,
                "Stored upload is unavailable",
                500,
            ) from None

        try:
            opened_identity = _file_identity_from_descriptor(descriptor)
            current_identity = _file_identity_from_path(self.path)
            resolved_path = self.path.resolve(strict=True)
            resolved_root = self.task_root.resolve(strict=True)
            _require_containment(resolved_path, resolved_root)
            _require_same_file(self.identity, opened_identity)
            _require_same_file(opened_identity, current_identity)
            if os.fstat(descriptor).st_size != self.size:
                raise AppError(
                    ErrorCode.REQUEST_INVALID,
                    "Stored upload changed before persistence",
                    422,
                )
            return os.fdopen(descriptor, "rb")
        except AppError:
            os.close(descriptor)
            raise
        except (OSError, ValueError):
            os.close(descriptor)
            raise AppError(
                ErrorCode.INTERNAL_ERROR,
                "Stored upload is unavailable",
                500,
            ) from None

    def delete_owned(self) -> None:
        """Delete only when the path still names the exact file this service created."""
        try:
            resolved_path = self.path.resolve(strict=False)
            resolved_root = self.task_root.resolve(strict=False)
            _require_containment(resolved_path, resolved_root)
            if _file_identity_from_path(self.path) == self.identity:
                self.path.unlink(missing_ok=True)
        except (AppError, OSError):
            return


def validate_upload_count(files: Sequence[object], max_files: int) -> None:
    if not files or len(files) > max_files:
        raise AppError(
            ErrorCode.TOO_MANY_FILES,
            f"Upload accepts 1 to {max_files} files",
            422,
        )


def _safe_basename(filename: str | None) -> str:
    return ntpath.basename(filename or "unnamed") or "unnamed"


async def _resolved_path(path: Path) -> Path:
    return Path(await anyio.Path(path).resolve(strict=False))


def _require_containment(path: Path, root: Path) -> None:
    if not path.is_relative_to(root):
        raise AppError(ErrorCode.REQUEST_INVALID, "Invalid upload path", 422)


def _file_identity_from_descriptor(descriptor: int) -> FileIdentity:
    stat = os.fstat(descriptor)
    return stat.st_dev, stat.st_ino


def _file_identity_from_path(path: Path) -> FileIdentity | None:
    try:
        stat = os.lstat(path)
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino


def _require_same_file(opened: FileIdentity, current: FileIdentity | None) -> None:
    if opened != current:
        raise AppError(ErrorCode.REQUEST_INVALID, "Invalid upload path", 422)


async def _remove_owned_target(target: Path, owned_identity: FileIdentity) -> None:
    try:
        current_identity = await anyio.to_thread.run_sync(_file_identity_from_path, target)
        if current_identity == owned_identity:
            await anyio.Path(target).unlink(missing_ok=True)
    except OSError:
        return


async def store_upload(upload: UploadFile, task_dir: Path, max_bytes: int) -> StoredUpload:
    target: Path | None = None
    owned_identity: FileIdentity | None = None
    primary_error: BaseException | None = None

    try:
        original_name = _safe_basename(upload.filename)
        if len(original_name) > 255 or any(
            ord(character) < 32 or ord(character) == 127 for character in original_name
        ):
            raise AppError(
                ErrorCode.REQUEST_INVALID,
                "Invalid document name",
                422,
            )
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise AppError(
                ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
                "Only PDF and Markdown are supported",
                415,
            )

        content_type = (upload.content_type or "").lower()
        if content_type not in ALLOWED_CONTENT_TYPES[suffix]:
            raise AppError(
                ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
                "File content type does not match its extension",
                415,
            )

        root = await _resolved_path(task_dir)
        await anyio.Path(root).mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex}{suffix}"
        target = root / stored_name
        _require_containment(await _resolved_path(target), root)

        digest = hashlib.sha256()
        total = 0
        header = b""
        decoder = codecs.getincrementaldecoder("utf-8")() if suffix in MARKDOWN_SUFFIXES else None

        try:
            with anyio.CancelScope(shield=True):
                destination = await anyio.open_file(target, "xb")
                try:
                    acquired_identity = await anyio.to_thread.run_sync(
                        _file_identity_from_descriptor,
                        destination.wrapped.fileno(),
                    )
                    owned_identity = acquired_identity
                except BaseException as identity_error:
                    try:
                        await destination.aclose()
                    except BaseException as descriptor_close_error:
                        identity_error.add_note(
                            f"Descriptor close also failed: {type(descriptor_close_error).__name__}"
                        )
                    raise

            destination_error: BaseException | None = None
            try:
                _require_containment(await _resolved_path(target), root)
                current_identity = await anyio.to_thread.run_sync(_file_identity_from_path, target)
                _require_same_file(acquired_identity, current_identity)
                while chunk := await upload.read(CHUNK_SIZE):
                    total += len(chunk)
                    if total > max_bytes:
                        raise AppError(
                            ErrorCode.FILE_TOO_LARGE,
                            "File exceeds configured size limit",
                            413,
                        )

                    if len(header) < 5:
                        header += chunk[: 5 - len(header)]
                    if decoder is not None:
                        if b"\x00" in chunk:
                            raise AppError(
                                ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
                                "Markdown must be text",
                                415,
                            )
                        decoder.decode(chunk, final=False)

                    digest.update(chunk)
                    await destination.write(chunk)

                if header.startswith(ZIP_SIGNATURES):
                    raise AppError(
                        ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
                        "Archive content is not an accepted document",
                        415,
                    )
                if suffix == ".pdf" and not header.startswith(b"%PDF-"):
                    raise AppError(
                        ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
                        "Invalid PDF signature",
                        415,
                    )
                if decoder is not None:
                    decoder.decode(b"", final=True)
                if total == 0:
                    raise AppError(
                        ErrorCode.REQUEST_INVALID,
                        "Uploaded document must not be empty",
                        422,
                    )
            except BaseException as exc:
                destination_error = exc
                raise
            finally:
                with anyio.CancelScope(shield=True):
                    try:
                        await destination.aclose()
                    except BaseException:
                        if destination_error is None:
                            raise
        except UnicodeDecodeError as exc:
            raise AppError(
                ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
                "Markdown must be UTF-8",
                415,
            ) from exc

        if owned_identity is None:
            raise AppError(ErrorCode.INTERNAL_ERROR, "Stored upload is unavailable", 500)
        return StoredUpload(
            original_name=original_name,
            stored_name=stored_name,
            path=target,
            content_type=content_type,
            size=total,
            sha256=digest.hexdigest(),
            task_root=root,
            identity=owned_identity,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close_error: BaseException | None = None
        with anyio.CancelScope(shield=True):
            try:
                await upload.close()
            except BaseException as exc:
                close_error = exc
            if (
                (primary_error is not None or close_error is not None)
                and target is not None
                and owned_identity is not None
            ):
                await _remove_owned_target(target, owned_identity)
        if primary_error is None and close_error is not None:
            raise close_error
