from __future__ import annotations

import builtins
import hashlib
import re
import threading
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import anyio
import pytest
from starlette.datastructures import Headers, UploadFile

from app.core.errors import AppError, ErrorCode
from app.services import upload_service
from app.services.upload_service import store_upload, validate_upload_count


def upload(
    name: str,
    body: bytes,
    content_type: str,
    *,
    content_length: int | None = None,
) -> UploadFile:
    values = {"content-type": content_type}
    if content_length is not None:
        values["content-length"] = str(content_length)
    return UploadFile(BytesIO(body), filename=name, headers=Headers(values))


def resolved(path: Path) -> Path:
    return path.resolve()


def directory_entries(path: Path) -> list[Path]:
    return list(path.iterdir())


def read_and_remove(path: Path) -> bytes:
    body = path.read_bytes()
    path.unlink()
    return body


def replace_owned_target(target: Path, moved: Path) -> None:
    target.replace(moved)
    target.write_bytes(b"replacement")


class ReadFailureUpload(UploadFile):
    def __init__(self, body: bytes) -> None:
        super().__init__(
            BytesIO(body),
            filename="failure.md",
            headers=Headers({"content-type": "text/markdown"}),
        )
        self.read_calls = 0
        self.close_called = False

    async def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        if self.read_calls == 2:
            raise OSError("simulated read failure")
        return await super().read(size)

    async def close(self) -> None:
        self.close_called = True
        await super().close()


class CancellableUpload(UploadFile):
    def __init__(self) -> None:
        super().__init__(
            BytesIO(),
            filename="cancel.md",
            headers=Headers({"content-type": "text/markdown"}),
        )
        self.read_calls = 0
        self.second_read_started = anyio.Event()
        self.read_cancellation: BaseException | None = None
        self.close_called = False

    async def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        if self.read_calls == 1:
            return b"partial"
        self.second_read_started.set()
        try:
            await anyio.sleep_forever()
        except BaseException as exc:
            self.read_cancellation = exc
            raise
        raise AssertionError("sleep_forever returned")

    async def close(self) -> None:
        self.close_called = True
        await anyio.lowlevel.checkpoint()
        await super().close()


class CloseFailureUpload(UploadFile):
    def __init__(self, name: str, body: bytes, content_type: str) -> None:
        super().__init__(
            BytesIO(body),
            filename=name,
            headers=Headers({"content-type": content_type}),
        )
        self.close_called = False

    async def close(self) -> None:
        self.close_called = True
        await super().close()
        raise RuntimeError("simulated close failure")


class ReplacingCloseUpload(UploadFile):
    def __init__(self, target: Path, moved: Path) -> None:
        super().__init__(
            BytesIO(b"not-a-pdf"),
            filename="x.pdf",
            headers=Headers({"content-type": "application/pdf"}),
        )
        self.target = target
        self.moved = moved
        self.close_called = False

    async def close(self) -> None:
        self.close_called = True
        replace_owned_target(self.target, self.moved)
        await super().close()
        raise RuntimeError("simulated replacement close failure")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_name", "safe_name"),
    [
        ("../../escape.pdf", "escape.pdf"),
        (r"..\..\escape.pdf", "escape.pdf"),
        ("/private/payroll.pdf", "payroll.pdf"),
        ("C:/private/payroll.pdf", "payroll.pdf"),
        (r"C:\private\payroll.pdf", "payroll.pdf"),
        ("C:payroll.pdf", "payroll.pdf"),
        (r"\\server\share\private\payroll.pdf", "payroll.pdf"),
        (r"\\server/share\private/payroll.pdf", "payroll.pdf"),
    ],
)
async def test_client_path_is_only_safe_basename_metadata(
    tmp_path: Path,
    client_name: str,
    safe_name: str,
) -> None:
    item = upload(client_name, b"%PDF-1.7\nbody", "application/pdf")

    result = await store_upload(item, tmp_path, 1024)

    assert result.original_name == safe_name
    assert result.path.parent == resolved(tmp_path)
    assert result.path == resolved(tmp_path) / result.stored_name
    assert result.path.read_bytes() == b"%PDF-1.7\nbody"
    assert not (tmp_path.parent / "escape.pdf").exists()
    assert item.file.closed


@pytest.mark.asyncio
async def test_stored_name_is_server_uuid_with_allowed_normalized_suffix(tmp_path: Path) -> None:
    item = upload("../../report.PDF", b"%PDF-1.7\nbody", "APPLICATION/PDF")

    result = await store_upload(item, tmp_path, 1024)

    assert re.fullmatch(r"[0-9a-f]{32}\.pdf", result.stored_name)
    assert result.stored_name != result.original_name
    assert result.content_type == "application/pdf"
    assert result.size == len(b"%PDF-1.7\nbody")
    assert result.sha256 == hashlib.sha256(b"%PDF-1.7\nbody").hexdigest()


@pytest.mark.asyncio
async def test_oversized_upload_counts_streamed_bytes_and_removes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(upload_service, "CHUNK_SIZE", 8)
    item = upload("x.md", b"x" * 21, "text/markdown", content_length=1)

    with pytest.raises(AppError) as captured:
        await store_upload(item, tmp_path, 20)

    assert captured.value.code is ErrorCode.FILE_TOO_LARGE
    assert captured.value.status_code == 413
    assert directory_entries(tmp_path) == []
    assert item.file.closed


@pytest.mark.asyncio
async def test_exact_size_is_accepted_without_trusting_large_content_length(tmp_path: Path) -> None:
    item = upload("x.md", b"hello", "text/markdown", content_length=10_000)

    result = await store_upload(item, tmp_path, 5)

    assert result.size == 5
    assert result.path.read_bytes() == b"hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not-a-pdf", b"PK\x03\x04" + b"x" * 128])
async def test_declared_pdf_rejects_non_pdf_payload_and_cleans_file(tmp_path: Path, body: bytes) -> None:
    item = upload("x.pdf", body, "application/pdf")

    with pytest.raises(AppError) as captured:
        await store_upload(item, tmp_path, 1024)

    assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
    assert captured.value.status_code == 415
    assert directory_entries(tmp_path) == []
    assert item.file.closed


@pytest.mark.asyncio
async def test_mime_mismatch_is_rejected_before_writing_and_upload_is_closed(tmp_path: Path) -> None:
    item = upload("x.pdf", b"%PDF-1.7", "text/plain")

    with pytest.raises(AppError) as captured:
        await store_upload(item, tmp_path, 1024)

    assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
    assert captured.value.status_code == 415
    assert directory_entries(tmp_path) == []
    assert item.file.closed


@pytest.mark.asyncio
async def test_invalid_utf8_markdown_removes_partial_file(tmp_path: Path) -> None:
    item = upload("x.md", b"valid\xffinvalid", "text/markdown")

    with pytest.raises(AppError) as captured:
        await store_upload(item, tmp_path, 1024)

    assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
    assert captured.value.status_code == 415
    assert directory_entries(tmp_path) == []
    assert item.file.closed


@pytest.mark.asyncio
async def test_invalid_utf8_split_across_chunks_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(upload_service, "CHUNK_SIZE", 4)
    item = upload("x.markdown", b"abc\xe2\x28\xa1", "text/plain")

    with pytest.raises(AppError) as captured:
        await store_upload(item, tmp_path, 1024)

    assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
    assert captured.value.status_code == 415
    assert directory_entries(tmp_path) == []
    assert item.file.closed


@pytest.mark.asyncio
async def test_markdown_nul_anywhere_in_stream_is_rejected(tmp_path: Path) -> None:
    item = upload("x.md", b"a" * 5000 + b"\x00tail", "text/markdown")

    with pytest.raises(AppError) as captured:
        await store_upload(item, tmp_path, 10_000)

    assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
    assert captured.value.status_code == 415
    assert directory_entries(tmp_path) == []
    assert item.file.closed


@pytest.mark.asyncio
async def test_zip_document_type_is_never_accepted(tmp_path: Path) -> None:
    item = upload("archive.zip", b"PK\x03\x04" + b"x" * 128, "application/zip")

    with pytest.raises(AppError) as captured:
        await store_upload(item, tmp_path, 1024)

    assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
    assert captured.value.status_code == 415
    assert directory_entries(tmp_path) == []
    assert item.file.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("magic", [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"])
async def test_zip_magic_disguised_as_markdown_is_rejected(tmp_path: Path, magic: bytes) -> None:
    item = upload("notes.md", magic + b"disguised archive", "text/markdown")

    with pytest.raises(AppError) as captured:
        await store_upload(item, tmp_path, 1024)

    assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
    assert captured.value.status_code == 415
    assert directory_entries(tmp_path) == []
    assert item.file.closed


@pytest.mark.asyncio
async def test_read_failure_removes_partial_file_and_closes_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(upload_service, "CHUNK_SIZE", 4)
    item = ReadFailureUpload(b"first chunk")

    with pytest.raises(OSError, match="simulated read failure"):
        await store_upload(item, tmp_path, 1024)

    assert item.read_calls == 2
    assert item.close_called
    assert item.file.closed
    assert directory_entries(tmp_path) == []


@pytest.mark.asyncio
async def test_cancellation_cleans_partial_file_and_closes_upload(tmp_path: Path) -> None:
    item = CancellableUpload()
    propagated: list[BaseException] = []

    async def run_upload() -> None:
        try:
            await store_upload(item, tmp_path, 1024)
        except BaseException as exc:
            propagated.append(exc)
            raise

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_upload)
        await item.second_read_started.wait()
        task_group.cancel_scope.cancel()

    assert len(propagated) == 1
    assert propagated[0] is item.read_cancellation
    assert isinstance(propagated[0], anyio.get_cancelled_exc_class())
    assert item.close_called
    assert item.file.closed
    assert directory_entries(tmp_path) == []


@pytest.mark.asyncio
async def test_cancellation_during_exclusive_open_cleans_acquired_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_id = uuid.UUID(int=5)
    target = resolved(tmp_path) / f"{fixed_id.hex}.pdf"
    created = threading.Event()
    release = threading.Event()
    real_open = builtins.open
    opened_handles: list[Any] = []
    propagated: list[BaseException] = []

    def blocking_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if mode == "xb" and Path(file) == target:
            opened_handles.append(handle)
            created.set()
            if not release.wait(timeout=10):
                handle.close()
                raise TimeoutError("exclusive open was not released")
        return handle

    async def run_upload(item: UploadFile) -> None:
        try:
            await store_upload(item, tmp_path, 1024)
        except BaseException as exc:
            propagated.append(exc)
            raise

    monkeypatch.setattr(upload_service.uuid, "uuid4", lambda: fixed_id)
    monkeypatch.setattr(builtins, "open", blocking_open)
    item = upload("x.pdf", b"%PDF-1.7\nbody", "application/pdf")

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_upload, item)
        assert await anyio.to_thread.run_sync(created.wait, 5)
        task_group.cancel_scope.cancel()
        release.set()

    assert len(propagated) == 1
    assert isinstance(propagated[0], anyio.get_cancelled_exc_class())
    assert item.file.closed
    assert opened_handles and all(handle.closed for handle in opened_handles)
    assert not target.exists()
    assert directory_entries(tmp_path) == []


@pytest.mark.asyncio
async def test_close_failure_does_not_mask_original_security_error(tmp_path: Path) -> None:
    item = CloseFailureUpload("x.pdf", b"not-a-pdf", "application/pdf")

    with pytest.raises(AppError) as captured:
        await store_upload(item, tmp_path, 1024)

    assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
    assert captured.value.status_code == 415
    assert item.close_called
    assert item.file.closed
    assert directory_entries(tmp_path) == []


@pytest.mark.asyncio
async def test_close_failure_after_success_removes_unreturned_file(tmp_path: Path) -> None:
    item = CloseFailureUpload("x.pdf", b"%PDF-1.7\nbody", "application/pdf")

    with pytest.raises(RuntimeError, match="simulated close failure"):
        await store_upload(item, tmp_path, 1024)

    assert item.close_called
    assert item.file.closed
    assert directory_entries(tmp_path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"%PDF-1.7\nbody", b"not-a-pdf"])
async def test_destination_close_error_obeys_primary_error_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
) -> None:
    real_open_file = upload_service.anyio.open_file

    async def close_failing_open(path, mode):
        destination = await real_open_file(path, mode)
        real_aclose = destination.aclose

        async def failing_close() -> None:
            await real_aclose()
            raise RuntimeError("simulated destination close failure")

        monkeypatch.setattr(destination, "aclose", failing_close)
        return destination

    monkeypatch.setattr(upload_service.anyio, "open_file", close_failing_open)
    item = upload("x.pdf", body, "application/pdf")

    if body.startswith(b"%PDF-"):
        with pytest.raises(RuntimeError, match="simulated destination close failure"):
            await store_upload(item, tmp_path, 1024)
    else:
        with pytest.raises(AppError) as captured:
            await store_upload(item, tmp_path, 1024)
        assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
        assert captured.value.status_code == 415

    assert item.file.closed
    assert directory_entries(tmp_path) == []


@pytest.mark.asyncio
async def test_identity_error_has_priority_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_id = uuid.UUID(int=6)
    target = resolved(tmp_path) / f"{fixed_id.hex}.pdf"
    real_open_file = upload_service.anyio.open_file
    opened_handles: list[Any] = []

    def failing_identity(descriptor: int) -> upload_service.FileIdentity:
        raise RuntimeError(f"simulated identity failure for {descriptor}")

    async def close_failing_open(path, mode):
        destination = await real_open_file(path, mode)
        opened_handles.append(destination.wrapped)
        real_aclose = destination.aclose

        async def failing_close() -> None:
            await real_aclose()
            raise OSError("simulated descriptor close failure")

        monkeypatch.setattr(destination, "aclose", failing_close)
        return destination

    monkeypatch.setattr(upload_service.uuid, "uuid4", lambda: fixed_id)
    monkeypatch.setattr(upload_service, "_file_identity_from_descriptor", failing_identity)
    monkeypatch.setattr(upload_service.anyio, "open_file", close_failing_open)
    item = upload("x.pdf", b"%PDF-1.7\nbody", "application/pdf")

    try:
        with pytest.raises(RuntimeError, match="simulated identity failure"):
            await store_upload(item, tmp_path, 1024)

        assert opened_handles and all(handle.closed for handle in opened_handles)
        assert item.file.closed
    finally:
        target.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_cleanup_preserves_replacement_and_primary_error_after_close_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_id = uuid.UUID(int=4)
    target = resolved(tmp_path) / f"{fixed_id.hex}.pdf"
    moved = resolved(tmp_path) / "moved-owned.pdf"
    monkeypatch.setattr(upload_service.uuid, "uuid4", lambda: fixed_id)
    item = ReplacingCloseUpload(target, moved)

    try:
        with pytest.raises(AppError) as captured:
            await store_upload(item, tmp_path, 1024)

        assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
        assert captured.value.status_code == 415
        assert item.close_called
        assert item.file.closed
        assert target.read_bytes() == b"replacement"
        assert moved.read_bytes() == b"not-a-pdf"
    finally:
        target.unlink(missing_ok=True)
        moved.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_target_must_resolve_within_root_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = resolved(tmp_path).parent / "escape.pdf"
    monkeypatch.setattr(upload_service.uuid, "uuid4", lambda: type("UnsafeUuid", (), {"hex": "../escape"})())
    item = upload("x.pdf", b"%PDF-1.7\nbody", "application/pdf")

    try:
        with pytest.raises(AppError) as captured:
            await store_upload(item, tmp_path, 1024)

        assert captured.value.code is ErrorCode.REQUEST_INVALID
        assert captured.value.status_code == 422
        assert not outside.exists()
        assert directory_entries(tmp_path) == []
        assert item.file.closed
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_target_containment_is_rechecked_after_exclusive_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_id = uuid.UUID(int=2)
    root = resolved(tmp_path)
    target = root / f"{fixed_id.hex}.pdf"
    outside = root.parent / "redirected.pdf"
    real_resolved_path = upload_service._resolved_path
    target_resolutions = 0

    async def redirected_after_open(path: Path) -> Path:
        nonlocal target_resolutions
        result = await real_resolved_path(path)
        if path == target:
            target_resolutions += 1
            if target_resolutions == 2:
                return outside
        return result

    monkeypatch.setattr(upload_service.uuid, "uuid4", lambda: fixed_id)
    monkeypatch.setattr(upload_service, "_resolved_path", redirected_after_open)
    item = upload("x.pdf", b"%PDF-1.7\nbody", "application/pdf")

    try:
        with pytest.raises(AppError) as captured:
            await store_upload(item, tmp_path, 1024)

        assert captured.value.code is ErrorCode.REQUEST_INVALID
        assert captured.value.status_code == 422
        assert target_resolutions == 2
        assert not outside.exists()
        assert directory_entries(tmp_path) == []
        assert item.file.closed
    finally:
        target.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_open_handle_identity_must_match_target_before_first_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_id = uuid.UUID(int=3)
    target = resolved(tmp_path) / f"{fixed_id.hex}.pdf"
    outside = resolved(tmp_path).parent / "misdirected.pdf"
    real_open_file = upload_service.anyio.open_file
    redirected_bodies: list[bytes] = []

    async def misdirected_open(path, mode):
        decoy = await real_open_file(path, mode)
        await decoy.aclose()
        redirected = await real_open_file(outside, mode)
        real_aclose = redirected.aclose

        async def recording_close() -> None:
            await real_aclose()
            redirected_bodies.append(read_and_remove(outside))

        monkeypatch.setattr(redirected, "aclose", recording_close)
        return redirected

    monkeypatch.setattr(upload_service.uuid, "uuid4", lambda: fixed_id)
    monkeypatch.setattr(upload_service.anyio, "open_file", misdirected_open)
    item = upload("x.pdf", b"%PDF-1.7\nbody", "application/pdf")

    try:
        with pytest.raises(AppError) as captured:
            await store_upload(item, tmp_path, 1024)

        assert captured.value.code is ErrorCode.REQUEST_INVALID
        assert captured.value.status_code == 422
        assert redirected_bodies == [b""]
        assert not outside.exists()
        assert item.file.closed
    finally:
        target.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_exclusive_creation_never_overwrites_existing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_id = uuid.UUID(int=0)
    existing = tmp_path / f"{fixed_id.hex}.pdf"
    existing.write_bytes(b"sentinel")
    monkeypatch.setattr(upload_service.uuid, "uuid4", lambda: fixed_id)
    item = upload("x.pdf", b"%PDF-1.7\nbody", "application/pdf")

    with pytest.raises(FileExistsError):
        await store_upload(item, tmp_path, 1024)

    assert existing.read_bytes() == b"sentinel"
    assert item.file.closed


@pytest.mark.asyncio
async def test_exclusive_creation_does_not_follow_or_remove_symlink_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_id = uuid.UUID(int=1)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"sentinel")
    candidate = tmp_path / f"{fixed_id.hex}.pdf"
    try:
        candidate.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable on this platform: {exc}")
    monkeypatch.setattr(upload_service.uuid, "uuid4", lambda: fixed_id)
    item = upload("x.pdf", b"%PDF-1.7\nbody", "application/pdf")

    with pytest.raises(FileExistsError):
        await store_upload(item, tmp_path, 1024)

    assert outside.read_bytes() == b"sentinel"
    assert candidate.is_symlink()
    assert item.file.closed


def test_zero_files_is_rejected() -> None:
    with pytest.raises(AppError) as captured:
        validate_upload_count([], 5)

    assert captured.value.code is ErrorCode.TOO_MANY_FILES
    assert captured.value.status_code == 422


def test_upload_count_at_limit_is_accepted() -> None:
    validate_upload_count([object()] * 5, 5)


def test_too_many_files_is_rejected() -> None:
    with pytest.raises(AppError) as captured:
        validate_upload_count([object()] * 6, 5)

    assert captured.value.code is ErrorCode.TOO_MANY_FILES
    assert captured.value.status_code == 422


def test_missing_path_has_no_file_identity(tmp_path: Path) -> None:
    assert upload_service._file_identity_from_path(tmp_path / "missing.pdf") is None


@pytest.mark.asyncio
async def test_cleanup_tolerates_path_identity_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inaccessible_path_identity(path: Path) -> upload_service.FileIdentity | None:
        raise PermissionError(path)

    monkeypatch.setattr(upload_service, "_file_identity_from_path", inaccessible_path_identity)

    await upload_service._remove_owned_target(tmp_path / "missing.pdf", (1, 2))


@pytest.mark.asyncio
async def test_empty_markdown_is_rejected_and_removed(tmp_path: Path) -> None:
    item = upload("empty.md", b"", "text/markdown")

    with pytest.raises(AppError) as captured:
        await store_upload(item, tmp_path, 1024)

    assert captured.value.code is ErrorCode.REQUEST_INVALID
    assert directory_entries(tmp_path) == []
    assert item.file.closed


@pytest.mark.asyncio
async def test_successful_upload_can_be_reopened_by_owned_identity(tmp_path: Path) -> None:
    body = b"verified content"
    result = await store_upload(upload("guide.md", body, "text/markdown"), tmp_path, 1024)

    with result.open_verified() as source:
        assert source.read() == body

    result.delete_owned()
    assert not result.path.exists()


@pytest.mark.asyncio
async def test_replaced_successful_upload_is_neither_opened_nor_deleted(tmp_path: Path) -> None:
    result = await store_upload(upload("guide.md", b"owned", "text/markdown"), tmp_path, 1024)
    moved = tmp_path / "moved-owned.md"
    result.path.replace(moved)
    result.path.write_bytes(b"replacement")

    try:
        with pytest.raises(AppError) as captured:
            result.open_verified()
        assert captured.value.code is ErrorCode.REQUEST_INVALID

        result.delete_owned()
        assert result.path.read_bytes() == b"replacement"
        assert moved.read_bytes() == b"owned"
    finally:
        result.path.unlink(missing_ok=True)
        moved.unlink(missing_ok=True)
