from __future__ import annotations

import hashlib
import uuid
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import pytest

from app.core.errors import AppError, ErrorCode
from app.infrastructure.minio_store import StoredObject, build_object_ref
from app.services.import_source import materialize_import_source
from app.services.import_workspace import ImportWorkspace


class Store:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls: list[tuple[str, StoredObject]] = []

    @contextmanager
    def open_stored(self, tenant_id: str, stored: StoredObject):
        self.calls.append((tenant_id, stored))
        yield BytesIO(self.body)


def source() -> StoredObject:
    return StoredObject(
        ref=build_object_ref("tenant-a", "documents", "document", "imports", "task", "source.md"),
        etag="etag-1",
        version_id="version-1",
    )


def test_materialize_reads_exact_source_and_verifies_digest(tmp_path: Path) -> None:
    body = b"# durable import\n"
    store = Store(body)
    workspace = ImportWorkspace(tmp_path / "imports")
    task_id = str(uuid.uuid4())

    task_dir, path = materialize_import_source(
        store,
        workspace,
        task_id=task_id,
        tenant_id="tenant-a",
        source=source(),
        original_name="../../guide.md",
        expected_size=len(body),
        expected_sha256=hashlib.sha256(body).hexdigest(),
    )

    assert task_dir.is_relative_to(workspace.root)
    assert path == task_dir / "source.md"
    assert path.read_bytes() == body
    assert store.calls == [("tenant-a", source())]


@pytest.mark.parametrize(
    ("body", "expected_size", "expected_sha256"),
    [
        (b"longer", 3, hashlib.sha256(b"lon").hexdigest()),
        (b"short", 6, hashlib.sha256(b"short!").hexdigest()),
        (b"same-size", 9, hashlib.sha256(b"different").hexdigest()),
    ],
)
def test_integrity_failure_removes_workspace(
    body: bytes,
    expected_size: int,
    expected_sha256: str,
    tmp_path: Path,
) -> None:
    workspace = ImportWorkspace(tmp_path / "imports")
    task_id = str(uuid.uuid4())

    with pytest.raises(AppError) as captured:
        materialize_import_source(
            Store(body),
            workspace,
            task_id=task_id,
            tenant_id="tenant-a",
            source=source(),
            original_name="guide.md",
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )

    assert captured.value.code is ErrorCode.REQUEST_INVALID
    assert not any(path.is_file() for path in workspace.root.rglob("*"))


def test_invalid_persisted_metadata_fails_before_workspace_creation(tmp_path: Path) -> None:
    workspace = ImportWorkspace(tmp_path / "imports")

    with pytest.raises(AppError) as captured:
        materialize_import_source(
            Store(b"x"),
            workspace,
            task_id=str(uuid.uuid4()),
            tenant_id="tenant-a",
            source=source(),
            original_name="guide.md",
            expected_size=1,
            expected_sha256="not-a-digest",
        )

    assert captured.value.code is ErrorCode.INTERNAL_ERROR
    assert not any(path.is_file() for path in workspace.root.rglob("*"))


def test_unsupported_persisted_name_is_removed(tmp_path: Path) -> None:
    workspace = ImportWorkspace(tmp_path / "imports")
    task_id = str(uuid.uuid4())

    with pytest.raises(AppError) as captured:
        materialize_import_source(
            Store(b"x"),
            workspace,
            task_id=task_id,
            tenant_id="tenant-a",
            source=source(),
            original_name="payload.exe",
            expected_size=1,
            expected_sha256=hashlib.sha256(b"x").hexdigest(),
        )

    assert captured.value.code is ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
    assert not any(path.is_file() for path in workspace.root.rglob("*"))
