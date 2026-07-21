from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from app.core.errors import AppError
from app.services.import_workspace import ImportWorkspace


def task_id_with_prefix(prefix: str) -> str:
    return str(uuid.UUID(f"{prefix}000000-0000-4000-8000-000000000000"))


def test_workspace_prepares_server_generated_sharded_directory(tmp_path: Path) -> None:
    workspace = ImportWorkspace(tmp_path / "imports")
    task_id = task_id_with_prefix("ab")

    task_dir = workspace.prepare(task_id)

    assert task_dir == (workspace.root / "ab" / uuid.UUID(task_id).hex).resolve()
    marker = task_dir / ".import-workspace-owner-v1"
    assert marker.is_dir()
    assert not any(marker.iterdir())
    workspace.cleanup_task(task_id)
    assert not task_dir.exists()


def test_sweeper_never_removes_active_workspace_even_with_old_marker(tmp_path: Path) -> None:
    workspace = ImportWorkspace(
        tmp_path / "imports",
        orphan_age_seconds=60,
        clock=lambda: 10_000,
    )
    task_id = task_id_with_prefix("aa")
    task_dir = workspace.prepare(task_id)
    marker = task_dir / ".import-workspace-owner-v1"
    os.utime(marker, (1, 1))

    report = workspace.sweep_orphans()

    assert task_dir.exists()
    assert report.removed == 0
    assert report.skipped_active == 1


def test_sweeper_removes_old_marked_workspace_after_process_restart(tmp_path: Path) -> None:
    root = tmp_path / "imports"
    task_id = task_id_with_prefix("aa")
    creator = ImportWorkspace(root)
    task_dir = creator.prepare(task_id)
    marker = task_dir / ".import-workspace-owner-v1"
    os.utime(marker, (1, 1))
    os.utime(task_dir, (1, 1))
    restarted = ImportWorkspace(root, orphan_age_seconds=60, clock=lambda: 10_000)

    report = restarted.sweep_orphans()

    assert report.removed == 1
    assert not task_dir.exists()


def test_sweeper_preserves_workspace_with_recent_file_activity(tmp_path: Path) -> None:
    root = tmp_path / "imports"
    task_id = task_id_with_prefix("ad")
    creator = ImportWorkspace(root)
    task_dir = creator.prepare(task_id)
    marker = task_dir / ".import-workspace-owner-v1"
    os.utime(marker, (1, 1))
    source = task_dir / "upload.part"
    source.write_bytes(b"still-uploading")
    restarted = ImportWorkspace(root, orphan_age_seconds=60)

    report = restarted.sweep_orphans()

    assert source.read_bytes() == b"still-uploading"
    assert report.removed == 0
    assert report.skipped_recent == 1


def test_sweeper_leaves_unmarked_uuid_directory_untouched(tmp_path: Path) -> None:
    workspace = ImportWorkspace(
        tmp_path / "imports",
        orphan_age_seconds=60,
        clock=lambda: 10_000,
    )
    task_id = task_id_with_prefix("ee")
    parsed = uuid.UUID(task_id)
    task_dir = workspace.root / parsed.hex[:2] / parsed.hex
    task_dir.mkdir(parents=True)
    sentinel = task_dir / "not-owned.txt"
    sentinel.write_text("keep", encoding="utf-8")

    report = workspace.sweep_orphans()

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert report.removed == 0
    assert report.skipped_unowned == 1


def test_sweeper_does_not_follow_replaced_owner_marker(tmp_path: Path) -> None:
    root = tmp_path / "imports"
    task_id = task_id_with_prefix("ef")
    creator = ImportWorkspace(root)
    task_dir = creator.prepare(task_id)
    marker = task_dir / ".import-workspace-owner-v1"
    marker.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    try:
        marker.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")
    restarted = ImportWorkspace(root, orphan_age_seconds=60, clock=lambda: 10_000)

    report = restarted.sweep_orphans()

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert marker.is_symlink()
    assert report.removed == 0
    assert report.skipped_unowned == 1


def test_sweeper_respects_bounded_batch_size(tmp_path: Path) -> None:
    root = tmp_path / "imports"
    creator = ImportWorkspace(root)
    task_dirs = []
    for prefix in ("10", "20", "30"):
        task_id = task_id_with_prefix(prefix)
        task_dir = creator.prepare(task_id)
        os.utime(task_dir / ".import-workspace-owner-v1", (1, 1))
        os.utime(task_dir, (1, 1))
        task_dirs.append(task_dir)
    restarted = ImportWorkspace(
        root,
        orphan_age_seconds=60,
        sweep_batch_size=2,
        clock=lambda: 10_000,
    )

    report = restarted.sweep_orphans()

    assert report.removed == 2
    assert sum(path.exists() for path in task_dirs) == 1


def test_active_workspace_heartbeat_refreshes_owner_marker(tmp_path: Path) -> None:
    workspace = ImportWorkspace(tmp_path / "imports")
    task_id = task_id_with_prefix("fa")
    task_dir = workspace.prepare(task_id)
    marker = task_dir / ".import-workspace-owner-v1"
    os.utime(marker, (1, 1))

    assert workspace.heartbeat_task(task_id) is True

    assert marker.stat().st_mtime > 1
    workspace.cleanup_task(task_id)
    assert workspace.heartbeat_task(task_id) is False


def test_symlink_shard_is_rejected_before_creating_outside_directory(tmp_path: Path) -> None:
    workspace = ImportWorkspace(tmp_path / "imports")
    outside = tmp_path / "outside"
    outside.mkdir()
    shard = workspace.root / "ab"
    try:
        shard.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")
    task_id = task_id_with_prefix("ab")

    with pytest.raises(AppError):
        workspace.prepare(task_id)

    assert not (outside / uuid.UUID(task_id).hex).exists()


def test_cleanup_unlinks_task_symlink_without_touching_its_target(tmp_path: Path) -> None:
    workspace = ImportWorkspace(tmp_path / "imports")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    task_id = task_id_with_prefix("cd")
    shard = workspace.root / "cd"
    shard.mkdir()
    task_path = shard / uuid.UUID(task_id).hex
    try:
        task_path.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")

    workspace.cleanup_task(task_id)

    assert not task_path.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
