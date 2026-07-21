from __future__ import annotations

import os
import stat
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from app.core.errors import AppError, DependencyError, ErrorCode
from app.core.settings import Settings

FileIdentity = tuple[int, int]

_OWNER_MARKER_NAME = ".import-workspace-owner-v1"
_HEX_CHARACTERS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class WorkspaceSweepReport:
    scanned: int
    removed: int
    skipped_active: int
    skipped_recent: int
    skipped_unowned: int
    failed: int


def _identity(path: Path) -> FileIdentity:
    info = path.lstat()
    return info.st_dev, info.st_ino


def _is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _remove_tree_no_links(path: Path) -> None:
    if _is_link_or_reparse(path):
        path.unlink(missing_ok=True)
        return
    if not path.is_dir():
        path.unlink(missing_ok=True)
        return
    for child in path.iterdir():
        _remove_tree_no_links(child)
    path.rmdir()


def _task_uuid(task_id: str) -> uuid.UUID:
    try:
        parsed = uuid.UUID(task_id)
    except (AttributeError, TypeError, ValueError):
        raise AppError(ErrorCode.REQUEST_INVALID, "Invalid import task identifier", 422) from None
    if str(parsed) != task_id:
        raise AppError(ErrorCode.REQUEST_INVALID, "Invalid import task identifier", 422)
    return parsed


def _safe_hex_segment(value: str, *, length: int) -> bool:
    return len(value) == length and all(character in _HEX_CHARACTERS for character in value)


class ImportWorkspace:
    def __init__(
        self,
        root: Path,
        *,
        orphan_age_seconds: int = 86_400,
        sweep_batch_size: int = 20,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(orphan_age_seconds, bool) or not isinstance(orphan_age_seconds, int) or orphan_age_seconds <= 0:
            raise ValueError("orphan_age_seconds must be a positive integer")
        if isinstance(sweep_batch_size, bool) or not isinstance(sweep_batch_size, int) or sweep_batch_size <= 0:
            raise ValueError("sweep_batch_size must be a positive integer")
        resolved = root.expanduser().resolve(strict=False)
        if resolved == Path(resolved.anchor):
            raise DependencyError("Import workspace is not configured")
        try:
            resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
            if _is_link_or_reparse(resolved) or not resolved.is_dir():
                raise OSError("unsafe workspace root")
            root_identity = _identity(resolved)
        except OSError:
            raise DependencyError("Import workspace is unavailable") from None
        self.root = resolved
        self._root_identity = root_identity
        self._orphan_age_seconds = orphan_age_seconds
        self._sweep_batch_size = sweep_batch_size
        self._clock = clock
        self._active_tasks: set[str] = set()
        self._lock = RLock()

    def _require_root(self) -> Path:
        try:
            if _is_link_or_reparse(self.root) or _identity(self.root) != self._root_identity:
                raise OSError("workspace root changed")
            resolved_root = self.root.resolve(strict=True)
        except OSError:
            raise DependencyError("Import workspace is unavailable") from None
        return resolved_root

    @staticmethod
    def _require_directory(path: Path, root: Path) -> Path:
        try:
            if _is_link_or_reparse(path) or not path.is_dir():
                raise OSError("unsafe workspace directory")
            resolved = path.resolve(strict=True)
        except OSError:
            raise AppError(ErrorCode.REQUEST_INVALID, "Invalid import workspace", 422) from None
        if not resolved.is_relative_to(root) or resolved == root:
            raise AppError(ErrorCode.REQUEST_INVALID, "Invalid import workspace", 422)
        return resolved

    def _task_path(self, task_id: str) -> Path:
        parsed = _task_uuid(task_id)
        return self.root / parsed.hex[:2] / parsed.hex

    @staticmethod
    def _write_marker(task_path: Path) -> None:
        marker = task_path / _OWNER_MARKER_NAME
        marker.mkdir(mode=0o700, exist_ok=False)

    @staticmethod
    def _read_valid_marker(marker: Path) -> os.stat_result | None:
        try:
            info = marker.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(info.st_mode) or _is_link_or_reparse(marker) or next(marker.iterdir(), None) is not None:
            return None
        return info

    @staticmethod
    def _latest_activity_mtime(task_path: Path, marker: Path) -> float | None:
        """Return direct workspace activity without following any child link."""
        latest = task_path.lstat().st_mtime
        for child in task_path.iterdir():
            if child == marker:
                continue
            info = child.lstat()
            if _is_link_or_reparse(child):
                return None
            latest = max(latest, info.st_mtime)
        return latest

    def probe(self) -> None:
        with self._lock:
            self._require_root()

    def prepare(self, task_id: str) -> Path:
        with self._lock:
            resolved_root = self._require_root()
            task_path = self._task_path(task_id)
            shard = task_path.parent
            shard.mkdir(mode=0o700, exist_ok=True)
            self._require_directory(shard, resolved_root)
            task_path.mkdir(mode=0o700, exist_ok=False)
            try:
                resolved_task = self._require_directory(task_path, resolved_root)
                self._write_marker(resolved_task)
            except BaseException:
                try:
                    _remove_tree_no_links(task_path)
                except OSError:
                    pass
                raise
            self._active_tasks.add(task_id)
            return resolved_task

    def heartbeat_task(self, task_id: str) -> bool:
        """Refresh an active workspace marker without following a replaced path."""
        with self._lock:
            resolved_root = self._require_root()
            task_path = self._task_path(task_id)
            if task_id not in self._active_tasks:
                return False
            resolved_task = self._require_directory(task_path, resolved_root)
            marker = resolved_task / _OWNER_MARKER_NAME
            try:
                info = self._read_valid_marker(marker)
                if info is None:
                    raise OSError("invalid workspace owner marker")
                try:
                    os.utime(marker, None, follow_symlinks=False)
                except NotImplementedError:
                    # Windows cannot combine directory timestamps with
                    # follow_symlinks=False. Revalidate immediately around the
                    # fallback; the workspace root is server-only (0700).
                    if _is_link_or_reparse(marker):
                        raise OSError("unsafe workspace owner marker") from None
                    os.utime(marker, None)
                current_info = self._read_valid_marker(marker)
                if current_info is None or (
                    current_info.st_dev,
                    current_info.st_ino,
                ) != (info.st_dev, info.st_ino):
                    raise OSError("workspace owner marker changed")
            except OSError:
                raise DependencyError("Import workspace is unavailable") from None
            return True

    def cleanup_task(self, task_id: str) -> None:
        with self._lock:
            resolved_root = self._require_root()
            task_path = self._task_path(task_id)
            shard = task_path.parent
            try:
                try:
                    shard.lstat()
                except FileNotFoundError:
                    return
                self._require_directory(shard, resolved_root)
                try:
                    task_path.lstat()
                except FileNotFoundError:
                    return
                if _is_link_or_reparse(task_path):
                    task_path.unlink(missing_ok=True)
                    return
                self._require_directory(task_path, resolved_root)
                _remove_tree_no_links(task_path)
                try:
                    shard.rmdir()
                except OSError:
                    pass
            finally:
                self._active_tasks.discard(task_id)

    def sweep_orphans(self) -> WorkspaceSweepReport:
        """Remove a bounded number of old, marked task directories under the workspace root."""
        with self._lock:
            resolved_root = self._require_root()
            cutoff = self._clock() - self._orphan_age_seconds
            scanned = 0
            removed = 0
            skipped_active = 0
            skipped_recent = 0
            skipped_unowned = 0
            failed = 0

            try:
                shards = sorted(resolved_root.iterdir(), key=lambda path: path.name)
            except OSError:
                raise DependencyError("Import workspace is unavailable") from None

            for shard in shards:
                if removed >= self._sweep_batch_size:
                    break
                if not _safe_hex_segment(shard.name, length=2):
                    continue
                try:
                    if _is_link_or_reparse(shard) or not shard.is_dir():
                        continue
                    resolved_shard = shard.resolve(strict=True)
                    if resolved_shard.parent != resolved_root:
                        continue
                    task_paths = sorted(resolved_shard.iterdir(), key=lambda path: path.name)
                except OSError:
                    failed += 1
                    continue

                for task_path in task_paths:
                    if removed >= self._sweep_batch_size:
                        break
                    if not _safe_hex_segment(task_path.name, length=32) or not task_path.name.startswith(shard.name):
                        continue
                    task_id = str(uuid.UUID(hex=task_path.name))
                    scanned += 1
                    if task_id in self._active_tasks:
                        skipped_active += 1
                        continue
                    try:
                        if _is_link_or_reparse(task_path) or not task_path.is_dir():
                            skipped_unowned += 1
                            continue
                        resolved_task = task_path.resolve(strict=True)
                        if resolved_task.parent != resolved_shard:
                            skipped_unowned += 1
                            continue
                        marker = resolved_task / _OWNER_MARKER_NAME
                        marker_info = self._read_valid_marker(marker)
                        if marker_info is None:
                            skipped_unowned += 1
                            continue
                        activity_mtime = self._latest_activity_mtime(resolved_task, marker)
                        if activity_mtime is None:
                            skipped_unowned += 1
                            continue
                        if max(marker_info.st_mtime, activity_mtime) > cutoff:
                            skipped_recent += 1
                            continue
                        # Re-read immediately before removal so a concurrent heartbeat
                        # observed during the scan protects the directory.
                        current_info = self._read_valid_marker(marker)
                        if current_info is None:
                            skipped_unowned += 1
                            continue
                        current_activity_mtime = self._latest_activity_mtime(
                            resolved_task,
                            marker,
                        )
                        if current_activity_mtime is None:
                            skipped_unowned += 1
                            continue
                        if (current_info.st_dev, current_info.st_ino) != (
                            marker_info.st_dev,
                            marker_info.st_ino,
                        ) or max(current_info.st_mtime, current_activity_mtime) > cutoff:
                            skipped_recent += 1
                            continue
                        _remove_tree_no_links(resolved_task)
                        removed += 1
                    except OSError:
                        failed += 1
                try:
                    resolved_shard.rmdir()
                except OSError:
                    pass

            return WorkspaceSweepReport(
                scanned=scanned,
                removed=removed,
                skipped_active=skipped_active,
                skipped_recent=skipped_recent,
                skipped_unowned=skipped_unowned,
                failed=failed,
            )


def build_import_workspace(settings: Settings) -> ImportWorkspace:
    configured = settings.api.import_work_root
    project_root = settings.api.project_root
    base = project_root if project_root is not None else Path(__file__).resolve().parents[2]
    root = configured if configured.is_absolute() else base / configured
    return ImportWorkspace(
        root,
        orphan_age_seconds=settings.import_runtime.workspace_orphan_age_seconds,
        sweep_batch_size=settings.import_runtime.workspace_sweep_batch_size,
    )
