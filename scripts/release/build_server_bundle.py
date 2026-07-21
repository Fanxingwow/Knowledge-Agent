from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dist"
ROOT_FILES = (
    ".dockerignore",
    "Dockerfile",
    "README.md",
    "README.zh-CN.md",
    "compose.gpu.yaml",
    "compose.server-reuse.yaml",
    "pyproject.toml",
    "uv.lock",
)
ROOT_DIRECTORIES = ("app", "deploy", "docs", "prompts", "scripts")
UPLOAD_README = "deploy/server-upload/README.md"
FORBIDDEN_NAMES = {".env", "compose.env", "server-reuse.env"}
FORBIDDEN_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".safetensors", ".pt", ".pth", ".onnx"}
GIT_EXECUTABLE = shutil.which("git")
if GIT_EXECUTABLE is None:
    raise RuntimeError("git is required to build the server bundle")
GIT_COMMAND = (
    GIT_EXECUTABLE,
    "-c",
    f"safe.directory={PROJECT_ROOT.as_posix()}",
)


def _git(*args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - arguments are internal constants
        [*GIT_COMMAND, *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _is_forbidden(path: PurePosixPath) -> bool:
    return (
        path.name in FORBIDDEN_NAMES
        or path.suffix.lower() in FORBIDDEN_SUFFIXES
        or "__pycache__" in path.parts
        or path.suffix.lower() in {".pyc", ".pyo"}
    )


def runtime_files(*, include_untracked: bool = False) -> list[PurePosixPath]:
    pathspecs = [*ROOT_FILES, *ROOT_DIRECTORIES]
    source_flags = ["--cached"]
    if include_untracked:
        source_flags.extend(["--others", "--exclude-standard"])
    output = subprocess.run(  # noqa: S603 - pathspecs are internal constants
        [*GIT_COMMAND, "ls-files", *source_flags, "-z", "--", *pathspecs],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    paths = [PurePosixPath(value.decode("utf-8")) for value in output.split(b"\0") if value]
    forbidden = [str(path) for path in paths if _is_forbidden(path)]
    if forbidden:
        raise RuntimeError("Refusing to package forbidden runtime files: " + ", ".join(forbidden))
    return sorted(paths, key=str)


def _assert_clean_runtime_files() -> None:
    pathspecs = [*ROOT_FILES, *ROOT_DIRECTORIES]
    dirty = _git("status", "--porcelain", "--untracked-files=all", "--", *pathspecs)
    if dirty:
        raise RuntimeError("Runtime files have uncommitted changes; commit them or pass --allow-dirty.\n" + dirty)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(bundle_dir: Path) -> None:
    entries = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        relative = path.relative_to(bundle_dir).as_posix()
        entries.append(f"{_sha256(path)}  {relative}")
    (bundle_dir / "MANIFEST.sha256").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_deterministic_zip(bundle_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(bundle_dir).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def build_bundle(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    allow_dirty: bool = False,
) -> tuple[Path, Path]:
    if not allow_dirty:
        _assert_clean_runtime_files()

    revision = _git("rev-parse", "--short=12", "HEAD")
    full_revision = _git("rev-parse", "HEAD")
    bundle_dir = output_root.resolve() / f"knowledge-agent-server-{revision}"
    zip_path = bundle_dir.with_suffix(".zip")
    if bundle_dir.exists() or zip_path.exists():
        raise FileExistsError(f"Bundle output already exists: {bundle_dir}")

    files = runtime_files(include_untracked=allow_dirty)
    for relative in files:
        source = PROJECT_ROOT.joinpath(*relative.parts)
        destination = bundle_dir.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    upload_readme = PROJECT_ROOT.joinpath(*PurePosixPath(UPLOAD_README).parts)
    shutil.copy2(upload_readme, bundle_dir / "SERVER_UPLOAD_README.md")
    release = {
        "schema_version": 1,
        "profile": "linux-server-reuse",
        "accelerators": ["cpu", "nvidia-gpu"],
        "source_commit": full_revision,
        "source_dirty": allow_dirty,
    }
    (bundle_dir / "RELEASE.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_manifest(bundle_dir)
    _write_deterministic_zip(bundle_dir, zip_path)
    return bundle_dir, zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the secret-free Linux CPU/GPU server upload bundle.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that receives the bundle directory and ZIP.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Package modified tracked runtime files and mark the release dirty.",
    )
    args = parser.parse_args()
    bundle_dir, zip_path = build_bundle(
        args.output_root,
        allow_dirty=args.allow_dirty,
    )
    print(bundle_dir)
    print(zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
