from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.release.build_server_bundle import build_bundle


def test_server_bundle_is_secret_free_and_manifest_is_valid(tmp_path: Path) -> None:
    bundle_dir, zip_path = build_bundle(tmp_path, allow_dirty=True)

    assert zip_path.is_file()
    assert (bundle_dir / "Dockerfile").is_file()
    assert (bundle_dir / "compose.server-reuse.yaml").is_file()
    assert (bundle_dir / "app/gateways/mineru_gateway.py").is_file()
    assert (bundle_dir / "SERVER_UPLOAD_README.md").is_file()
    assert not (bundle_dir / ".env").exists()
    assert not (bundle_dir / "deploy/compose.env").exists()
    assert not (bundle_dir / "deploy/server-reuse.env").exists()
    assert not (bundle_dir / "tests").exists()

    manifest = (bundle_dir / "MANIFEST.sha256").read_text(encoding="utf-8")
    entries = [line.split("  ", maxsplit=1) for line in manifest.splitlines()]
    assert entries
    for expected, relative in entries:
        payload = (bundle_dir / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
