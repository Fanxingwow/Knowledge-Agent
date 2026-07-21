from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def _safe_text(value: object, *, fallback: str) -> str:
    if not isinstance(value, str) or not value:
        return fallback
    sanitized = "".join(
        character if character.isprintable() and character not in "\r\n" else "?"
        for character in value
    )
    return sanitized[:240] or fallback


def _safe_line(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def scan_tracked_files(project_root: Path) -> tuple[int, tuple[str, ...]]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "detect_secrets",
            "scan",
            "--no-verify",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        return 2, ("Secret scanner could not complete.",)
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        return 2, ("Secret scanner returned an invalid report.",)
    if not isinstance(payload, Mapping):
        return 2, ("Secret scanner returned an invalid report.",)
    results = payload.get("results")
    if not isinstance(results, Mapping):
        return 2, ("Secret scanner returned an invalid report.",)

    findings: list[str] = []
    for raw_path, raw_items in sorted(results.items(), key=lambda item: str(item[0])):
        path = _safe_text(raw_path, fallback="unknown-file")
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
            return 2, ("Secret scanner returned an invalid report.",)
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                return 2, ("Secret scanner returned an invalid report.",)
            detector = _safe_text(raw_item.get("type"), fallback="potential-secret")
            line = _safe_line(raw_item.get("line_number"))
            location = f"{path}:{line}" if line is not None else path
            findings.append(f"{location}: {detector}")
    if findings:
        return 1, tuple(findings)
    return 0, ()


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    exit_code, messages = scan_tracked_files(project_root)
    if exit_code == 0:
        print("No potential secrets detected in tracked files.")
        return 0
    print("Secret scan failed; only file, line and detector are shown:", file=sys.stderr)
    for message in messages:
        print(message, file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
