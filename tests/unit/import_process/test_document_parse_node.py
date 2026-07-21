from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from app.core.errors import AppError
from app.import_process.agent.nodes.node_document_parse import DocumentParseNode
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.infrastructure.mineru_client import MineruParseArtifact

_TASK_ID = "11111111-1111-4111-8111-111111111111"


class FakeParser:
    def __init__(
        self,
        *,
        artifact: MineruParseArtifact | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self._artifact = artifact or MineruParseArtifact(
            markdown="# parsed\n",
            sha256=hashlib.sha256(b"# parsed\n").hexdigest(),
            size=len(b"# parsed\n"),
            artifact_path="artifact/result.md",
        )
        self._failure = failure
        self.calls: list[tuple[Path, str]] = []
        self.probe_calls = 0
        self.close_calls = 0
        self.max_request_bytes = 1024 * 1024

    def parse_pdf(self, source: Path, *, source_sha256: str) -> MineruParseArtifact:
        self.calls.append((source, source_sha256))
        if self._failure is not None:
            raise self._failure
        return self._artifact

    def probe(self) -> None:
        self.probe_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _state(
    tmp_path: Path,
    *,
    kind: str,
    content: bytes,
    source_override: Path | None = None,
) -> tuple[ImportGraphState, Path, Path]:
    task_hex = _TASK_ID.replace("-", "")
    workspace = tmp_path / task_hex[:2] / task_hex
    workspace.mkdir(parents=True)
    (workspace / ".import-workspace-owner-v1").mkdir()
    source = workspace / ("source.pdf" if kind == "pdf" else "source.md")
    source.write_bytes(content)
    selected_source = source_override or source
    state = create_default_state(
        task_id=_TASK_ID,
        document_id="22222222-2222-4222-8222-222222222222",
        tenant_id="tenant-a",
        owner_subject="user-a",
        workspace_dir=str(workspace),
        source_path=str(selected_source),
        local_dir=str(workspace),
        local_file_path=str(selected_source),
        source_ref="tenants/tenant-a/source",
        original_name="manual.pdf" if kind == "pdf" else "manual.md",
        content_type="application/pdf" if kind == "pdf" else "text/markdown",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        source_kind=kind,
        is_pdf_read_enabled=kind == "pdf",
        is_md_read_enabled=kind == "markdown",
        pdf_path=str(selected_source) if kind == "pdf" else "",
        md_path=str(selected_source) if kind == "markdown" else "",
    )
    return state, workspace, source


def test_markdown_is_parsed_locally_without_calling_mineru(tmp_path: Path) -> None:
    parser = FakeParser()
    state, _, _ = _state(tmp_path, kind="markdown", content="# 本地文档\n".encode())

    update = DocumentParseNode(parser)(state)

    assert update == {"md_content": "# 本地文档\n", "md_path": "source.md"}
    assert parser.calls == []
    serialized = repr(update)
    assert state["source_path"] not in serialized
    assert state["workspace_dir"] not in serialized


@pytest.mark.parametrize("content", [b"invalid-utf8-\xff", b"before\x00after"])
def test_markdown_rejects_invalid_utf8_and_nul(tmp_path: Path, content: bytes) -> None:
    state, _, _ = _state(tmp_path, kind="markdown", content=content)

    with pytest.raises(AppError, match="Import source is invalid"):
        DocumentParseNode(FakeParser())(state)


def test_source_path_cannot_escape_the_server_workspace(tmp_path: Path) -> None:
    outside = tmp_path / "source.pdf"
    outside.write_bytes(b"%PDF-outside")
    state, _, _ = _state(
        tmp_path,
        kind="pdf",
        content=b"%PDF-outside",
        source_override=outside,
    )

    with pytest.raises(AppError, match="Import source is invalid") as captured:
        DocumentParseNode(FakeParser())(state)

    assert str(outside) not in repr(captured.value)


def test_source_name_must_be_the_fixed_server_name(tmp_path: Path) -> None:
    state, workspace, source = _state(tmp_path, kind="pdf", content=b"%PDF-test")
    renamed = workspace / "client-name.pdf"
    source.rename(renamed)
    state["source_path"] = str(renamed)
    state["local_file_path"] = str(renamed)

    with pytest.raises(AppError, match="Import source is invalid"):
        DocumentParseNode(FakeParser())(state)


def test_pdf_parse_writes_only_fixed_artifact_and_cleans_temporary_directory(
    tmp_path: Path,
) -> None:
    parser = FakeParser()
    state, workspace, source = _state(tmp_path, kind="pdf", content=b"%PDF-test")

    update = DocumentParseNode(parser)(state)

    assert update == {"md_content": "# parsed\n", "md_path": "parsed.md"}
    assert (workspace / "parsed.md").read_text(encoding="utf-8") == "# parsed\n"
    assert list(workspace.glob(".mineru-*")) == []
    assert parser.calls == [(source.resolve(), hashlib.sha256(b"%PDF-test").hexdigest())]
    serialized = repr(update)
    assert state["source_path"] not in serialized
    assert state["workspace_dir"] not in serialized


def test_cancellation_is_not_wrapped_and_temporary_directory_is_removed(
    tmp_path: Path,
) -> None:
    state, workspace, _ = _state(tmp_path, kind="pdf", content=b"%PDF-test")
    node = DocumentParseNode(FakeParser(failure=asyncio.CancelledError()))

    with pytest.raises(asyncio.CancelledError):
        node(state)

    assert list(workspace.glob(".mineru-*")) == []
    assert not (workspace / "parsed.md").exists()


def test_probe_is_explicit_and_not_run_during_construction() -> None:
    parser = FakeParser()
    node = DocumentParseNode(parser)
    assert parser.probe_calls == 0

    node.probe()

    assert parser.probe_calls == 1

    node.close()
    assert parser.close_calls == 1


def test_source_digest_is_verified_before_remote_call(tmp_path: Path) -> None:
    parser = FakeParser()
    state, _, source = _state(tmp_path, kind="pdf", content=b"%PDF-test")
    source.write_bytes(b"tampered")

    with pytest.raises(AppError, match="Import source is invalid"):
        DocumentParseNode(parser)(state)

    assert parser.calls == []
