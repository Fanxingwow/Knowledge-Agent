from __future__ import annotations

import hashlib
import io
import json
import socket
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.gateways.mineru_gateway import (
    GatewayFailure,
    MineruCloudAdapter,
    MineruGatewaySettings,
    ParsedMarkdown,
    _read_markdown_archive,
    create_app,
)

_INTERNAL_TOKEN = "internal-gateway-token-that-is-long-enough"  # noqa: S105
_UPSTREAM_TOKEN = "upstream-mineru-token-that-is-long-enough"  # noqa: S105


def _settings(**overrides: object) -> MineruGatewaySettings:
    values: dict[str, object] = {
        "auth_token": SecretStr(_INTERNAL_TOKEN),
        "upstream_api_token": SecretStr(_UPSTREAM_TOKEN),
        "upstream_base_url": "https://mineru.net/api/v4",
        "model_version": "vlm",
        "connect_timeout_seconds": 1.0,
        "read_timeout_seconds": 2.0,
        "total_timeout_seconds": 30.0,
        "poll_interval_seconds": 0.5,
        "max_request_bytes": 1024,
        "max_archive_bytes": 4096,
        "max_archive_members": 10,
        "max_extracted_bytes": 4096,
        "max_markdown_bytes": 2048,
        "max_compression_ratio": 100.0,
        "max_concurrency": 1,
        "allowed_download_hosts_raw": ".aliyuncs.com,.openxlab.org.cn,mineru.net",
    }
    values.update(overrides)
    return MineruGatewaySettings.model_construct(**values)


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []
        self.closed = False

    def parse_pdf(self, source: Path, *, source_sha256: str) -> ParsedMarkdown:
        assert source.read_bytes().startswith(b"%PDF-")
        self.calls.append((source, source_sha256))
        content = "# parsed\n"
        encoded = content.encode()
        return ParsedMarkdown(
            content=content,
            size=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def close(self) -> None:
        self.closed = True


def _headers(content: bytes, digest: str | None = None) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_INTERNAL_TOKEN}",
        "Content-Type": "application/pdf",
        "Content-Length": str(len(content)),
        "X-Source-SHA256": digest or hashlib.sha256(content).hexdigest(),
    }


def test_gateway_requires_internal_auth_and_returns_the_strict_client_contract() -> None:
    content = b"%PDF-enterprise-test"
    digest = hashlib.sha256(content).hexdigest()
    adapter = FakeAdapter()
    app = create_app(_settings(), adapter=adapter)

    with TestClient(app) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/health").status_code == 401
        assert client.get("/health", headers={"Authorization": f"Bearer {_INTERNAL_TOKEN}"}).status_code == 200
        response = client.post("/api/v1/parse", content=content, headers=_headers(content))

    assert response.status_code == 200
    assert response.json() == {
        "status": "succeeded",
        "source_sha256": digest,
        "artifact": {
            "path": "artifacts/document.md",
            "media_type": "text/markdown",
            "size": len(b"# parsed\n"),
            "sha256": hashlib.sha256(b"# parsed\n").hexdigest(),
            "content": "# parsed\n",
        },
    }
    assert len(adapter.calls) == 1
    assert not adapter.calls[0][0].exists()
    assert adapter.closed is True


@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Content-Type": "text/plain"}, 415),
        ({"X-Source-SHA256": "0" * 64}, 422),
    ],
)
def test_gateway_rejects_untrusted_requests_before_calling_upstream(
    headers: dict[str, str],
    expected_status: int,
) -> None:
    content = b"%PDF-enterprise-test"
    adapter = FakeAdapter()
    request_headers = _headers(content)
    request_headers.update(headers)
    app = create_app(_settings(), adapter=adapter)

    with TestClient(app) as client:
        response = client.post("/api/v1/parse", content=content, headers=request_headers)

    assert response.status_code == expected_status
    assert adapter.calls == []


def _zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_archive_reader_prefers_full_markdown_and_calculates_integrity(tmp_path: Path) -> None:
    archive = tmp_path / "result.zip"
    _zip(
        archive,
        {
            "nested/other.md": b"other",
            "document/full.md": b"# expected\n",
        },
    )

    parsed = _read_markdown_archive(archive, _settings())

    assert parsed.content == "# expected\n"
    assert parsed.size == len(b"# expected\n")
    assert parsed.sha256 == hashlib.sha256(b"# expected\n").hexdigest()


def test_archive_reader_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    _zip(archive, {"../full.md": b"unsafe"})

    with pytest.raises(GatewayFailure) as captured:
        _read_markdown_archive(archive, _settings())

    assert captured.value.reason == "unsafe_archive"


def _response(status: int, payload: dict[str, Any] | bytes) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    if isinstance(payload, dict):
        response._content = json.dumps(payload).encode()
        response.headers["Content-Type"] = "application/json"
    else:
        response._content = payload
    response.raw = io.BytesIO(response.content)
    response.headers["Content-Length"] = str(len(response.content))
    return response


class FakeCloudSession:
    def __init__(self, responses: list[requests.Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.trust_env = True
        self.closed = False

    def _next(self, method: str, url: str, kwargs: dict[str, object]) -> requests.Response:
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)

    def post(self, url: str, **kwargs: object) -> requests.Response:
        return self._next("POST", url, kwargs)

    def put(self, url: str, **kwargs: object) -> requests.Response:
        return self._next("PUT", url, kwargs)

    def get(self, url: str, **kwargs: object) -> requests.Response:
        return self._next("GET", url, kwargs)

    def close(self) -> None:
        self.closed = True


def test_cloud_adapter_uses_official_v4_batch_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("result/full.md", "# cloud result\n")
    session = FakeCloudSession(
        [
            _response(
                200,
                {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["https://mineru.oss-cn-shanghai.aliyuncs.com/upload.pdf"],
                    },
                },
            ),
            _response(200, b""),
            _response(
                200,
                {
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "state": "done",
                                "full_zip_url": "https://cdn-mineru.openxlab.org.cn/result.zip",
                            }
                        ]
                    },
                },
            ),
            _response(200, archive_buffer.getvalue()),
        ]
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
    )
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-cloud")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    adapter = MineruCloudAdapter(_settings(), session=session, sleeper=lambda _delay: None)  # type: ignore[arg-type]

    parsed = adapter.parse_pdf(source, source_sha256=digest)

    assert parsed.content == "# cloud result\n"
    assert session.trust_env is False
    assert [(method, url) for method, url, _kwargs in session.calls] == [
        ("POST", "https://mineru.net/api/v4/file-urls/batch"),
        ("PUT", "https://mineru.oss-cn-shanghai.aliyuncs.com/upload.pdf"),
        ("GET", "https://mineru.net/api/v4/extract-results/batch/batch-1"),
        ("GET", "https://cdn-mineru.openxlab.org.cn/result.zip"),
    ]
    submit = session.calls[0][2]
    assert submit["allow_redirects"] is False
    assert submit["json"] == {
        "files": [{"name": f"{digest}.pdf", "data_id": digest}],
        "model_version": "vlm",
    }
    authorization = str(cast(dict[str, str], submit["headers"])["Authorization"])
    assert authorization == f"Bearer {_UPSTREAM_TOKEN}"
