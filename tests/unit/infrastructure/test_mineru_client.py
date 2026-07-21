from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path
from typing import Any, cast

import pytest
import requests
from pydantic import SecretStr
from urllib3.util import Timeout as Urllib3Timeout

from app.core.errors import DependencyError
from app.core.settings import MineruSettings
from app.infrastructure.mineru_client import (
    MineruClient,
    MineruRequestTimeout,
    MineruTransportError,
    RequestsMineruTransport,
    build_mineru_client,
)


class FakeTransport:
    def __init__(self, outcomes: list[bytes | BaseException]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []
        self.probe_outcome: BaseException | None = None
        self.probe_calls: list[dict[str, object]] = []
        self.closed = False

    def parse_pdf(
        self,
        *,
        base_url: str,
        api_key: str,
        source: Path,
        source_size: int,
        source_sha256: str,
        timeout: MineruRequestTimeout,
        max_response_bytes: int,
    ) -> bytes:
        self.calls.append(
            {
                "base_url": base_url,
                "api_key": api_key,
                "source": source,
                "source_size": source_size,
                "source_sha256": source_sha256,
                "timeout": timeout,
                "max_response_bytes": max_response_bytes,
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def probe(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None:
        self.probe_calls.append(
            {
                "base_url": base_url,
                "api_key": api_key,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.probe_outcome is not None:
            raise self.probe_outcome

    def close(self) -> None:
        self.closed = True


class FakeHttpSession:
    def __init__(self, response: requests.Response) -> None:
        self.response = response
        self.trust_env = True
        self.post_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []
        self.closed = False

    def post(self, url: str, **kwargs: object) -> requests.Response:
        self.post_calls.append({"url": url, **kwargs})
        return self.response

    def get(self, url: str, **kwargs: object) -> requests.Response:
        self.get_calls.append({"url": url, **kwargs})
        return self.response

    def close(self) -> None:
        self.closed = True


def _http_response(status: int, content: bytes = b"") -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = content
    response.raw = io.BytesIO(content)
    response.headers["Content-Length"] = str(len(content))
    return response


def _settings(**overrides: object) -> MineruSettings:
    values: dict[str, object] = {
        "base_url": "https://mineru.internal",
        "api_key": SecretStr("mineru-secret-token"),
        "connect_timeout_seconds": 1.5,
        "read_timeout_seconds": 4.0,
        "total_timeout_seconds": 20.0,
        "probe_timeout_seconds": 2.0,
        "max_retries": 2,
        "retry_base_seconds": 0.25,
        "retry_max_seconds": 1.0,
        "max_request_bytes": 1024,
        "max_response_bytes": 4096,
    }
    values.update(overrides)
    return MineruSettings.model_construct(**values)


def _response(source_sha256: str, content: str = "# parsed\n") -> bytes:
    encoded = content.encode()
    return json.dumps(
        {
            "status": "succeeded",
            "source_sha256": source_sha256,
            "artifact": {
                "path": "artifacts/document.md",
                "media_type": "text/markdown",
                "size": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "content": content,
            },
        }
    ).encode()


def _source(tmp_path: Path, content: bytes = b"%PDF-test") -> tuple[Path, str]:
    source = tmp_path / "source.pdf"
    source.write_bytes(content)
    return source, hashlib.sha256(content).hexdigest()


def test_parse_pdf_validates_response_and_passes_explicit_timeouts(tmp_path: Path) -> None:
    source, source_sha256 = _source(tmp_path)
    transport = FakeTransport([_response(source_sha256)])
    client = build_mineru_client(_settings(), transport=transport)

    artifact = client.parse_pdf(source, source_sha256=source_sha256)

    assert artifact.markdown == "# parsed\n"
    assert artifact.artifact_path == "artifacts/document.md"
    assert artifact.sha256 == hashlib.sha256(b"# parsed\n").hexdigest()
    call = transport.calls[0]
    timeout = call["timeout"]
    assert isinstance(timeout, MineruRequestTimeout)
    assert timeout.connect_seconds == 1.5
    assert timeout.read_seconds == 4.0
    assert 0 < timeout.remaining_total_seconds <= 20.0
    assert call["source_size"] == len(b"%PDF-test")
    assert call["max_response_bytes"] == 4096


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("socket timed out at https://secret.invalid"),
        MineruTransportError("rate_limited", retryable=True),
        MineruTransportError("upstream_unavailable", retryable=True),
    ],
)
def test_retryable_failures_use_finite_exponential_backoff(
    tmp_path: Path,
    failure: BaseException,
) -> None:
    source, source_sha256 = _source(tmp_path)
    transport = FakeTransport([failure, failure, _response(source_sha256)])
    sleeps: list[float] = []
    client = MineruClient(
        _settings(),
        transport=transport,
        sleeper=sleeps.append,
    )

    artifact = client.parse_pdf(source, source_sha256=source_sha256)

    assert artifact.markdown == "# parsed\n"
    assert len(transport.calls) == 3
    assert sleeps == [0.25, 0.5]


def test_retry_exhaustion_returns_sanitized_dependency_error(tmp_path: Path) -> None:
    source, source_sha256 = _source(tmp_path)
    failure = MineruTransportError("upstream_unavailable", retryable=True)
    transport = FakeTransport([failure, failure, failure])
    client = MineruClient(_settings(), transport=transport, sleeper=lambda _: None)

    with pytest.raises(DependencyError) as captured:
        client.parse_pdf(source, source_sha256=source_sha256)

    assert captured.value.details == {
        "reason": "upstream_unavailable",
        "retryable": True,
    }
    assert len(transport.calls) == 3


def test_total_deadline_stops_before_an_unbounded_retry(tmp_path: Path) -> None:
    source, source_sha256 = _source(tmp_path)
    failure = MineruTransportError("rate_limited", retryable=True)
    transport = FakeTransport([failure])
    clock_values = iter([0.0, 0.0, 19.9])
    client = MineruClient(
        _settings(),
        transport=transport,
        monotonic=lambda: next(clock_values),
        sleeper=lambda _: pytest.fail("deadline must prevent sleeping"),
    )

    with pytest.raises(DependencyError) as captured:
        client.parse_pdf(source, source_sha256=source_sha256)

    assert captured.value.details == {"reason": "timeout", "retryable": True}
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(status="done"),
        lambda payload: payload.update(source_sha256="0" * 64),
        lambda payload: payload["artifact"].update(path="../../escape.md"),
        lambda payload: payload["artifact"].update(path="C:\\escape.md"),
        lambda payload: payload["artifact"].update(path="nested//document.md"),
        lambda payload: payload["artifact"].update(path="volume:document.md"),
        lambda payload: payload["artifact"].update(sha256="0" * 64),
        lambda payload: payload["artifact"].update(size=999),
        lambda payload: payload["artifact"].update(extra="unexpected"),
    ],
)
def test_malformed_or_untrusted_artifact_response_is_rejected(
    tmp_path: Path,
    mutate,
) -> None:
    source, source_sha256 = _source(tmp_path)
    payload = json.loads(_response(source_sha256))
    mutate(payload)
    transport = FakeTransport([json.dumps(payload).encode()])
    client = MineruClient(_settings(), transport=transport)

    with pytest.raises(DependencyError) as captured:
        client.parse_pdf(source, source_sha256=source_sha256)

    assert captured.value.details == {
        "reason": "invalid_response",
        "retryable": False,
    }


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    source, source_sha256 = _source(tmp_path)
    response = _response(source_sha256)
    duplicate = response.replace(b'"status": "succeeded"', b'"status": "succeeded", "status": "succeeded"')
    client = MineruClient(_settings(), transport=FakeTransport([duplicate]))

    with pytest.raises(DependencyError, match="Document parsing service") as captured:
        client.parse_pdf(source, source_sha256=source_sha256)

    assert captured.value.details["reason"] == "invalid_response"


def test_request_size_limit_is_enforced_before_transport(tmp_path: Path) -> None:
    source, source_sha256 = _source(tmp_path, b"x" * 11)
    transport = FakeTransport([])
    client = MineruClient(_settings(max_request_bytes=10), transport=transport)

    with pytest.raises(DependencyError) as captured:
        client.parse_pdf(source, source_sha256=source_sha256)

    assert captured.value.details == {"reason": "source_invalid", "retryable": False}
    assert transport.calls == []


def test_source_digest_is_checked_before_transport(tmp_path: Path) -> None:
    source, source_sha256 = _source(tmp_path)
    transport = FakeTransport([])
    client = MineruClient(_settings(), transport=transport)

    with pytest.raises(DependencyError) as captured:
        client.parse_pdf(source, source_sha256="0" * 64)

    assert captured.value.details == {"reason": "source_invalid", "retryable": False}
    assert source_sha256 != "0" * 64
    assert transport.calls == []


def test_transport_cancellation_is_not_wrapped(tmp_path: Path) -> None:
    source, source_sha256 = _source(tmp_path)
    transport = FakeTransport([asyncio.CancelledError()])
    client = MineruClient(_settings(), transport=transport)

    with pytest.raises(asyncio.CancelledError):
        client.parse_pdf(source, source_sha256=source_sha256)

    assert len(transport.calls) == 1


def test_transport_exception_cannot_leak_url_token_or_source_path(tmp_path: Path) -> None:
    source, source_sha256 = _source(tmp_path)
    configured_api_key = _settings().api_key
    assert configured_api_key is not None
    secret = configured_api_key.get_secret_value()
    url = "https://mineru.internal"
    failure = RuntimeError(f"{secret} {url} {source}")
    transport = FakeTransport([failure, failure, failure])
    client = MineruClient(_settings(), transport=transport, sleeper=lambda _: None)

    with pytest.raises(DependencyError) as captured:
        client.parse_pdf(source, source_sha256=source_sha256)

    public_error = f"{captured.value!r} {captured.value.details}"
    assert secret not in public_error
    assert url not in public_error
    assert str(source) not in public_error


def test_untrusted_transport_reason_is_reduced_to_a_safe_code(tmp_path: Path) -> None:
    source, source_sha256 = _source(tmp_path)
    configured_api_key = _settings().api_key
    assert configured_api_key is not None
    sensitive_value = configured_api_key.get_secret_value()
    transport = FakeTransport(
        [MineruTransportError(sensitive_value, retryable=False)]
    )
    client = MineruClient(_settings(), transport=transport)

    with pytest.raises(DependencyError) as captured:
        client.parse_pdf(source, source_sha256=source_sha256)

    assert captured.value.details == {
        "reason": "transport_failure",
        "retryable": False,
    }
    assert sensitive_value not in repr(captured.value)


def test_probe_is_lazy_sanitized_and_closeable() -> None:
    transport = FakeTransport([])
    node_client = build_mineru_client(_settings(), transport=transport)
    assert transport.probe_calls == []

    transport.probe_outcome = RuntimeError(
        "mineru-secret-token https://mineru.internal/private"
    )
    with pytest.raises(DependencyError) as captured:
        node_client.probe()

    public_error = f"{captured.value!r} {captured.value.details}"
    assert "mineru-secret-token" not in public_error
    assert "https://mineru.internal" not in public_error
    node_client.close()
    assert transport.closed is True


@pytest.mark.parametrize(
    ("status", "reason", "retryable"),
    [
        (429, "rate_limited", True),
        (500, "upstream_unavailable", True),
        (503, "upstream_unavailable", True),
        (400, "request_rejected", False),
    ],
)
def test_requests_transport_classifies_http_status_without_response_leakage(
    tmp_path: Path,
    status: int,
    reason: str,
    retryable: bool,
) -> None:
    source, source_sha256 = _source(tmp_path)
    session = FakeHttpSession(_http_response(status, b"upstream secret body"))
    transport = RequestsMineruTransport(cast(Any, session))

    with pytest.raises(MineruTransportError) as captured:
        transport.parse_pdf(
            base_url="https://mineru.internal",
            api_key="secret-token",  # pragma: allowlist secret
            source=source,
            source_size=source.stat().st_size,
            source_sha256=source_sha256,
            timeout=MineruRequestTimeout(1.0, 2.0, 3.0),
            max_response_bytes=1024,
        )

    assert captured.value.reason == reason
    assert captured.value.retryable is retryable
    assert "upstream secret body" not in repr(captured.value)
    assert session.trust_env is False
    call = session.post_calls[0]
    assert call["allow_redirects"] is False
    assert call["stream"] is True
    request_timeout = call["timeout"]
    assert isinstance(request_timeout, Urllib3Timeout)
    assert request_timeout.connect_timeout == 1.0
    assert request_timeout.read_timeout == 2.0
    assert request_timeout.total == 3.0
    assert session.response.raw.closed is True
