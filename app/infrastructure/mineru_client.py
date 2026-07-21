from __future__ import annotations

import hashlib
import json
import re
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast

import requests
from pydantic import BaseModel, ConfigDict, ValidationError
from urllib3.util import Timeout as Urllib3Timeout

from app.core.errors import DependencyError
from app.core.logger import logger
from app.core.settings import MineruSettings

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARTIFACT_PATH_LENGTH = 240
_PARSE_PATH = "/api/v1/parse"
_PROBE_PATH = "/health"
_SAFE_REASONS = frozenset(
    {
        "invalid_response",
        "probe_failed",
        "rate_limited",
        "request_rejected",
        "response_too_large",
        "source_invalid",
        "source_unavailable",
        "timeout",
        "transport_failure",
        "upstream_unavailable",
    }
)


@dataclass(frozen=True, slots=True)
class MineruRequestTimeout:
    connect_seconds: float
    read_seconds: float
    remaining_total_seconds: float


@dataclass(frozen=True, slots=True)
class MineruParseArtifact:
    markdown: str
    sha256: str
    size: int
    artifact_path: str


class MineruTransportError(Exception):
    """Sanitized transport classification; never contains an upstream payload."""

    def __init__(self, reason: str, *, retryable: bool) -> None:
        safe_reason = reason if reason in _SAFE_REASONS else "transport_failure"
        super().__init__(safe_reason)
        self.reason = safe_reason
        self.retryable = retryable


class MineruTransport(Protocol):
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
    ) -> bytes: ...

    def probe(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None: ...

    def close(self) -> None: ...


class _ArtifactPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str
    media_type: Literal["text/markdown"]
    size: int
    sha256: str
    content: str


class _ParsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["succeeded"]
    source_sha256: str
    artifact: _ArtifactPayload


def _bounded_response(response: requests.Response, maximum: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            raise MineruTransportError("invalid_response", retryable=False) from None
        if declared < 0 or declared > maximum:
            raise MineruTransportError("response_too_large", retryable=False)

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > maximum:
            raise MineruTransportError("response_too_large", retryable=False)
        chunks.append(chunk)
    return b"".join(chunks)


def _status_error(status_code: int) -> MineruTransportError:
    if status_code == 429:
        return MineruTransportError("rate_limited", retryable=True)
    if status_code == 408 or 500 <= status_code < 600:
        return MineruTransportError("upstream_unavailable", retryable=True)
    return MineruTransportError("request_rejected", retryable=False)


class RequestsMineruTransport:
    """Streaming HTTP adapter for the hardened MinerU gateway contract.

    The gateway accepts a raw PDF at ``/api/v1/parse`` and returns the strict
    JSON schema validated by :class:`MineruClient`. Redirects and environment
    proxy inheritance are disabled so credentials and source bytes cannot be
    forwarded to an untrusted destination.
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._session.trust_env = False

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
        request_timeout = Urllib3Timeout(
            connect=min(timeout.connect_seconds, timeout.remaining_total_seconds),
            read=min(timeout.read_seconds, timeout.remaining_total_seconds),
            total=timeout.remaining_total_seconds,
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/pdf",
            "Content-Length": str(source_size),
            "X-Source-SHA256": source_sha256,
            "Accept": "application/json",
        }
        try:
            with source.open("rb") as body:
                response = self._session.post(
                    f"{base_url}{_PARSE_PATH}",
                    headers=headers,
                    data=body,
                    timeout=cast(Any, request_timeout),
                    allow_redirects=False,
                    stream=True,
                )
            try:
                if response.status_code != 200:
                    raise _status_error(response.status_code)
                return _bounded_response(response, max_response_bytes)
            finally:
                response.close()
        except MineruTransportError:
            raise
        except requests.Timeout:
            raise MineruTransportError("timeout", retryable=True) from None
        except (OSError, requests.RequestException):
            raise MineruTransportError("transport_failure", retryable=True) from None

    def probe(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None:
        try:
            response = self._session.get(
                f"{base_url}{_PROBE_PATH}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=cast(
                    Any,
                    Urllib3Timeout(
                        connect=timeout_seconds,
                        read=timeout_seconds,
                        total=timeout_seconds,
                    ),
                ),
                allow_redirects=False,
                stream=True,
            )
            try:
                if response.status_code not in {200, 204}:
                    raise _status_error(response.status_code)
            finally:
                response.close()
        except MineruTransportError:
            raise
        except requests.Timeout:
            raise MineruTransportError("timeout", retryable=True) from None
        except requests.RequestException:
            raise MineruTransportError("transport_failure", retryable=True) from None

    def close(self) -> None:
        self._session.close()


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _safe_artifact_path(value: str) -> str:
    if (
        not value
        or len(value) > _MAX_ARTIFACT_PATH_LENGTH
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("invalid artifact path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or path.suffix.lower() != ".md"
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise ValueError("invalid artifact path")
    return path.as_posix()


def _parse_artifact(
    raw_response: bytes,
    *,
    expected_source_sha256: str,
    maximum_bytes: int,
) -> MineruParseArtifact:
    try:
        decoded = raw_response.decode("utf-8", errors="strict")
        raw_payload = json.loads(decoded, object_pairs_hook=_object_without_duplicate_keys)
        if not isinstance(raw_payload, Mapping):
            raise ValueError("response must be an object")
        payload = _ParsePayload.model_validate(raw_payload)
        artifact_path = _safe_artifact_path(payload.artifact.path)
        content = payload.artifact.content
        content_bytes = content.encode("utf-8")
        if "\x00" in content:
            raise ValueError("NUL is forbidden")
        if payload.source_sha256 != expected_source_sha256:
            raise ValueError("source digest mismatch")
        if (
            _SHA256.fullmatch(payload.source_sha256) is None
            or _SHA256.fullmatch(payload.artifact.sha256) is None
            or isinstance(payload.artifact.size, bool)
            or payload.artifact.size <= 0
            or payload.artifact.size > maximum_bytes
            or len(content_bytes) != payload.artifact.size
            or hashlib.sha256(content_bytes).hexdigest() != payload.artifact.sha256
        ):
            raise ValueError("artifact integrity mismatch")
    except (UnicodeDecodeError, ValueError, ValidationError):
        raise MineruTransportError("invalid_response", retryable=False) from None
    return MineruParseArtifact(
        markdown=content,
        sha256=payload.artifact.sha256,
        size=payload.artifact.size,
        artifact_path=artifact_path,
    )


def _dependency_error(reason: str, *, retryable: bool) -> DependencyError:
    safe_reason = reason if reason in _SAFE_REASONS else "transport_failure"
    return DependencyError(
        "Document parsing service is unavailable",
        {"reason": safe_reason, "retryable": retryable},
    )


def _inspect_pdf_source(source: Path, maximum: int) -> tuple[int, str]:
    try:
        info = source.lstat()
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        attributes = getattr(info, "st_file_attributes", 0)
        if (
            source.name != "source.pdf"
            or not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or bool(attributes & reparse_flag)
            or info.st_size <= 0
            or info.st_size > maximum
        ):
            raise OSError("unsafe source")
        digest = hashlib.sha256()
        total = 0
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                if total > maximum:
                    raise OSError("oversized source")
                digest.update(chunk)
    except OSError:
        raise _dependency_error("source_invalid", retryable=False) from None
    if total != info.st_size:
        raise _dependency_error("source_invalid", retryable=False)
    return total, digest.hexdigest()


class MineruClient:
    """Bounded, retrying MinerU client with a strict response contract."""

    def __init__(
        self,
        settings: MineruSettings,
        *,
        transport: MineruTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if settings.base_url is None or settings.api_key is None:
            raise ValueError("MinerU capability is not configured")
        self._base_url = settings.base_url
        self._api_key = settings.api_key
        self._connect_timeout = settings.connect_timeout_seconds
        self._read_timeout = settings.read_timeout_seconds
        self._total_timeout = settings.total_timeout_seconds
        self._probe_timeout = settings.probe_timeout_seconds
        self._max_retries = settings.max_retries
        self._retry_base = settings.retry_base_seconds
        self._retry_max = settings.retry_max_seconds
        self._max_request_bytes = settings.max_request_bytes
        self._max_response_bytes = settings.max_response_bytes
        self._transport = transport or RequestsMineruTransport()
        self._monotonic = monotonic
        self._sleeper = sleeper

    @property
    def max_request_bytes(self) -> int:
        return self._max_request_bytes

    def parse_pdf(self, source: Path, *, source_sha256: str) -> MineruParseArtifact:
        if _SHA256.fullmatch(source_sha256) is None:
            raise _dependency_error("source_invalid", retryable=False)
        source_size, observed_sha256 = _inspect_pdf_source(
            source,
            self._max_request_bytes,
        )
        if observed_sha256 != source_sha256:
            raise _dependency_error("source_invalid", retryable=False)

        started = self._monotonic()
        api_key = self._api_key.get_secret_value()
        for attempt in range(self._max_retries + 1):
            current_size, current_sha256 = _inspect_pdf_source(
                source,
                self._max_request_bytes,
            )
            if current_size != source_size or current_sha256 != source_sha256:
                raise _dependency_error("source_invalid", retryable=False)
            remaining = self._total_timeout - (self._monotonic() - started)
            if remaining <= 0:
                raise _dependency_error("timeout", retryable=True)
            timeout = MineruRequestTimeout(
                connect_seconds=self._connect_timeout,
                read_seconds=self._read_timeout,
                remaining_total_seconds=remaining,
            )
            try:
                response = self._transport.parse_pdf(
                    base_url=self._base_url,
                    api_key=api_key,
                    source=source,
                    source_size=source_size,
                    source_sha256=source_sha256,
                    timeout=timeout,
                    max_response_bytes=self._max_response_bytes,
                )
                if len(response) > self._max_response_bytes:
                    raise MineruTransportError("response_too_large", retryable=False)
                return _parse_artifact(
                    response,
                    expected_source_sha256=source_sha256,
                    maximum_bytes=self._max_response_bytes,
                )
            except MineruTransportError as exc:
                reason = exc.reason
                retryable = exc.retryable
            except TimeoutError:
                reason = "timeout"
                retryable = True
            except Exception as exc:
                logger.bind(error_type=type(exc).__name__).warning(
                    "Document parsing transport failed"
                )
                reason = "transport_failure"
                retryable = True

            if not retryable or attempt >= self._max_retries:
                raise _dependency_error(reason, retryable=retryable) from None
            delay = min(self._retry_base * (2**attempt), self._retry_max)
            remaining = self._total_timeout - (self._monotonic() - started)
            if remaining <= delay:
                raise _dependency_error("timeout", retryable=True) from None
            self._sleeper(delay)

        raise _dependency_error("transport_failure", retryable=True)

    def probe(self) -> None:
        try:
            self._transport.probe(
                base_url=self._base_url,
                api_key=self._api_key.get_secret_value(),
                timeout_seconds=self._probe_timeout,
            )
        except MineruTransportError as exc:
            raise _dependency_error(exc.reason, retryable=exc.retryable) from None
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).warning(
                "Document parsing dependency probe failed"
            )
            raise _dependency_error("probe_failed", retryable=True) from None

    def close(self) -> None:
        self._transport.close()


def build_mineru_client(
    settings: MineruSettings,
    *,
    transport: MineruTransport | None = None,
) -> MineruClient:
    """Construct a client without performing network or filesystem I/O."""

    return MineruClient(settings, transport=transport)


__all__ = [
    "MineruClient",
    "MineruParseArtifact",
    "MineruRequestTimeout",
    "MineruTransport",
    "MineruTransportError",
    "RequestsMineruTransport",
    "build_mineru_client",
]
