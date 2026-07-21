from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import re
import socket
import stat
import tempfile
import time
import zipfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlsplit

import anyio
import requests
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.logger import logger

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PDF_MAGIC = b"%PDF-"
_UPLOAD_PATH = "/file-urls/batch"
_RESULT_PATH = "/extract-results/batch"
_SAFE_STATES = frozenset({"waiting-file", "pending", "running", "converting"})


class MineruGatewaySettings(BaseSettings):
    """Runtime settings for the private MinerU cloud adapter."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    auth_token: SecretStr | None = Field(None, validation_alias="MINERU_GATEWAY_AUTH_TOKEN")
    upstream_api_token: SecretStr | None = Field(
        None,
        validation_alias="MINERU_UPSTREAM_API_TOKEN",
    )
    upstream_base_url: str = Field(
        "https://mineru.net/api/v4",
        validation_alias="MINERU_UPSTREAM_BASE_URL",
    )
    model_version: Literal["pipeline", "vlm"] = Field(
        "vlm",
        validation_alias="MINERU_UPSTREAM_MODEL_VERSION",
    )
    connect_timeout_seconds: float = Field(
        5.0,
        ge=0.1,
        le=30.0,
        validation_alias="MINERU_GATEWAY_CONNECT_TIMEOUT_SECONDS",
    )
    read_timeout_seconds: float = Field(
        60.0,
        ge=1.0,
        le=300.0,
        validation_alias="MINERU_GATEWAY_READ_TIMEOUT_SECONDS",
    )
    total_timeout_seconds: float = Field(
        540.0,
        ge=30.0,
        le=1800.0,
        validation_alias="MINERU_GATEWAY_TOTAL_TIMEOUT_SECONDS",
    )
    poll_interval_seconds: float = Field(
        3.0,
        ge=0.5,
        le=30.0,
        validation_alias="MINERU_GATEWAY_POLL_INTERVAL_SECONDS",
    )
    max_request_bytes: int = Field(
        100 * 1024 * 1024,
        ge=1,
        le=200 * 1024 * 1024,
        validation_alias="MINERU_GATEWAY_MAX_REQUEST_BYTES",
    )
    max_archive_bytes: int = Field(
        500 * 1024 * 1024,
        ge=1,
        le=1024 * 1024 * 1024,
        validation_alias="MINERU_GATEWAY_MAX_ARCHIVE_BYTES",
    )
    max_archive_members: int = Field(
        5000,
        ge=1,
        le=20000,
        validation_alias="MINERU_GATEWAY_MAX_ARCHIVE_MEMBERS",
    )
    max_extracted_bytes: int = Field(
        1024 * 1024 * 1024,
        ge=1,
        le=2 * 1024 * 1024 * 1024,
        validation_alias="MINERU_GATEWAY_MAX_EXTRACTED_BYTES",
    )
    max_markdown_bytes: int = Field(
        60 * 1024 * 1024,
        ge=1,
        le=64 * 1024 * 1024,
        validation_alias="MINERU_GATEWAY_MAX_MARKDOWN_BYTES",
    )
    max_compression_ratio: float = Field(
        100.0,
        ge=1.0,
        le=1000.0,
        validation_alias="MINERU_GATEWAY_MAX_COMPRESSION_RATIO",
    )
    max_concurrency: int = Field(
        2,
        ge=1,
        le=16,
        validation_alias="MINERU_GATEWAY_MAX_CONCURRENCY",
    )
    allowed_download_hosts_raw: str = Field(
        ".aliyuncs.com,.openxlab.org.cn,mineru.net",
        validation_alias="MINERU_GATEWAY_ALLOWED_DOWNLOAD_HOSTS",
    )

    @field_validator("auth_token", "upstream_api_token")
    @classmethod
    def validate_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        if len(secret) < 32 or len(secret) > 4096 or not secret.isprintable():
            raise ValueError("MinerU gateway secrets must contain 32-4096 printable characters")
        return value

    @field_validator("upstream_base_url")
    @classmethod
    def validate_upstream_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path != "/api/v4"
        ):
            raise ValueError("MINERU_UPSTREAM_BASE_URL must be an HTTPS /api/v4 service origin")
        return normalized

    @model_validator(mode="after")
    def validate_limits(self) -> MineruGatewaySettings:
        if self.max_markdown_bytes > self.max_extracted_bytes:
            raise ValueError("Markdown limit cannot exceed the extracted archive limit")
        if not self.allowed_download_hosts:
            raise ValueError("At least one MinerU download host suffix is required")
        return self

    @property
    def configured(self) -> bool:
        return self.auth_token is not None and self.upstream_api_token is not None

    @property
    def allowed_download_hosts(self) -> tuple[str, ...]:
        return tuple(
            item.strip().lower()
            for item in self.allowed_download_hosts_raw.split(",")
            if item.strip()
        )


@dataclass(frozen=True, slots=True)
class ParsedMarkdown:
    content: str
    size: int
    sha256: str


class GatewayFailure(Exception):
    def __init__(self, reason: str, *, status_code: int, retryable: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
        self.retryable = retryable


class MineruAdapter(Protocol):
    def parse_pdf(self, source: Path, *, source_sha256: str) -> ParsedMarkdown: ...

    def close(self) -> None: ...


def _failure(reason: str, status_code: int, *, retryable: bool) -> GatewayFailure:
    return GatewayFailure(reason, status_code=status_code, retryable=retryable)


def _remaining(deadline: float, monotonic: Callable[[], float]) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise _failure("timeout", 504, retryable=True)
    return remaining


def _response_error(status_code: int) -> GatewayFailure:
    if status_code == 429:
        return _failure("rate_limited", 429, retryable=True)
    if status_code in {408, 425} or 500 <= status_code < 600:
        return _failure("upstream_unavailable", 503, retryable=True)
    return _failure("upstream_rejected", 502, retryable=False)


def _json_object(response: requests.Response, *, maximum: int = 1024 * 1024) -> dict[str, Any]:
    declared = response.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) > maximum:
                raise _failure("invalid_upstream_response", 502, retryable=False)
        except ValueError:
            raise _failure("invalid_upstream_response", 502, retryable=False) from None
    body = response.content
    if len(body) > maximum:
        raise _failure("invalid_upstream_response", 502, retryable=False)
    try:
        payload = response.json()
    except (requests.JSONDecodeError, ValueError):
        raise _failure("invalid_upstream_response", 502, retryable=False) from None
    if not isinstance(payload, dict):
        raise _failure("invalid_upstream_response", 502, retryable=False)
    return cast(dict[str, Any], payload)


def _require_success_payload(response: requests.Response) -> dict[str, Any]:
    if response.status_code != 200:
        raise _response_error(response.status_code)
    payload = _json_object(response)
    if payload.get("code") != 0 or not isinstance(payload.get("data"), dict):
        raise _failure("upstream_rejected", 502, retryable=False)
    return cast(dict[str, Any], payload["data"])


def _host_allowed(hostname: str, suffixes: tuple[str, ...]) -> bool:
    host = hostname.rstrip(".").lower()
    for suffix in suffixes:
        normalized = suffix.rstrip(".")
        if normalized.startswith("."):
            if host.endswith(normalized) and host != normalized[1:]:
                return True
        elif host == normalized:
            return True
    return False


def _validate_download_url(url: object, suffixes: tuple[str, ...]) -> str:
    if not isinstance(url, str) or len(url) > 8192:
        raise _failure("invalid_upstream_response", 502, retryable=False)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not _host_allowed(parsed.hostname, suffixes)
    ):
        raise _failure("invalid_upstream_response", 502, retryable=False)
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except OSError:
        raise _failure("upstream_unavailable", 503, retryable=True) from None
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address[4][0])
        except ValueError:
            raise _failure("invalid_upstream_response", 502, retryable=False) from None
        if not ip.is_global:
            raise _failure("invalid_upstream_response", 502, retryable=False)
    return url


def _safe_archive_member(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return not (
        info.is_dir()
        or not info.filename
        or info.filename.startswith(("/", "\\"))
        or "\\" in info.filename
        or ":" in info.filename
        or any(part in {"", ".", ".."} for part in path.parts)
        or stat.S_ISLNK(unix_mode)
    )


def _read_markdown_archive(path: Path, settings: MineruGatewaySettings) -> ParsedMarkdown:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > settings.max_archive_members:
                raise _failure("unsafe_archive", 502, retryable=False)
            total = 0
            markdown: list[zipfile.ZipInfo] = []
            for info in members:
                if not _safe_archive_member(info):
                    if not info.is_dir():
                        raise _failure("unsafe_archive", 502, retryable=False)
                    continue
                total += info.file_size
                if total > settings.max_extracted_bytes:
                    raise _failure("unsafe_archive", 502, retryable=False)
                if info.compress_size == 0 and info.file_size > 0:
                    raise _failure("unsafe_archive", 502, retryable=False)
                if info.compress_size > 0 and info.file_size / info.compress_size > settings.max_compression_ratio:
                    raise _failure("unsafe_archive", 502, retryable=False)
                if info.filename.lower().endswith(".md"):
                    markdown.append(info)
            if not markdown:
                raise _failure("markdown_missing", 502, retryable=False)
            markdown.sort(
                key=lambda item: (
                    PurePosixPath(item.filename).name.lower() != "full.md",
                    len(PurePosixPath(item.filename).parts),
                    item.filename,
                )
            )
            selected = markdown[0]
            if selected.file_size <= 0 or selected.file_size > settings.max_markdown_bytes:
                raise _failure("markdown_invalid", 502, retryable=False)
            with archive.open(selected) as stream:
                content = stream.read(settings.max_markdown_bytes + 1)
            if len(content) != selected.file_size or len(content) > settings.max_markdown_bytes:
                raise _failure("markdown_invalid", 502, retryable=False)
    except GatewayFailure:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError):
        raise _failure("invalid_archive", 502, retryable=False) from None
    try:
        decoded = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _failure("markdown_invalid", 502, retryable=False) from None
    if "\x00" in decoded:
        raise _failure("markdown_invalid", 502, retryable=False)
    return ParsedMarkdown(
        content=decoded,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


class MineruCloudAdapter:
    def __init__(
        self,
        settings: MineruGatewaySettings,
        *,
        session: requests.Session | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.configured:
            raise ValueError("MinerU gateway is not configured")
        self._settings = settings
        self._session = session or requests.Session()
        self._session.trust_env = False
        self._monotonic = monotonic
        self._sleeper = sleeper

    def _timeout(self, deadline: float) -> tuple[float, float]:
        remaining = _remaining(deadline, self._monotonic)
        return (
            min(self._settings.connect_timeout_seconds, remaining),
            min(self._settings.read_timeout_seconds, remaining),
        )

    def _upstream_headers(self) -> dict[str, str]:
        token = self._settings.upstream_api_token
        if token is None:
            raise _failure("gateway_not_configured", 503, retryable=False)
        return {
            "Authorization": f"Bearer {token.get_secret_value()}",
            "Accept": "application/json",
        }

    def _submit(self, source_sha256: str, deadline: float) -> tuple[str, str]:
        try:
            response = self._session.post(
                f"{self._settings.upstream_base_url}{_UPLOAD_PATH}",
                headers={**self._upstream_headers(), "Content-Type": "application/json"},
                json={
                    "files": [{"name": f"{source_sha256}.pdf", "data_id": source_sha256}],
                    "model_version": self._settings.model_version,
                },
                timeout=self._timeout(deadline),
                allow_redirects=False,
            )
        except requests.Timeout:
            raise _failure("timeout", 504, retryable=True) from None
        except requests.RequestException:
            raise _failure("upstream_unavailable", 503, retryable=True) from None
        try:
            data = _require_success_payload(response)
        finally:
            response.close()
        batch_id = data.get("batch_id")
        urls = data.get("file_urls")
        if not isinstance(batch_id, str) or not batch_id or not isinstance(urls, list) or len(urls) != 1:
            raise _failure("invalid_upstream_response", 502, retryable=False)
        upload_url = _validate_download_url(urls[0], self._settings.allowed_download_hosts)
        return batch_id, upload_url

    def _upload(self, source: Path, upload_url: str, deadline: float) -> None:
        try:
            with source.open("rb") as stream:
                response = self._session.put(
                    upload_url,
                    data=stream,
                    timeout=self._timeout(deadline),
                    allow_redirects=False,
                )
        except requests.Timeout:
            raise _failure("timeout", 504, retryable=True) from None
        except (OSError, requests.RequestException):
            raise _failure("upstream_unavailable", 503, retryable=True) from None
        try:
            if response.status_code not in {200, 201, 204}:
                raise _response_error(response.status_code)
        finally:
            response.close()

    def _poll(self, batch_id: str, deadline: float) -> str:
        while True:
            try:
                response = self._session.get(
                    f"{self._settings.upstream_base_url}{_RESULT_PATH}/{batch_id}",
                    headers=self._upstream_headers(),
                    timeout=self._timeout(deadline),
                    allow_redirects=False,
                )
            except requests.Timeout:
                raise _failure("timeout", 504, retryable=True) from None
            except requests.RequestException:
                raise _failure("upstream_unavailable", 503, retryable=True) from None
            try:
                data = _require_success_payload(response)
            finally:
                response.close()
            results = data.get("extract_result")
            if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
                raise _failure("invalid_upstream_response", 502, retryable=False)
            result = cast(dict[str, Any], results[0])
            state = result.get("state")
            if state == "done":
                return _validate_download_url(
                    result.get("full_zip_url"),
                    self._settings.allowed_download_hosts,
                )
            if state == "failed":
                raise _failure("parse_failed", 502, retryable=False)
            if state not in _SAFE_STATES:
                raise _failure("invalid_upstream_response", 502, retryable=False)
            remaining = _remaining(deadline, self._monotonic)
            delay = min(self._settings.poll_interval_seconds, remaining)
            self._sleeper(delay)

    def _download(self, result_url: str, destination: Path, deadline: float) -> None:
        try:
            response = self._session.get(
                result_url,
                headers={"Accept": "application/zip"},
                timeout=self._timeout(deadline),
                allow_redirects=False,
                stream=True,
            )
        except requests.Timeout:
            raise _failure("timeout", 504, retryable=True) from None
        except requests.RequestException:
            raise _failure("upstream_unavailable", 503, retryable=True) from None
        try:
            if response.status_code != 200:
                raise _response_error(response.status_code)
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    if int(declared) > self._settings.max_archive_bytes:
                        raise _failure("archive_too_large", 502, retryable=False)
                except ValueError:
                    raise _failure("invalid_upstream_response", 502, retryable=False) from None
            total = 0
            with destination.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > self._settings.max_archive_bytes:
                        raise _failure("archive_too_large", 502, retryable=False)
                    stream.write(chunk)
            if total <= 0:
                raise _failure("invalid_archive", 502, retryable=False)
        except OSError:
            raise _failure("gateway_storage_failure", 500, retryable=True) from None
        finally:
            response.close()

    def parse_pdf(self, source: Path, *, source_sha256: str) -> ParsedMarkdown:
        deadline = self._monotonic() + self._settings.total_timeout_seconds
        batch_id, upload_url = self._submit(source_sha256, deadline)
        self._upload(source, upload_url, deadline)
        result_url = self._poll(batch_id, deadline)
        descriptor, archive_name = tempfile.mkstemp(prefix="mineru-result-", suffix=".zip")
        os.close(descriptor)
        archive = Path(archive_name)
        try:
            self._download(result_url, archive, deadline)
            return _read_markdown_archive(archive, self._settings)
        finally:
            archive.unlink(missing_ok=True)

    def close(self) -> None:
        self._session.close()


class ArtifactResponse(BaseModel):
    path: Literal["artifacts/document.md"] = "artifacts/document.md"
    media_type: Literal["text/markdown"] = "text/markdown"
    size: int
    sha256: str
    content: str


class ParseResponse(BaseModel):
    status: Literal["succeeded"] = "succeeded"
    source_sha256: str
    artifact: ArtifactResponse


def _authorize(authorization: str | None, settings: MineruGatewaySettings) -> None:
    configured = settings.auth_token
    if configured is None or authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    candidate = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(candidate, configured.get_secret_value()):
        raise HTTPException(status_code=401, detail="Authentication required")


async def _receive_pdf(
    request: Request,
    *,
    expected_sha256: str,
    maximum: int,
) -> Path:
    declared = request.headers.get("content-length")
    if declared is None:
        raise HTTPException(status_code=411, detail="Content-Length required")
    try:
        declared_size = int(declared)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Content-Length") from None
    if declared_size <= 0 or declared_size > maximum:
        raise HTTPException(status_code=413, detail="PDF payload is too large")
    descriptor, source_name = tempfile.mkstemp(prefix="mineru-source-", suffix=".pdf")
    source = Path(source_name)
    digest = hashlib.sha256()
    size = 0
    prefix = bytearray()
    try:
        with os.fdopen(descriptor, "wb") as stream:
            async for chunk in request.stream():
                if not chunk:
                    continue
                size += len(chunk)
                if size > maximum or size > declared_size:
                    raise HTTPException(status_code=413, detail="PDF payload is too large")
                if len(prefix) < len(_PDF_MAGIC):
                    prefix.extend(chunk[: len(_PDF_MAGIC) - len(prefix)])
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if size != declared_size or bytes(prefix) != _PDF_MAGIC or digest.hexdigest() != expected_sha256:
            raise HTTPException(status_code=422, detail="PDF payload validation failed")
        return source
    except BaseException:
        await anyio.Path(source).unlink(missing_ok=True)
        raise


def create_app(
    settings: MineruGatewaySettings | None = None,
    *,
    adapter: MineruAdapter | None = None,
) -> FastAPI:
    runtime_settings = settings or MineruGatewaySettings()
    runtime_adapter = adapter
    if runtime_adapter is None and runtime_settings.configured:
        runtime_adapter = MineruCloudAdapter(runtime_settings)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if runtime_adapter is not None:
                runtime_adapter.close()

    application = FastAPI(
        title="Knowledge Agent MinerU Gateway",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.adapter = runtime_adapter
    application.state.limiter = anyio.Semaphore(runtime_settings.max_concurrency)

    @application.get("/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/health")
    async def health(authorization: str | None = Header(None)) -> dict[str, str]:
        _authorize(authorization, runtime_settings)
        if application.state.adapter is None:
            raise HTTPException(status_code=503, detail="Gateway not configured")
        return {"status": "ready"}

    @application.post("/api/v1/parse", response_model=ParseResponse)
    async def parse_pdf(
        request: Request,
        authorization: str | None = Header(None),
        x_source_sha256: str | None = Header(None),
    ) -> ParseResponse:
        _authorize(authorization, runtime_settings)
        if application.state.adapter is None:
            raise HTTPException(status_code=503, detail="Gateway not configured")
        if request.headers.get("content-type", "").partition(";")[0].strip().lower() != "application/pdf":
            raise HTTPException(status_code=415, detail="Only application/pdf is accepted")
        if x_source_sha256 is None or _SHA256.fullmatch(x_source_sha256) is None:
            raise HTTPException(status_code=422, detail="Invalid source digest")
        source = await _receive_pdf(
            request,
            expected_sha256=x_source_sha256,
            maximum=runtime_settings.max_request_bytes,
        )
        try:
            async with application.state.limiter:
                try:
                    parsed = await anyio.to_thread.run_sync(
                        lambda: application.state.adapter.parse_pdf(
                            source,
                            source_sha256=x_source_sha256,
                        )
                    )
                except GatewayFailure as exc:
                    logger.bind(reason=exc.reason, retryable=exc.retryable).warning(
                        "MinerU gateway request failed"
                    )
                    raise HTTPException(status_code=exc.status_code, detail=exc.reason) from None
            return ParseResponse(
                source_sha256=x_source_sha256,
                artifact=ArtifactResponse(
                    size=parsed.size,
                    sha256=parsed.sha256,
                    content=parsed.content,
                ),
            )
        finally:
            await anyio.Path(source).unlink(missing_ok=True)

    return application


__all__ = [
    "GatewayFailure",
    "MineruAdapter",
    "MineruCloudAdapter",
    "MineruGatewaySettings",
    "ParsedMarkdown",
    "create_app",
]
