from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO, Protocol, TypeVar, cast
from urllib.parse import urlsplit

import certifi
from minio import Minio
from urllib3 import PoolManager, Retry, Timeout

from app.core.errors import AppError, DependencyError, ErrorCode, RetryableError, ValidationError
from app.core.settings import AppEnvironment, Settings

MAX_PRESIGNED_SECONDS = 15 * 60
_OBJECT_SCHEME = "kb-object"
_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_KEY_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_METADATA_KEY = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_BUCKET_NAME = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_ALLOWED_METADATA_KEYS = frozenset(
    {
        "document-id",
        "import-id",
        "object-kind",
        "schema-version",
        "sha256",
    }
)
_MAX_METADATA_ITEMS = 8
_MAX_METADATA_BYTES = 2048
_NO_POLICY_CODES = frozenset({"NoSuchBucketPolicy", "NoSuchPolicy"})
_NOT_FOUND_CODES = frozenset({"NoSuchKey", "NoSuchObject", "XMinioInvalidObjectName"})
_RETRYABLE_CODES = frozenset(
    {
        "InternalError",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "SlowDown",
    }
)


class MinioClient(Protocol):
    def bucket_exists(self, bucket_name: str) -> bool: ...

    def get_bucket_policy(self, bucket_name: str) -> str: ...

    def put_object(self, *args: Any, **kwargs: Any) -> Any: ...

    def get_object(self, *args: Any, **kwargs: Any) -> Any: ...

    def stat_object(self, *args: Any, **kwargs: Any) -> Any: ...

    def get_bucket_versioning(self, bucket_name: str) -> Any: ...

    def presigned_get_object(self, *args: Any, **kwargs: Any) -> str: ...

    def remove_object(self, *args: Any, **kwargs: Any) -> None: ...


@dataclass(frozen=True)
class ObjectRef:
    tenant_id: str
    relative_key: str

    def __post_init__(self) -> None:
        _validate_tenant_id(self.tenant_id)
        _validate_relative_key(self.relative_key)

    @property
    def object_name(self) -> str:
        return f"{self.tenant_id}/{self.relative_key}"

    @property
    def uri(self) -> str:
        return f"{_OBJECT_SCHEME}://{self.object_name}"


@dataclass(frozen=True)
class StoredObject:
    ref: ObjectRef
    etag: str | None
    version_id: str | None


@dataclass(frozen=True)
class ObservedObject:
    stored: StoredObject
    size: int
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class _Failure:
    code: str | None
    retryable: bool = False


@dataclass(frozen=True)
class _Outcome:
    value: object | None = None
    failure: _Failure | None = None


_T = TypeVar("_T")


def _validate_tenant_id(tenant_id: str) -> None:
    if not isinstance(tenant_id, str) or not _TENANT_ID.fullmatch(tenant_id):
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid tenant identifier")


def _validate_relative_key(relative_key: str) -> None:
    if not isinstance(relative_key, str) or len(relative_key) > 1024:
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object reference")
    if relative_key.startswith("/") or relative_key.endswith("/") or "\\" in relative_key:
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object reference")
    segments = relative_key.split("/")
    if not segments or any(segment in {"", ".", ".."} or not _KEY_SEGMENT.fullmatch(segment) for segment in segments):
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object reference")


def build_object_ref(tenant_id: str, *parts: str) -> ObjectRef:
    if not parts:
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object reference")
    if any(not isinstance(part, str) or not _KEY_SEGMENT.fullmatch(part) or part in {".", ".."} for part in parts):
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object reference")
    return ObjectRef(tenant_id=tenant_id, relative_key="/".join(parts))


def parse_object_ref(uri: str) -> ObjectRef:
    if not isinstance(uri, str):
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object reference")
    parsed = None
    try:
        parsed = urlsplit(uri)
    except ValueError:
        parsed = None
    if parsed is None:
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object reference")
    if parsed.scheme != _OBJECT_SCHEME or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object reference")
    relative_key = parsed.path.removeprefix("/")
    if not relative_key or parsed.path.startswith("//"):
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object reference")
    return ObjectRef(tenant_id=parsed.netloc, relative_key=relative_key)


def _exception_code(exc: Exception) -> str | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, str) else None


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in {
        "ConnectTimeoutError",
        "MaxRetryError",
        "NewConnectionError",
        "ProtocolError",
        "ReadTimeoutError",
    }


def _capture(call: Callable[[], _T]) -> _Outcome:
    try:
        return _Outcome(value=call())
    except AppError:
        raise
    except Exception as exc:
        failure = _Failure(_exception_code(exc), retryable=_is_retryable_exception(exc))
    return _Outcome(failure=failure)


def _raise_dependency(failure: _Failure | None) -> None:
    code = failure.code if failure else None
    if code in _NOT_FOUND_CODES:
        raise AppError(ErrorCode.RESOURCE_NOT_FOUND, "Object not found", 404)
    if (failure and failure.retryable) or code in _RETRYABLE_CODES:
        raise RetryableError("Object storage is temporarily unavailable")
    raise DependencyError("Object storage is unavailable")


def _require_value(outcome: _Outcome) -> object | None:
    if outcome.failure is not None:
        _raise_dependency(outcome.failure)
    return outcome.value


def _safe_optional_header(value: object) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any(ord(character) < 32 or ord(character) > 126 for character in value)
    ):
        raise DependencyError("Object storage is unavailable")
    return value


def _normalized_observed_metadata(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DependencyError("Object storage is unavailable")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise DependencyError("Object storage is unavailable")
        key = raw_key.lower().removeprefix("x-amz-meta-")
        if key in _ALLOWED_METADATA_KEYS:
            normalized[key] = raw_value
    return _validated_metadata(normalized) or {}


def _safely_close(callback: Callable[[], None]) -> bool:
    """Run cleanup without allowing a secondary failure to mask the result."""
    try:
        callback()
    except Exception:
        return False
    return True


def _read_policy(client: MinioClient, bucket_name: str) -> _Outcome:
    try:
        return _Outcome(value=client.get_bucket_policy(bucket_name))
    except AppError:
        raise
    except Exception as exc:
        code = _exception_code(exc)
        if code in _NO_POLICY_CODES:
            return _Outcome(value=None)
        failure = _Failure(code, retryable=_is_retryable_exception(exc))
    return _Outcome(failure=failure)


def _contains_public_principal(value: object) -> bool:
    if isinstance(value, str):
        return "*" in value or "?" in value
    if isinstance(value, dict):
        return any(_contains_public_principal(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_public_principal(item) for item in value)
    return False


def _policy_is_public(policy: str | None) -> bool | None:
    if not policy:
        return False
    try:
        document = json.loads(policy)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    statements = document.get("Statement", [])
    if isinstance(statements, dict):
        statements = [statements]
    if not isinstance(statements, list):
        return None
    for statement in statements:
        if not isinstance(statement, dict):
            return None
        if str(statement.get("Effect", "")).lower() != "allow":
            continue
        principal = statement.get("Principal")
        if _contains_public_principal(principal) or "NotPrincipal" in statement:
            return True
    return False


def _validate_content_type(content_type: str) -> None:
    if not content_type or any(character in content_type for character in "\r\n\x00"):
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid content type")


def _validated_metadata(metadata: Mapping[str, str] | None) -> dict[str, str] | None:
    if metadata is None:
        return None
    if len(metadata) > _MAX_METADATA_ITEMS:
        raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object metadata")
    copied: dict[str, str] = {}
    encoded_size = 0
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object metadata")
        if (
            not _METADATA_KEY.fullmatch(key)
            or key not in _ALLOWED_METADATA_KEYS
            or len(value) > 1024
            or any(ord(character) < 32 or ord(character) > 126 for character in value)
        ):
            raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object metadata")
        encoded_size += len(key.encode("ascii")) + len(value.encode("ascii")) + 2
        if encoded_size > _MAX_METADATA_BYTES:
            raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object metadata")
        copied[key] = value
    return copied


class MinioStore:
    def __init__(
        self,
        client: MinioClient,
        bucket_name: str,
        *,
        require_versioning: bool = False,
        close_transport: Callable[[], None] | None = None,
    ) -> None:
        if not _BUCKET_NAME.fullmatch(bucket_name):
            raise DependencyError("Object storage is not configured")
        self._client = client
        self.bucket_name = bucket_name
        self._require_versioning = require_versioning
        self._close_transport = close_transport
        self._close_lock = Lock()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise DependencyError("Object storage is unavailable")

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        if self._close_transport is not None:
            _safely_close(self._close_transport)

    def _readiness_status(self) -> str:
        if self._closed:
            return "unavailable"
        exists = _capture(lambda: self._client.bucket_exists(self.bucket_name))
        if exists.failure is not None:
            return "unavailable"
        if exists.value is not True:
            return "missing"
        policy = _read_policy(self._client, self.bucket_name)
        if policy.failure is not None:
            return "unavailable"
        public = _policy_is_public(cast(str | None, policy.value))
        if public is None:
            return "unavailable"
        if public:
            return "public"
        if self._require_versioning:
            versioning = _capture(lambda: self._client.get_bucket_versioning(self.bucket_name))
            if versioning.failure is not None:
                return "unavailable"
            status = getattr(versioning.value, "status", None)
            if status != "Enabled":
                return "unversioned"
        return "ready"

    def is_ready(self) -> bool:
        return self._readiness_status() == "ready"

    def require_ready(self) -> None:
        status = self._readiness_status()
        if status == "ready":
            return
        if status == "public":
            raise DependencyError("Object storage bucket must remain private")
        if status == "unversioned":
            raise DependencyError("Object storage bucket versioning must be enabled")
        raise DependencyError("Object storage is unavailable")

    @staticmethod
    def _owned_ref(tenant_id: str, ref: ObjectRef | str) -> ObjectRef:
        _validate_tenant_id(tenant_id)
        resolved = parse_object_ref(ref) if isinstance(ref, str) else ref
        if resolved.tenant_id != tenant_id:
            raise AppError(ErrorCode.RESOURCE_NOT_FOUND, "Object not found", 404)
        return resolved

    def put_stream(
        self,
        tenant_id: str,
        ref: ObjectRef,
        data: BinaryIO,
        *,
        length: int,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> StoredObject:
        self._ensure_open()
        owned = self._owned_ref(tenant_id, ref)
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object length")
        _validate_content_type(content_type)
        safe_metadata = _validated_metadata(metadata)
        outcome = _capture(
            lambda: self._client.put_object(
                self.bucket_name,
                owned.object_name,
                data=data,
                length=length,
                content_type=content_type,
                metadata=safe_metadata,
            )
        )
        result = _require_value(outcome)
        stored = StoredObject(
            ref=owned,
            etag=cast(str | None, getattr(result, "etag", None)),
            version_id=cast(str | None, getattr(result, "version_id", None)),
        )
        if self._require_versioning and stored.version_id is None:
            raise DependencyError("Object storage is unavailable")
        return stored

    def put_file(
        self,
        tenant_id: str,
        ref: ObjectRef,
        path: Path,
        *,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> StoredObject:
        self._ensure_open()
        try:
            source = path.open("rb")
        except OSError:
            unavailable = True
        else:
            unavailable = False
        if unavailable:
            raise AppError(ErrorCode.INTERNAL_ERROR, "Source file is unavailable", 500)
        with source:
            size = None
            try:
                size = os.fstat(source.fileno()).st_size
            except OSError:
                size = None
            if size is None:
                raise AppError(ErrorCode.INTERNAL_ERROR, "Source file is unavailable", 500)
            return self.put_stream(
                tenant_id,
                ref,
                source,
                length=size,
                content_type=content_type,
                metadata=metadata,
            )

    def presigned_get(
        self,
        tenant_id: str,
        ref: ObjectRef | str,
        *,
        expires: timedelta = timedelta(seconds=MAX_PRESIGNED_SECONDS),
    ) -> str:
        self._ensure_open()
        owned = self._owned_ref(tenant_id, ref)
        seconds = expires.total_seconds()
        if seconds < 1 or seconds > MAX_PRESIGNED_SECONDS:
            raise ValidationError(ErrorCode.REQUEST_INVALID, "Invalid object URL lifetime")
        outcome = _capture(
            lambda: self._client.presigned_get_object(
                self.bucket_name,
                owned.object_name,
                expires=expires,
            )
        )
        return cast(str, _require_value(outcome))

    def delete(self, tenant_id: str, ref: ObjectRef | str) -> None:
        self._ensure_open()
        owned = self._owned_ref(tenant_id, ref)
        outcome = _capture(lambda: self._client.remove_object(self.bucket_name, owned.object_name))
        _require_value(outcome)

    def delete_stored(self, tenant_id: str, stored: StoredObject) -> None:
        """Delete the exact object version created by a prior successful write."""
        self._ensure_open()
        owned = self._owned_ref(tenant_id, stored.ref)
        version_id = stored.version_id
        if version_id is not None and (
            not version_id
            or len(version_id) > 1024
            or any(ord(character) < 32 or ord(character) > 126 for character in version_id)
        ):
            raise DependencyError("Object storage is unavailable")
        if self._require_versioning and version_id is None:
            raise DependencyError("Object storage is unavailable")
        if version_id is None:
            outcome = _capture(lambda: self._client.remove_object(self.bucket_name, owned.object_name))
        else:
            outcome = _capture(
                lambda: self._client.remove_object(
                    self.bucket_name,
                    owned.object_name,
                    version_id=version_id,
                )
            )
        _require_value(outcome)

    def inspect_object(self, tenant_id: str, ref: ObjectRef | str) -> ObservedObject | None:
        """Inspect the current version of a server-generated object for crash reconciliation."""
        self._ensure_open()
        owned = self._owned_ref(tenant_id, ref)
        failure: _Failure | None = None
        try:
            result = self._client.stat_object(self.bucket_name, owned.object_name)
        except AppError:
            raise
        except Exception as exc:
            code = _exception_code(exc)
            if code in _NOT_FOUND_CODES:
                return None
            failure = _Failure(code, retryable=_is_retryable_exception(exc))
            result = None
        if failure is not None:
            _raise_dependency(failure)
        if result is None:
            raise DependencyError("Object storage is unavailable")

        size = getattr(result, "size", None)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DependencyError("Object storage is unavailable")
        stored = StoredObject(
            ref=owned,
            etag=_safe_optional_header(getattr(result, "etag", None)),
            version_id=_safe_optional_header(getattr(result, "version_id", None)),
        )
        if self._require_versioning and stored.version_id is None:
            raise DependencyError("Object storage is unavailable")
        metadata = _normalized_observed_metadata(getattr(result, "metadata", None))
        return ObservedObject(stored=stored, size=size, metadata=metadata)

    @contextmanager
    def open_stored(self, tenant_id: str, stored: StoredObject) -> Iterator[BinaryIO]:
        """Open exactly the admitted source version and always release the SDK response."""
        self._ensure_open()
        owned = self._owned_ref(tenant_id, stored.ref)
        version_id = _safe_optional_header(stored.version_id)
        if self._require_versioning and version_id is None:
            raise DependencyError("Object storage is unavailable")
        kwargs = {"version_id": version_id} if version_id is not None else {}
        outcome = _capture(
            lambda: self._client.get_object(
                self.bucket_name,
                owned.object_name,
                **kwargs,
            )
        )
        response = _require_value(outcome)
        if response is None or not callable(getattr(response, "read", None)):
            raise DependencyError("Object storage is unavailable")
        try:
            yield cast(BinaryIO, response)
        finally:
            close = getattr(response, "close", None)
            release = getattr(response, "release_conn", None)
            if callable(close):
                _safely_close(close)
            if callable(release):
                _safely_close(release)


def _valid_endpoint(endpoint: str) -> bool:
    if not endpoint or "://" in endpoint or any(character.isspace() for character in endpoint):
        return False
    try:
        parsed = urlsplit(f"//{endpoint}")
        _ = parsed.port
    except ValueError:
        return False
    return (
        bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
    )


def build_minio_store(
    settings: Settings,
    *,
    client_factory: Callable[..., MinioClient] = Minio,
    http_client_factory: Callable[..., PoolManager] = PoolManager,
) -> MinioStore:
    minio = settings.minio
    endpoint = minio.endpoint or ""
    access_key = minio.access_key.get_secret_value() if minio.access_key else ""
    secret_key = minio.secret_key.get_secret_value() if minio.secret_key else ""
    bucket_name = minio.bucket_name or ""
    invalid = (
        not _valid_endpoint(endpoint) or not access_key or not secret_key or not _BUCKET_NAME.fullmatch(bucket_name)
    )
    if settings.api.environment is AppEnvironment.PRODUCTION and not minio.secure:
        invalid = True
    if invalid:
        raise DependencyError("Object storage is not configured")
    timeout = Timeout(
        connect=minio.connect_timeout_seconds,
        read=minio.read_timeout_seconds,
    )
    retries = Retry(
        total=minio.max_retries,
        connect=minio.max_retries,
        read=minio.max_retries,
        redirect=False,
        status=minio.max_retries,
        other=0,
        allowed_methods=frozenset({"DELETE", "GET", "HEAD", "PUT"}),
        backoff_factor=0.2,
        backoff_max=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
        respect_retry_after_header=True,
        retry_after_max=2,
    )
    http_client = None
    try:
        http_client = http_client_factory(
            timeout=timeout,
            retries=retries,
            maxsize=10,
            block=False,
            cert_reqs="CERT_REQUIRED",
            ca_certs=os.environ.get("SSL_CERT_FILE") or certifi.where(),
        )
    except Exception:
        http_client = None
    if http_client is None:
        raise DependencyError("Object storage is not configured")
    try:
        outcome = _capture(
            lambda: client_factory(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=minio.secure,
                http_client=http_client,
            )
        )
    except AppError:
        _safely_close(http_client.clear)
        raise
    if outcome.failure is not None:
        _safely_close(http_client.clear)
    client = cast(MinioClient, _require_value(outcome))
    return MinioStore(
        client,
        bucket_name,
        require_versioning=(
            settings.api.environment is AppEnvironment.PRODUCTION
            and settings.api.import_enabled
        ),
        close_transport=http_client.clear,
    )
