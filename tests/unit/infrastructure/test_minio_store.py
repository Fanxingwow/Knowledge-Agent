from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr
from urllib3 import PoolManager, Retry, Timeout

import app.infrastructure.minio_store as minio_store_module
from app.core.errors import AppError, ErrorCode
from app.core.settings import ApiSettings, AppEnvironment, MinioSettings, Settings
from app.infrastructure.minio_store import (
    MAX_PRESIGNED_SECONDS,
    MinioStore,
    ObjectRef,
    ObservedObject,
    StoredObject,
    build_minio_store,
    build_object_ref,
    parse_object_ref,
)


class CodedError(Exception):
    def __init__(self, code: str, message: str = "upstream-secret") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class WriteResult:
    etag: str = "etag-1"
    version_id: str | None = "version-1"


@dataclass
class StatResult:
    size: int = 5
    etag: str = "etag-1"
    version_id: str | None = "version-1"
    metadata: dict[str, str] | None = None


@dataclass
class VersioningResult:
    status: str | None = "Enabled"


class ReadResponse(BytesIO):
    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.released = False

    def release_conn(self) -> None:
        self.released = True


class FakeMinioClient:
    def __init__(self) -> None:
        self.bucket_present = True
        self.policy: str | None = None
        self.policy_error: Exception | None = None
        self.failure: Exception | None = None
        self.stat_result = StatResult(metadata={})
        self.read_response = ReadResponse(b"hello")
        self.versioning_status: str | None = "Enabled"
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def bucket_exists(self, bucket_name: str) -> bool:
        self.calls.append(("bucket_exists", (bucket_name,), {}))
        if self.failure:
            raise self.failure
        return self.bucket_present

    def get_bucket_policy(self, bucket_name: str) -> str:
        self.calls.append(("get_bucket_policy", (bucket_name,), {}))
        if self.policy_error:
            raise self.policy_error
        if self.failure:
            raise self.failure
        return self.policy or ""

    def put_object(self, *args: Any, **kwargs: Any) -> WriteResult:
        self.calls.append(("put_object", args, kwargs))
        if self.failure:
            raise self.failure
        return WriteResult()

    def get_object(self, *args: Any, **kwargs: Any) -> ReadResponse:
        self.calls.append(("get_object", args, kwargs))
        if self.failure:
            raise self.failure
        return self.read_response

    def stat_object(self, *args: Any, **kwargs: Any) -> StatResult:
        self.calls.append(("stat_object", args, kwargs))
        if self.failure:
            raise self.failure
        return self.stat_result

    def get_bucket_versioning(self, bucket_name: str) -> VersioningResult:
        self.calls.append(("get_bucket_versioning", (bucket_name,), {}))
        if self.failure:
            raise self.failure
        return VersioningResult(self.versioning_status)

    def presigned_get_object(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append(("presigned_get_object", args, kwargs))
        if self.failure:
            raise self.failure
        return "https://storage.invalid/private?X-Amz-Signature=secret"

    def remove_object(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("remove_object", args, kwargs))
        if self.failure:
            raise self.failure


def private_store(client: FakeMinioClient | None = None) -> tuple[MinioStore, FakeMinioClient]:
    fake = client or FakeMinioClient()
    return MinioStore(fake, "knowledge-private"), fake


def test_readiness_requires_preprovisioned_bucket_without_mutating_it() -> None:
    store, client = private_store()
    client.bucket_present = False

    assert store.is_ready() is False
    with pytest.raises(AppError) as captured:
        store.require_ready()

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert [name for name, _, _ in client.calls] == ["bucket_exists", "bucket_exists"]


def test_no_bucket_policy_is_private() -> None:
    store, client = private_store()
    client.policy_error = CodedError("NoSuchBucketPolicy")

    assert store.is_ready() is True


def test_ready_bucket_passes_explicit_requirement() -> None:
    store, _ = private_store()

    store.require_ready()


def test_production_import_readiness_requires_bucket_versioning() -> None:
    client = FakeMinioClient()
    client.versioning_status = "Suspended"
    store = MinioStore(client, "knowledge-private", require_versioning=True)

    assert store.is_ready() is False
    with pytest.raises(AppError, match="versioning") as captured:
        store.require_ready()

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert client.calls[-1][0] == "get_bucket_versioning"


def test_restricted_bucket_policy_is_allowed() -> None:
    store, client = private_store()
    client.policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": {
                "Effect": "Allow",
                "Principal": {"AWS": ["arn:aws:iam::123456789012:role/kb-api"]},
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::knowledge-private/*",
            },
        }
    )

    assert store.is_ready() is True


@pytest.mark.parametrize(
    ("principal", "action_field"),
    [
        ("*", {"Action": "s3:GetObject"}),
        ({"AWS": "*"}, {"Action": "s3:Get*"}),
        ({"AWS": ["arn:example:role", "*"]}, {"NotAction": "s3:DeleteObject"}),
        ({"AWS": "arn:aws:iam::*:root"}, {"Action": "s3:ListBucket"}),
    ],
)
def test_public_bucket_policy_fails_closed(principal: object, action_field: dict[str, str]) -> None:
    store, client = private_store()
    client.policy = json.dumps(
        {
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": principal,
                    "Resource": "arn:aws:s3:::knowledge-private/*",
                    **action_field,
                }
            ]
        }
    )

    assert store.is_ready() is False
    with pytest.raises(AppError, match="private") as captured:
        store.require_ready()
    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE


def test_malformed_policy_fails_closed_without_policy_text() -> None:
    store, client = private_store()
    client.policy = "not-json private-policy-secret"

    with pytest.raises(AppError) as captured:
        store.require_ready()

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert "private-policy-secret" not in str(captured.value)


@pytest.mark.parametrize(
    "policy",
    [
        "[]",
        '{"Statement": "invalid"}',
        '{"Statement": ["invalid"]}',
    ],
)
def test_structurally_invalid_policy_fails_closed(policy: str) -> None:
    store, client = private_store()
    client.policy = policy

    assert store.is_ready() is False


def test_policy_lookup_failure_is_unavailable() -> None:
    store, client = private_store()
    client.policy_error = CodedError("AccessDenied", "credential-secret")

    assert store.is_ready() is False
    with pytest.raises(AppError) as captured:
        store.require_ready()
    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert "secret" not in str(captured.value)


def test_object_ref_round_trip_and_server_generated_key_shape() -> None:
    ref = build_object_ref("tenant-a", "images", "task-1", "aabbcc.png")

    assert ref == ObjectRef("tenant-a", "images/task-1/aabbcc.png")
    assert ref.object_name == "tenant-a/images/task-1/aabbcc.png"
    assert ref.uri == "kb-object://tenant-a/images/task-1/aabbcc.png"
    assert parse_object_ref(ref.uri) == ref


@pytest.mark.parametrize(
    ("tenant_id", "relative_key"),
    [
        ("tenant-a", "/images/x.png"),
        ("tenant-a", "images/x.png/"),
        ("tenant-a", "images\\x.png"),
        ("tenant-a", "a" * 1025),
        ("bad tenant", "images/x.png"),
    ],
)
def test_direct_object_ref_construction_is_fail_closed(tenant_id: str, relative_key: str) -> None:
    with pytest.raises(AppError) as captured:
        ObjectRef(tenant_id, relative_key)
    assert captured.value.code is ErrorCode.REQUEST_INVALID


def test_object_ref_requires_at_least_one_key_part() -> None:
    with pytest.raises(AppError) as captured:
        build_object_ref("tenant-a")
    assert captured.value.code is ErrorCode.REQUEST_INVALID


@pytest.mark.parametrize(
    ("tenant_id", "parts"),
    [
        ('tenant-a" or true', ("images", "task", "file.png")),
        (" tenant-a", ("images", "task", "file.png")),
        ("tenant-a", ("../images", "task", "file.png")),
        ("tenant-a", ("images", "task/other", "file.png")),
        ("tenant-a", ("images", "task", "..")),
        ("tenant-a", ("images", "task", "file\\name.png")),
    ],
)
def test_object_ref_rejects_injection_and_path_segments(tenant_id: str, parts: tuple[str, ...]) -> None:
    with pytest.raises(AppError) as captured:
        build_object_ref(tenant_id, *parts)
    assert captured.value.code is ErrorCode.REQUEST_INVALID


@pytest.mark.parametrize(
    "uri",
    [
        "https://tenant-a/images/x.png",
        "kb-object:///images/x.png",
        "kb-object://tenant-a/../tenant-b/x.png",
        "kb-object://tenant-a/images/x.png?X-Amz-Signature=secret",
        "kb-object://tenant-a/images//x.png",
        "kb-object://tenant-a//images/x.png",
        "kb-object://[invalid/images/x.png",
    ],
)
def test_object_ref_parser_rejects_unsafe_or_signed_values(uri: str) -> None:
    with pytest.raises(AppError) as captured:
        parse_object_ref(uri)
    assert captured.value.code is ErrorCode.REQUEST_INVALID
    assert captured.value.__context__ is None


def test_put_stream_uses_tenant_scoped_name_and_returns_stable_ref() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.pdf")
    data = BytesIO(b"%PDF-1.7")

    stored = store.put_stream("tenant-a", ref, data, length=8, content_type="application/pdf")

    assert stored.ref == ref
    assert stored.etag == "etag-1"
    operation, args, kwargs = client.calls[-1]
    assert operation == "put_object"
    assert args[:2] == ("knowledge-private", ref.object_name)
    assert kwargs["data"] is data
    assert kwargs["length"] == 8


def test_put_stream_validates_length_content_type_and_metadata_before_sdk() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.pdf")

    for kwargs in (
        {"length": True, "content_type": "application/pdf"},
        {"length": 1, "content_type": "application/pdf\r\nX-Leak: yes"},
        {"length": 1, "content_type": "application/pdf", "metadata": {"name": "bad\x00value"}},
    ):
        with pytest.raises(AppError) as captured:
            store.put_stream("tenant-a", ref, BytesIO(b"x"), **kwargs)  # type: ignore[arg-type]
        assert captured.value.code is ErrorCode.REQUEST_INVALID

    assert client.calls == []


def test_put_stream_copies_safe_metadata() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.pdf")
    metadata = {"object-kind": "source"}

    store.put_stream("tenant-a", ref, BytesIO(b"x"), length=1, content_type="application/pdf", metadata=metadata)

    assert client.calls[-1][2]["metadata"] == metadata


def test_put_file_streams_local_file(tmp_path: Path) -> None:
    store, client = private_store()
    source = tmp_path / "source.md"
    source.write_bytes(b"hello")
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.md")

    stored = store.put_file("tenant-a", ref, source, content_type="text/markdown")

    assert stored.ref == ref
    _, _, kwargs = client.calls[-1]
    assert kwargs["length"] == 5
    assert kwargs["data"].closed is True


def test_missing_local_file_returns_safe_internal_error(tmp_path: Path) -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "documents", "task-1", "missing.md")

    with pytest.raises(AppError) as captured:
        store.put_file("tenant-a", ref, tmp_path / "missing.md", content_type="text/markdown")

    assert captured.value.code is ErrorCode.INTERNAL_ERROR
    assert client.calls == []


def test_local_file_stat_failure_does_not_retain_sensitive_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, client = private_store()
    source = tmp_path / "source.md"
    source.write_bytes(b"hello")
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.md")

    def fail_stat(file_descriptor: int) -> object:
        del file_descriptor
        raise OSError("sensitive-local-path")

    monkeypatch.setattr(minio_store_module.os, "fstat", fail_stat)

    with pytest.raises(AppError) as captured:
        store.put_file("tenant-a", ref, source, content_type="text/markdown")

    assert captured.value.code is ErrorCode.INTERNAL_ERROR
    assert captured.value.__context__ is None
    assert client.calls == []


def test_cross_tenant_ref_is_hidden_before_presigning() -> None:
    store, client = private_store()
    other_ref = build_object_ref("tenant-b", "images", "task-1", "image.png")

    with pytest.raises(AppError) as captured:
        store.presigned_get("tenant-a", other_ref)

    assert captured.value.code is ErrorCode.RESOURCE_NOT_FOUND
    assert not client.calls


def test_presigned_get_is_capped_at_fifteen_minutes() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "images", "task-1", "image.png")

    url = store.presigned_get("tenant-a", ref)

    assert "X-Amz-Signature" in url
    _, args, kwargs = client.calls[-1]
    assert args[:2] == ("knowledge-private", ref.object_name)
    assert kwargs["expires"] == timedelta(seconds=MAX_PRESIGNED_SECONDS)

    with pytest.raises(AppError) as captured:
        store.presigned_get("tenant-a", ref, expires=timedelta(seconds=MAX_PRESIGNED_SECONDS + 1))
    assert captured.value.code is ErrorCode.REQUEST_INVALID


def test_exact_delete_rejects_cross_tenant_reference() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-b", "images", "task-1", "image.png")

    with pytest.raises(AppError) as captured:
        store.delete("tenant-a", ref)

    assert captured.value.code is ErrorCode.RESOURCE_NOT_FOUND
    assert not client.calls


def test_exact_delete_calls_sdk_for_owned_reference() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "images", "task-1", "image.png")

    store.delete("tenant-a", ref.uri)

    assert client.calls[-1] == ("remove_object", ("knowledge-private", ref.object_name), {})


def test_dependency_failures_are_sanitized_and_retryable_when_appropriate() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "images", "task-1", "image.png")
    client.failure = CodedError("SlowDown", "https://secret.invalid/?X-Amz-Signature=leak")

    with pytest.raises(AppError) as captured:
        store.presigned_get("tenant-a", ref)

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert captured.value.details == {"retryable": True}
    assert "secret" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_network_timeout_without_s3_code_is_retryable() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "images", "task-1", "image.png")
    client.failure = TimeoutError("https://secret.invalid/?X-Amz-Signature=leak")

    with pytest.raises(AppError) as captured:
        store.presigned_get("tenant-a", ref)

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert captured.value.details == {"retryable": True}
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize(
    ("code", "expected"),
    [("NoSuchKey", ErrorCode.RESOURCE_NOT_FOUND), ("AccessDenied", ErrorCode.DEPENDENCY_UNAVAILABLE)],
)
def test_dependency_error_codes_are_normalized(code: str, expected: ErrorCode) -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "images", "task-1", "image.png")
    client.failure = CodedError(code, "upstream-secret")

    with pytest.raises(AppError) as captured:
        store.presigned_get("tenant-a", ref)

    assert captured.value.code is expected
    assert "secret" not in str(captured.value)


def test_application_errors_from_adapter_boundary_are_preserved() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "images", "task-1", "image.png")
    sentinel = AppError(ErrorCode.PERMISSION_DENIED, "safe denial", 403)
    client.failure = sentinel

    with pytest.raises(AppError) as captured:
        store.presigned_get("tenant-a", ref)

    assert captured.value is sentinel


def minio_settings(**overrides: Any) -> MinioSettings:
    configured = MinioSettings.model_construct(
        endpoint="minio.internal:9000",
        access_key=SecretStr("access-secret"),
        secret_key=SecretStr("secret-secret"),
        bucket_name="knowledge-private",
        secure=True,
    )
    return configured.model_copy(update=overrides)


def test_factory_uses_typed_secrets_bounded_transport_and_no_network_side_effect() -> None:
    captured: dict[str, object] = {}
    fake = FakeMinioClient()

    def factory(**kwargs: object) -> FakeMinioClient:
        captured.update(kwargs)
        return fake

    settings = Settings.model_construct(
        api=ApiSettings.model_construct(environment=AppEnvironment.PRODUCTION),
        minio=minio_settings(),
    )
    store = build_minio_store(settings, client_factory=factory)

    assert isinstance(store, MinioStore)
    http_client = captured.pop("http_client")
    assert captured == {
        "endpoint": "minio.internal:9000",
        "access_key": "access-secret",
        "secret_key": "secret-secret",
        "secure": True,
    }
    assert isinstance(http_client, PoolManager)
    timeout = http_client.connection_pool_kw["timeout"]
    retries = http_client.connection_pool_kw["retries"]
    assert isinstance(timeout, Timeout)
    assert timeout.connect_timeout == 3.0
    assert timeout.read_timeout == 10.0
    assert isinstance(retries, Retry)
    assert retries.total == 2
    assert retries.connect == 2
    assert retries.read == 2
    assert retries.status == 2
    assert retries.other == 0
    assert retries.redirect == 0
    assert retries.raise_on_redirect is False
    assert retries.allowed_methods == frozenset({"DELETE", "GET", "HEAD", "PUT"})
    assert retries.backoff_max == 2.0
    assert retries.retry_after_max == 2
    assert retries.raise_on_status is False
    assert retries.is_retry("GET", 503) is True
    assert retries.is_retry("POST", 503) is False
    assert http_client.connection_pool_kw["maxsize"] == 10
    assert http_client.connection_pool_kw["block"] is False
    assert http_client.connection_pool_kw["cert_reqs"] == "CERT_REQUIRED"
    assert isinstance(http_client.connection_pool_kw["ca_certs"], str)
    assert fake.calls == []
    store.close()


def test_factory_transport_is_closed_once_and_store_fails_readiness_after_close() -> None:
    class PoolStub:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear(self) -> None:
            self.clear_calls += 1

    pool = PoolStub()
    settings = Settings.model_construct(
        api=ApiSettings.model_construct(environment=AppEnvironment.PRODUCTION),
        minio=minio_settings(),
    )
    store = build_minio_store(
        settings,
        client_factory=lambda **_: FakeMinioClient(),
        http_client_factory=lambda **_: cast(PoolManager, pool),
    )

    store.close()
    store.close()

    assert pool.clear_calls == 1
    assert store.is_ready() is False
    with pytest.raises(AppError) as captured:
        store.require_ready()
    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    ref = build_object_ref("tenant-a", "images", "task-1", "closed.png")
    with pytest.raises(AppError) as operation_error:
        store.presigned_get("tenant-a", ref)
    assert operation_error.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE


def test_factory_cleans_transport_when_client_construction_fails() -> None:
    class PoolStub:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear(self) -> None:
            self.clear_calls += 1

    pool = PoolStub()
    settings = Settings.model_construct(
        api=ApiSettings.model_construct(environment=AppEnvironment.PRODUCTION),
        minio=minio_settings(),
    )

    def failing_factory(**_: object) -> FakeMinioClient:
        raise RuntimeError("secret")

    with pytest.raises(AppError):
        build_minio_store(
            settings,
            client_factory=failing_factory,
            http_client_factory=lambda **_: cast(PoolManager, pool),
        )

    assert pool.clear_calls == 1


def test_http_transport_factory_failure_is_sanitized_without_context() -> None:
    settings = Settings.model_construct(
        api=ApiSettings.model_construct(environment=AppEnvironment.PRODUCTION),
        minio=minio_settings(),
    )

    def failing_http_factory(**_: object) -> PoolManager:
        raise RuntimeError("SSL_CERT_FILE=sensitive-path")

    with pytest.raises(AppError) as captured:
        build_minio_store(settings, http_client_factory=failing_http_factory)

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert "sensitive" not in str(captured.value)
    assert captured.value.__context__ is None


def test_factory_cleans_transport_when_client_raises_application_error() -> None:
    class PoolStub:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear(self) -> None:
            self.clear_calls += 1

    pool = PoolStub()
    sentinel = AppError(ErrorCode.PERMISSION_DENIED, "safe denial", 403)
    settings = Settings.model_construct(
        api=ApiSettings.model_construct(environment=AppEnvironment.PRODUCTION),
        minio=minio_settings(),
    )

    def failing_factory(**_: object) -> FakeMinioClient:
        raise sentinel

    with pytest.raises(AppError) as captured:
        build_minio_store(
            settings,
            client_factory=failing_factory,
            http_client_factory=lambda **_: cast(PoolManager, pool),
        )

    assert captured.value is sentinel
    assert pool.clear_calls == 1


def test_transport_cleanup_failure_does_not_mask_constructor_error() -> None:
    class FailingPool:
        def clear(self) -> None:
            raise RuntimeError("cleanup-secret")

    settings = Settings.model_construct(
        api=ApiSettings.model_construct(environment=AppEnvironment.PRODUCTION),
        minio=minio_settings(),
    )

    def failing_factory(**_: object) -> FakeMinioClient:
        raise RuntimeError("constructor-secret")

    with pytest.raises(AppError) as captured:
        build_minio_store(
            settings,
            client_factory=failing_factory,
            http_client_factory=lambda **_: cast(PoolManager, FailingPool()),
        )

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert "secret" not in str(captured.value)


def test_factory_allows_insecure_transport_only_outside_production() -> None:
    fake = FakeMinioClient()
    settings = Settings.model_construct(
        api=ApiSettings.model_construct(environment=AppEnvironment.DEVELOPMENT),
        minio=minio_settings(secure=False),
    )

    store = build_minio_store(settings, client_factory=lambda **_: fake)

    assert isinstance(store, MinioStore)


@pytest.mark.parametrize(
    "minio",
    [
        minio_settings(endpoint="https://minio.internal:9000"),
        minio_settings(endpoint="minio.internal:9000/path"),
        minio_settings(endpoint="minio.internal:not-a-port"),
        minio_settings(secure=False),
    ],
)
def test_factory_rejects_unsafe_endpoint_or_production_transport(minio: MinioSettings) -> None:
    settings = Settings.model_construct(
        api=ApiSettings.model_construct(environment=AppEnvironment.PRODUCTION),
        minio=minio,
    )

    with pytest.raises(AppError) as captured:
        build_minio_store(settings, client_factory=lambda **_: FakeMinioClient())

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert "minio.internal" not in str(captured.value)


def test_factory_constructor_failure_is_sanitized() -> None:
    settings = Settings.model_construct(
        api=ApiSettings.model_construct(environment=AppEnvironment.PRODUCTION),
        minio=minio_settings(),
    )

    def failing_factory(**_: object) -> FakeMinioClient:
        raise RuntimeError("access-secret secret-secret endpoint-secret")

    with pytest.raises(AppError) as captured:
        build_minio_store(settings, client_factory=failing_factory)

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert "secret" not in str(captured.value)
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"X-Name": "value"},
        {"x-amz-acl": "public-read"},
        {"content-disposition": "attachment"},
        {"cache-control": "public, max-age=31536000"},
        {"content-encoding": "gzip"},
        {"content-language": "en"},
        {"source": "中文"},
        {"source": "bad\tvalue"},
        {f"key-{index}": "value" for index in range(9)},
        {"source": "a" * 1024, "digest": "b" * 1024},
    ],
)
def test_metadata_control_headers_and_oversized_values_are_rejected_before_sdk(
    metadata: dict[str, str],
) -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.pdf")

    with pytest.raises(AppError) as captured:
        store.put_stream(
            "tenant-a",
            ref,
            BytesIO(b"x"),
            length=1,
            content_type="application/pdf",
            metadata=metadata,
        )

    assert captured.value.code is ErrorCode.REQUEST_INVALID
    assert client.calls == []


def test_delete_stored_uses_exact_version_id() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.pdf")
    stored = StoredObject(ref=ref, etag="etag-1", version_id="version-1")

    store.delete_stored("tenant-a", stored)

    assert client.calls[-1] == (
        "remove_object",
        ("knowledge-private", ref.object_name),
        {"version_id": "version-1"},
    )


def test_delete_stored_hides_cross_tenant_version_before_sdk() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-b", "documents", "task-1", "source.pdf")

    with pytest.raises(AppError) as captured:
        store.delete_stored(
            "tenant-a",
            StoredObject(ref=ref, etag=None, version_id="version-secret"),
        )

    assert captured.value.code is ErrorCode.RESOURCE_NOT_FOUND
    assert client.calls == []


def test_inspect_object_returns_exact_version_and_normalized_metadata() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.md")
    client.stat_result = StatResult(
        size=5,
        etag="etag-2",
        version_id="version-2",
        metadata={
            "x-amz-meta-import-id": "a" * 32,
            "x-amz-meta-sha256": "b" * 64,
            "server": "must-not-be-forwarded",
        },
    )

    observed = store.inspect_object("tenant-a", ref)

    assert observed == ObservedObject(
        stored=StoredObject(ref=ref, etag="etag-2", version_id="version-2"),
        size=5,
        metadata={"import-id": "a" * 32, "sha256": "b" * 64},
    )
    assert client.calls[-1] == ("stat_object", ("knowledge-private", ref.object_name), {})


def test_inspect_object_returns_none_only_for_not_found() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.md")
    client.failure = CodedError("NoSuchKey")

    assert store.inspect_object("tenant-a", ref) is None

    client.failure = CodedError("AccessDenied", "credential-secret")
    with pytest.raises(AppError) as captured:
        store.inspect_object("tenant-a", ref)
    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert "secret" not in str(captured.value)


def test_open_stored_reads_exact_version_and_releases_response() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.md")
    stored = StoredObject(ref=ref, etag="etag-1", version_id="version-1")

    with store.open_stored("tenant-a", stored) as source:
        assert source.read() == b"hello"

    assert client.calls[-1] == (
        "get_object",
        ("knowledge-private", ref.object_name),
        {"version_id": "version-1"},
    )
    assert client.read_response.closed is True
    assert client.read_response.released is True


def test_versioned_store_rejects_missing_version_before_read() -> None:
    client = FakeMinioClient()
    store = MinioStore(client, "knowledge-private", require_versioning=True)
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.md")

    with pytest.raises(AppError) as captured:
        with store.open_stored(
            "tenant-a",
            StoredObject(ref=ref, etag="etag-1", version_id=None),
        ):
            pass

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert client.calls == []


def test_versioned_store_rejects_missing_put_version_and_unversioned_delete() -> None:
    client = FakeMinioClient()
    store = MinioStore(client, "knowledge-private", require_versioning=True)
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.md")

    original_put = client.put_object

    def unversioned_put(*args: Any, **kwargs: Any) -> WriteResult:
        result = original_put(*args, **kwargs)
        result.version_id = None
        return result

    client.put_object = unversioned_put  # type: ignore[method-assign]
    with pytest.raises(AppError) as put_error:
        store.put_stream(
            "tenant-a",
            ref,
            BytesIO(b"x"),
            length=1,
            content_type="text/markdown",
        )
    assert put_error.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE

    calls_before_delete = len(client.calls)
    with pytest.raises(AppError) as delete_error:
        store.delete_stored(
            "tenant-a",
            StoredObject(ref=ref, etag="etag-1", version_id=None),
        )
    assert delete_error.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert len(client.calls) == calls_before_delete


def test_inspect_object_sanitizes_exception_context() -> None:
    store, client = private_store()
    ref = build_object_ref("tenant-a", "documents", "task-1", "source.md")
    client.failure = CodedError("AccessDenied", "credential-secret")

    with pytest.raises(AppError) as captured:
        store.inspect_object("tenant-a", ref)

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert captured.value.__context__ is None
