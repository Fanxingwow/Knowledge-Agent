import pytest
from pydantic import ValidationError

from app.core.settings import get_settings, reset_settings_cache


@pytest.fixture(autouse=True)
def isolate_settings_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_secrets_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "top-secret")
    reset_settings_cache()
    settings = get_settings()
    assert settings.llm.api_key is not None
    assert settings.llm.api_key.get_secret_value() == "top-secret"
    assert "top-secret" not in repr(settings)


def test_production_rejects_disabled_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUTH_MODE", "disabled")
    reset_settings_cache()
    with pytest.raises(ValidationError, match="AUTH_MODE"):
        get_settings()


def test_production_metrics_require_a_scrape_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("AUTH_ISSUER", "https://issuer.example")
    monkeypatch.setenv("AUTH_AUDIENCE", "kb-api")
    monkeypatch.setenv("AUTH_OIDC_CLIENT_ID", "kb-console")
    monkeypatch.setenv("AUTH_JWKS_URL", "https://issuer.example/.well-known/jwks.json")
    monkeypatch.setenv("METRICS_ENABLED", "true")
    monkeypatch.delenv("METRICS_AUTH_TOKEN", raising=False)

    with pytest.raises(ValidationError, match="METRICS_AUTH_TOKEN"):
        get_settings()


def test_metrics_configuration_is_typed_and_secret_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "metrics-" + "x" * 32
    monkeypatch.setenv("METRICS_ENABLED", "true")
    monkeypatch.setenv("METRICS_PATH", "/internal/metrics")
    monkeypatch.setenv("METRICS_AUTH_TOKEN", token)

    settings = get_settings()

    assert settings.observability.metrics_enabled is True
    assert settings.observability.metrics_path == "/internal/metrics"
    assert settings.observability.metrics_auth_token is not None
    assert settings.observability.metrics_auth_token.get_secret_value() == token
    assert token not in repr(settings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("METRICS_PATH", "metrics"),
        ("METRICS_PATH", "/metrics/"),
        ("METRICS_PATH", "/metrics/{tenant_id}"),
        ("METRICS_AUTH_TOKEN", "too-short"),
    ],
)
def test_metrics_configuration_rejects_unsafe_values(
    name: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        get_settings()


def test_production_jwt_requires_jwks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("AUTH_ISSUER", "https://issuer.example")
    monkeypatch.setenv("AUTH_AUDIENCE", "kb-api")
    monkeypatch.setenv("AUTH_OIDC_CLIENT_ID", "kb-console")
    monkeypatch.setenv("AUTH_PUBLIC_KEY", "test-only-key")
    monkeypatch.setenv("MINERU_BASE_URL", "https://mineru.example/api/v4")
    monkeypatch.delenv("AUTH_JWKS_URL", raising=False)
    with pytest.raises(ValidationError, match="AUTH_JWKS_URL"):
        get_settings()


def test_jwt_mode_requires_verification_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("AUTH_ISSUER", "https://issuer.example")
    monkeypatch.setenv("AUTH_AUDIENCE", "kb-api")
    monkeypatch.setenv("AUTH_OIDC_CLIENT_ID", "kb-console")
    monkeypatch.delenv("AUTH_JWKS_URL", raising=False)
    monkeypatch.delenv("AUTH_PUBLIC_KEY", raising=False)
    reset_settings_cache()
    with pytest.raises(ValidationError, match="AUTH_JWKS_URL"):
        get_settings()


def test_upload_hard_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_MAX_UPLOAD_BYTES", str(501 * 1024 * 1024))
    reset_settings_cache()
    with pytest.raises(ValidationError, match="500 MiB"):
        get_settings()


def test_wildcard_cors_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_CORS_ORIGINS", "*")
    with pytest.raises(ValidationError, match="wildcard"):
        get_settings()


def test_empty_values_in_env_template_use_typed_defaults(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("APP_ENV=\nAPI_MAX_UPLOAD_FILES=\nMINIO_SECURE=\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IMPORT_ENABLED", "false")
    reset_settings_cache()
    settings = get_settings()
    assert settings.api.environment.value == "development"
    assert settings.api.import_enabled is False
    assert settings.api.max_upload_files == 5
    assert settings.minio.secure is False


def test_minio_network_and_probe_limits_are_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIO_CONNECT_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("MINIO_READ_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setenv("MINIO_MAX_RETRIES", "1")
    monkeypatch.setenv("MINIO_READINESS_INTERVAL_SECONDS", "12.5")
    monkeypatch.setenv("MINIO_READINESS_PROBE_TIMEOUT_SECONDS", "4.5")
    monkeypatch.setenv("MINIO_READINESS_TTL_SECONDS", "20")

    settings = get_settings()

    assert settings.minio.connect_timeout_seconds == 1.5
    assert settings.minio.read_timeout_seconds == 7.5
    assert settings.minio.max_retries == 1
    assert settings.minio.readiness_interval_seconds == 12.5
    assert settings.minio.readiness_probe_timeout_seconds == 4.5
    assert settings.minio.readiness_ttl_seconds == 20


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MINIO_CONNECT_TIMEOUT_SECONDS", "31"),
        ("MINIO_READ_TIMEOUT_SECONDS", "121"),
        ("MINIO_MAX_RETRIES", "6"),
        ("MINIO_READINESS_INTERVAL_SECONDS", "0"),
        ("MINIO_READINESS_PROBE_TIMEOUT_SECONDS", "61"),
        ("MINIO_READINESS_TTL_SECONDS", "901"),
    ],
)
def test_minio_network_and_probe_limits_are_bounded(
    name: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        get_settings()


def test_minio_readiness_ttl_must_cover_interval_and_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINIO_READINESS_INTERVAL_SECONDS", "10")
    monkeypatch.setenv("MINIO_READINESS_PROBE_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("MINIO_READINESS_TTL_SECONDS", "24.9")

    with pytest.raises(ValidationError, match="cover the probe interval"):
        get_settings()


def test_legacy_leading_slash_object_prefixes_are_safely_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINIO_IMG_DIR", "/upload-images/")
    monkeypatch.setenv("MINIO_PDF_DIR", "/documents/")

    settings = get_settings()

    assert settings.minio.image_prefix == "upload-images"
    assert settings.minio.pdf_prefix == "documents"


@pytest.mark.parametrize("value", ["../documents", "nested/documents", "/", "bad prefix"])
def test_unsafe_object_prefix_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIO_PDF_DIR", value)

    with pytest.raises(ValidationError, match="one safe path segment"):
        get_settings()


def test_import_work_root_cannot_escape_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPORT_WORK_ROOT", "../outside")

    with pytest.raises(ValidationError, match="cannot escape"):
        get_settings()


def test_upload_batch_limit_cannot_be_smaller_than_single_file_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_MAX_UPLOAD_BYTES", "1024")
    monkeypatch.setenv("API_MAX_UPLOAD_BATCH_BYTES", "1023")

    with pytest.raises(ValidationError, match="cannot be smaller"):
        get_settings()


def test_import_worker_heartbeat_must_fit_inside_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMPORT_WORKER_LEASE_SECONDS", "30")
    monkeypatch.setenv("IMPORT_WORKER_HEARTBEAT_SECONDS", "15")

    with pytest.raises(ValidationError, match="less than half"):
        get_settings()


def test_import_worker_heartbeat_must_fit_inside_write_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMPORT_WORKER_LEASE_SECONDS", "120")
    monkeypatch.setenv("IMPORT_WORKER_HEARTBEAT_SECONDS", "30")
    monkeypatch.setenv("IMPORT_WRITE_LEASE_SECONDS", "60")

    with pytest.raises(ValidationError, match="less than half the write lease"):
        get_settings()


def test_import_write_lease_must_cover_bounded_minio_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMPORT_ENABLED", "true")
    monkeypatch.setenv("IMPORT_WORKER_LEASE_SECONDS", "60")
    monkeypatch.setenv("IMPORT_WORKER_HEARTBEAT_SECONDS", "10")
    monkeypatch.setenv("IMPORT_WRITE_LEASE_SECONDS", "100")
    monkeypatch.setenv("MINIO_CONNECT_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("MINIO_READ_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("MINIO_MAX_RETRIES", "2")

    with pytest.raises(ValidationError, match="bounded MinIO PUT window"):
        get_settings()


def test_import_retry_window_is_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPORT_WORKER_RETRY_BASE_SECONDS", "20")
    monkeypatch.setenv("IMPORT_WORKER_RETRY_MAX_SECONDS", "10")

    with pytest.raises(ValidationError, match="cannot be smaller"):
        get_settings()


def test_import_workspace_orphan_age_must_cover_two_worker_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMPORT_WORKER_LEASE_SECONDS", "300")
    monkeypatch.setenv("IMPORT_WORKER_HEARTBEAT_SECONDS", "100")
    monkeypatch.setenv("IMPORT_WORKSPACE_ORPHAN_AGE_SECONDS", "599")

    with pytest.raises(ValidationError, match="cover at least two worker leases"):
        get_settings()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("IMPORT_WORKSPACE_ORPHAN_AGE_SECONDS", "299"),
        ("IMPORT_WORKSPACE_ORPHAN_AGE_SECONDS", "2592001"),
        ("IMPORT_WORKSPACE_SWEEP_BATCH_SIZE", "0"),
        ("IMPORT_WORKSPACE_SWEEP_BATCH_SIZE", "1001"),
    ],
)
def test_import_workspace_sweep_settings_are_bounded(
    name: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        get_settings()


def _configure_production_jwt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("AUTH_ISSUER", "https://issuer.example")
    monkeypatch.setenv("AUTH_AUDIENCE", "kb-api")
    monkeypatch.setenv("AUTH_OIDC_CLIENT_ID", "kb-console")
    monkeypatch.setenv("AUTH_JWKS_URL", "https://issuer.example/.well-known/jwks.json")
    monkeypatch.setenv("MINERU_BASE_URL", "https://mineru.example/api/v4")


def test_jwt_mode_requires_console_oidc_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("AUTH_ISSUER", "https://issuer.example")
    monkeypatch.setenv("AUTH_AUDIENCE", "kb-api")
    monkeypatch.setenv("AUTH_JWKS_URL", "https://issuer.example/.well-known/jwks.json")
    monkeypatch.delenv("AUTH_OIDC_CLIENT_ID", raising=False)

    with pytest.raises(ValidationError, match="AUTH_OIDC_CLIENT_ID"):
        get_settings()


def test_production_rejects_insecure_oidc_issuer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_production_jwt(monkeypatch)
    monkeypatch.setenv("AUTH_ISSUER", "http://issuer.example")

    with pytest.raises(ValidationError, match="AUTH_ISSUER must use HTTPS"):
        get_settings()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("IMPORT_REPOSITORY", "memory", "IMPORT_REPOSITORY"),
        ("IMPORT_REQUIRE_IDEMPOTENCY_KEY", "false", "IMPORT_REQUIRE_IDEMPOTENCY_KEY"),
        ("IMPORT_WORKER_ENABLED", "false", "IMPORT_WORKER_ENABLED"),
        ("MONGO_REQUIRE_REPLICA_SET", "false", "MONGO_REQUIRE_REPLICA_SET"),
    ],
)
def test_production_import_runtime_is_fail_closed(
    name: str,
    value: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_production_jwt(monkeypatch)
    monkeypatch.setenv("IMPORT_ENABLED", "true")
    monkeypatch.setenv("IMPORT_REPOSITORY", "mongo")
    monkeypatch.setenv("IMPORT_REQUIRE_IDEMPOTENCY_KEY", "true")
    monkeypatch.setenv("IMPORT_WORKER_ENABLED", "true")
    monkeypatch.setenv("MONGO_URL", "mongodb://mongo.invalid:27017")
    monkeypatch.setenv("MONGO_REQUIRE_REPLICA_SET", "true")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=message):
        get_settings()


def test_production_import_runtime_accepts_durable_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_production_jwt(monkeypatch)
    monkeypatch.setenv("IMPORT_ENABLED", "true")
    monkeypatch.setenv("IMPORT_REPOSITORY", "mongo")
    monkeypatch.setenv("IMPORT_REQUIRE_IDEMPOTENCY_KEY", "true")
    monkeypatch.setenv("IMPORT_WORKER_ENABLED", "true")
    monkeypatch.setenv("MONGO_URL", "mongodb://mongo.invalid:27017")
    monkeypatch.setenv("MONGO_REQUIRE_REPLICA_SET", "true")

    settings = get_settings()

    assert settings.import_runtime.repository == "mongo"
    assert settings.mongo.require_replica_set is True


def test_knowledge_search_requires_authoritative_mongo_milvus_and_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEARCH_ENABLED", "true")
    monkeypatch.setenv("IMPORT_REPOSITORY", "memory")

    with pytest.raises(ValidationError, match="IMPORT_REPOSITORY"):
        get_settings()

    monkeypatch.setenv("IMPORT_REPOSITORY", "mongo")
    monkeypatch.setenv("MONGO_URL", "mongodb://mongo.invalid:27017")
    monkeypatch.setenv("MILVUS_URL", "http://milvus.invalid:19530")
    monkeypatch.setenv("BGE_M3_PATH", "relative/model")
    reset_settings_cache()

    with pytest.raises(ValidationError, match="absolute local model"):
        get_settings()


def test_knowledge_search_typed_policy_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setenv("SEARCH_MAX_TOP_K", "12")
    monkeypatch.setenv("SEARCH_MAX_CANDIDATES", "120")
    monkeypatch.setenv("SEARCH_MAX_CONCURRENT_REQUESTS", "7")
    monkeypatch.setenv("SEARCH_INITIAL_OVERFETCH_FACTOR", "4")
    monkeypatch.setenv("SEARCH_GROWTH_FACTOR", "3")
    monkeypatch.setenv("SEARCH_DENSE_WEIGHT", "0.7")
    monkeypatch.setenv("SEARCH_SPARSE_WEIGHT", "0.3")
    monkeypatch.setenv("SEARCH_READINESS_INTERVAL_SECONDS", "2")
    monkeypatch.setenv("SEARCH_READINESS_PROBE_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("SEARCH_READINESS_TTL_SECONDS", "5")

    settings = get_settings()

    assert settings.search.timeout_seconds == 7.5
    assert settings.search.max_top_k == 12
    assert settings.search.max_candidates == 120
    assert settings.search.max_concurrent_requests == 7
    assert settings.search.initial_overfetch_factor == 4
    assert settings.search.growth_factor == 3
    assert settings.search.dense_weight == 0.7
    assert settings.search.sparse_weight == 0.3
    assert settings.search.readiness_ttl_seconds == 5


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("SEARCH_MAX_TOP_K", "21", "less than or equal to 20"),
        ("SEARCH_DENSE_WEIGHT", "0", "ranker weight"),
        ("SEARCH_READINESS_TTL_SECONDS", "69.9", "cover the probe interval"),
    ],
)
def test_knowledge_search_policy_fails_closed(
    name: str,
    value: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if name == "SEARCH_DENSE_WEIGHT":
        monkeypatch.setenv("SEARCH_SPARSE_WEIGHT", "0")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=message):
        get_settings()


def test_chat_requires_transactional_mongo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_ENABLED", "true")
    monkeypatch.setenv("IMPORT_REPOSITORY", "mongo")
    monkeypatch.setenv("MONGO_URL", "mongodb://mongo.invalid:27017")
    monkeypatch.setenv("MONGO_REQUIRE_REPLICA_SET", "false")

    with pytest.raises(ValidationError, match="MONGO_REQUIRE_REPLICA_SET"):
        get_settings()


def test_production_chat_requires_distributed_redis_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_production_jwt(monkeypatch)
    monkeypatch.setenv("IMPORT_ENABLED", "false")
    monkeypatch.setenv("SEARCH_ENABLED", "false")
    monkeypatch.setenv("CHAT_ENABLED", "true")
    monkeypatch.setenv("IMPORT_REPOSITORY", "mongo")
    monkeypatch.setenv("MONGO_URL", "mongodb://mongo.invalid:27017")
    monkeypatch.setenv("MONGO_REQUIRE_REPLICA_SET", "true")
    monkeypatch.setenv("CHAT_SSE_BROKER", "memory")

    with pytest.raises(ValidationError, match="CHAT_SSE_BROKER"):
        get_settings()

    monkeypatch.setenv("CHAT_SSE_BROKER", "redis")
    reset_settings_cache()
    with pytest.raises(ValidationError, match="REDIS_URL"):
        get_settings()

    monkeypatch.setenv("REDIS_URL", "rediss://redis.invalid:6379/0")
    reset_settings_cache()
    with pytest.raises(ValidationError, match="REDIS_CHAT_KEY_HMAC_SECRET"):
        get_settings()

    monkeypatch.setenv("REDIS_CHAT_KEY_HMAC_SECRET", "k" * 32)
    settings = get_settings()
    assert settings.chat.sse_broker == "redis"
    assert settings.redis.chat_key_hmac_secret is not None
    assert "kkkk" not in repr(settings)


def test_chat_runtime_policy_is_typed_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_SSE_REPLAY_CAPACITY", "100")
    monkeypatch.setenv("CHAT_SSE_MAX_REPLAY_EVENTS", "101")

    with pytest.raises(ValidationError, match="replay capacity"):
        get_settings()

    monkeypatch.setenv("CHAT_SSE_MAX_REPLAY_EVENTS", "50")
    monkeypatch.setenv("CHAT_READINESS_INTERVAL_SECONDS", "3")
    monkeypatch.setenv("CHAT_READINESS_PROBE_TIMEOUT_SECONDS", "4")
    monkeypatch.setenv("CHAT_READINESS_TTL_SECONDS", "7")
    reset_settings_cache()

    settings = get_settings()

    assert settings.chat.replay_capacity == 100
    assert settings.chat.max_replay_events == 50
    assert settings.chat.readiness_ttl_seconds == 7


@pytest.mark.parametrize("value", ["bad.name", "$system", "nested/name", "bad-name"])
def test_mongo_collection_names_are_safe(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONGO_IMPORT_TASKS_COLLECTION", value)

    with pytest.raises(ValidationError, match="safe identifiers"):
        get_settings()


def test_mineru_network_limits_are_typed_and_secret_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_BASE_URL", "https://mineru.example/")
    monkeypatch.setenv("MINERU_API_TOKEN", "mineru-secret")
    monkeypatch.setenv("MINERU_CONNECT_TIMEOUT_SECONDS", "1.25")
    monkeypatch.setenv("MINERU_READ_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("MINERU_TOTAL_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("MINERU_PROBE_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("MINERU_MAX_RETRIES", "3")
    monkeypatch.setenv("MINERU_RETRY_BASE_SECONDS", "0.2")
    monkeypatch.setenv("MINERU_RETRY_MAX_SECONDS", "2")
    monkeypatch.setenv("MINERU_MAX_REQUEST_BYTES", "1024")
    monkeypatch.setenv("MINERU_MAX_RESPONSE_BYTES", "2048")

    settings = get_settings()

    assert settings.mineru.base_url == "https://mineru.example"
    assert settings.mineru.connect_timeout_seconds == 1.25
    assert settings.mineru.read_timeout_seconds == 12.5
    assert settings.mineru.total_timeout_seconds == 60
    assert settings.mineru.probe_timeout_seconds == 2.5
    assert settings.mineru.max_retries == 3
    assert settings.mineru.retry_base_seconds == 0.2
    assert settings.mineru.retry_max_seconds == 2
    assert settings.mineru.max_request_bytes == 1024
    assert settings.mineru.max_response_bytes == 2048
    assert "mineru-secret" not in repr(settings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MINERU_CONNECT_TIMEOUT_SECONDS", "31"),
        ("MINERU_READ_TIMEOUT_SECONDS", "301"),
        ("MINERU_TOTAL_TIMEOUT_SECONDS", "3601"),
        ("MINERU_PROBE_TIMEOUT_SECONDS", "31"),
        ("MINERU_MAX_RETRIES", "6"),
        ("MINERU_MAX_REQUEST_BYTES", str(501 * 1024 * 1024)),
        ("MINERU_MAX_RESPONSE_BYTES", str(251 * 1024 * 1024)),
    ],
)
def test_mineru_network_limits_are_bounded(
    name: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        get_settings()


def test_mineru_total_timeout_must_exceed_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_READ_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("MINERU_TOTAL_TIMEOUT_SECONDS", "60")

    with pytest.raises(ValidationError, match="must exceed"):
        get_settings()


def test_mineru_retry_window_is_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_RETRY_BASE_SECONDS", "3")
    monkeypatch.setenv("MINERU_RETRY_MAX_SECONDS", "2")

    with pytest.raises(ValidationError, match="cannot be smaller"):
        get_settings()


@pytest.mark.parametrize(
    "url",
    [
        "ftp://mineru.example",
        "https://user:password@mineru.example",  # pragma: allowlist secret
        "https://mineru.example?token=secret",
        "https://mineru.example#fragment",
        "https://mineru.example:invalid-port",
        "https://mineru.example/../admin",
        " https://mineru.example",
    ],
)
def test_mineru_url_rejects_unsafe_forms(
    url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_BASE_URL", url)

    with pytest.raises(ValidationError, match="service origin"):
        get_settings()


def test_mineru_token_rejects_whitespace_only_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_TOKEN", "   ")

    with pytest.raises(ValidationError, match="MINERU_API_TOKEN"):
        get_settings()


def test_production_mineru_requires_https(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_production_jwt(monkeypatch)
    monkeypatch.setenv("MINERU_BASE_URL", "http://mineru.internal")

    with pytest.raises(ValidationError, match="HTTPS"):
        get_settings()


def test_embedding_and_milvus_pipeline_settings_are_typed_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("BGE_MODEL_BATCH_SIZE", "16")
    monkeypatch.setenv("BGE_MAX_TOKENS", "4096")
    monkeypatch.setenv("MILVUS_URL", "https://milvus.example")
    monkeypatch.setenv("MILVUS_TOKEN", "milvus-secret")
    monkeypatch.setenv("IMPORT_PIPELINE_VERSION", "bge-m3.v2")
    monkeypatch.setenv("MILVUS_WRITE_BATCH_SIZE", "64")
    monkeypatch.setenv("MILVUS_MAX_CHUNKS_PER_IMPORT", "2000")
    monkeypatch.setenv("MILVUS_TIMEOUT_SECONDS", "7.5")

    settings = get_settings()

    assert settings.embedding.dense_dimension == 768
    assert settings.embedding.model_batch_size == 16
    assert settings.embedding.max_tokens == 4096
    assert settings.milvus.pipeline_version == "bge-m3.v2"
    assert settings.milvus.write_batch_size == 64
    assert settings.milvus.max_chunks_per_import == 2000
    assert settings.milvus.timeout_seconds == 7.5
    assert "milvus-secret" not in repr(settings)


def test_milvus_credentials_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILVUS_TOKEN", "token")
    monkeypatch.setenv("MILVUS_USERNAME", "user")
    monkeypatch.setenv("MILVUS_PASSWORD", "password")

    with pytest.raises(ValidationError, match="cannot be combined"):
        get_settings()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("IMPORT_PIPELINE_VERSION", "bad version", "PIPELINE_VERSION"),
        ("CHUNKS_COLLECTION", "bad-name", "collection names"),
        ("MILVUS_TOKEN", "   ", "credential"),
        ("MILVUS_WRITE_BATCH_SIZE", "10", "chunk limit"),
    ],
)
def test_milvus_pipeline_rejects_invalid_configuration(
    name: str,
    value: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if name == "MILVUS_WRITE_BATCH_SIZE":
        monkeypatch.setenv("MILVUS_MAX_CHUNKS_PER_IMPORT", "5")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=message):
        get_settings()


def test_production_forbids_application_side_milvus_schema_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_production_jwt(monkeypatch)
    monkeypatch.setenv("MILVUS_ALLOW_SCHEMA_CREATE", "true")

    with pytest.raises(ValidationError, match="MILVUS_ALLOW_SCHEMA_CREATE"):
        get_settings()
