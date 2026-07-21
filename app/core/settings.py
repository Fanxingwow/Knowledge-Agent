from __future__ import annotations

import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_OBJECT_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_COLLECTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
_PIPELINE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ACCOUNT_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class AppEnvironment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class DomainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


class ApiSettings(DomainSettings):
    environment: AppEnvironment = Field(
        AppEnvironment.DEVELOPMENT,
        validation_alias="APP_ENV",
    )
    import_enabled: bool = Field(False, validation_alias="IMPORT_ENABLED")
    project_root: Path | None = Field(None, validation_alias="PROJECT_ROOT")
    import_work_root: Path = Field(
        Path("output/imports"),
        validation_alias="IMPORT_WORK_ROOT",
    )
    cors_origins_raw: str = Field(
        "http://127.0.0.1:8000,http://127.0.0.1:8001",
        validation_alias="API_CORS_ORIGINS",
    )
    max_upload_files: int = Field(
        5,
        ge=1,
        le=20,
        validation_alias="API_MAX_UPLOAD_FILES",
    )
    max_upload_bytes: int = Field(
        100 * 1024 * 1024,
        ge=1,
        validation_alias="API_MAX_UPLOAD_BYTES",
    )
    max_upload_batch_bytes: int = Field(
        500 * 1024 * 1024,
        ge=1,
        validation_alias="API_MAX_UPLOAD_BATCH_BYTES",
    )

    @field_validator("max_upload_bytes")
    @classmethod
    def validate_upload_hard_limit(cls, value: int) -> int:
        if value > 500 * 1024 * 1024:
            raise ValueError("API_MAX_UPLOAD_BYTES cannot exceed 500 MiB")
        return value

    @field_validator("max_upload_batch_bytes")
    @classmethod
    def validate_upload_batch_hard_limit(cls, value: int) -> int:
        if value > 2 * 1024 * 1024 * 1024:
            raise ValueError("API_MAX_UPLOAD_BATCH_BYTES cannot exceed 2 GiB")
        return value

    @model_validator(mode="after")
    def validate_upload_limits(self) -> ApiSettings:
        if self.max_upload_batch_bytes < self.max_upload_bytes:
            raise ValueError("API_MAX_UPLOAD_BATCH_BYTES cannot be smaller than API_MAX_UPLOAD_BYTES")
        return self

    @field_validator("import_work_root")
    @classmethod
    def validate_import_work_root(cls, value: Path) -> Path:
        if value in {Path("."), Path("..")}:
            raise ValueError("IMPORT_WORK_ROOT must identify a dedicated directory")
        if not value.is_absolute() and ".." in value.parts:
            raise ValueError("IMPORT_WORK_ROOT cannot escape the project root")
        if value.is_absolute() and value == Path(value.anchor):
            raise ValueError("IMPORT_WORK_ROOT cannot be a filesystem root")
        return value

    @field_validator("cors_origins_raw")
    @classmethod
    def reject_wildcard_cors(cls, value: str) -> str:
        if "*" in {item.strip() for item in value.split(",")}:
            raise ValueError("CORS wildcard is forbidden when credentials are enabled")
        return value

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


class ImportRuntimeSettings(DomainSettings):
    repository: Literal["memory", "mongo"] = Field(
        "memory",
        validation_alias="IMPORT_REPOSITORY",
    )
    worker_enabled: bool = Field(True, validation_alias="IMPORT_WORKER_ENABLED")
    require_idempotency_key: bool = Field(
        False,
        validation_alias="IMPORT_REQUIRE_IDEMPOTENCY_KEY",
    )
    poll_interval_seconds: float = Field(
        0.5,
        ge=0.05,
        le=30.0,
        validation_alias="IMPORT_WORKER_POLL_INTERVAL_SECONDS",
    )
    lease_seconds: int = Field(
        120,
        ge=15,
        le=3600,
        validation_alias="IMPORT_WORKER_LEASE_SECONDS",
    )
    heartbeat_seconds: float = Field(
        30.0,
        ge=1.0,
        le=1200.0,
        validation_alias="IMPORT_WORKER_HEARTBEAT_SECONDS",
    )
    write_lease_seconds: int = Field(
        600,
        ge=60,
        le=7200,
        validation_alias="IMPORT_WRITE_LEASE_SECONDS",
    )
    max_attempts: int = Field(
        5,
        ge=1,
        le=20,
        validation_alias="IMPORT_WORKER_MAX_ATTEMPTS",
    )
    retry_base_seconds: float = Field(
        2.0,
        ge=0.1,
        le=300.0,
        validation_alias="IMPORT_WORKER_RETRY_BASE_SECONDS",
    )
    retry_max_seconds: float = Field(
        300.0,
        ge=1.0,
        le=3600.0,
        validation_alias="IMPORT_WORKER_RETRY_MAX_SECONDS",
    )
    cleanup_poll_interval_seconds: float = Field(
        2.0,
        ge=0.1,
        le=300.0,
        validation_alias="IMPORT_CLEANUP_POLL_INTERVAL_SECONDS",
    )
    workspace_orphan_age_seconds: int = Field(
        86_400,
        ge=300,
        le=2_592_000,
        validation_alias="IMPORT_WORKSPACE_ORPHAN_AGE_SECONDS",
    )
    workspace_sweep_batch_size: int = Field(
        20,
        ge=1,
        le=1_000,
        validation_alias="IMPORT_WORKSPACE_SWEEP_BATCH_SIZE",
    )
    shutdown_grace_seconds: float = Field(
        30.0,
        ge=1.0,
        le=600.0,
        validation_alias="IMPORT_SHUTDOWN_GRACE_SECONDS",
    )

    @model_validator(mode="after")
    def validate_worker_windows(self) -> ImportRuntimeSettings:
        if self.heartbeat_seconds * 2 >= self.lease_seconds:
            raise ValueError("IMPORT_WORKER_HEARTBEAT_SECONDS must be less than half the lease")
        if self.heartbeat_seconds * 2 >= self.write_lease_seconds:
            raise ValueError("IMPORT_WORKER_HEARTBEAT_SECONDS must be less than half the write lease")
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("IMPORT_WORKER_RETRY_MAX_SECONDS cannot be smaller than the base")
        if self.workspace_orphan_age_seconds < self.lease_seconds * 2:
            raise ValueError("IMPORT_WORKSPACE_ORPHAN_AGE_SECONDS must cover at least two worker leases")
        return self


class AuthSettings(DomainSettings):
    mode: Literal["disabled", "jwt"] = Field("disabled", validation_alias="AUTH_MODE")
    issuer: str | None = Field(None, validation_alias="AUTH_ISSUER")
    audience: str | None = Field(None, validation_alias="AUTH_AUDIENCE")
    oidc_client_id: str | None = Field(None, validation_alias="AUTH_OIDC_CLIENT_ID")
    jwks_url: str | None = Field(None, validation_alias="AUTH_JWKS_URL")
    public_key: SecretStr | None = Field(None, validation_alias="AUTH_PUBLIC_KEY")
    algorithms_raw: str = Field("RS256", validation_alias="AUTH_ALGORITHMS")
    dev_tenant_id: str = Field("local-dev", validation_alias="AUTH_DEV_TENANT_ID")

    @property
    def algorithms(self) -> list[str]:
        return [item.strip() for item in self.algorithms_raw.split(",") if item.strip()]

    @field_validator("oidc_client_id")
    @classmethod
    def validate_oidc_client_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not 1 <= len(normalized) <= 128 or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
            for character in normalized
        ):
            raise ValueError("AUTH_OIDC_CLIENT_ID is invalid")
        return normalized


class LlmSettings(DomainSettings):
    api_key: SecretStr | None = Field(None, validation_alias="OPENAI_API_KEY")
    base_url: str | None = Field(None, validation_alias="OPENAI_BASE_URL")
    default_model: str | None = Field(None, validation_alias="LLM_DEFAULT_MODEL")
    temperature: float = Field(
        0.1,
        ge=0,
        le=2,
        validation_alias="LLM_DEFAULT_TEMPERATURE",
    )
    vision_model: str | None = Field(None, validation_alias="VL_MODEL")


class MinioSettings(DomainSettings):
    endpoint: str | None = Field(None, validation_alias="MINIO_ENDPOINT")
    access_key: SecretStr | None = Field(None, validation_alias="MINIO_ACCESS_KEY")
    secret_key: SecretStr | None = Field(None, validation_alias="MINIO_SECRET_KEY")
    bucket_name: str | None = Field(None, validation_alias="MINIO_BUCKET_NAME")
    image_prefix: str = Field("images", validation_alias="MINIO_IMG_DIR")
    pdf_prefix: str = Field("pdf_files", validation_alias="MINIO_PDF_DIR")
    secure: bool = Field(False, validation_alias="MINIO_SECURE")
    connect_timeout_seconds: float = Field(
        3.0,
        ge=0.1,
        le=30.0,
        validation_alias="MINIO_CONNECT_TIMEOUT_SECONDS",
    )
    read_timeout_seconds: float = Field(
        10.0,
        ge=0.1,
        le=120.0,
        validation_alias="MINIO_READ_TIMEOUT_SECONDS",
    )
    max_retries: int = Field(
        2,
        ge=0,
        le=5,
        validation_alias="MINIO_MAX_RETRIES",
    )
    readiness_interval_seconds: float = Field(
        10.0,
        ge=0.1,
        le=300.0,
        validation_alias="MINIO_READINESS_INTERVAL_SECONDS",
    )
    readiness_probe_timeout_seconds: float = Field(
        15.0,
        ge=0.1,
        le=60.0,
        validation_alias="MINIO_READINESS_PROBE_TIMEOUT_SECONDS",
    )
    readiness_ttl_seconds: float = Field(
        30.0,
        ge=0.2,
        le=900.0,
        validation_alias="MINIO_READINESS_TTL_SECONDS",
    )

    @field_validator("image_prefix", "pdf_prefix")
    @classmethod
    def validate_object_prefix(cls, value: str) -> str:
        normalized = value.strip("/")
        if not _OBJECT_PREFIX.fullmatch(normalized):
            raise ValueError("MinIO object prefixes must be one safe path segment")
        return normalized

    @model_validator(mode="after")
    def validate_readiness_window(self) -> MinioSettings:
        minimum_ttl = self.readiness_interval_seconds + self.readiness_probe_timeout_seconds
        if self.readiness_ttl_seconds < minimum_ttl:
            raise ValueError("MINIO_READINESS_TTL_SECONDS must cover the probe interval and timeout")
        return self


class EmbeddingSettings(DomainSettings):
    model_path: str | None = Field(None, validation_alias="BGE_M3_PATH")
    model_id: str = Field("BAAI/bge-m3", validation_alias="BGE_M3")
    device: str = Field("cpu", validation_alias="BGE_DEVICE")
    fp16: bool = Field(False, validation_alias="BGE_FP16")
    dense_dimension: int = Field(
        1024,
        ge=1,
        le=32768,
        validation_alias="EMBEDDING_DIM",
    )
    model_batch_size: int = Field(
        32,
        ge=1,
        le=1024,
        validation_alias="BGE_MODEL_BATCH_SIZE",
    )
    max_tokens: int = Field(
        8192,
        ge=1,
        le=131072,
        validation_alias="BGE_MAX_TOKENS",
    )


class RerankerSettings(DomainSettings):
    model_path: str | None = Field(None, validation_alias="BGE_RERANKER_LARGE")
    device: str = Field("cpu", validation_alias="BGE_RERANKER_DEVICE")
    fp16: bool = Field(False, validation_alias="BGE_RERANKER_FP16")


class MilvusSettings(DomainSettings):
    url: str | None = Field(None, validation_alias="MILVUS_URL")
    token: SecretStr | None = Field(None, validation_alias="MILVUS_TOKEN")
    username: str | None = Field(None, validation_alias="MILVUS_USERNAME")
    password: SecretStr | None = Field(None, validation_alias="MILVUS_PASSWORD")
    database: str | None = Field(None, validation_alias="MILVUS_DATABASE")
    chunks_collection: str = Field("kb_chunks_v2", validation_alias="CHUNKS_COLLECTION")
    item_name_collection: str = Field(
        "kb_item_names_v2",
        validation_alias="ITEM_NAME_COLLECTION",
    )
    entity_name_collection: str | None = Field(None, validation_alias="ENTITY_NAME_COLLECTION")
    pipeline_version: str = Field(
        "bge-m3.v1",
        validation_alias="IMPORT_PIPELINE_VERSION",
    )
    allow_schema_create: bool = Field(
        False,
        validation_alias="MILVUS_ALLOW_SCHEMA_CREATE",
    )
    write_batch_size: int = Field(
        128,
        ge=1,
        le=1000,
        validation_alias="MILVUS_WRITE_BATCH_SIZE",
    )
    max_chunks_per_import: int = Field(
        10000,
        ge=1,
        le=100000,
        validation_alias="MILVUS_MAX_CHUNKS_PER_IMPORT",
    )
    timeout_seconds: float = Field(
        5.0,
        ge=0.1,
        le=120.0,
        validation_alias="MILVUS_TIMEOUT_SECONDS",
    )

    @field_validator(
        "chunks_collection",
        "item_name_collection",
        "entity_name_collection",
    )
    @classmethod
    def validate_collection_name(cls, value: str | None) -> str | None:
        if value is not None and _COLLECTION_NAME.fullmatch(value) is None:
            raise ValueError("Milvus collection names must be safe identifiers")
        return value

    @field_validator("pipeline_version")
    @classmethod
    def validate_pipeline_version(cls, value: str) -> str:
        if _PIPELINE_VERSION.fullmatch(value) is None:
            raise ValueError("IMPORT_PIPELINE_VERSION is invalid")
        return value

    @field_validator("username")
    @classmethod
    def validate_optional_username(cls, value: str | None) -> str | None:
        if value is not None and _ACCOUNT_IDENTIFIER.fullmatch(value) is None:
            raise ValueError("Milvus account identifier must be safe")
        return value

    @field_validator("database")
    @classmethod
    def validate_optional_database(cls, value: str | None) -> str | None:
        if value is not None and _COLLECTION_NAME.fullmatch(value) is None:
            raise ValueError("Milvus database identifier must be safe")
        return value

    @field_validator("token", "password")
    @classmethod
    def validate_optional_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        if not secret.strip() or len(secret) > 4096 or "\x00" in secret:
            raise ValueError("Milvus credential is invalid")
        return value

    @model_validator(mode="after")
    def validate_credentials_and_limits(self) -> MilvusSettings:
        if self.token is not None and (self.username is not None or self.password is not None):
            raise ValueError("MILVUS_TOKEN cannot be combined with username/password")
        if (self.username is None) != (self.password is None):
            raise ValueError("MILVUS_USERNAME and MILVUS_PASSWORD must be configured together")
        if self.write_batch_size > self.max_chunks_per_import:
            raise ValueError("MILVUS_WRITE_BATCH_SIZE cannot exceed the import chunk limit")
        return self


class KnowledgeSearchSettings(DomainSettings):
    enabled: bool = Field(False, validation_alias="SEARCH_ENABLED")
    timeout_seconds: float = Field(
        10.0,
        ge=0.1,
        le=60.0,
        validation_alias="SEARCH_TIMEOUT_SECONDS",
    )
    max_top_k: int = Field(
        20,
        ge=1,
        le=20,
        validation_alias="SEARCH_MAX_TOP_K",
    )
    max_candidates: int = Field(
        500,
        ge=1,
        le=500,
        validation_alias="SEARCH_MAX_CANDIDATES",
    )
    max_concurrent_requests: int = Field(
        16,
        ge=1,
        le=256,
        validation_alias="SEARCH_MAX_CONCURRENT_REQUESTS",
    )
    initial_overfetch_factor: int = Field(
        3,
        ge=1,
        le=20,
        validation_alias="SEARCH_INITIAL_OVERFETCH_FACTOR",
    )
    growth_factor: int = Field(
        2,
        ge=2,
        le=10,
        validation_alias="SEARCH_GROWTH_FACTOR",
    )
    dense_weight: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        validation_alias="SEARCH_DENSE_WEIGHT",
    )
    sparse_weight: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        validation_alias="SEARCH_SPARSE_WEIGHT",
    )
    readiness_interval_seconds: float = Field(
        10.0,
        ge=0.1,
        le=300.0,
        validation_alias="SEARCH_READINESS_INTERVAL_SECONDS",
    )
    readiness_probe_timeout_seconds: float = Field(
        60.0,
        ge=0.1,
        le=300.0,
        validation_alias="SEARCH_READINESS_PROBE_TIMEOUT_SECONDS",
    )
    readiness_ttl_seconds: float = Field(
        80.0,
        ge=0.2,
        le=900.0,
        validation_alias="SEARCH_READINESS_TTL_SECONDS",
    )

    @model_validator(mode="after")
    def validate_search_policy(self) -> KnowledgeSearchSettings:
        if self.max_top_k > self.max_candidates:
            raise ValueError("SEARCH_MAX_TOP_K cannot exceed SEARCH_MAX_CANDIDATES")
        if self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("At least one search ranker weight must be positive")
        minimum_ttl = self.readiness_interval_seconds + self.readiness_probe_timeout_seconds
        if self.readiness_ttl_seconds < minimum_ttl:
            raise ValueError("SEARCH_READINESS_TTL_SECONDS must cover the probe interval and timeout")
        return self


class ChatSettings(DomainSettings):
    enabled: bool = Field(False, validation_alias="CHAT_ENABLED")
    sse_broker: Literal["memory", "redis"] = Field(
        "memory",
        validation_alias="CHAT_SSE_BROKER",
    )
    max_sessions: int = Field(
        1_000,
        ge=1,
        le=100_000,
        validation_alias="CHAT_SSE_MAX_SESSIONS",
    )
    replay_capacity: int = Field(
        256,
        ge=1,
        le=10_000,
        validation_alias="CHAT_SSE_REPLAY_CAPACITY",
    )
    max_replay_events: int = Field(
        128,
        ge=1,
        le=10_000,
        validation_alias="CHAT_SSE_MAX_REPLAY_EVENTS",
    )
    subscriber_queue_capacity: int = Field(
        64,
        ge=1,
        le=10_000,
        validation_alias="CHAT_SSE_SUBSCRIBER_QUEUE_CAPACITY",
    )
    max_subscribers_per_session: int = Field(
        8,
        ge=1,
        le=1_000,
        validation_alias="CHAT_SSE_MAX_SUBSCRIBERS_PER_SESSION",
    )
    heartbeat_seconds: float = Field(
        15.0,
        ge=0.1,
        le=300.0,
        validation_alias="CHAT_SSE_HEARTBEAT_SECONDS",
    )
    readiness_interval_seconds: float = Field(
        10.0,
        ge=0.1,
        le=300.0,
        validation_alias="CHAT_READINESS_INTERVAL_SECONDS",
    )
    readiness_probe_timeout_seconds: float = Field(
        5.0,
        ge=0.1,
        le=60.0,
        validation_alias="CHAT_READINESS_PROBE_TIMEOUT_SECONDS",
    )
    readiness_ttl_seconds: float = Field(
        20.0,
        ge=0.2,
        le=900.0,
        validation_alias="CHAT_READINESS_TTL_SECONDS",
    )

    @model_validator(mode="after")
    def validate_chat_runtime(self) -> ChatSettings:
        if self.max_replay_events > self.replay_capacity:
            raise ValueError("CHAT_SSE_MAX_REPLAY_EVENTS cannot exceed replay capacity")
        minimum_ttl = self.readiness_interval_seconds + self.readiness_probe_timeout_seconds
        if self.readiness_ttl_seconds < minimum_ttl:
            raise ValueError("CHAT_READINESS_TTL_SECONDS must cover the probe interval and timeout")
        return self


class MongoSettings(DomainSettings):
    url: SecretStr | None = Field(None, validation_alias="MONGO_URL")
    database: str = Field("knowledge_base", validation_alias="MONGO_DB_NAME")
    import_admissions_collection: str = Field(
        "import_admissions_v1",
        validation_alias="MONGO_IMPORT_ADMISSIONS_COLLECTION",
    )
    import_tasks_collection: str = Field(
        "import_tasks_v1",
        validation_alias="MONGO_IMPORT_TASKS_COLLECTION",
    )
    import_outbox_collection: str = Field(
        "import_outbox_v1",
        validation_alias="MONGO_IMPORT_OUTBOX_COLLECTION",
    )
    chat_sessions_collection: str = Field(
        "chat_sessions_v1",
        validation_alias="MONGO_CHAT_SESSIONS_COLLECTION",
    )
    chat_messages_collection: str = Field(
        "chat_messages_v1",
        validation_alias="MONGO_CHAT_MESSAGES_COLLECTION",
    )
    connect_timeout_seconds: float = Field(
        3.0,
        ge=0.1,
        le=30.0,
        validation_alias="MONGO_CONNECT_TIMEOUT_SECONDS",
    )
    server_selection_timeout_seconds: float = Field(
        5.0,
        ge=0.1,
        le=60.0,
        validation_alias="MONGO_SERVER_SELECTION_TIMEOUT_SECONDS",
    )
    socket_timeout_seconds: float = Field(
        30.0,
        ge=1.0,
        le=300.0,
        validation_alias="MONGO_SOCKET_TIMEOUT_SECONDS",
    )
    health_timeout_seconds: float = Field(
        2.0,
        ge=0.1,
        le=30.0,
        validation_alias="MONGO_HEALTH_TIMEOUT_SECONDS",
    )
    transaction_timeout_seconds: float = Field(
        10.0,
        ge=1.0,
        le=120.0,
        validation_alias="MONGO_TRANSACTION_TIMEOUT_SECONDS",
    )
    wait_queue_timeout_seconds: float = Field(
        5.0,
        ge=0.1,
        le=60.0,
        validation_alias="MONGO_WAIT_QUEUE_TIMEOUT_SECONDS",
    )
    max_pool_size: int = Field(
        50,
        ge=1,
        le=500,
        validation_alias="MONGO_MAX_POOL_SIZE",
    )
    terminal_retention_days: int = Field(
        30,
        ge=1,
        le=3650,
        validation_alias="MONGO_IMPORT_TERMINAL_RETENTION_DAYS",
    )
    require_replica_set: bool = Field(
        False,
        validation_alias="MONGO_REQUIRE_REPLICA_SET",
    )

    @field_validator(
        "database",
        "import_admissions_collection",
        "import_tasks_collection",
        "import_outbox_collection",
        "chat_sessions_collection",
        "chat_messages_collection",
    )
    @classmethod
    def validate_database_identifier(cls, value: str) -> str:
        if not _COLLECTION_NAME.fullmatch(value):
            raise ValueError("MongoDB database and collection names must be safe identifiers")
        return value

    @model_validator(mode="after")
    def validate_distinct_import_collections(self) -> MongoSettings:
        collections = {
            self.import_admissions_collection,
            self.import_tasks_collection,
            self.import_outbox_collection,
            self.chat_sessions_collection,
            self.chat_messages_collection,
        }
        if len(collections) != 5:
            raise ValueError("MongoDB application collections must be distinct")
        return self


class Neo4jSettings(DomainSettings):
    uri: str | None = Field(None, validation_alias="NEO4J_URI")
    username: str | None = Field(None, validation_alias="NEO4J_USERNAME")
    password: SecretStr | None = Field(None, validation_alias="NEO4J_PASSWORD")
    database: str = Field("neo4j", validation_alias="NEO4J_DATABASE")
    enabled: bool = Field(False, validation_alias="KG_ENABLED")


class MineruSettings(DomainSettings):
    base_url: str | None = Field(None, validation_alias="MINERU_BASE_URL")
    api_key: SecretStr | None = Field(None, validation_alias="MINERU_API_TOKEN")
    connect_timeout_seconds: float = Field(
        3.0,
        ge=0.1,
        le=30.0,
        validation_alias="MINERU_CONNECT_TIMEOUT_SECONDS",
    )
    read_timeout_seconds: float = Field(
        60.0,
        ge=0.1,
        le=300.0,
        validation_alias="MINERU_READ_TIMEOUT_SECONDS",
    )
    total_timeout_seconds: float = Field(
        600.0,
        ge=1.0,
        le=3600.0,
        validation_alias="MINERU_TOTAL_TIMEOUT_SECONDS",
    )
    probe_timeout_seconds: float = Field(
        5.0,
        ge=0.1,
        le=30.0,
        validation_alias="MINERU_PROBE_TIMEOUT_SECONDS",
    )
    max_retries: int = Field(
        2,
        ge=0,
        le=5,
        validation_alias="MINERU_MAX_RETRIES",
    )
    retry_base_seconds: float = Field(
        0.5,
        ge=0.05,
        le=30.0,
        validation_alias="MINERU_RETRY_BASE_SECONDS",
    )
    retry_max_seconds: float = Field(
        5.0,
        ge=0.1,
        le=120.0,
        validation_alias="MINERU_RETRY_MAX_SECONDS",
    )
    max_request_bytes: int = Field(
        100 * 1024 * 1024,
        ge=1,
        le=500 * 1024 * 1024,
        validation_alias="MINERU_MAX_REQUEST_BYTES",
    )
    max_response_bytes: int = Field(
        64 * 1024 * 1024,
        ge=1,
        le=250 * 1024 * 1024,
        validation_alias="MINERU_MAX_RESPONSE_BYTES",
    )
    max_archive_bytes: int = Field(
        500 * 1024 * 1024,
        ge=1,
        validation_alias="MINERU_MAX_ARCHIVE_BYTES",
    )
    max_archive_entry_bytes: int = Field(
        250 * 1024 * 1024,
        ge=1,
        validation_alias="MINERU_MAX_ARCHIVE_ENTRY_BYTES",
    )
    max_extracted_bytes: int = Field(
        1024 * 1024 * 1024,
        ge=1,
        validation_alias="MINERU_MAX_EXTRACTED_BYTES",
    )
    max_archive_members: int = Field(
        5000,
        ge=1,
        le=20000,
        validation_alias="MINERU_MAX_ARCHIVE_MEMBERS",
    )
    max_compression_ratio: float = Field(
        100.0,
        ge=1,
        validation_alias="MINERU_MAX_COMPRESSION_RATIO",
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.rstrip("/")
        parsed = urlsplit(normalized)
        try:
            _ = parsed.port
        except ValueError:
            raise ValueError("MINERU_BASE_URL must be an HTTP(S) service origin") from None
        if (
            not normalized.isascii()
            or not normalized.isprintable()
            or any(character.isspace() for character in normalized)
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "\\" in parsed.path
            or ".." in Path(parsed.path).parts
        ):
            raise ValueError("MINERU_BASE_URL must be an HTTP(S) service origin")
        return normalized

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        if not secret.strip() or len(secret) > 4096 or "\x00" in secret:
            raise ValueError("MINERU_API_TOKEN is invalid")
        return value

    @model_validator(mode="after")
    def validate_network_limits(self) -> MineruSettings:
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("MINERU_RETRY_MAX_SECONDS cannot be smaller than the base")
        minimum_request_window = max(
            self.connect_timeout_seconds,
            self.read_timeout_seconds,
        )
        if self.total_timeout_seconds <= minimum_request_window:
            raise ValueError("MINERU_TOTAL_TIMEOUT_SECONDS must exceed the per-request timeout")
        return self


class McpSettings(DomainSettings):
    base_url: str | None = Field(None, validation_alias="MCP_DASHSCOPE_BASE_URL")


class LoggingSettings(DomainSettings):
    console_enabled: bool = Field(True, validation_alias="LOG_CONSOLE_ENABLE")
    console_level: str = Field("INFO", validation_alias="LOG_CONSOLE_LEVEL")
    file_enabled: bool = Field(True, validation_alias="LOG_FILE_ENABLE")
    file_level: str = Field("INFO", validation_alias="LOG_FILE_LEVEL")
    retention: str = Field("7 days", validation_alias="LOG_FILE_RETENTION")
    json_output: bool = Field(False, validation_alias="LOG_JSON")
    diagnose: bool = Field(False, validation_alias="LOG_DIAGNOSE")


class ObservabilitySettings(DomainSettings):
    metrics_enabled: bool = Field(False, validation_alias="METRICS_ENABLED")
    metrics_path: str = Field("/metrics", validation_alias="METRICS_PATH")
    metrics_auth_token: SecretStr | None = Field(
        None,
        validation_alias="METRICS_AUTH_TOKEN",
    )

    @field_validator("metrics_path")
    @classmethod
    def validate_metrics_path(cls, value: str) -> str:
        if (
            not 2 <= len(value) <= 64
            or not value.startswith("/")
            or value.endswith("/")
            or not re.fullmatch(r"/[a-z][a-z0-9/_-]*", value)
            or "{" in value
            or "}" in value
        ):
            raise ValueError("METRICS_PATH is invalid")
        return value

    @field_validator("metrics_auth_token")
    @classmethod
    def validate_metrics_auth_token(
        cls,
        value: SecretStr | None,
    ) -> SecretStr | None:
        if value is None:
            return None
        token = value.get_secret_value()
        try:
            encoded = token.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ValueError("METRICS_AUTH_TOKEN is invalid") from None
        if not 32 <= len(encoded) <= 4_096 or "\x00" in token:
            raise ValueError("METRICS_AUTH_TOKEN is invalid")
        return value


class PostgresSettings(DomainSettings):
    dsn: SecretStr | None = Field(None, validation_alias="POSTGRES_DSN")


class RedisSettings(DomainSettings):
    url: SecretStr | None = Field(None, validation_alias="REDIS_URL")
    chat_key_hmac_secret: SecretStr | None = Field(
        None,
        validation_alias="REDIS_CHAT_KEY_HMAC_SECRET",
    )
    key_prefix: str = Field(
        "knowledge-agent:chat:v1",
        validation_alias="REDIS_KEY_PREFIX",
    )
    connect_timeout_seconds: float = Field(
        3.0,
        ge=0.1,
        le=30.0,
        validation_alias="REDIS_CONNECT_TIMEOUT_SECONDS",
    )
    socket_timeout_seconds: float = Field(
        20.0,
        ge=0.1,
        le=120.0,
        validation_alias="REDIS_SOCKET_TIMEOUT_SECONDS",
    )
    max_connections: int = Field(
        50,
        ge=1,
        le=1_000,
        validation_alias="REDIS_MAX_CONNECTIONS",
    )
    stream_ttl_seconds: int = Field(
        86_400,
        ge=60,
        le=2_592_000,
        validation_alias="REDIS_CHAT_STREAM_TTL_SECONDS",
    )

    @field_validator("key_prefix")
    @classmethod
    def validate_key_prefix(cls, value: str) -> str:
        if (
            not 1 <= len(value) <= 64
            or not value[0].isalpha()
            or value != value.strip(":")
            or any(
                character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
                for character in value
            )
        ):
            raise ValueError("REDIS_KEY_PREFIX is invalid")
        return value

    @field_validator("chat_key_hmac_secret")
    @classmethod
    def validate_chat_key_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        try:
            encoded = secret.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ValueError("REDIS_CHAT_KEY_HMAC_SECRET is invalid") from None
        if not 32 <= len(encoded) <= 4_096 or "\x00" in secret:
            raise ValueError("REDIS_CHAT_KEY_HMAC_SECRET is invalid")
        return value


class Settings(BaseModel):
    api: ApiSettings = Field(default_factory=ApiSettings)
    import_runtime: ImportRuntimeSettings = Field(default_factory=ImportRuntimeSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    search: KnowledgeSearchSettings = Field(default_factory=KnowledgeSearchSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    mineru: MineruSettings = Field(default_factory=MineruSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)

    @model_validator(mode="after")
    def validate_cross_domain_rules(self) -> Settings:
        if self.api.environment is AppEnvironment.PRODUCTION and self.auth.mode == "disabled":
            raise ValueError("AUTH_MODE cannot be disabled in production")
        if self.api.environment is AppEnvironment.PRODUCTION and self.logging.diagnose:
            raise ValueError("LOG_DIAGNOSE must be false in production")
        if (
            self.api.environment is AppEnvironment.PRODUCTION
            and self.observability.metrics_enabled
            and self.observability.metrics_auth_token is None
        ):
            raise ValueError("METRICS_AUTH_TOKEN is required when metrics are enabled in production")
        if self.api.environment is AppEnvironment.PRODUCTION and self.milvus.allow_schema_create:
            raise ValueError("MILVUS_ALLOW_SCHEMA_CREATE must be false in production")
        if self.search.enabled:
            if self.import_runtime.repository != "mongo":
                raise ValueError("IMPORT_REPOSITORY must be mongo when knowledge search is enabled")
            if not self.mongo.url:
                raise ValueError("MONGO_URL is required when knowledge search is enabled")
            if not self.milvus.url:
                raise ValueError("MILVUS_URL is required when knowledge search is enabled")
            model_path = self.embedding.model_path
            if not isinstance(model_path, str) or not model_path or not Path(model_path).is_absolute():
                raise ValueError(
                    "BGE_M3_PATH must be an absolute local model directory when knowledge search is enabled"
                )
        if self.chat.enabled:
            if self.import_runtime.repository != "mongo":
                raise ValueError("IMPORT_REPOSITORY must be mongo when chat is enabled")
            if not self.mongo.url:
                raise ValueError("MONGO_URL is required when chat is enabled")
            if not self.mongo.require_replica_set:
                raise ValueError("MONGO_REQUIRE_REPLICA_SET must be true when chat is enabled")
            if self.chat.sse_broker == "redis" and not self.redis.url:
                raise ValueError("REDIS_URL is required for the Redis chat broker")
            if self.chat.sse_broker == "redis" and self.redis.chat_key_hmac_secret is None:
                raise ValueError("REDIS_CHAT_KEY_HMAC_SECRET is required for the Redis chat broker")
            if self.chat.sse_broker == "redis" and self.redis.socket_timeout_seconds <= self.chat.heartbeat_seconds:
                raise ValueError("REDIS_SOCKET_TIMEOUT_SECONDS must exceed the chat heartbeat interval")
            if self.api.environment is AppEnvironment.PRODUCTION and self.chat.sse_broker != "redis":
                raise ValueError("CHAT_SSE_BROKER must be redis in production")
        if (
            self.api.environment is AppEnvironment.PRODUCTION
            and self.mineru.base_url is not None
            and not self.mineru.base_url.startswith("https://")
        ):
            raise ValueError("MINERU_BASE_URL must use HTTPS in production")
        if self.api.import_enabled:
            maximum_put_window = (self.minio.connect_timeout_seconds + self.minio.read_timeout_seconds) * (
                self.minio.max_retries + 1
            )
            minimum_write_lease = maximum_put_window + self.import_runtime.heartbeat_seconds * 2
            if self.import_runtime.write_lease_seconds <= minimum_write_lease:
                raise ValueError("IMPORT_WRITE_LEASE_SECONDS must exceed the bounded MinIO PUT window")
        if self.auth.mode == "jwt":
            if not self.auth.issuer or not self.auth.audience:
                raise ValueError("AUTH_ISSUER and AUTH_AUDIENCE are required in jwt mode")
            if not self.auth.oidc_client_id:
                raise ValueError("AUTH_OIDC_CLIENT_ID is required in jwt mode")
            if not self.auth.jwks_url and not self.auth.public_key:
                raise ValueError("AUTH_JWKS_URL or AUTH_PUBLIC_KEY is required in jwt mode")
            if self.api.environment is AppEnvironment.PRODUCTION and not self.auth.jwks_url:
                raise ValueError("AUTH_JWKS_URL is required for production JWT verification")
            if self.api.environment is AppEnvironment.PRODUCTION and not self.auth.issuer.startswith("https://"):
                raise ValueError("AUTH_ISSUER must use HTTPS in production")
        if self.api.environment is AppEnvironment.PRODUCTION and self.api.import_enabled:
            if self.import_runtime.repository != "mongo":
                raise ValueError("IMPORT_REPOSITORY must be mongo when production imports are enabled")
            if not self.import_runtime.require_idempotency_key:
                raise ValueError("IMPORT_REQUIRE_IDEMPOTENCY_KEY must be true when production imports are enabled")
            if not self.import_runtime.worker_enabled:
                raise ValueError("IMPORT_WORKER_ENABLED must be true when production imports are enabled")
            if not self.mongo.url:
                raise ValueError("MONGO_URL is required when production imports are enabled")
            if not self.mongo.require_replica_set:
                raise ValueError("MONGO_REQUIRE_REPLICA_SET must be true when production imports are enabled")
        return self

    def validate_capabilities(self, *names: str) -> None:
        missing: list[str] = []
        if "llm" in names and (not self.llm.api_key or not self.llm.base_url):
            missing.append("llm")
        if "minio" in names and (
            not self.minio.endpoint
            or not self.minio.access_key
            or not self.minio.secret_key
            or not self.minio.bucket_name
        ):
            missing.append("minio")
        if "milvus" in names and not self.milvus.url:
            missing.append("milvus")
        if "mongo" in names and not self.mongo.url:
            missing.append("mongo")
        if "mineru" in names and (not self.mineru.base_url or not self.mineru.api_key):
            missing.append("mineru")
        if missing:
            raise ValueError(f"missing required capabilities: {', '.join(missing)}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
