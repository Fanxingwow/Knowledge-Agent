from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from app.core.settings import Settings

_COLLECTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
_FIELD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,254}$")
_INDEX_NAME = _FIELD_NAME
_MAX_FILTER_LENGTH = 16_384

MILVUS_CANDIDATE_OUTPUT_FIELDS = (
    "chunk_id",
    "tenant_id",
    "document_id",
    "import_version",
    "commit_token",
    "content_sha256",
    "content",
    "title",
    "parent_title",
    "file_title",
    "item_name",
)
_MILVUS_CANDIDATE_OUTPUT_FIELD_ALLOWLIST = frozenset(MILVUS_CANDIDATE_OUTPUT_FIELDS)

_TYPE_BY_CODE = {
    5: "INT64",
    21: "VARCHAR",
    101: "FLOAT_VECTOR",
    104: "SPARSE_FLOAT_VECTOR",
}


class MilvusRuntimeError(RuntimeError):
    """Base error for the managed Milvus runtime."""


class MilvusConfigurationError(MilvusRuntimeError):
    """The runtime was not given a complete, safe configuration."""


class MilvusSchemaNotReady(MilvusRuntimeError):
    """An operation was attempted before the startup schema gate passed."""


class MilvusSchemaMissing(MilvusRuntimeError):
    """The enterprise collection is absent and creation is not authorized."""


class MilvusSchemaIncompatible(MilvusRuntimeError):
    """An existing collection or index cannot safely serve this runtime."""


class MilvusRuntimeUnavailable(MilvusRuntimeError):
    """Milvus could not complete a connection or data-plane operation."""


class MilvusRuntimeClosed(MilvusRuntimeError):
    """The runtime has already entered terminal shutdown."""


class MilvusHybridSearchClient(Protocol):
    """Managed hybrid-search surface consumed by ``MilvusTenantVectorSource``."""

    def hybrid_search(
        self,
        *,
        collection_name: str,
        reqs: list[object],
        ranker: object,
        limit: int,
        output_fields: list[str],
        consistency_level: str,
    ) -> object: ...


class MilvusAdminClient(Protocol):
    """Synchronous MilvusClient surface used by the runtime and schema gate."""

    def has_collection(self, collection_name: str, **kwargs: object) -> object: ...

    def describe_collection(self, collection_name: str, **kwargs: object) -> object: ...

    def create_collection(self, collection_name: str, **kwargs: object) -> object: ...

    def list_indexes(self, collection_name: str, **kwargs: object) -> object: ...

    def describe_index(
        self,
        collection_name: str,
        index_name: str,
        **kwargs: object,
    ) -> object: ...

    def create_index(
        self,
        collection_name: str,
        index_params: object,
        **kwargs: object,
    ) -> object: ...

    def load_collection(self, collection_name: str, **kwargs: object) -> object: ...

    def get_load_state(self, collection_name: str, **kwargs: object) -> object: ...

    def query(self, collection_name: str, **kwargs: object) -> object: ...

    def hybrid_search(
        self,
        *,
        collection_name: str,
        reqs: list[object],
        ranker: object,
        limit: int,
        output_fields: list[str],
        consistency_level: str,
        timeout: float,
    ) -> object: ...

    def upsert(self, collection_name: str, data: object, **kwargs: object) -> object: ...

    def delete(self, collection_name: str, **kwargs: object) -> object: ...

    def close(self) -> object: ...


@dataclass(frozen=True, slots=True)
class MilvusRuntimeConfig:
    """Explicit connection and collection policy for one runtime instance.

    Critical production values intentionally have no implicit defaults.  Secrets
    are excluded from repr so a config object can be logged without exposing
    credentials.
    """

    uri: str
    collection_name: str
    dense_dimension: int
    allow_create: bool
    production: bool
    token: str | None = field(default=None, repr=False)
    user: str | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)
    database: str | None = None
    timeout_seconds: float = 5.0
    close_wait_seconds: float = 10.0
    max_query_limit: int = 10_001

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass(frozen=True, slots=True)
class MilvusSchemaReport:
    collection_name: str
    schema_fingerprint: str
    collection_created: bool
    indexes_created: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MilvusFieldSpec:
    name: str
    datatype: str
    params: Mapping[str, int] = field(default_factory=dict)
    primary: bool = False
    auto_id: bool = False


@dataclass(frozen=True, slots=True)
class MilvusIndexSpec:
    field_name: str
    index_name: str
    index_type: str
    metric_type: str | None = None
    params: Mapping[str, int | float] = field(default_factory=dict)


class MilvusArtifactFactory(Protocol):
    """Build native pymilvus schema/index objects without importing eagerly."""

    def schema(self, fields: Sequence[MilvusFieldSpec]) -> object: ...

    def indexes(self, indexes: Sequence[MilvusIndexSpec]) -> object: ...


ClientFactory = Callable[[MilvusRuntimeConfig], MilvusAdminClient]


def _validate_config(config: MilvusRuntimeConfig) -> None:
    if not isinstance(config.uri, str) or not config.uri or config.uri != config.uri.strip():
        raise MilvusConfigurationError("Milvus URI is required")
    parsed = urlsplit(config.uri)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MilvusConfigurationError("Milvus URI must be an HTTP(S) endpoint")
    if parsed.username is not None or parsed.password is not None:
        raise MilvusConfigurationError("Milvus credentials must not be embedded in the URI")
    if parsed.query or parsed.fragment:
        raise MilvusConfigurationError("Milvus URI query and fragment components are forbidden")
    if (
        not isinstance(config.collection_name, str)
        or _COLLECTION_NAME.fullmatch(config.collection_name) is None
    ):
        raise MilvusConfigurationError("Milvus collection name is invalid")
    if (
        isinstance(config.dense_dimension, bool)
        or not isinstance(config.dense_dimension, int)
        or not 1 <= config.dense_dimension <= 32_768
    ):
        raise MilvusConfigurationError("Milvus dense vector dimension is invalid")
    if not isinstance(config.allow_create, bool) or not isinstance(config.production, bool):
        raise MilvusConfigurationError("Milvus runtime policy is invalid")
    for name, value in (("token", config.token), ("user", config.user), ("password", config.password)):
        if value is not None and (not isinstance(value, str) or not value):
            raise MilvusConfigurationError(f"Milvus {name} is invalid")
    if (config.user is None) != (config.password is None):
        raise MilvusConfigurationError("Milvus user and password must be configured together")
    if config.token is not None and config.user is not None:
        raise MilvusConfigurationError("Milvus authentication methods are mutually exclusive")
    if config.database is not None and (
        not isinstance(config.database, str) or _COLLECTION_NAME.fullmatch(config.database) is None
    ):
        raise MilvusConfigurationError("Milvus database name is invalid")
    if (
        isinstance(config.timeout_seconds, bool)
        or not isinstance(config.timeout_seconds, (int, float))
        or not math.isfinite(float(config.timeout_seconds))
        or config.timeout_seconds <= 0
    ):
        raise MilvusConfigurationError("Milvus timeout is invalid")
    if (
        isinstance(config.close_wait_seconds, bool)
        or not isinstance(config.close_wait_seconds, (int, float))
        or not math.isfinite(float(config.close_wait_seconds))
        or config.close_wait_seconds <= 0
    ):
        raise MilvusConfigurationError("Milvus close timeout is invalid")
    if (
        isinstance(config.max_query_limit, bool)
        or not isinstance(config.max_query_limit, int)
        or config.max_query_limit <= 0
    ):
        raise MilvusConfigurationError("Milvus query limit is invalid")


def _field_specs(dense_dimension: int) -> tuple[MilvusFieldSpec, ...]:
    return (
        MilvusFieldSpec("chunk_id", "VARCHAR", {"max_length": 64}, primary=True),
        MilvusFieldSpec("tenant_id", "VARCHAR", {"max_length": 128}),
        MilvusFieldSpec("document_id", "VARCHAR", {"max_length": 36}),
        MilvusFieldSpec("import_version", "VARCHAR", {"max_length": 36}),
        MilvusFieldSpec("pipeline_version", "VARCHAR", {"max_length": 128}),
        MilvusFieldSpec("ordinal", "INT64"),
        MilvusFieldSpec("content_sha256", "VARCHAR", {"max_length": 64}),
        MilvusFieldSpec("record_sha256", "VARCHAR", {"max_length": 64}),
        MilvusFieldSpec("manifest_sha256", "VARCHAR", {"max_length": 64}),
        MilvusFieldSpec("commit_token", "VARCHAR", {"max_length": 128}),
        MilvusFieldSpec("content", "VARCHAR", {"max_length": 65_535}),
        MilvusFieldSpec("title", "VARCHAR", {"max_length": 65_535}),
        MilvusFieldSpec("parent_title", "VARCHAR", {"max_length": 65_535}),
        MilvusFieldSpec("file_title", "VARCHAR", {"max_length": 65_535}),
        MilvusFieldSpec("item_name", "VARCHAR", {"max_length": 65_535}),
        MilvusFieldSpec("dense_vector", "FLOAT_VECTOR", {"dim": dense_dimension}),
        MilvusFieldSpec("sparse_vector", "SPARSE_FLOAT_VECTOR"),
    )


def _index_specs() -> tuple[MilvusIndexSpec, ...]:
    return (
        MilvusIndexSpec("tenant_id", "tenant_id_idx", "INVERTED"),
        MilvusIndexSpec("document_id", "document_id_idx", "INVERTED"),
        MilvusIndexSpec("import_version", "import_version_idx", "INVERTED"),
        MilvusIndexSpec("pipeline_version", "pipeline_version_idx", "INVERTED"),
        MilvusIndexSpec("commit_token", "commit_token_idx", "INVERTED"),
        MilvusIndexSpec(
            "dense_vector",
            "dense_vector_idx",
            "HNSW",
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 200},
        ),
        MilvusIndexSpec(
            "sparse_vector",
            "sparse_vector_idx",
            "SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        ),
    )


class _PymilvusArtifacts:
    """Default artifact builder; importing pymilvus is deferred until use."""

    @staticmethod
    def schema(fields: Sequence[MilvusFieldSpec]) -> object:
        from pymilvus import DataType, MilvusClient  # type: ignore[import-untyped]

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="Tenant-fenced immutable knowledge chunks v1",
        )
        for spec in fields:
            datatype = getattr(DataType, spec.datatype)
            schema.add_field(
                field_name=spec.name,
                datatype=datatype,
                is_primary=spec.primary,
                auto_id=spec.auto_id,
                nullable=False,
                **dict(spec.params),
            )
        return schema

    @staticmethod
    def indexes(indexes: Sequence[MilvusIndexSpec]) -> object:
        from pymilvus import MilvusClient

        params = MilvusClient.prepare_index_params()
        for spec in indexes:
            kwargs: dict[str, object] = {
                "field_name": spec.field_name,
                "index_name": spec.index_name,
                "index_type": spec.index_type,
            }
            if spec.metric_type is not None:
                kwargs["metric_type"] = spec.metric_type
            if spec.params:
                kwargs["params"] = dict(spec.params)
            params.add_index(**kwargs)
        return params


def _default_client_factory(config: MilvusRuntimeConfig) -> MilvusAdminClient:
    from pymilvus import MilvusClient

    kwargs: dict[str, object] = {
        "uri": config.uri,
        "timeout": float(config.timeout_seconds),
    }
    if config.token is not None:
        kwargs["token"] = config.token
    if config.user is not None and config.password is not None:
        kwargs["user"] = config.user
        kwargs["password"] = config.password
    if config.database is not None:
        kwargs["db_name"] = config.database
    return cast(MilvusAdminClient, MilvusClient(**kwargs))


def _type_name(value: object) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return _TYPE_BY_CODE.get(value, str(value))
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str):
        return enum_name.upper()
    text = str(value).upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MilvusSchemaIncompatible("Milvus metadata response is invalid")
    return cast(Mapping[str, object], value)


def _params(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)


def _equivalent_number(actual: object, expected: int | float) -> bool:
    if isinstance(actual, bool):
        return False
    try:
        number = float(cast(int | float | str, actual))
    except (TypeError, ValueError):
        return False
    return math.isclose(number, float(expected), rel_tol=0, abs_tol=1e-12)


def _schema_fingerprint(
    fields: Sequence[MilvusFieldSpec],
    indexes: Sequence[MilvusIndexSpec],
) -> str:
    payload = {
        "dynamic": False,
        "auto_id": False,
        "fields": [
            {
                "name": spec.name,
                "datatype": spec.datatype,
                "params": dict(spec.params),
                "primary": spec.primary,
                "auto_id": spec.auto_id,
            }
            for spec in fields
        ],
        "indexes": [
            {
                "field": spec.field_name,
                "name": spec.index_name,
                "type": spec.index_type,
                "metric": spec.metric_type,
                "params": dict(spec.params),
            }
            for spec in indexes
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_collection(description: object, fields: Sequence[MilvusFieldSpec]) -> None:
    metadata = _mapping(description)
    if metadata.get("auto_id", False) is not False:
        raise MilvusSchemaIncompatible("Milvus collection auto-id policy is incompatible")
    dynamic = metadata.get("enable_dynamic_field", metadata.get("enableDynamicField", False))
    if dynamic is not False:
        raise MilvusSchemaIncompatible("Milvus dynamic fields must be disabled")
    raw_fields = metadata.get("fields")
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes, bytearray)):
        raise MilvusSchemaIncompatible("Milvus collection fields are invalid")
    observed: dict[str, Mapping[str, object]] = {}
    for raw_field in raw_fields:
        item = _mapping(raw_field)
        name = item.get("name", item.get("field_name"))
        if not isinstance(name, str) or _FIELD_NAME.fullmatch(name) is None or name in observed:
            raise MilvusSchemaIncompatible("Milvus collection fields are invalid")
        observed[name] = item
    expected_names = {spec.name for spec in fields}
    if set(observed) != expected_names:
        raise MilvusSchemaIncompatible("Milvus collection field set is incompatible")
    for spec in fields:
        actual = observed[spec.name]
        if _type_name(actual.get("type", actual.get("datatype"))) != spec.datatype:
            raise MilvusSchemaIncompatible("Milvus collection field type is incompatible")
        if bool(actual.get("is_primary", actual.get("isPrimary", False))) is not spec.primary:
            raise MilvusSchemaIncompatible("Milvus collection primary key is incompatible")
        if bool(actual.get("auto_id", actual.get("autoID", False))) is not spec.auto_id:
            raise MilvusSchemaIncompatible("Milvus collection field auto-id policy is incompatible")
        if actual.get("nullable", False) is True:
            raise MilvusSchemaIncompatible("Milvus nullable field policy is incompatible")
        actual_params: dict[str, object] = {}
        for key in ("params", "type_params", "typeParams"):
            actual_params.update(_params(actual.get(key)))
        for key, value in spec.params.items():
            if not _equivalent_number(actual_params.get(key), value):
                raise MilvusSchemaIncompatible("Milvus collection field parameters are incompatible")


def _index_names(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise MilvusSchemaIncompatible("Milvus index metadata is invalid")
    names: list[str] = []
    for item in raw:
        name: object
        if isinstance(item, Mapping):
            name = item.get("index_name", item.get("name"))
        else:
            name = item
        if not isinstance(name, str) or _INDEX_NAME.fullmatch(name) is None or name in names:
            raise MilvusSchemaIncompatible("Milvus index metadata is invalid")
        names.append(name)
    return tuple(names)


def _index_property(metadata: Mapping[str, object], name: str) -> object:
    if name in metadata:
        return metadata[name]
    for key in ("params", "index_param", "index_params"):
        nested = _params(metadata.get(key))
        if name in nested:
            return nested[name]
    return None


def _verify_index(description: object, spec: MilvusIndexSpec) -> None:
    metadata = _mapping(description)
    if metadata.get("index_name", metadata.get("name")) != spec.index_name:
        raise MilvusSchemaIncompatible("Milvus index identity is incompatible")
    if metadata.get("field_name", metadata.get("field")) != spec.field_name:
        raise MilvusSchemaIncompatible("Milvus index field is incompatible")
    index_type = str(_index_property(metadata, "index_type") or "").upper()
    if index_type != spec.index_type:
        raise MilvusSchemaIncompatible("Milvus index type is incompatible")
    if spec.metric_type is not None:
        metric = str(_index_property(metadata, "metric_type") or "").upper()
        if metric != spec.metric_type:
            raise MilvusSchemaIncompatible("Milvus index metric is incompatible")
    for key, expected in spec.params.items():
        if not _equivalent_number(_index_property(metadata, key), expected):
            raise MilvusSchemaIncompatible("Milvus index parameters are incompatible")
    state = metadata.get("state")
    if state is not None:
        state_name = getattr(state, "name", str(state)).rsplit(".", 1)[-1].lower()
        if state_name not in {"finished", "loaded"}:
            raise MilvusSchemaIncompatible("Milvus index is not ready")


def _load_state_is_ready(raw: object) -> bool:
    if isinstance(raw, Mapping):
        raw = raw.get("state", raw.get("load_state"))
    name = getattr(raw, "name", str(raw)).rsplit(".", 1)[-1].lower()
    return name in {"loaded", "loadstate_loaded"}


class MilvusRuntimeAdapter:
    """Lazy, schema-gated, thread-safe adapter around ``MilvusClient``.

    Data-plane calls remain unavailable until :meth:`ensure_schema` proves the
    live collection contract.  This prevents a process from serving or writing
    against an old dynamic-field collection by accident.
    """

    def __init__(
        self,
        config: MilvusRuntimeConfig,
        *,
        client_factory: ClientFactory | None = None,
        artifact_factory: MilvusArtifactFactory | None = None,
    ) -> None:
        self._config = config
        self._client_factory = client_factory or _default_client_factory
        self._artifacts = artifact_factory or _PymilvusArtifacts()
        self._fields = _field_specs(config.dense_dimension)
        self._indexes = _index_specs()
        self._fingerprint = _schema_fingerprint(self._fields, self._indexes)
        self._condition = threading.Condition(threading.RLock())
        self._gate_lock = threading.Lock()
        self._client: MilvusAdminClient | None = None
        self._active_operations = 0
        self._schema_ready = False
        self._closing = False
        self._closed = False

    @property
    def collection_name(self) -> str:
        return self._config.collection_name

    @property
    def schema_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def schema_ready(self) -> bool:
        with self._condition:
            return self._schema_ready and not self._closing and not self._closed

    def _acquire_client(self, *, require_schema: bool) -> MilvusAdminClient:
        with self._condition:
            if self._closing or self._closed:
                raise MilvusRuntimeClosed("Milvus runtime is closed")
            if require_schema and not self._schema_ready:
                raise MilvusSchemaNotReady("Milvus schema gate has not passed")
            if self._client is None:
                try:
                    self._client = self._client_factory(self._config)
                except Exception:
                    raise MilvusRuntimeUnavailable("Milvus connection is unavailable") from None
            self._active_operations += 1
            return self._client

    def _release_client(self) -> None:
        with self._condition:
            self._active_operations -= 1
            if self._active_operations == 0:
                self._condition.notify_all()

    @contextmanager
    def _operation(self, *, require_schema: bool) -> Iterator[MilvusAdminClient]:
        client = self._acquire_client(require_schema=require_schema)
        try:
            yield client
        finally:
            self._release_client()

    def _call(self, operation: Callable[[MilvusAdminClient], object], *, require_schema: bool) -> object:
        try:
            with self._operation(require_schema=require_schema) as client:
                return operation(client)
        except MilvusRuntimeError:
            raise
        except Exception:
            raise MilvusRuntimeUnavailable("Milvus operation is unavailable") from None

    def _has_collection(self) -> bool:
        result = self._call(
            lambda client: client.has_collection(
                self._config.collection_name,
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=False,
        )
        if not isinstance(result, bool):
            raise MilvusRuntimeUnavailable("Milvus collection state is unavailable")
        return result

    def _describe_collection(self) -> object:
        return self._call(
            lambda client: client.describe_collection(
                self._config.collection_name,
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=False,
        )

    def _create_collection(self) -> None:
        try:
            schema = self._artifacts.schema(self._fields)
            indexes = self._artifacts.indexes(self._indexes)
        except Exception:
            raise MilvusRuntimeUnavailable("Milvus schema artifacts could not be built") from None
        self._call(
            lambda client: client.create_collection(
                self._config.collection_name,
                schema=schema,
                index_params=indexes,
                consistency_level="Strong",
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=False,
        )

    def _observed_indexes(self) -> dict[str, object]:
        raw_names = self._call(
            lambda client: client.list_indexes(
                self._config.collection_name,
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=False,
        )
        descriptions: dict[str, object] = {}
        for name in _index_names(raw_names):
            def describe(client: MilvusAdminClient, index_name: str = name) -> object:
                return client.describe_index(
                    self._config.collection_name,
                    index_name,
                    timeout=float(self._config.timeout_seconds),
                )

            descriptions[name] = self._call(
                describe,
                require_schema=False,
            )
        return descriptions

    def _missing_indexes(self, observed: Mapping[str, object]) -> tuple[MilvusIndexSpec, ...]:
        fields_with_indexes: dict[str, str] = {}
        for name, raw in observed.items():
            metadata = _mapping(raw)
            raw_field = metadata.get("field_name", metadata.get("field"))
            if isinstance(raw_field, str):
                fields_with_indexes[raw_field] = name
        missing: list[MilvusIndexSpec] = []
        for spec in self._indexes:
            current = observed.get(spec.index_name)
            if current is not None:
                _verify_index(current, spec)
                continue
            conflicting_name = fields_with_indexes.get(spec.field_name)
            if conflicting_name is not None:
                raise MilvusSchemaIncompatible("Milvus field has a conflicting index")
            missing.append(spec)
        return tuple(missing)

    def _create_indexes(self, indexes: Sequence[MilvusIndexSpec]) -> None:
        try:
            params = self._artifacts.indexes(indexes)
        except Exception:
            raise MilvusRuntimeUnavailable("Milvus index artifacts could not be built") from None
        self._call(
            lambda client: client.create_index(
                self._config.collection_name,
                params,
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=False,
        )

    def _load_and_probe(self) -> None:
        self._call(
            lambda client: client.load_collection(
                self._config.collection_name,
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=False,
        )
        state = self._call(
            lambda client: client.get_load_state(
                self._config.collection_name,
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=False,
        )
        if not _load_state_is_ready(state):
            raise MilvusRuntimeUnavailable("Milvus collection is not loaded")

    def ensure_schema(self) -> MilvusSchemaReport:
        """Verify or non-destructively create the exact enterprise contract."""

        with self._gate_lock:
            with self._condition:
                if self._closing or self._closed:
                    raise MilvusRuntimeClosed("Milvus runtime is closed")
                self._schema_ready = False
            collection_created = False
            indexes_created: tuple[str, ...] = ()
            exists = self._has_collection()
            if not exists:
                if not self._config.allow_create:
                    raise MilvusSchemaMissing("Milvus enterprise collection is missing")
                try:
                    self._create_collection()
                    collection_created = True
                    indexes_created = tuple(spec.index_name for spec in self._indexes)
                except MilvusRuntimeUnavailable:
                    # Reconcile an acknowledgement loss or a concurrent creator.
                    if not self._has_collection():
                        raise
            _verify_collection(self._describe_collection(), self._fields)
            observed = self._observed_indexes()
            missing = self._missing_indexes(observed)
            if missing:
                if not self._config.allow_create:
                    raise MilvusSchemaMissing("Milvus enterprise indexes are missing")
                try:
                    self._create_indexes(missing)
                except MilvusRuntimeUnavailable:
                    # An index may have been committed despite an unavailable ACK.
                    pass
                observed = self._observed_indexes()
                remaining = self._missing_indexes(observed)
                if remaining:
                    raise MilvusRuntimeUnavailable("Milvus indexes could not be created")
                created_now = tuple(spec.index_name for spec in missing)
                indexes_created = tuple(dict.fromkeys((*indexes_created, *created_now)))
            self._load_and_probe()
            with self._condition:
                if self._closing or self._closed:
                    raise MilvusRuntimeClosed("Milvus runtime is closed")
                self._schema_ready = True
            return MilvusSchemaReport(
                collection_name=self._config.collection_name,
                schema_fingerprint=self._fingerprint,
                collection_created=collection_created,
                indexes_created=indexes_created,
            )

    def probe(self) -> None:
        """Fail unless the gated collection still exists and is loaded."""

        if not self.schema_ready:
            raise MilvusSchemaNotReady("Milvus schema gate has not passed")
        if not self._has_collection():
            with self._condition:
                self._schema_ready = False
            raise MilvusRuntimeUnavailable("Milvus collection is unavailable")
        state = self._call(
            lambda client: client.get_load_state(
                self._config.collection_name,
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=True,
        )
        if not _load_state_is_ready(state):
            raise MilvusRuntimeUnavailable("Milvus collection is not loaded")

    def _validate_collection(self, collection_name: str) -> None:
        if collection_name != self._config.collection_name:
            raise ValueError("Milvus collection is outside the managed runtime")

    def query(
        self,
        *,
        collection_name: str,
        filter: str,
        output_fields: list[str],
        limit: int,
        consistency_level: str,
    ) -> Sequence[Mapping[str, object]]:
        self._validate_collection(collection_name)
        if (
            not isinstance(filter, str)
            or not filter
            or len(filter) > _MAX_FILTER_LENGTH
            or consistency_level != "Strong"
        ):
            raise ValueError("Milvus query contract is invalid")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= self._config.max_query_limit
        ):
            raise ValueError("Milvus query limit is invalid")
        allowed_fields = {spec.name for spec in self._fields}
        if (
            not isinstance(output_fields, list)
            or not output_fields
            or len(output_fields) != len(set(output_fields))
            or any(not isinstance(name, str) or name not in allowed_fields for name in output_fields)
        ):
            raise ValueError("Milvus output fields are invalid")
        result = self._call(
            lambda client: client.query(
                collection_name,
                filter=filter,
                output_fields=list(output_fields),
                limit=limit,
                consistency_level="Strong",
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=True,
        )
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
            raise MilvusRuntimeUnavailable("Milvus query response is invalid")
        records: list[Mapping[str, object]] = []
        for item in result:
            if not isinstance(item, Mapping):
                raise MilvusRuntimeUnavailable("Milvus query response is invalid")
            records.append(cast(Mapping[str, object], item))
        return tuple(records)

    def hybrid_search(
        self,
        *,
        collection_name: str,
        reqs: list[object],
        ranker: object,
        limit: int,
        output_fields: list[str],
        consistency_level: str,
    ) -> object:
        """Run one schema-gated dense+sparse search against the managed collection."""

        self._validate_collection(collection_name)
        if consistency_level != "Strong":
            raise ValueError("Milvus hybrid search consistency is invalid")
        if not isinstance(reqs, list) or len(reqs) != 2 or any(request is None for request in reqs) or ranker is None:
            raise ValueError("Milvus hybrid search request is invalid")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self._config.max_query_limit:
            raise ValueError("Milvus hybrid search limit is invalid")
        if (
            not isinstance(output_fields, list)
            or not output_fields
            or any(not isinstance(name, str) for name in output_fields)
            or len(output_fields) != len(set(output_fields))
            or any(
                name not in _MILVUS_CANDIDATE_OUTPUT_FIELD_ALLOWLIST
                for name in output_fields
            )
        ):
            raise ValueError("Milvus hybrid search output fields are invalid")
        requests = list(reqs)
        selected_fields = list(output_fields)
        return self._call(
            lambda client: client.hybrid_search(
                collection_name=self._config.collection_name,
                reqs=requests,
                ranker=ranker,
                limit=limit,
                output_fields=selected_fields,
                consistency_level="Strong",
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=True,
        )

    def upsert(
        self,
        *,
        collection_name: str,
        data: list[dict[str, object]],
    ) -> object:
        self._validate_collection(collection_name)
        if not isinstance(data, list) or not data or any(not isinstance(item, dict) for item in data):
            raise ValueError("Milvus upsert payload is invalid")
        return self._call(
            lambda client: client.upsert(
                collection_name,
                data=data,
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=True,
        )

    def delete(self, *, collection_name: str, filter: str) -> object:
        self._validate_collection(collection_name)
        if not isinstance(filter, str) or not filter or len(filter) > _MAX_FILTER_LENGTH:
            raise ValueError("Milvus delete filter is invalid")
        return self._call(
            lambda client: client.delete(
                collection_name,
                filter=filter,
                timeout=float(self._config.timeout_seconds),
            ),
            require_schema=True,
        )

    def close(self) -> None:
        """Wait for in-flight calls, then close the lazy client exactly once."""

        with self._condition:
            if self._closed:
                return
            self._closing = True
            completed = self._condition.wait_for(
                lambda: self._active_operations == 0,
                timeout=float(self._config.close_wait_seconds),
            )
            if not completed:
                self._closing = False
                self._condition.notify_all()
                raise MilvusRuntimeUnavailable("Milvus runtime shutdown timed out")
            client = self._client
            self._client = None
            self._schema_ready = False
            self._closed = True
        if client is not None:
            try:
                client.close()
            except Exception:
                raise MilvusRuntimeUnavailable("Milvus runtime could not be closed") from None


def build_milvus_runtime(
    config: MilvusRuntimeConfig,
    *,
    client_factory: ClientFactory | None = None,
    artifact_factory: MilvusArtifactFactory | None = None,
) -> MilvusRuntimeAdapter:
    """Build a lazy runtime.  The caller must invoke ``ensure_schema`` at startup."""

    if not isinstance(config, MilvusRuntimeConfig):
        raise MilvusConfigurationError("Milvus runtime config is required")
    return MilvusRuntimeAdapter(
        config,
        client_factory=client_factory,
        artifact_factory=artifact_factory,
    )


def _secret_text(value: object) -> str | None:
    if value is None:
        return None
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        value = getter()
    if not isinstance(value, str):
        raise MilvusConfigurationError("Milvus credential is invalid")
    return value


def build_milvus_runtime_from_settings(
    settings: Settings,
    *,
    dense_dimension: int,
    allow_create: bool,
    token: object = None,
    user: object = None,
    password: object = None,
    database: str | None = None,
    timeout_seconds: float = 5.0,
    close_wait_seconds: float = 10.0,
    max_query_limit: int = 10_001,
    client_factory: ClientFactory | None = None,
    artifact_factory: MilvusArtifactFactory | None = None,
) -> MilvusRuntimeAdapter:
    """Adapt the central Settings object without reading environment variables."""

    try:
        milvus = settings.milvus
        api = settings.api
        uri = milvus.url
        collection_name = milvus.chunks_collection
        production = str(api.environment).lower() == "production"
    except (AttributeError, TypeError):
        raise MilvusConfigurationError("Application settings are incompatible") from None
    if not isinstance(uri, str) or not uri:
        raise MilvusConfigurationError("MILVUS_URL is required")
    if not isinstance(collection_name, str) or not collection_name:
        raise MilvusConfigurationError("CHUNKS_COLLECTION is required")
    config = MilvusRuntimeConfig(
        uri=uri,
        collection_name=collection_name,
        dense_dimension=dense_dimension,
        allow_create=allow_create,
        production=production,
        token=_secret_text(token),
        user=_secret_text(user),
        password=_secret_text(password),
        database=database,
        timeout_seconds=timeout_seconds,
        close_wait_seconds=close_wait_seconds,
        max_query_limit=max_query_limit,
    )
    return build_milvus_runtime(
        config,
        client_factory=client_factory,
        artifact_factory=artifact_factory,
    )
