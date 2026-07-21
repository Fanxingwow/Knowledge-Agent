from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Protocol, cast

import pytest
from pydantic import SecretStr

from app.infrastructure.milvus_runtime import (
    MILVUS_CANDIDATE_OUTPUT_FIELDS,
    MilvusConfigurationError,
    MilvusFieldSpec,
    MilvusHybridSearchClient,
    MilvusIndexSpec,
    MilvusRuntimeAdapter,
    MilvusRuntimeClosed,
    MilvusRuntimeConfig,
    MilvusRuntimeUnavailable,
    MilvusSchemaIncompatible,
    MilvusSchemaMissing,
    MilvusSchemaNotReady,
    build_milvus_runtime,
    build_milvus_runtime_from_settings,
)

_COLLECTION = "knowledge_chunks_v1"
_EXPECTED_FIELDS = {
    "chunk_id",
    "tenant_id",
    "document_id",
    "import_version",
    "pipeline_version",
    "ordinal",
    "content_sha256",
    "record_sha256",
    "manifest_sha256",
    "commit_token",
    "content",
    "title",
    "parent_title",
    "file_title",
    "item_name",
    "dense_vector",
    "sparse_vector",
}
_EXPECTED_INDEXES = {
    "tenant_id_idx",
    "document_id_idx",
    "import_version_idx",
    "pipeline_version_idx",
    "commit_token_idx",
    "dense_vector_idx",
    "sparse_vector_idx",
}


class NativeSchema(Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


class NativeIndexParam(Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


class FakeArtifacts:
    def __init__(self) -> None:
        self.schema_calls: list[tuple[MilvusFieldSpec, ...]] = []
        self.index_calls: list[tuple[MilvusIndexSpec, ...]] = []

    def schema(self, fields: Sequence[MilvusFieldSpec]) -> object:
        captured = tuple(fields)
        self.schema_calls.append(captured)
        return {
            "auto_id": False,
            "enable_dynamic_field": False,
            "fields": [
                {
                    "name": spec.name,
                    "type": spec.datatype,
                    "params": dict(spec.params),
                    "is_primary": spec.primary,
                    "auto_id": spec.auto_id,
                    "nullable": False,
                }
                for spec in captured
            ],
        }

    def indexes(self, indexes: Sequence[MilvusIndexSpec]) -> object:
        captured = tuple(indexes)
        self.index_calls.append(captured)
        return [
            {
                "field_name": spec.field_name,
                "index_name": spec.index_name,
                "index_type": spec.index_type,
                "metric_type": spec.metric_type,
                "params": dict(spec.params),
                "state": "Finished",
            }
            for spec in captured
        ]


class FakeClient:
    def __init__(
        self,
        *,
        exists: bool = False,
        description: Mapping[str, object] | None = None,
        indexes: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self.exists = exists
        self.description = dict(description or {})
        self.indexes = {name: dict(value) for name, value in (indexes or {}).items()}
        self.loaded = False
        self.closed = False
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.query_result: object = []
        self.hybrid_search_result: object = [[]]
        self.fail_operation: str | None = None
        self.create_collection_then_raise = False
        self.create_index_then_raise = False

    def _call(self, name: str, **kwargs: object) -> None:
        self.calls.append((name, kwargs))
        if self.fail_operation == name:
            raise RuntimeError(
                "tcp://root:secret@10.0.0.8:19530 private detail"  # pragma: allowlist secret
            )

    def has_collection(self, collection_name: str, **kwargs: object) -> object:
        self._call("has_collection", collection_name=collection_name, **kwargs)
        return self.exists

    def describe_collection(self, collection_name: str, **kwargs: object) -> object:
        self._call("describe_collection", collection_name=collection_name, **kwargs)
        return self.description

    def create_collection(self, collection_name: str, **kwargs: object) -> object:
        self._call("create_collection", collection_name=collection_name, **kwargs)
        schema = kwargs["schema"]
        index_params = kwargs["index_params"]
        assert isinstance(schema, Mapping)
        assert isinstance(index_params, list)
        self.exists = True
        self.description = dict(schema)
        self.indexes = {
            str(index["index_name"]): dict(index)
            for index in index_params
        }
        if self.create_collection_then_raise:
            self.create_collection_then_raise = False
            raise TimeoutError("creation acknowledgement contained a secret")
        return None

    def list_indexes(self, collection_name: str, **kwargs: object) -> object:
        self._call("list_indexes", collection_name=collection_name, **kwargs)
        return list(self.indexes)

    def describe_index(
        self,
        collection_name: str,
        index_name: str,
        **kwargs: object,
    ) -> object:
        self._call(
            "describe_index",
            collection_name=collection_name,
            index_name=index_name,
            **kwargs,
        )
        return self.indexes[index_name]

    def create_index(
        self,
        collection_name: str,
        index_params: object,
        **kwargs: object,
    ) -> object:
        self._call(
            "create_index",
            collection_name=collection_name,
            index_params=index_params,
            **kwargs,
        )
        assert isinstance(index_params, list)
        for index in index_params:
            self.indexes[str(index["index_name"])] = dict(index)
        if self.create_index_then_raise:
            self.create_index_then_raise = False
            raise TimeoutError("index acknowledgement contained a secret")
        return None

    def load_collection(self, collection_name: str, **kwargs: object) -> object:
        self._call("load_collection", collection_name=collection_name, **kwargs)
        self.loaded = True
        return None

    def get_load_state(self, collection_name: str, **kwargs: object) -> object:
        self._call("get_load_state", collection_name=collection_name, **kwargs)
        return {"state": "Loaded" if self.loaded else "NotLoad"}

    def query(self, collection_name: str, **kwargs: object) -> object:
        self._call("query", collection_name=collection_name, **kwargs)
        return self.query_result

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
    ) -> object:
        self._call(
            "hybrid_search",
            collection_name=collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=output_fields,
            consistency_level=consistency_level,
            timeout=timeout,
        )
        return self.hybrid_search_result

    def upsert(self, collection_name: str, data: object, **kwargs: object) -> object:
        self._call("upsert", collection_name=collection_name, data=data, **kwargs)
        return {"upsert_count": len(data) if isinstance(data, list) else 0}

    def delete(self, collection_name: str, **kwargs: object) -> object:
        self._call("delete", collection_name=collection_name, **kwargs)
        return {"delete_count": 1}

    def close(self) -> object:
        self._call("close")
        self.closed = True
        return None


class NativeArtifactClient(FakeClient):
    """Accept the real CollectionSchema/IndexParams objects from pymilvus."""

    def create_collection(self, collection_name: str, **kwargs: object) -> object:
        self._call("create_collection", collection_name=collection_name, **kwargs)
        schema = cast(NativeSchema, kwargs["schema"])
        index_params = cast(Iterable[NativeIndexParam], kwargs["index_params"])
        self.exists = True
        self.description = dict(schema.to_dict())
        self.indexes = {
            str(index.to_dict()["index_name"]): {**dict(index.to_dict()), "state": "Finished"}
            for index in index_params
        }
        return None


def config(**overrides: object) -> MilvusRuntimeConfig:
    values: dict[str, object] = {
        "uri": "https://milvus.internal:19530",
        "collection_name": _COLLECTION,
        "dense_dimension": 1024,
        "allow_create": True,
        "production": True,
        "token": "root:credential",
        "timeout_seconds": 3.0,
        "close_wait_seconds": 1.0,
    }
    values.update(overrides)
    return MilvusRuntimeConfig(**values)  # type: ignore[arg-type]


def runtime(
    client: FakeClient,
    *,
    runtime_config: MilvusRuntimeConfig | None = None,
    artifacts: FakeArtifacts | None = None,
) -> tuple[MilvusRuntimeAdapter, FakeArtifacts, list[MilvusRuntimeConfig]]:
    resolved_artifacts = artifacts or FakeArtifacts()
    factory_calls: list[MilvusRuntimeConfig] = []

    def factory(received: MilvusRuntimeConfig) -> FakeClient:
        factory_calls.append(received)
        return client

    adapter = build_milvus_runtime(
        runtime_config or config(),
        client_factory=factory,
        artifact_factory=resolved_artifacts,
    )
    return adapter, resolved_artifacts, factory_calls


def compatible_state(*, dimension: int = 1024) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    seed = FakeClient()
    adapter, _, _ = runtime(seed, runtime_config=config(dense_dimension=dimension))
    adapter.ensure_schema()
    adapter.close()
    return dict(seed.description), {name: dict(value) for name, value in seed.indexes.items()}


def test_runtime_is_lazy_and_data_plane_fails_closed_before_gate() -> None:
    client = FakeClient()
    adapter, artifacts, factory_calls = runtime(client)

    assert factory_calls == []
    assert artifacts.schema_calls == []
    with pytest.raises(MilvusSchemaNotReady):
        adapter.query(
            collection_name=_COLLECTION,
            filter='tenant_id == "tenant-a"',
            output_fields=["chunk_id"],
            limit=1,
            consistency_level="Strong",
        )

    assert factory_calls == []
    assert client.calls == []

    with pytest.raises(MilvusSchemaNotReady):
        adapter.hybrid_search(
            collection_name=_COLLECTION,
            reqs=[object(), object()],
            ranker=object(),
            limit=1,
            output_fields=list(MILVUS_CANDIDATE_OUTPUT_FIELDS),
            consistency_level="Strong",
        )

    assert factory_calls == []
    assert client.calls == []


def test_missing_collection_is_created_with_exact_fixed_schema_and_indexes() -> None:
    client = FakeClient()
    adapter, artifacts, factory_calls = runtime(client)

    report = adapter.ensure_schema()

    assert len(factory_calls) == 1
    assert report.collection_created is True
    assert set(report.indexes_created) == _EXPECTED_INDEXES
    assert re.fullmatch(r"[0-9a-f]{64}", report.schema_fingerprint)
    assert adapter.schema_ready is True
    assert client.description["auto_id"] is False
    assert client.description["enable_dynamic_field"] is False
    fields = client.description["fields"]
    assert isinstance(fields, list)
    assert {field["name"] for field in fields} == _EXPECTED_FIELDS
    primary = next(field for field in fields if field["name"] == "chunk_id")
    assert primary["type"] == "VARCHAR"
    assert primary["params"] == {"max_length": 64}
    assert primary["is_primary"] is True
    assert primary["auto_id"] is False
    dense = next(field for field in fields if field["name"] == "dense_vector")
    assert dense["params"] == {"dim": 1024}
    assert set(client.indexes) == _EXPECTED_INDEXES
    assert client.indexes["dense_vector_idx"] == {
        "field_name": "dense_vector",
        "index_name": "dense_vector_idx",
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 32, "efConstruction": 200},
        "state": "Finished",
    }
    assert client.indexes["sparse_vector_idx"]["index_type"] == "SPARSE_INVERTED_INDEX"
    assert client.indexes["sparse_vector_idx"]["metric_type"] == "IP"
    assert len(artifacts.schema_calls) == 1
    assert len(artifacts.index_calls) == 1
    create_call = next(kwargs for name, kwargs in client.calls if name == "create_collection")
    assert create_call["consistency_level"] == "Strong"
    assert create_call["timeout"] == 3.0


def test_default_lazy_artifacts_match_the_installed_pymilvus_sdk_contract() -> None:
    client = NativeArtifactClient()
    factory_calls = 0

    def factory(_: MilvusRuntimeConfig) -> NativeArtifactClient:
        nonlocal factory_calls
        factory_calls += 1
        return client

    adapter = build_milvus_runtime(config(), client_factory=factory)
    assert factory_calls == 0

    report = adapter.ensure_schema()

    assert report.collection_created is True
    assert factory_calls == 1
    raw_fields = client.description["fields"]
    assert isinstance(raw_fields, list)
    fields = cast(list[Mapping[str, object]], raw_fields)
    assert {field["name"] for field in fields} == _EXPECTED_FIELDS
    assert set(client.indexes) == _EXPECTED_INDEXES
    assert client.indexes["dense_vector_idx"]["M"] == 32
    assert client.indexes["sparse_vector_idx"]["drop_ratio_build"] == 0.2


def test_compatible_existing_collection_is_reused_without_mutation() -> None:
    description, indexes = compatible_state()
    client = FakeClient(exists=True, description=description, indexes=indexes)
    adapter, artifacts, _ = runtime(client)

    first = adapter.ensure_schema()
    second = adapter.ensure_schema()

    assert first.collection_created is False
    assert first.indexes_created == ()
    assert second.collection_created is False
    assert artifacts.schema_calls == []
    assert artifacts.index_calls == []
    assert all(name not in {"create_collection", "create_index"} for name, _ in client.calls)


def test_missing_collection_without_migration_authority_fails_closed() -> None:
    client = FakeClient()
    adapter, artifacts, _ = runtime(client, runtime_config=config(allow_create=False))

    with pytest.raises(MilvusSchemaMissing) as raised:
        adapter.ensure_schema()

    assert str(raised.value) == "Milvus enterprise collection is missing"
    assert artifacts.schema_calls == []
    assert all(name != "create_collection" for name, _ in client.calls)
    assert adapter.schema_ready is False


@pytest.mark.parametrize(
    "mutation",
    [
        "dynamic",
        "auto_id",
        "extra_field",
        "wrong_primary",
        "wrong_dimension",
        "nullable",
    ],
)
def test_incompatible_existing_schema_never_triggers_destructive_migration(mutation: str) -> None:
    description, indexes = compatible_state()
    fields = description["fields"]
    assert isinstance(fields, list)
    if mutation == "dynamic":
        description["enable_dynamic_field"] = True
    elif mutation == "auto_id":
        description["auto_id"] = True
    elif mutation == "extra_field":
        fields.append({"name": "legacy", "type": "VARCHAR", "params": {"max_length": 8}})
    elif mutation == "wrong_primary":
        next(field for field in fields if field["name"] == "chunk_id")["is_primary"] = False
    elif mutation == "wrong_dimension":
        next(field for field in fields if field["name"] == "dense_vector")["params"]["dim"] = 768
    else:
        next(field for field in fields if field["name"] == "content")["nullable"] = True
    client = FakeClient(exists=True, description=description, indexes=indexes)
    adapter, _, _ = runtime(client)

    with pytest.raises(MilvusSchemaIncompatible):
        adapter.ensure_schema()

    assert all(name not in {"create_collection", "create_index"} for name, _ in client.calls)
    assert adapter.schema_ready is False


@pytest.mark.parametrize(
    ("index_name", "property_name", "bad_value"),
    [
        ("dense_vector_idx", "index_type", "IVF_FLAT"),
        ("dense_vector_idx", "metric_type", "L2"),
        ("dense_vector_idx", "params", {"M": 8, "efConstruction": 200}),
        ("sparse_vector_idx", "metric_type", "COSINE"),
    ],
)
def test_incompatible_existing_index_fails_closed_without_replacement(
    index_name: str,
    property_name: str,
    bad_value: object,
) -> None:
    description, indexes = compatible_state()
    indexes[index_name][property_name] = bad_value
    client = FakeClient(exists=True, description=description, indexes=indexes)
    adapter, _, _ = runtime(client)

    with pytest.raises(MilvusSchemaIncompatible):
        adapter.ensure_schema()

    assert all(name != "create_index" for name, _ in client.calls)


def test_missing_indexes_are_idempotently_created_only_when_authorized() -> None:
    description, indexes = compatible_state()
    indexes.pop("dense_vector_idx")
    indexes.pop("tenant_id_idx")
    client = FakeClient(exists=True, description=description, indexes=indexes)
    adapter, artifacts, _ = runtime(client)

    report = adapter.ensure_schema()

    assert set(report.indexes_created) == {"dense_vector_idx", "tenant_id_idx"}
    assert set(client.indexes) == _EXPECTED_INDEXES
    assert len(artifacts.index_calls) == 1
    assert {spec.index_name for spec in artifacts.index_calls[0]} == {
        "dense_vector_idx",
        "tenant_id_idx",
    }


def test_missing_indexes_without_authority_fail_without_create() -> None:
    description, indexes = compatible_state()
    indexes.pop("commit_token_idx")
    client = FakeClient(exists=True, description=description, indexes=indexes)
    adapter, _, _ = runtime(client, runtime_config=config(allow_create=False))

    with pytest.raises(MilvusSchemaMissing, match="indexes are missing"):
        adapter.ensure_schema()

    assert all(name != "create_index" for name, _ in client.calls)


def test_conflicting_index_name_on_managed_field_is_not_overwritten() -> None:
    description, indexes = compatible_state()
    dense = indexes.pop("dense_vector_idx")
    dense["index_name"] = "legacy_dense_idx"
    indexes["legacy_dense_idx"] = dense
    client = FakeClient(exists=True, description=description, indexes=indexes)
    adapter, _, _ = runtime(client)

    with pytest.raises(MilvusSchemaIncompatible, match="conflicting index"):
        adapter.ensure_schema()

    assert all(name != "create_index" for name, _ in client.calls)


def test_unknown_create_acknowledgements_are_reconciled_without_duplicates() -> None:
    client = FakeClient()
    client.create_collection_then_raise = True
    adapter, artifacts, _ = runtime(client)

    report = adapter.ensure_schema()

    assert report.collection_created is False
    assert set(client.indexes) == _EXPECTED_INDEXES
    assert len(artifacts.schema_calls) == 1
    assert sum(name == "create_collection" for name, _ in client.calls) == 1


def test_unknown_index_acknowledgement_is_reconciled_by_metadata_readback() -> None:
    description, indexes = compatible_state()
    indexes.pop("sparse_vector_idx")
    client = FakeClient(exists=True, description=description, indexes=indexes)
    client.create_index_then_raise = True
    adapter, _, _ = runtime(client)

    report = adapter.ensure_schema()

    assert report.indexes_created == ("sparse_vector_idx",)
    assert "sparse_vector_idx" in client.indexes
    assert sum(name == "create_index" for name, _ in client.calls) == 1


def test_data_plane_adapter_enforces_managed_collection_and_strong_reads() -> None:
    client = FakeClient()
    client.query_result = [{"chunk_id": "1" * 64}]
    adapter, _, _ = runtime(client)
    adapter.ensure_schema()

    records = adapter.query(
        collection_name=_COLLECTION,
        filter='tenant_id == "tenant-a"',
        output_fields=["chunk_id"],
        limit=5,
        consistency_level="Strong",
    )
    upserted = adapter.upsert(collection_name=_COLLECTION, data=[{"chunk_id": "1" * 64}])
    deleted = adapter.delete(
        collection_name=_COLLECTION,
        filter='tenant_id == "tenant-a" and import_version == "v"',
    )

    assert records == ({"chunk_id": "1" * 64},)
    assert upserted == {"upsert_count": 1}
    assert deleted == {"delete_count": 1}
    query_call = [kwargs for name, kwargs in client.calls if name == "query"][-1]
    assert query_call["consistency_level"] == "Strong"
    assert query_call["timeout"] == 3.0
    with pytest.raises(ValueError, match="outside"):
        adapter.delete(collection_name="other_collection", filter="chunk_id != \"\"")
    with pytest.raises(ValueError, match="contract"):
        adapter.query(
            collection_name=_COLLECTION,
            filter="chunk_id != \"\"",
            output_fields=["chunk_id"],
            limit=1,
            consistency_level="Eventually",
        )
    with pytest.raises(ValueError, match="output"):
        adapter.query(
            collection_name=_COLLECTION,
            filter="chunk_id != \"\"",
            output_fields=["dynamic_secret"],
            limit=1,
            consistency_level="Strong",
        )


def _as_hybrid_search_client(
    client: MilvusHybridSearchClient,
) -> MilvusHybridSearchClient:
    return client


def test_hybrid_search_implements_source_protocol_and_injects_runtime_policy() -> None:
    client = FakeClient()
    expected_response = [[{"id": "1" * 64, "distance": 0.9, "entity": {}}]]
    client.hybrid_search_result = expected_response
    adapter, _, _ = runtime(client)
    adapter.ensure_schema()
    assert _as_hybrid_search_client(adapter) is adapter
    requests = [object(), object()]
    ranker = object()
    output_fields = list(MILVUS_CANDIDATE_OUTPUT_FIELDS)

    response = adapter.hybrid_search(
        collection_name=_COLLECTION,
        reqs=requests,
        ranker=ranker,
        limit=7,
        output_fields=output_fields,
        consistency_level="Strong",
    )
    requests.clear()
    output_fields.clear()

    assert response is expected_response
    call = [kwargs for name, kwargs in client.calls if name == "hybrid_search"][-1]
    assert call["collection_name"] == _COLLECTION
    called_requests = call["reqs"]
    assert isinstance(called_requests, list)
    assert len(called_requests) == 2
    assert call["ranker"] is ranker
    assert call["limit"] == 7
    assert call["output_fields"] == list(MILVUS_CANDIDATE_OUTPUT_FIELDS)
    assert call["consistency_level"] == "Strong"
    assert call["timeout"] == 3.0


@pytest.mark.parametrize(
    "invalid_contract",
    [
        "collection",
        "consistency",
        "one_request",
        "request_tuple",
        "null_request",
        "null_ranker",
        "zero_limit",
        "oversized_limit",
        "boolean_limit",
        "dynamic_field",
        "vector_field",
        "unhashable_field",
        "duplicate_field",
        "field_tuple",
    ],
)
def test_hybrid_search_rejects_unmanaged_or_unbounded_contract_before_client(
    invalid_contract: str,
) -> None:
    client = FakeClient()
    adapter, _, _ = runtime(client, runtime_config=config(max_query_limit=10))
    adapter.ensure_schema()
    collection_name = _COLLECTION
    requests: list[object] = [object(), object()]
    ranker: object = object()
    limit = 5
    output_fields = list(MILVUS_CANDIDATE_OUTPUT_FIELDS)
    consistency_level = "Strong"
    if invalid_contract == "collection":
        collection_name = "another_collection"
    elif invalid_contract == "consistency":
        consistency_level = "Eventually"
    elif invalid_contract == "one_request":
        requests = [object()]
    elif invalid_contract == "request_tuple":
        requests = cast(list[object], (object(), object()))
    elif invalid_contract == "null_request":
        requests[1] = None
    elif invalid_contract == "null_ranker":
        ranker = None
    elif invalid_contract == "zero_limit":
        limit = 0
    elif invalid_contract == "oversized_limit":
        limit = 11
    elif invalid_contract == "boolean_limit":
        limit = True
    elif invalid_contract == "dynamic_field":
        output_fields = ["dynamic_secret"]
    elif invalid_contract == "vector_field":
        output_fields = ["dense_vector"]
    elif invalid_contract == "unhashable_field":
        output_fields = cast(list[str], [{}])
    elif invalid_contract == "duplicate_field":
        output_fields = ["chunk_id", "chunk_id"]
    else:
        output_fields = cast(list[str], ("chunk_id",))
    calls_before = len(client.calls)

    with pytest.raises(ValueError, match="Milvus"):
        adapter.hybrid_search(
            collection_name=collection_name,
            reqs=requests,
            ranker=ranker,
            limit=limit,
            output_fields=output_fields,
            consistency_level=consistency_level,
        )

    assert len(client.calls) == calls_before


def test_hybrid_search_dependency_failure_is_sanitized_without_cause() -> None:
    client = FakeClient()
    adapter, _, _ = runtime(client)
    adapter.ensure_schema()
    client.fail_operation = "hybrid_search"

    with pytest.raises(MilvusRuntimeUnavailable) as raised:
        adapter.hybrid_search(
            collection_name=_COLLECTION,
            reqs=[object(), object()],
            ranker=object(),
            limit=1,
            output_fields=list(MILVUS_CANDIDATE_OUTPUT_FIELDS),
            consistency_level="Strong",
        )

    assert str(raised.value) == "Milvus operation is unavailable"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_probe_detects_lost_collection_and_revokes_schema_gate() -> None:
    client = FakeClient()
    adapter, _, _ = runtime(client)
    adapter.ensure_schema()
    adapter.probe()
    client.exists = False

    with pytest.raises(MilvusRuntimeUnavailable, match="collection is unavailable"):
        adapter.probe()

    assert adapter.schema_ready is False
    with pytest.raises(MilvusSchemaNotReady):
        adapter.upsert(collection_name=_COLLECTION, data=[{"chunk_id": "1" * 64}])


def test_external_failures_are_sanitized_and_have_no_cause() -> None:
    client = FakeClient()
    client.fail_operation = "has_collection"
    adapter, _, _ = runtime(client)

    with pytest.raises(MilvusRuntimeUnavailable) as raised:
        adapter.ensure_schema()

    assert str(raised.value) == "Milvus operation is unavailable"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_lazy_connection_failure_is_sanitized() -> None:
    def failing_factory(_: MilvusRuntimeConfig) -> FakeClient:
        raise RuntimeError(
            "https://admin:password@private-host"  # pragma: allowlist secret
        )

    adapter = build_milvus_runtime(config(), client_factory=failing_factory)

    with pytest.raises(MilvusRuntimeUnavailable) as raised:
        adapter.ensure_schema()

    assert str(raised.value) == "Milvus connection is unavailable"
    assert raised.value.__cause__ is None


def test_client_is_constructed_once_under_concurrent_gate_calls() -> None:
    client = FakeClient()
    artifacts = FakeArtifacts()
    calls = 0
    lock = threading.Lock()

    def factory(_: MilvusRuntimeConfig) -> FakeClient:
        nonlocal calls
        with lock:
            calls += 1
        return client

    adapter = build_milvus_runtime(config(), client_factory=factory, artifact_factory=artifacts)
    errors: list[BaseException] = []

    def gate() -> None:
        try:
            adapter.ensure_schema()
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=gate) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert calls == 1
    assert sum(name == "create_collection" for name, _ in client.calls) == 1


def test_close_is_idempotent_and_terminal() -> None:
    client = FakeClient()
    adapter, _, _ = runtime(client)
    adapter.ensure_schema()

    adapter.close()
    adapter.close()

    assert client.closed is True
    assert sum(name == "close" for name, _ in client.calls) == 1
    assert adapter.schema_ready is False
    with pytest.raises(MilvusRuntimeClosed):
        adapter.ensure_schema()


@pytest.mark.parametrize(
    "overrides",
    [
        {"uri": ""},
        {"uri": "milvus.internal:19530"},
        {"uri": "https://root:secret@milvus.internal:19530"},  # pragma: allowlist secret
        {"collection_name": "bad-name"},
        {"dense_dimension": 0},
        {"allow_create": "yes"},
        {  # pragma: allowlist secret
            "token": "token",
            "user": "root",
            "password": "secret",  # pragma: allowlist secret
        },
        {"user": "root", "password": None},
        {"timeout_seconds": float("nan")},
    ],
)
def test_invalid_or_incomplete_runtime_config_fails_before_connection(
    overrides: Mapping[str, object],
) -> None:
    with pytest.raises(MilvusConfigurationError):
        config(**overrides)


def test_config_repr_never_contains_credentials() -> None:
    credential_text = "credential-value"
    configured = config(token=None, user="operator", password=credential_text)

    rendered = repr(configured)

    assert "operator" not in rendered
    assert "credential-value" not in rendered


def test_settings_factory_requires_explicit_capability_and_unwraps_secret() -> None:
    client = FakeClient()
    received: list[MilvusRuntimeConfig] = []

    def factory(runtime_config: MilvusRuntimeConfig) -> FakeClient:
        received.append(runtime_config)
        return client

    settings = SimpleNamespace(
        api=SimpleNamespace(environment="production"),
        milvus=SimpleNamespace(url="https://milvus.prod:19530", chunks_collection=_COLLECTION),
    )
    credential_text = "credential-token"
    adapter = build_milvus_runtime_from_settings(
        settings,  # type: ignore[arg-type]
        dense_dimension=1024,
        allow_create=False,
        token=SecretStr(credential_text),
        client_factory=factory,
        artifact_factory=FakeArtifacts(),
    )

    assert received == []
    with pytest.raises(MilvusSchemaMissing):
        adapter.ensure_schema()
    assert received[0].production is True
    assert received[0].token == credential_text
    assert credential_text not in repr(received[0])


def test_settings_factory_fails_closed_when_production_url_is_missing() -> None:
    settings = SimpleNamespace(
        api=SimpleNamespace(environment="production"),
        milvus=SimpleNamespace(url=None, chunks_collection=_COLLECTION),
    )

    with pytest.raises(MilvusConfigurationError, match="MILVUS_URL is required"):
        build_milvus_runtime_from_settings(
            settings,  # type: ignore[arg-type]
            dense_dimension=1024,
            allow_create=False,
        )


def test_schema_fingerprint_changes_with_dense_dimension() -> None:
    first, _, _ = runtime(FakeClient(), runtime_config=config(dense_dimension=1024))
    second, _, _ = runtime(FakeClient(), runtime_config=config(dense_dimension=768))

    assert first.schema_fingerprint != second.schema_fingerprint
    alternate_credential = "other"
    assert first.schema_fingerprint == build_milvus_runtime(
        replace(config(), token=alternate_credential)
    ).schema_fingerprint
