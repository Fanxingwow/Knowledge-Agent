from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest
from pydantic import SecretStr
from pymongo import IndexModel, MongoClient, ReadPreference
from pymongo.client_session import ClientSession
from pymongo.errors import DuplicateKeyError, PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern

from app.core.errors import DependencyError, ErrorCode
from app.core.settings import MongoSettings, Settings
from app.infrastructure.mongo_import_store import (
    MongoImportStore,
    build_mongo_import_store,
)


class FakeCollection:
    def __init__(self) -> None:
        self.indexes: list[IndexModel] = []
        self.failure: Exception | None = None

    def create_indexes(self, indexes: list[IndexModel]) -> list[str]:
        if self.failure is not None:
            raise self.failure
        self.indexes.extend(indexes)
        return [str(index.document["name"]) for index in indexes]


class FakeDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection())


class FakeAdmin:
    def __init__(self) -> None:
        self.hello: dict[str, object] = {
            "ok": 1,
            "logicalSessionTimeoutMinutes": 30,
            "setName": "rs0",
        }
        self.failure: Exception | None = None
        self.calls: list[tuple[str, dict[str, object]]] = []

    def command(self, command: str, **kwargs: object) -> dict[str, object]:
        self.calls.append((command, kwargs))
        if self.failure is not None:
            raise self.failure
        if command == "ping":
            return {"ok": 1}
        if command == "hello":
            return dict(self.hello)
        raise AssertionError(f"unexpected command: {command}")


class FakeSession:
    def __init__(self) -> None:
        self.transaction_options: dict[str, object] | None = None
        self.callback_session: object | None = None
        self.failure: Exception | None = None

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def with_transaction(
        self,
        callback: Callable[[object], object],
        **kwargs: object,
    ) -> object:
        if self.failure is not None:
            raise self.failure
        self.transaction_options = kwargs
        self.callback_session = self
        return callback(self)


class FakeClient:
    def __init__(self) -> None:
        self.admin = FakeAdmin()
        self.database = FakeDatabase()
        self.database_name: str | None = None
        self.database_options: dict[str, object] = {}
        self.session = FakeSession()
        self.close_count = 0

    def get_database(self, name: str, **kwargs: object) -> FakeDatabase:
        self.database_name = name
        self.database_options = kwargs
        return self.database

    def start_session(self) -> FakeSession:
        return self.session

    def close(self) -> None:
        self.close_count += 1


class RecordingClientFactory:
    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.uri: str | None = None
        self.kwargs: dict[str, object] = {}
        self.failure: Exception | None = None

    def __call__(self, uri: str, **kwargs: object) -> MongoClient[dict[str, Any]]:
        self.uri = uri
        self.kwargs = kwargs
        if self.failure is not None:
            raise self.failure
        return cast(MongoClient[dict[str, Any]], self.client)


def mongo_settings(**changes: object) -> MongoSettings:
    values: dict[str, object] = {
        "url": SecretStr("mongodb://mongo-user:top-secret@db.internal:27017/kb"),
        "database": "kb_runtime",
        "import_admissions_collection": "import_admissions_v1",
        "import_tasks_collection": "import_tasks_v1",
        "import_outbox_collection": "import_outbox_v1",
        "connect_timeout_seconds": 1.25,
        "server_selection_timeout_seconds": 2.5,
        "socket_timeout_seconds": 12.0,
        "health_timeout_seconds": 1.5,
        "transaction_timeout_seconds": 7.25,
        "wait_queue_timeout_seconds": 3.75,
        "max_pool_size": 17,
        "require_replica_set": True,
    }
    values.update(changes)
    aliases = {
        "url": "MONGO_URL",
        "database": "MONGO_DB_NAME",
        "import_admissions_collection": "MONGO_IMPORT_ADMISSIONS_COLLECTION",
        "import_tasks_collection": "MONGO_IMPORT_TASKS_COLLECTION",
        "import_outbox_collection": "MONGO_IMPORT_OUTBOX_COLLECTION",
        "connect_timeout_seconds": "MONGO_CONNECT_TIMEOUT_SECONDS",
        "server_selection_timeout_seconds": "MONGO_SERVER_SELECTION_TIMEOUT_SECONDS",
        "socket_timeout_seconds": "MONGO_SOCKET_TIMEOUT_SECONDS",
        "health_timeout_seconds": "MONGO_HEALTH_TIMEOUT_SECONDS",
        "transaction_timeout_seconds": "MONGO_TRANSACTION_TIMEOUT_SECONDS",
        "wait_queue_timeout_seconds": "MONGO_WAIT_QUEUE_TIMEOUT_SECONDS",
        "max_pool_size": "MONGO_MAX_POOL_SIZE",
        "require_replica_set": "MONGO_REQUIRE_REPLICA_SET",
    }
    aliased_values = {aliases[name]: value for name, value in values.items()}
    return MongoSettings.model_validate(aliased_values)


def build_settings(**changes: object) -> Settings:
    return Settings(mongo=mongo_settings(**changes))


def direct_store(
    client: FakeClient | None = None,
    **changes: object,
) -> tuple[MongoImportStore, FakeClient]:
    fake = client or FakeClient()
    store = MongoImportStore(
        cast(MongoClient[dict[str, Any]], fake),
        mongo_settings(**changes),
    )
    return store, fake


def index_documents(collection: FakeCollection) -> dict[str, dict[str, object]]:
    return {
        str(index.document["name"]): dict(index.document)
        for index in collection.indexes
    }


def test_builder_uses_configured_client_parameters_and_exposes_collections() -> None:
    client = FakeClient()
    factory = RecordingClientFactory(client)

    store = build_mongo_import_store(build_settings(), client_factory=factory)

    assert store.durable is True
    assert factory.uri == (
        "mongodb://mongo-user:top-secret@db.internal:27017/kb"  # pragma: allowlist secret
    )
    assert factory.kwargs == {
        "appname": "knowledge-agent",
        "connectTimeoutMS": 1250,
        "serverSelectionTimeoutMS": 2500,
        "socketTimeoutMS": 12000,
        "waitQueueTimeoutMS": 3750,
        "maxPoolSize": 17,
        "retryReads": True,
        "retryWrites": True,
        "tz_aware": True,
        "uuidRepresentation": "standard",
    }
    assert client.database_name == "kb_runtime"
    assert store.admissions is client.database.collections["import_admissions_v1"]
    assert store.tasks is client.database.collections["import_tasks_v1"]
    assert store.outbox is client.database.collections["import_outbox_v1"]
    assert [call[0] for call in client.admin.calls] == ["ping", "hello"]
    assert all(call[1] == {"maxTimeMS": 1500} for call in client.admin.calls)


def test_ensure_indexes_defines_unique_claim_and_ttl_indexes() -> None:
    store, client = direct_store()

    store.ensure_indexes()

    admissions = index_documents(client.database.collections["import_admissions_v1"])
    tasks = index_documents(client.database.collections["import_tasks_v1"])
    outbox = index_documents(client.database.collections["import_outbox_v1"])

    assert set(admissions) == {
        "uq_import_admission_idempotency",
        "ix_import_admission_stale",
        "ix_import_admission_cleanup",
        "ttl_import_admission",
    }
    assert admissions["uq_import_admission_idempotency"]["unique"] is True
    admission_key = cast(
        Mapping[str, int],
        admissions["uq_import_admission_idempotency"]["key"],
    )
    assert list(admission_key.items()) == [
        ("tenant_id", 1),
        ("owner_subject", 1),
        ("idempotency_key", 1),
    ]
    assert admissions["ttl_import_admission"]["expireAfterSeconds"] == 0

    task_claim_key = cast(Mapping[str, int], tasks["ix_import_task_claim"]["key"])
    assert list(task_claim_key.items()) == [
        ("status", 1),
        ("next_attempt_at", 1),
        ("lease_expires_at", 1),
        ("created_at", 1),
    ]
    assert tasks["ttl_import_task"]["expireAfterSeconds"] == 0

    assert outbox["uq_import_outbox_dedupe"]["unique"] is True
    outbox_dispatch_key = cast(
        Mapping[str, int],
        outbox["ix_import_outbox_dispatch"]["key"],
    )
    assert list(outbox_dispatch_key.items()) == [
        ("status", 1),
        ("available_at", 1),
        ("lease_expires_at", 1),
        ("created_at", 1),
    ]
    assert outbox["ttl_import_outbox"]["expireAfterSeconds"] == 0


@pytest.mark.parametrize(
    "hello",
    [
        {"ok": 1, "setName": "rs0"},
        {"ok": 1, "logicalSessionTimeoutMinutes": 30},
        {"ok": 1, "logicalSessionTimeoutMinutes": 0, "setName": "rs0"},
    ],
)
def test_replica_gate_requires_sessions_and_supported_topology(
    hello: dict[str, object],
) -> None:
    client = FakeClient()
    client.admin.hello = hello
    factory = RecordingClientFactory(client)

    with pytest.raises(DependencyError) as captured:
        build_mongo_import_store(build_settings(), client_factory=factory)

    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert captured.value.message == "MongoDB import store is unavailable"
    assert client.close_count == 1


def test_replica_gate_accepts_mongos_and_can_be_disabled() -> None:
    mongos_client = FakeClient()
    mongos_client.admin.hello = {
        "ok": 1,
        "logicalSessionTimeoutMinutes": 30,
        "msg": "isdbgrid",
    }
    mongos_store = build_mongo_import_store(
        build_settings(),
        client_factory=RecordingClientFactory(mongos_client),
    )

    standalone_client = FakeClient()
    standalone_client.admin.hello = {"ok": 1}
    standalone_store = build_mongo_import_store(
        build_settings(require_replica_set=False),
        client_factory=RecordingClientFactory(standalone_client),
    )

    assert mongos_store.is_ready() is True
    assert standalone_store.is_ready() is True


def test_close_is_idempotent_and_closed_store_is_not_ready() -> None:
    store, client = direct_store()

    store.close()
    store.close()

    assert client.close_count == 1
    assert store.is_ready() is False
    with pytest.raises(DependencyError):
        store.require_ready()


def test_with_transaction_uses_required_concerns_and_timeout() -> None:
    store, client = direct_store()
    observed: list[object] = []

    def callback(session: ClientSession) -> str:
        observed.append(session)
        return "committed"

    result = store.with_transaction(callback)

    assert result == "committed"
    assert observed == [client.session]
    options = client.session.transaction_options
    assert options is not None
    read_concern = cast(ReadConcern, options["read_concern"])
    write_concern = cast(WriteConcern, options["write_concern"])
    assert read_concern.level == "snapshot"
    assert write_concern.document == {
        "w": "majority",
        "wtimeout": 7250,
    }
    assert options["read_preference"] is ReadPreference.PRIMARY
    assert options["max_commit_time_ms"] == 7250


def test_shared_database_and_transaction_runner_reuse_the_owned_client() -> None:
    store, client = direct_store()
    observed: list[object] = []

    result = store.run(lambda session: observed.append(session) or "committed")

    assert store.database is client.database
    assert result == "committed"
    assert observed == [client.session]
    assert client.close_count == 0

    store.close()
    with pytest.raises(DependencyError, match="MongoDB import store is unavailable"):
        _ = store.database


def test_shared_transaction_runner_preserves_duplicate_signal_for_idempotency() -> None:
    store, client = direct_store()
    duplicate = DuplicateKeyError("private duplicate index detail")
    client.session.failure = duplicate

    with pytest.raises(DuplicateKeyError) as raised:
        store.run(lambda _session: None)

    assert raised.value is duplicate

    with pytest.raises(DependencyError, match="MongoDB import store is unavailable"):
        store.with_transaction(lambda _session: None)


@pytest.mark.parametrize("failure_stage", ["construct", "probe", "index"])
def test_builder_failures_close_client_and_do_not_expose_upstream_details(
    failure_stage: str,
) -> None:
    client = FakeClient()
    factory = RecordingClientFactory(client)
    upstream_detail = (
        "mongodb://mongo-user:leaked-password@db.internal:27017/kb"  # pragma: allowlist secret
    )
    failure = RuntimeError(f"upstream failed at {upstream_detail}")
    if failure_stage == "construct":
        factory.failure = failure
    elif failure_stage == "probe":
        client.admin.failure = failure
    else:
        client.database["import_tasks_v1"].failure = failure

    with pytest.raises(DependencyError) as captured:
        build_mongo_import_store(build_settings(), client_factory=factory)

    rendered = f"{captured.value} {captured.value.message} {captured.value.details}"
    assert upstream_detail not in rendered
    assert "leaked-password" not in rendered
    assert captured.value.message == "MongoDB import store is unavailable"
    expected_close_count = 0 if failure_stage == "construct" else 1
    assert client.close_count == expected_close_count


def test_is_ready_fails_closed_without_leaking_probe_error() -> None:
    client = FakeClient()
    client.admin.failure = RuntimeError("private topology details")
    store, _ = direct_store(client)

    assert store.is_ready() is False
    with pytest.raises(DependencyError, match="MongoDB import store is unavailable") as captured:
        store.require_ready()
    assert "private topology details" not in str(captured.value)


def test_transaction_failure_is_sanitized() -> None:
    store, client = direct_store()
    client.session.failure = PyMongoError("private transaction details")

    with pytest.raises(DependencyError, match="MongoDB import store is unavailable") as captured:
        store.with_transaction(lambda _session: None)

    assert "private transaction details" not in str(captured.value)


def test_transaction_callback_error_is_not_misclassified_as_dependency_failure() -> None:
    store, _ = direct_store()
    domain_error = ValueError("domain conflict")

    def callback(_session: ClientSession) -> None:
        raise domain_error

    with pytest.raises(ValueError) as captured:
        store.with_transaction(callback)

    assert captured.value is domain_error
