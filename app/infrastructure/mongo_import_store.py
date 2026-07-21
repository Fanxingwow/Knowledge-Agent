from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from threading import Lock
from typing import Any, TypeVar, cast

from pymongo import ASCENDING, DESCENDING, IndexModel, MongoClient, ReadPreference
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern

from app.core.errors import DependencyError
from app.core.settings import MongoSettings, Settings

MongoDocument = dict[str, Any]
MongoClientFactory = Callable[..., MongoClient[MongoDocument]]

_T = TypeVar("_T")
_UNAVAILABLE_MESSAGE = "MongoDB import store is unavailable"


def import_index_contracts() -> Mapping[str, tuple[IndexModel, ...]]:
    """Return the canonical durable-import index contract used by runtime and ops."""

    return {
        "admissions": (
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("owner_subject", ASCENDING),
                    ("idempotency_key", ASCENDING),
                ],
                unique=True,
                name="uq_import_admission_idempotency",
            ),
            IndexModel(
                [("state", ASCENDING), ("updated_at", ASCENDING)],
                name="ix_import_admission_stale",
            ),
            IndexModel(
                [
                    ("state", ASCENDING),
                    ("cleanup_next_attempt_at", ASCENDING),
                    ("cleanup_lease_expires_at", ASCENDING),
                ],
                name="ix_import_admission_cleanup",
            ),
            IndexModel(
                [("expire_at", ASCENDING)],
                expireAfterSeconds=0,
                name="ttl_import_admission",
            ),
        ),
        "tasks": (
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("owner_subject", ASCENDING),
                    ("created_at", DESCENDING),
                ],
                name="ix_import_task_owned",
            ),
            IndexModel(
                [("admission_id", ASCENDING), ("created_at", ASCENDING)],
                name="ix_import_task_admission",
            ),
            IndexModel(
                [
                    ("status", ASCENDING),
                    ("next_attempt_at", ASCENDING),
                    ("lease_expires_at", ASCENDING),
                    ("created_at", ASCENDING),
                ],
                name="ix_import_task_claim",
            ),
            IndexModel(
                [("expire_at", ASCENDING)],
                expireAfterSeconds=0,
                name="ttl_import_task",
            ),
        ),
        "outbox": (
            IndexModel(
                [("dedupe_key", ASCENDING)],
                unique=True,
                name="uq_import_outbox_dedupe",
            ),
            IndexModel(
                [
                    ("status", ASCENDING),
                    ("available_at", ASCENDING),
                    ("lease_expires_at", ASCENDING),
                    ("created_at", ASCENDING),
                ],
                name="ix_import_outbox_dispatch",
            ),
            IndexModel(
                [
                    ("aggregate_type", ASCENDING),
                    ("aggregate_id", ASCENDING),
                    ("event_type", ASCENDING),
                ],
                name="ix_import_outbox_aggregate",
            ),
            IndexModel(
                [("expire_at", ASCENDING)],
                expireAfterSeconds=0,
                name="ttl_import_outbox",
            ),
        ),
    }


def _milliseconds(seconds: float) -> int:
    return max(1, math.ceil(seconds * 1000))


def _close_client(client: MongoClient[MongoDocument]) -> None:
    try:
        client.close()
    except Exception:
        return


class MongoImportStore:
    """Own the MongoDB resources used by the durable import subsystem."""

    durable = True

    def __init__(
        self,
        client: MongoClient[MongoDocument],
        settings: MongoSettings,
    ) -> None:
        self._client = client
        self._settings = settings
        self._health_timeout_ms = _milliseconds(settings.health_timeout_seconds)
        self._transaction_timeout_ms = _milliseconds(settings.transaction_timeout_seconds)
        self._close_lock = Lock()
        self._closed = False

        self._database: Database[MongoDocument] = client.get_database(
            settings.database,
            read_concern=ReadConcern("majority"),
            write_concern=WriteConcern(
                "majority",
                wtimeout=self._transaction_timeout_ms,
            ),
            read_preference=ReadPreference.PRIMARY,
        )
        self._admissions = self._database[settings.import_admissions_collection]
        self._tasks = self._database[settings.import_tasks_collection]
        self._outbox = self._database[settings.import_outbox_collection]

    @property
    def admissions(self) -> Collection[MongoDocument]:
        return self._admissions

    @property
    def tasks(self) -> Collection[MongoDocument]:
        return self._tasks

    @property
    def outbox(self) -> Collection[MongoDocument]:
        return self._outbox

    @property
    def database(self) -> Database[MongoDocument]:
        """Expose the already configured majority database to sibling repositories."""

        self._ensure_open()
        return self._database

    def _ensure_open(self) -> None:
        if self._closed:
            raise DependencyError(_UNAVAILABLE_MESSAGE)

    def ensure_indexes(self) -> None:
        self._ensure_open()
        try:
            contracts = import_index_contracts()
            self._admissions.create_indexes(list(contracts["admissions"]))
            self._tasks.create_indexes(list(contracts["tasks"]))
            self._outbox.create_indexes(list(contracts["outbox"]))
        except DependencyError:
            raise
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None

    @staticmethod
    def _successful_command(response: object) -> bool:
        return isinstance(response, Mapping) and response.get("ok") == 1

    @staticmethod
    def _supports_transactions(hello: Mapping[str, object]) -> bool:
        logical_session_timeout = hello.get("logicalSessionTimeoutMinutes")
        has_logical_sessions = (
            isinstance(logical_session_timeout, int)
            and not isinstance(logical_session_timeout, bool)
            and logical_session_timeout > 0
        )
        set_name = hello.get("setName")
        is_replica_set = isinstance(set_name, str) and bool(set_name)
        is_mongos = hello.get("msg") == "isdbgrid"
        return has_logical_sessions and (is_replica_set or is_mongos)

    def require_ready(self) -> None:
        self._ensure_open()
        try:
            ping = self._client.admin.command(
                "ping",
                maxTimeMS=self._health_timeout_ms,
            )
            hello = self._client.admin.command(
                "hello",
                maxTimeMS=self._health_timeout_ms,
            )
            if not self._successful_command(ping) or not self._successful_command(hello):
                raise DependencyError(_UNAVAILABLE_MESSAGE)
            if self._settings.require_replica_set and not self._supports_transactions(
                cast(Mapping[str, object], hello)
            ):
                raise DependencyError(_UNAVAILABLE_MESSAGE)
        except DependencyError:
            raise
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None

    def is_ready(self) -> bool:
        try:
            self.require_ready()
        except DependencyError:
            return False
        return True

    def _execute_transaction(self, callback: Callable[[ClientSession], _T]) -> _T:
        self._ensure_open()
        with self._client.start_session() as session:
            return session.with_transaction(
                callback,
                read_concern=ReadConcern("snapshot"),
                write_concern=WriteConcern(
                    "majority",
                    wtimeout=self._transaction_timeout_ms,
                ),
                read_preference=ReadPreference.PRIMARY,
                max_commit_time_ms=self._transaction_timeout_ms,
            )

    def with_transaction(self, callback: Callable[[ClientSession], _T]) -> _T:
        try:
            return self._execute_transaction(callback)
        except PyMongoError:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None

    def run(self, operation: Callable[[object], _T]) -> _T:
        """Implement the shared repository transaction-runner contract."""

        try:
            return self._execute_transaction(operation)
        except DuplicateKeyError:
            raise
        except PyMongoError:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._client.close()
        except Exception:
            return


def build_mongo_import_store(
    settings: Settings,
    *,
    client_factory: MongoClientFactory = MongoClient,
) -> MongoImportStore:
    client: MongoClient[MongoDocument] | None = None
    store: MongoImportStore | None = None
    try:
        settings.validate_capabilities("mongo")
        mongo = settings.mongo
        if mongo.url is None:
            raise DependencyError(_UNAVAILABLE_MESSAGE)
        client = client_factory(
            mongo.url.get_secret_value(),
            appname="knowledge-agent",
            connectTimeoutMS=_milliseconds(mongo.connect_timeout_seconds),
            serverSelectionTimeoutMS=_milliseconds(mongo.server_selection_timeout_seconds),
            socketTimeoutMS=_milliseconds(mongo.socket_timeout_seconds),
            waitQueueTimeoutMS=_milliseconds(mongo.wait_queue_timeout_seconds),
            maxPoolSize=mongo.max_pool_size,
            retryReads=True,
            retryWrites=True,
            tz_aware=True,
            uuidRepresentation="standard",
        )
        store = MongoImportStore(client, mongo)
        store.require_ready()
        store.ensure_indexes()
        return store
    except Exception:
        if store is not None:
            store.close()
        elif client is not None:
            _close_client(client)
        raise DependencyError(_UNAVAILABLE_MESSAGE) from None
