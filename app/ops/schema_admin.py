from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol, cast

from pymongo import IndexModel, MongoClient

from app.core.settings import AppEnvironment, MongoSettings, Settings, get_settings
from app.infrastructure.milvus_runtime import (
    MilvusRuntimeAdapter,
    MilvusSchemaIncompatible,
    MilvusSchemaMissing,
    MilvusSchemaReport,
    build_milvus_runtime_from_settings,
)
from app.infrastructure.mongo_chat_repository import (
    MongoChatSchemaManager,
    chat_index_contracts,
    chat_schema_contracts,
)
from app.infrastructure.mongo_import_store import MongoImportStore, import_index_contracts

EXIT_READY = 0
EXIT_CHANGES_REQUIRED = 2
EXIT_INCOMPATIBLE = 3
EXIT_DEPENDENCY_FAILURE = 4
EXIT_PRODUCTION_CONFIRMATION_REQUIRED = 5
EXIT_USAGE = 64

_OUTPUT_SCHEMA_VERSION = 1
_GENERIC_FAILURE_MESSAGE = "Schema operation failed"
_PRODUCTION_CONFIRMATION_ERROR = "PRODUCTION_APPLY_CONFIRMATION_REQUIRED"
_CONFIGURATION_ERROR = "SCHEMA_CONFIGURATION_INVALID"
_DEPENDENCY_ERROR = "SCHEMA_DEPENDENCY_UNAVAILABLE"
_INCOMPATIBLE_ERROR = "SCHEMA_INCOMPATIBLE"
_CHANGES_ERROR = "SCHEMA_CHANGES_REQUIRED"
_APPLY_INCOMPLETE_ERROR = "SCHEMA_APPLY_INCOMPLETE"
_HELP_USAGE = (
    "schema-admin [--target all|mongo|milvus] [--apply] "
    "[--confirm-production]"
)

JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class SchemaChange:
    action: str
    resource: str

    def to_dict(self) -> JsonObject:
        return {"action": self.action, "resource": self.resource}


@dataclass(frozen=True, slots=True)
class TargetReport:
    target: str
    status: str
    fingerprint: str | None
    changes: tuple[SchemaChange, ...] = ()
    issues: tuple[SchemaChange, ...] = ()
    error_code: str | None = None

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "target": self.target,
            "status": self.status,
            "changes": [change.to_dict() for change in self.changes],
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if self.fingerprint is not None:
            payload["fingerprint"] = self.fingerprint
        if self.error_code is not None:
            payload["error_code"] = self.error_code
            payload["message"] = _GENERIC_FAILURE_MESSAGE
        return payload


@dataclass(frozen=True, slots=True)
class SchemaAdminReport:
    mode: str
    environment: str
    status: str
    exit_code: int
    targets: tuple[TargetReport, ...] = ()
    error_code: str | None = None

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "schema_version": _OUTPUT_SCHEMA_VERSION,
            "mode": self.mode,
            "environment": self.environment,
            "status": self.status,
            "exit_code": self.exit_code,
            "targets": [target.to_dict() for target in self.targets],
        }
        if self.error_code is not None:
            payload["error_code"] = self.error_code
            payload["message"] = _GENERIC_FAILURE_MESSAGE
        return payload


class SchemaOperator(Protocol):
    target: str

    def inspect(self) -> TargetReport: ...

    def apply(self) -> TargetReport: ...

    def close(self) -> None: ...


class MongoOpsCollection(Protocol):
    def index_information(self) -> Mapping[str, Mapping[str, object]]: ...

    def options(self) -> Mapping[str, object]: ...

    def find_one(
        self,
        filter: Mapping[str, object],
        projection: Mapping[str, int] | None = None,
        **kwargs: object,
    ) -> Mapping[str, object] | None: ...


class MongoOpsDatabase(Protocol):
    def __getitem__(self, name: str) -> MongoOpsCollection: ...

    def list_collection_names(self) -> list[str]: ...


class MongoOpsStore(Protocol):
    @property
    def database(self) -> MongoOpsDatabase: ...

    def require_ready(self) -> None: ...

    def ensure_indexes(self) -> None: ...

    def close(self) -> None: ...


class MilvusSchemaRuntime(Protocol):
    @property
    def collection_name(self) -> str: ...

    @property
    def schema_fingerprint(self) -> str: ...

    def ensure_schema(self) -> MilvusSchemaReport: ...

    def close(self) -> None: ...


MilvusRuntimeFactory = Callable[[bool], MilvusSchemaRuntime]
SchemaOperatorFactory = Callable[[Settings, str], SchemaOperator]
SettingsLoader = Callable[[], Settings]


@dataclass(frozen=True, slots=True)
class _MongoCollectionContract:
    name: str
    indexes: tuple[IndexModel, ...]
    validator: Mapping[str, object] | None = None


def _mongo_collection_contracts(
    settings: MongoSettings,
) -> tuple[_MongoCollectionContract, ...]:
    import_indexes = import_index_contracts()
    chat_indexes = chat_index_contracts()
    chat_schemas = chat_schema_contracts()
    return (
        _MongoCollectionContract(
            settings.import_admissions_collection,
            import_indexes["admissions"],
        ),
        _MongoCollectionContract(
            settings.import_tasks_collection,
            import_indexes["tasks"],
        ),
        _MongoCollectionContract(
            settings.import_outbox_collection,
            import_indexes["outbox"],
        ),
        _MongoCollectionContract(
            settings.chat_sessions_collection,
            chat_indexes["sessions"],
            chat_schemas["sessions"],
        ),
        _MongoCollectionContract(
            settings.chat_messages_collection,
            chat_indexes["messages"],
            chat_schemas["messages"],
        ),
    )


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_plain(item) for item in value]
    return value


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        _plain(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _index_document(index: IndexModel) -> Mapping[str, object]:
    document = index.document
    if not isinstance(document, Mapping):
        raise ValueError("invalid Mongo index contract")
    return cast(Mapping[str, object], document)


def _index_name(index: IndexModel) -> str:
    name = _index_document(index).get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("invalid Mongo index contract")
    return name


def _ordered_keys(value: object) -> tuple[tuple[str, object], ...] | None:
    if isinstance(value, Mapping):
        pairs = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        pairs = tuple(value)
    else:
        return None
    normalized: list[tuple[str, object]] = []
    for pair in pairs:
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes, bytearray))
            or len(pair) != 2
            or not isinstance(pair[0], str)
        ):
            return None
        normalized.append((pair[0], pair[1]))
    return tuple(normalized)


def _index_matches(
    actual: Mapping[str, object],
    expected: IndexModel,
) -> bool:
    document = _index_document(expected)
    if _ordered_keys(actual.get("key")) != _ordered_keys(document.get("key")):
        return False
    if bool(actual.get("unique", False)) is not bool(
        document.get("unique", False)
    ):
        return False
    if bool(actual.get("sparse", False)) is not bool(
        document.get("sparse", False)
    ):
        return False
    if actual.get("expireAfterSeconds") != document.get("expireAfterSeconds"):
        return False
    if _plain(actual.get("partialFilterExpression")) != _plain(
        document.get("partialFilterExpression")
    ):
        return False
    return True


def _resource(collection: str, index: str | None = None) -> str:
    base = f"mongo.collection.{collection}"
    return base if index is None else f"{base}.index.{index}"


def _sorted_changes(changes: Sequence[SchemaChange]) -> tuple[SchemaChange, ...]:
    return tuple(sorted(changes, key=lambda item: (item.resource, item.action)))


class MongoSchemaOperator:
    target = "mongo"

    def __init__(
        self,
        store: MongoOpsStore,
        settings: MongoSettings,
    ) -> None:
        self._store = store
        self._settings = settings
        self._contracts = _mongo_collection_contracts(settings)
        self._document_check_timeout_ms = _milliseconds(
            settings.health_timeout_seconds
        )
        self._fingerprint = _fingerprint(
            [
                {
                    "name": contract.name,
                    "validator": contract.validator,
                    "indexes": [
                        _index_document(index)
                        for index in contract.indexes
                    ],
                }
                for contract in self._contracts
            ]
        )
        self._closed = False

    def _inspect(self) -> TargetReport:
        try:
            self._store.require_ready()
            database = self._store.database
            existing = set(database.list_collection_names())
            changes: list[SchemaChange] = []
            issues: list[SchemaChange] = []
            for contract in self._contracts:
                collection_resource = _resource(contract.name)
                if contract.name not in existing:
                    changes.append(
                        SchemaChange("create_collection", collection_resource)
                    )
                    changes.extend(
                        SchemaChange(
                            "create_index",
                            _resource(contract.name, _index_name(index)),
                        )
                        for index in contract.indexes
                    )
                    continue
                collection = database[contract.name]
                if contract.validator is None:
                    incompatible_filter: Mapping[str, object] = {
                        "schema_version": {"$ne": 1}
                    }
                else:
                    incompatible_filter = {
                        "$nor": [dict(contract.validator)]
                    }
                incompatible_document = collection.find_one(
                    incompatible_filter,
                    {"_id": 1},
                    max_time_ms=self._document_check_timeout_ms,
                )
                if incompatible_document is not None:
                    issues.append(
                        SchemaChange(
                            "manual_data_migration_required",
                            collection_resource,
                        )
                    )
                options = collection.options()
                if contract.validator is None:
                    if options.get("validator") not in (None, {}):
                        issues.append(
                            SchemaChange(
                                "unexpected_validator_requires_review",
                                collection_resource,
                            )
                        )
                else:
                    if (
                        _plain(options.get("validator"))
                        != _plain(contract.validator)
                        or options.get("validationLevel") != "strict"
                        or options.get("validationAction") != "error"
                    ):
                        changes.append(
                            SchemaChange(
                                "update_validator",
                                collection_resource,
                            )
                        )
                actual_indexes = collection.index_information()
                expected = {
                    _index_name(index): index
                    for index in contract.indexes
                }
                actual_names = set(actual_indexes)
                for name, index in expected.items():
                    observed = actual_indexes.get(name)
                    if observed is None:
                        changes.append(
                            SchemaChange(
                                "create_index",
                                _resource(contract.name, name),
                            )
                        )
                    elif not _index_matches(observed, index):
                        issues.append(
                            SchemaChange(
                                "manual_index_migration_required",
                                _resource(contract.name, name),
                            )
                        )
                for name in sorted(actual_names - set(expected) - {"_id_"}):
                    issues.append(
                        SchemaChange(
                            "unexpected_index_requires_review",
                            _resource(contract.name, name),
                        )
                    )
            if issues:
                return TargetReport(
                    target=self.target,
                    status="incompatible",
                    fingerprint=self._fingerprint,
                    changes=_sorted_changes(changes),
                    issues=_sorted_changes(issues),
                    error_code=_INCOMPATIBLE_ERROR,
                )
            status = "changes_required" if changes else "ready"
            return TargetReport(
                target=self.target,
                status=status,
                fingerprint=self._fingerprint,
                changes=_sorted_changes(changes),
            )
        except Exception:
            return TargetReport(
                target=self.target,
                status="failed",
                fingerprint=self._fingerprint,
                error_code=_DEPENDENCY_ERROR,
            )

    def inspect(self) -> TargetReport:
        if self._closed:
            return TargetReport(
                target=self.target,
                status="failed",
                fingerprint=self._fingerprint,
                error_code=_DEPENDENCY_ERROR,
            )
        return self._inspect()

    def apply(self) -> TargetReport:
        planned = self.inspect()
        if planned.status in {"failed", "incompatible"}:
            return planned
        if planned.status == "ready":
            return planned
        try:
            self._store.ensure_indexes()
            manager = MongoChatSchemaManager(
                cast(Any, self._store.database),
                sessions_collection=self._settings.chat_sessions_collection,
                messages_collection=self._settings.chat_messages_collection,
            )
            manager.ensure_schema()
        except Exception:
            return TargetReport(
                target=self.target,
                status="failed",
                fingerprint=self._fingerprint,
                changes=planned.changes,
                error_code=_DEPENDENCY_ERROR,
            )
        verified = self.inspect()
        if verified.status != "ready":
            return TargetReport(
                target=self.target,
                status="failed",
                fingerprint=self._fingerprint,
                changes=planned.changes,
                issues=verified.issues,
                error_code=_APPLY_INCOMPLETE_ERROR,
            )
        return TargetReport(
            target=self.target,
            status="applied",
            fingerprint=self._fingerprint,
            changes=planned.changes,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._store.close()


class MilvusSchemaOperator:
    target = "milvus"

    def __init__(self, runtime_factory: MilvusRuntimeFactory) -> None:
        if not callable(runtime_factory):
            raise ValueError("invalid Milvus schema runtime factory")
        self._runtime_factory = runtime_factory

    @staticmethod
    def _safe_close(runtime: MilvusSchemaRuntime | None) -> bool:
        if runtime is None:
            return True
        try:
            runtime.close()
        except Exception:
            return False
        return True

    def inspect(self) -> TargetReport:
        runtime: MilvusSchemaRuntime | None = None
        report: TargetReport
        fingerprint: str | None = None
        try:
            runtime = self._runtime_factory(False)
            fingerprint = runtime.schema_fingerprint
            runtime.ensure_schema()
            report = TargetReport(
                target=self.target,
                status="ready",
                fingerprint=fingerprint,
            )
        except MilvusSchemaMissing:
            fingerprint = (
                runtime.schema_fingerprint
                if runtime is not None
                else None
            )
            collection = (
                runtime.collection_name
                if runtime is not None
                else "managed"
            )
            report = TargetReport(
                target=self.target,
                status="changes_required",
                fingerprint=fingerprint,
                changes=(
                    SchemaChange(
                        "ensure_enterprise_schema",
                        f"milvus.collection.{collection}",
                    ),
                ),
            )
        except MilvusSchemaIncompatible:
            report = TargetReport(
                target=self.target,
                status="incompatible",
                fingerprint=(
                    runtime.schema_fingerprint
                    if runtime is not None
                    else None
                ),
                issues=(
                    SchemaChange(
                        "manual_schema_migration_required",
                        (
                            f"milvus.collection.{runtime.collection_name}"
                            if runtime is not None
                            else "milvus.collection.managed"
                        ),
                    ),
                ),
                error_code=_INCOMPATIBLE_ERROR,
            )
        except Exception:
            report = TargetReport(
                target=self.target,
                status="failed",
                fingerprint=(
                    runtime.schema_fingerprint
                    if runtime is not None
                    else None
                ),
                error_code=_DEPENDENCY_ERROR,
            )
        if not self._safe_close(runtime):
            return TargetReport(
                target=self.target,
                status="failed",
                fingerprint=report.fingerprint,
                changes=report.changes,
                issues=report.issues,
                error_code=_DEPENDENCY_ERROR,
            )
        return report

    def apply(self) -> TargetReport:
        runtime: MilvusSchemaRuntime | None = None
        result: TargetReport
        try:
            runtime = self._runtime_factory(True)
            report = runtime.ensure_schema()
            if not isinstance(report, MilvusSchemaReport):
                raise TypeError("invalid Milvus schema report")
            changes: list[SchemaChange] = []
            if report.collection_created:
                changes.append(
                    SchemaChange(
                        "create_collection",
                        f"milvus.collection.{report.collection_name}",
                    )
                )
            changes.extend(
                SchemaChange(
                    "create_index",
                    (
                        f"milvus.collection.{report.collection_name}"
                        f".index.{index_name}"
                    ),
                )
                for index_name in report.indexes_created
            )
            result = TargetReport(
                target=self.target,
                status="applied" if changes else "ready",
                fingerprint=report.schema_fingerprint,
                changes=_sorted_changes(changes),
            )
        except MilvusSchemaIncompatible:
            result = TargetReport(
                target=self.target,
                status="incompatible",
                fingerprint=(
                    runtime.schema_fingerprint
                    if runtime is not None
                    else None
                ),
                issues=(
                    SchemaChange(
                        "manual_schema_migration_required",
                        (
                            f"milvus.collection.{runtime.collection_name}"
                            if runtime is not None
                            else "milvus.collection.managed"
                        ),
                    ),
                ),
                error_code=_INCOMPATIBLE_ERROR,
            )
        except Exception:
            result = TargetReport(
                target=self.target,
                status="failed",
                fingerprint=(
                    runtime.schema_fingerprint
                    if runtime is not None
                    else None
                ),
                error_code=_DEPENDENCY_ERROR,
            )
        if not self._safe_close(runtime):
            return TargetReport(
                target=self.target,
                status="failed",
                fingerprint=result.fingerprint,
                changes=result.changes,
                issues=result.issues,
                error_code=_DEPENDENCY_ERROR,
            )
        return result

    def close(self) -> None:
        return None


def _milliseconds(seconds: float) -> int:
    return max(1, math.ceil(seconds * 1_000))


def _close_mongo_client(client: MongoClient[dict[str, Any]]) -> None:
    try:
        client.close()
    except Exception:
        return


def _build_mongo_store(settings: Settings) -> MongoImportStore:
    client: MongoClient[dict[str, Any]] | None = None
    try:
        settings.validate_capabilities("mongo")
        mongo = settings.mongo
        if mongo.url is None:
            raise ValueError("Mongo URL is required")
        client = MongoClient(
            mongo.url.get_secret_value(),
            appname="knowledge-agent-schema-admin",
            connectTimeoutMS=_milliseconds(mongo.connect_timeout_seconds),
            serverSelectionTimeoutMS=_milliseconds(
                mongo.server_selection_timeout_seconds
            ),
            socketTimeoutMS=_milliseconds(mongo.socket_timeout_seconds),
            waitQueueTimeoutMS=_milliseconds(
                mongo.wait_queue_timeout_seconds
            ),
            maxPoolSize=mongo.max_pool_size,
            retryReads=True,
            retryWrites=True,
            tz_aware=True,
            uuidRepresentation="standard",
        )
        store = MongoImportStore(client, mongo)
        store.require_ready()
        return store
    except Exception:
        if client is not None:
            _close_mongo_client(client)
        raise RuntimeError("Mongo schema dependency is unavailable") from None


def _build_milvus_runtime(
    settings: Settings,
    *,
    allow_create: bool,
) -> MilvusRuntimeAdapter:
    settings.validate_capabilities("milvus")
    return build_milvus_runtime_from_settings(
        settings,
        dense_dimension=settings.embedding.dense_dimension,
        allow_create=allow_create,
        token=settings.milvus.token,
        user=settings.milvus.username,
        password=settings.milvus.password,
        database=settings.milvus.database,
        timeout_seconds=settings.milvus.timeout_seconds,
        close_wait_seconds=settings.import_runtime.shutdown_grace_seconds,
        max_query_limit=max(
            settings.milvus.max_chunks_per_import + 1,
            settings.search.max_candidates,
        ),
    )


def build_schema_operator(settings: Settings, target: str) -> SchemaOperator:
    if target == "mongo":
        return MongoSchemaOperator(
            _build_mongo_store(settings),
            settings.mongo,
        )
    if target == "milvus":
        return MilvusSchemaOperator(
            lambda allow_create: _build_milvus_runtime(
                settings,
                allow_create=allow_create,
            )
        )
    raise ValueError("invalid schema target")


def _environment_name(settings: Settings) -> str:
    environment = settings.api.environment
    if isinstance(environment, AppEnvironment):
        return environment.value
    value = str(environment).lower()
    return value if value in {item.value for item in AppEnvironment} else "unknown"


def _failed_target(target: str) -> TargetReport:
    return TargetReport(
        target=target,
        status="failed",
        fingerprint=None,
        error_code=_DEPENDENCY_ERROR,
    )


def _aggregate(
    *,
    mode: str,
    environment: str,
    reports: Sequence[TargetReport],
) -> SchemaAdminReport:
    targets = tuple(reports)
    if any(report.status == "failed" for report in targets):
        return SchemaAdminReport(
            mode=mode,
            environment=environment,
            status="failed",
            exit_code=EXIT_DEPENDENCY_FAILURE,
            targets=targets,
            error_code=_DEPENDENCY_ERROR,
        )
    if any(report.status == "incompatible" for report in targets):
        return SchemaAdminReport(
            mode=mode,
            environment=environment,
            status="failed",
            exit_code=EXIT_INCOMPATIBLE,
            targets=targets,
            error_code=_INCOMPATIBLE_ERROR,
        )
    if mode == "dry-run" and any(
        report.status == "changes_required" for report in targets
    ):
        return SchemaAdminReport(
            mode=mode,
            environment=environment,
            status="changes_required",
            exit_code=EXIT_CHANGES_REQUIRED,
            targets=targets,
            error_code=_CHANGES_ERROR,
        )
    if any(report.status == "applied" for report in targets):
        status = "applied"
    else:
        status = "ready"
    return SchemaAdminReport(
        mode=mode,
        environment=environment,
        status=status,
        exit_code=EXIT_READY,
        targets=targets,
    )


def execute_schema_admin(
    settings: Settings,
    *,
    targets: Sequence[str],
    apply: bool,
    confirm_production: bool,
    operator_factory: SchemaOperatorFactory = build_schema_operator,
) -> SchemaAdminReport:
    mode = "apply" if apply else "dry-run"
    if not isinstance(settings, Settings):
        return SchemaAdminReport(
            mode=mode,
            environment="unknown",
            status="failed",
            exit_code=EXIT_USAGE,
            error_code=_CONFIGURATION_ERROR,
        )
    environment = _environment_name(settings)
    if (
        apply
        and environment == AppEnvironment.PRODUCTION.value
        and not confirm_production
    ):
        return SchemaAdminReport(
            mode=mode,
            environment=environment,
            status="failed",
            exit_code=EXIT_PRODUCTION_CONFIRMATION_REQUIRED,
            error_code=_PRODUCTION_CONFIRMATION_ERROR,
        )
    resolved_targets = tuple(targets)
    if (
        not resolved_targets
        or len(resolved_targets) != len(set(resolved_targets))
        or any(target not in {"mongo", "milvus"} for target in resolved_targets)
    ):
        return SchemaAdminReport(
            mode=mode,
            environment=environment,
            status="failed",
            exit_code=EXIT_USAGE,
            error_code=_CONFIGURATION_ERROR,
        )

    operators: dict[str, SchemaOperator] = {}
    preflight: list[TargetReport] = []
    result: SchemaAdminReport
    close_failures: set[str] = set()
    try:
        for target in resolved_targets:
            try:
                operator = operator_factory(settings, target)
                operators[target] = operator
                preflight.append(operator.inspect())
            except Exception:
                preflight.append(_failed_target(target))

        preflight_report = _aggregate(
            mode="dry-run",
            environment=environment,
            reports=preflight,
        )
        if not apply:
            result = preflight_report
        elif preflight_report.exit_code not in {
            EXIT_READY,
            EXIT_CHANGES_REQUIRED,
        }:
            result = SchemaAdminReport(
                mode=mode,
                environment=environment,
                status=preflight_report.status,
                exit_code=preflight_report.exit_code,
                targets=preflight_report.targets,
                error_code=preflight_report.error_code,
            )
        else:
            applied: list[TargetReport] = []
            for planned in preflight:
                planned_operator = operators.get(planned.target)
                if planned_operator is None:
                    applied.append(_failed_target(planned.target))
                elif planned.status == "changes_required":
                    applied.append(planned_operator.apply())
                else:
                    applied.append(planned)
            result = _aggregate(
                mode=mode,
                environment=environment,
                reports=applied,
            )
    except Exception:
        result = SchemaAdminReport(
            mode=mode,
            environment=environment,
            status="failed",
            exit_code=EXIT_DEPENDENCY_FAILURE,
            targets=tuple(preflight),
            error_code=_DEPENDENCY_ERROR,
        )
    finally:
        for target, operator in operators.items():
            try:
                operator.close()
            except Exception:
                close_failures.add(target)
    if close_failures:
        by_target = {
            report.target: report
            for report in result.targets
        }
        closed_reports: list[TargetReport] = []
        for target in resolved_targets:
            current = by_target.get(target)
            if target in close_failures:
                closed_reports.append(
                    TargetReport(
                        target=target,
                        status="failed",
                        fingerprint=(
                            current.fingerprint
                            if current is not None
                            else None
                        ),
                        changes=(
                            current.changes
                            if current is not None
                            else ()
                        ),
                        issues=(
                            current.issues
                            if current is not None
                            else ()
                        ),
                        error_code=_DEPENDENCY_ERROR,
                    )
                )
            elif current is not None:
                closed_reports.append(current)
        return _aggregate(
            mode=mode,
            environment=environment,
            reports=closed_reports,
        )
    return result


class _ArgumentError(ValueError):
    pass


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _ArgumentError("invalid schema admin arguments") from None


@dataclass(frozen=True, slots=True)
class _Arguments:
    target: str
    apply: bool
    confirm_production: bool
    help: bool


def _parse_arguments(argv: Sequence[str]) -> _Arguments:
    parser = _JsonArgumentParser(add_help=False, prog="schema-admin")
    parser.add_argument(
        "--target",
        choices=("all", "mongo", "milvus"),
        default="all",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-production", action="store_true")
    parser.add_argument("--help", action="store_true")
    parsed = parser.parse_args(list(argv))
    if parsed.confirm_production and not parsed.apply:
        raise _ArgumentError("production confirmation requires apply")
    return _Arguments(
        target=cast(str, parsed.target),
        apply=cast(bool, parsed.apply),
        confirm_production=cast(bool, parsed.confirm_production),
        help=cast(bool, parsed.help),
    )


def _emit(payload: Mapping[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    settings_loader: SettingsLoader = get_settings,
    operator_factory: SchemaOperatorFactory = build_schema_operator,
) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        parsed = _parse_arguments(arguments)
    except Exception:
        report = SchemaAdminReport(
            mode="dry-run",
            environment="unknown",
            status="failed",
            exit_code=EXIT_USAGE,
            error_code=_CONFIGURATION_ERROR,
        )
        _emit(report.to_dict())
        return report.exit_code
    if parsed.help:
        payload: JsonObject = {
            "schema_version": _OUTPUT_SCHEMA_VERSION,
            "status": "help",
            "exit_code": EXIT_READY,
            "usage": _HELP_USAGE,
        }
        _emit(payload)
        return EXIT_READY
    try:
        settings = settings_loader()
    except Exception:
        report = SchemaAdminReport(
            mode="apply" if parsed.apply else "dry-run",
            environment="unknown",
            status="failed",
            exit_code=EXIT_USAGE,
            error_code=_CONFIGURATION_ERROR,
        )
        _emit(report.to_dict())
        return report.exit_code
    targets = (
        ("mongo", "milvus")
        if parsed.target == "all"
        else (parsed.target,)
    )
    report = execute_schema_admin(
        settings,
        targets=targets,
        apply=parsed.apply,
        confirm_production=parsed.confirm_production,
        operator_factory=operator_factory,
    )
    _emit(report.to_dict())
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_CHANGES_REQUIRED",
    "EXIT_DEPENDENCY_FAILURE",
    "EXIT_INCOMPATIBLE",
    "EXIT_PRODUCTION_CONFIRMATION_REQUIRED",
    "EXIT_READY",
    "EXIT_USAGE",
    "MilvusSchemaOperator",
    "MongoSchemaOperator",
    "SchemaAdminReport",
    "SchemaChange",
    "TargetReport",
    "build_schema_operator",
    "execute_schema_admin",
    "main",
]
