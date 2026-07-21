from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

import pytest
from pymongo import IndexModel

from app.core.settings import (
    ApiSettings,
    AppEnvironment,
    MongoSettings,
    Settings,
)
from app.infrastructure.milvus_runtime import (
    MilvusSchemaIncompatible,
    MilvusSchemaMissing,
    MilvusSchemaReport,
)
from app.ops import schema_admin as ops


def app_settings(
    environment: AppEnvironment = AppEnvironment.DEVELOPMENT,
) -> Settings:
    return Settings.model_construct(
        api=ApiSettings.model_construct(environment=environment),
    )


def mongo_settings() -> MongoSettings:
    return MongoSettings.model_construct(
        import_admissions_collection="import_admissions_v1",
        import_tasks_collection="import_tasks_v1",
        import_outbox_collection="import_outbox_v1",
        chat_sessions_collection="chat_sessions_v1",
        chat_messages_collection="chat_messages_v1",
        health_timeout_seconds=2.0,
    )


def _index_information(index: IndexModel) -> dict[str, object]:
    document = cast(Mapping[str, object], index.document)
    keys = cast(Mapping[str, object], document["key"])
    information: dict[str, object] = {"key": list(keys.items())}
    for option in (
        "unique",
        "sparse",
        "expireAfterSeconds",
        "partialFilterExpression",
    ):
        if option in document:
            information[option] = document[option]
    return information


class FakeCollection:
    def __init__(self, options: Mapping[str, object] | None = None) -> None:
        self.options_value = dict(options or {})
        self.indexes: dict[str, dict[str, object]] = {
            "_id_": {"key": [("_id", 1)]}
        }
        self.incompatible_document = False

    def index_information(self) -> Mapping[str, Mapping[str, object]]:
        return self.indexes

    def options(self) -> Mapping[str, object]:
        return dict(self.options_value)

    def find_one(
        self,
        filter: Mapping[str, object],
        projection: Mapping[str, int] | None = None,
        **kwargs: object,
    ) -> Mapping[str, object] | None:
        return {"_id": "incompatible"} if self.incompatible_document else None

    def create_indexes(self, indexes: list[IndexModel]) -> list[str]:
        names: list[str] = []
        for index in indexes:
            document = cast(Mapping[str, object], index.document)
            name = cast(str, document["name"])
            self.indexes[name] = _index_information(index)
            names.append(name)
        return names


class FakeDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.commands: list[Mapping[str, object]] = []
        self.failure: Exception | None = None

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection())

    def list_collection_names(self) -> list[str]:
        if self.failure is not None:
            raise self.failure
        return list(self.collections)

    def create_collection(
        self,
        name: str,
        **kwargs: object,
    ) -> FakeCollection:
        collection = FakeCollection(kwargs)
        self.collections[name] = collection
        return collection

    def command(
        self,
        command: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.commands.append(command)
        name = cast(str, command["collMod"])
        self.collections[name].options_value = {
            key: value
            for key, value in command.items()
            if key != "collMod"
        }
        return {"ok": 1}


class FakeStore:
    def __init__(self, settings: MongoSettings) -> None:
        self.settings = settings
        self.database = FakeDatabase()
        self.require_calls = 0
        self.ensure_calls = 0
        self.close_calls = 0

    def require_ready(self) -> None:
        self.require_calls += 1

    def ensure_indexes(self) -> None:
        self.ensure_calls += 1
        for contract in ops._mongo_collection_contracts(self.settings)[:3]:
            self.database[contract.name].create_indexes(
                list(contract.indexes)
            )

    def close(self) -> None:
        self.close_calls += 1


def install_ready_mongo_schema(
    store: FakeStore,
    settings: MongoSettings,
) -> None:
    for contract in ops._mongo_collection_contracts(settings):
        options: dict[str, object] = {}
        if contract.validator is not None:
            options = {
                "validator": contract.validator,
                "validationLevel": "strict",
                "validationAction": "error",
            }
        collection = FakeCollection(options)
        collection.create_indexes(list(contract.indexes))
        store.database.collections[contract.name] = collection


def test_mongo_dry_run_reports_changes_without_mutation() -> None:
    settings = mongo_settings()
    store = FakeStore(settings)
    operator = ops.MongoSchemaOperator(store, settings)

    report = operator.inspect()

    assert report.status == "changes_required"
    assert report.fingerprint is not None
    assert len(report.fingerprint) == 64
    assert store.ensure_calls == 0
    assert store.database.collections == {}
    assert {change.action for change in report.changes} == {
        "create_collection",
        "create_index",
    }


def test_mongo_apply_installs_and_reverifies_indexes_and_validators() -> None:
    settings = mongo_settings()
    store = FakeStore(settings)
    operator = ops.MongoSchemaOperator(store, settings)

    report = operator.apply()

    assert report.status == "applied"
    assert store.ensure_calls == 1
    assert operator.inspect().status == "ready"
    for name in (
        settings.chat_sessions_collection,
        settings.chat_messages_collection,
    ):
        options = store.database.collections[name].options_value
        assert options["validationLevel"] == "strict"
        assert options["validationAction"] == "error"


@pytest.mark.parametrize(
    "drift",
    ["wrong_index", "extra_index", "document", "unexpected_validator"],
)
def test_mongo_incompatible_state_requires_manual_migration_without_writes(
    drift: str,
) -> None:
    settings = mongo_settings()
    store = FakeStore(settings)
    install_ready_mongo_schema(store, settings)
    collection = store.database.collections[
        settings.import_admissions_collection
    ]
    if drift == "wrong_index":
        collection.indexes["uq_import_admission_idempotency"]["key"] = [
            ("tenant_id", 1)
        ]
    elif drift == "extra_index":
        collection.indexes["legacy_unique"] = {
            "key": [("legacy", 1)],
            "unique": True,
        }
    elif drift == "unexpected_validator":
        collection.options_value["validator"] = {
            "$jsonSchema": {"bsonType": "object"}
        }
    else:
        collection.incompatible_document = True
    operator = ops.MongoSchemaOperator(store, settings)

    inspected = operator.inspect()
    applied = operator.apply()

    assert inspected.status == "incompatible"
    assert inspected.error_code == "SCHEMA_INCOMPATIBLE"
    assert inspected.issues
    assert applied.status == "incompatible"
    assert store.ensure_calls == 0


def test_mongo_validator_drift_is_a_safe_upgrade() -> None:
    settings = mongo_settings()
    store = FakeStore(settings)
    install_ready_mongo_schema(store, settings)
    store.database.collections[
        settings.chat_sessions_collection
    ].options_value["validationLevel"] = "moderate"
    operator = ops.MongoSchemaOperator(store, settings)

    planned = operator.inspect()
    applied = operator.apply()

    assert planned.status == "changes_required"
    assert any(change.action == "update_validator" for change in planned.changes)
    assert applied.status == "applied"
    assert operator.inspect().status == "ready"


def test_mongo_inspection_failure_is_structured_and_contains_no_exception_text() -> None:
    leaked_detail = "mongodb://admin:credential@private-host"  # pragma: allowlist secret
    settings = mongo_settings()
    store = FakeStore(settings)
    store.database.failure = RuntimeError(leaked_detail)
    operator = ops.MongoSchemaOperator(store, settings)

    report = operator.inspect()
    rendered = json.dumps(report.to_dict(), sort_keys=True)

    assert report.status == "failed"
    assert report.error_code == "SCHEMA_DEPENDENCY_UNAVAILABLE"
    assert leaked_detail not in rendered


class FakeMilvusState:
    def __init__(self) -> None:
        self.ready = False
        self.incompatible = False
        self.failure: Exception | None = None
        self.close_failure: Exception | None = None
        self.allow_create_calls: list[bool] = []
        self.close_calls = 0


class FakeMilvusRuntime:
    collection_name = "kb_chunks_v2"
    schema_fingerprint = "f" * 64

    def __init__(self, state: FakeMilvusState, allow_create: bool) -> None:
        self.state = state
        self.allow_create = allow_create

    def ensure_schema(self) -> MilvusSchemaReport:
        if self.state.failure is not None:
            raise self.state.failure
        if self.state.incompatible:
            raise MilvusSchemaIncompatible("private incompatible detail")
        if not self.state.ready and not self.allow_create:
            raise MilvusSchemaMissing("private missing detail")
        created = not self.state.ready
        self.state.ready = True
        return MilvusSchemaReport(
            collection_name=self.collection_name,
            schema_fingerprint=self.schema_fingerprint,
            collection_created=created,
            indexes_created=("dense_vector_idx",) if created else (),
        )

    def close(self) -> None:
        self.state.close_calls += 1
        if self.state.close_failure is not None:
            raise self.state.close_failure


def milvus_operator(state: FakeMilvusState) -> ops.MilvusSchemaOperator:
    def factory(allow_create: bool) -> FakeMilvusRuntime:
        state.allow_create_calls.append(allow_create)
        return FakeMilvusRuntime(state, allow_create)

    return ops.MilvusSchemaOperator(factory)


def test_milvus_dry_run_never_authorizes_creation_and_apply_does() -> None:
    state = FakeMilvusState()
    operator = milvus_operator(state)

    planned = operator.inspect()
    applied = operator.apply()

    assert planned.status == "changes_required"
    assert planned.changes[0].action == "ensure_enterprise_schema"
    assert applied.status == "applied"
    assert {change.action for change in applied.changes} == {
        "create_collection",
        "create_index",
    }
    assert state.allow_create_calls == [False, True]
    assert state.close_calls == 2


def test_milvus_incompatible_and_dependency_failures_are_sanitized() -> None:
    incompatible_state = FakeMilvusState()
    incompatible_state.incompatible = True
    incompatible = milvus_operator(incompatible_state).inspect()
    assert incompatible.status == "incompatible"
    assert "private incompatible detail" not in json.dumps(
        incompatible.to_dict()
    )

    leaked_detail = "https://milvus.internal/?token=must-not-leak"
    failed_state = FakeMilvusState()
    failed_state.failure = RuntimeError(leaked_detail)
    failed = milvus_operator(failed_state).inspect()
    assert failed.status == "failed"
    assert leaked_detail not in json.dumps(failed.to_dict())


def test_milvus_close_failure_is_a_failed_operation() -> None:
    state = FakeMilvusState()
    state.ready = True
    state.close_failure = RuntimeError("private close detail")

    report = milvus_operator(state).inspect()

    assert report.status == "failed"
    assert report.error_code == "SCHEMA_DEPENDENCY_UNAVAILABLE"


class FakeOperator:
    def __init__(
        self,
        target: str,
        inspected: ops.TargetReport,
        applied: ops.TargetReport | None = None,
        *,
        close_failure: Exception | None = None,
    ) -> None:
        self.target = target
        self.inspected = inspected
        self.applied = applied or inspected
        self.close_failure = close_failure
        self.inspect_calls = 0
        self.apply_calls = 0
        self.close_calls = 0

    def inspect(self) -> ops.TargetReport:
        self.inspect_calls += 1
        return self.inspected

    def apply(self) -> ops.TargetReport:
        self.apply_calls += 1
        return self.applied

    def close(self) -> None:
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure


def target_report(
    target: str,
    status: str,
    *,
    error_code: str | None = None,
) -> ops.TargetReport:
    changes = (
        (ops.SchemaChange("create_index", f"{target}.schema"),)
        if status == "changes_required"
        else ()
    )
    return ops.TargetReport(
        target=target,
        status=status,
        fingerprint="a" * 64,
        changes=changes,
        error_code=error_code,
    )


def test_execute_is_two_phase_and_blocks_all_apply_on_incompatibility() -> None:
    mongo = FakeOperator(
        "mongo",
        target_report("mongo", "changes_required"),
        target_report("mongo", "applied"),
    )
    milvus = FakeOperator(
        "milvus",
        target_report(
            "milvus",
            "incompatible",
            error_code="SCHEMA_INCOMPATIBLE",
        ),
    )
    operators = {"mongo": mongo, "milvus": milvus}

    report = ops.execute_schema_admin(
        app_settings(),
        targets=("mongo", "milvus"),
        apply=True,
        confirm_production=False,
        operator_factory=lambda settings, target: operators[target],
    )

    assert report.exit_code == ops.EXIT_INCOMPATIBLE
    assert mongo.apply_calls == milvus.apply_calls == 0
    assert mongo.close_calls == milvus.close_calls == 1


def test_production_apply_requires_explicit_confirmation_before_building() -> None:
    builds: list[str] = []

    def forbidden_factory(
        settings: Settings,
        target: str,
    ) -> ops.SchemaOperator:
        builds.append(target)
        raise AssertionError("operator must not be built")

    report = ops.execute_schema_admin(
        app_settings(AppEnvironment.PRODUCTION),
        targets=("mongo",),
        apply=True,
        confirm_production=False,
        operator_factory=forbidden_factory,
    )

    assert report.exit_code == ops.EXIT_PRODUCTION_CONFIRMATION_REQUIRED
    assert report.error_code == "PRODUCTION_APPLY_CONFIRMATION_REQUIRED"
    assert builds == []


def test_apply_executes_only_targets_with_planned_changes() -> None:
    mongo = FakeOperator("mongo", target_report("mongo", "ready"))
    milvus = FakeOperator(
        "milvus",
        target_report("milvus", "changes_required"),
        target_report("milvus", "applied"),
    )
    operators = {"mongo": mongo, "milvus": milvus}

    report = ops.execute_schema_admin(
        app_settings(),
        targets=("mongo", "milvus"),
        apply=True,
        confirm_production=False,
        operator_factory=lambda settings, target: operators[target],
    )

    assert report.exit_code == ops.EXIT_READY
    assert report.status == "applied"
    assert mongo.apply_calls == 0
    assert milvus.apply_calls == 1


def test_cli_defaults_to_dry_run_and_emits_one_structured_json_document(
    capsys: pytest.CaptureFixture[str],
) -> None:
    operator = FakeOperator(
        "mongo",
        target_report("mongo", "changes_required"),
    )

    exit_code = ops.main(
        ("--target", "mongo"),
        settings_loader=app_settings,
        operator_factory=lambda settings, target: operator,
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == ops.EXIT_CHANGES_REQUIRED
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert payload["mode"] == "dry-run"
    assert payload["status"] == "changes_required"
    assert payload["targets"][0]["target"] == "mongo"
    assert operator.apply_calls == 0


def test_cli_failure_output_never_contains_factory_exception_details(
    capsys: pytest.CaptureFixture[str],
) -> None:
    leaked_detail = "mongodb://root:secret@private-host"  # pragma: allowlist secret

    def failing_factory(settings: Settings, target: str) -> ops.SchemaOperator:
        raise RuntimeError(leaked_detail)

    exit_code = ops.main(
        ("--target", "mongo"),
        settings_loader=app_settings,
        operator_factory=failing_factory,
    )
    captured = capsys.readouterr()

    assert exit_code == ops.EXIT_DEPENDENCY_FAILURE
    assert leaked_detail not in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ("--unknown",),
        ("--target", "invalid"),
        ("--confirm-production",),
    ],
)
def test_cli_argument_errors_are_json_and_nonzero(
    arguments: Sequence[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = ops.main(arguments, settings_loader=app_settings)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == ops.EXIT_USAGE
    assert payload["error_code"] == "SCHEMA_CONFIGURATION_INVALID"
    assert captured.err == ""


def test_cli_help_is_structured_and_does_not_load_settings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def forbidden_settings() -> Settings:
        nonlocal calls
        calls += 1
        raise AssertionError

    exit_code = ops.main(("--help",), settings_loader=forbidden_settings)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == ops.EXIT_READY
    assert payload["status"] == "help"
    assert "usage" in payload
    assert calls == 0


def test_operator_close_failure_makes_the_run_fail() -> None:
    operator = FakeOperator(
        "mongo",
        target_report("mongo", "ready"),
        close_failure=RuntimeError("private close failure"),
    )

    report = ops.execute_schema_admin(
        app_settings(),
        targets=("mongo",),
        apply=False,
        confirm_production=False,
        operator_factory=lambda settings, target: operator,
    )

    assert report.exit_code == ops.EXIT_DEPENDENCY_FAILURE
    assert report.status == "failed"
