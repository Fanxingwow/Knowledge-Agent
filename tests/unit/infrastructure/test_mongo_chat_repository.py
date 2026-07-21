from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar, cast

import pytest
from pymongo import IndexModel
from pymongo.errors import DuplicateKeyError

from app.core.errors import AppError, DependencyError, ErrorCode
from app.infrastructure.mongo_chat_repository import (
    MongoChatRepository,
    MongoChatSchemaManager,
)
from app.services.chat_history import (
    ChatMessageRole,
    ChatOwnerScope,
    ChatSessionScope,
    NewChatMessage,
    NewChatSession,
)

_T = TypeVar("_T")
_SESSION_ID = "00000000-0000-4000-8000-000000000001"
_MESSAGE_1 = "10000000-0000-4000-8000-000000000001"
_MESSAGE_2 = "10000000-0000-4000-8000-000000000002"
_MESSAGE_3 = "10000000-0000-4000-8000-000000000003"


@dataclass
class FakeResult:
    matched_count: int = 0


def _matches(document: Mapping[str, object], query: Mapping[str, object]) -> bool:
    for key, expected in query.items():
        if key == "$or":
            alternatives = cast(Sequence[Mapping[str, object]], expected)
            if not any(_matches(document, alternative) for alternative in alternatives):
                return False
            continue
        actual = document.get(key)
        if isinstance(expected, Mapping):
            if "$gt" in expected:
                if actual is None or not actual > expected["$gt"]:  # type: ignore[operator]
                    return False
                continue
            if "$lt" in expected:
                if actual is None or not actual < expected["$lt"]:  # type: ignore[operator]
                    return False
                continue
        if actual != expected:
            return False
    return True


def _project(
    document: Mapping[str, object],
    projection: Mapping[str, int] | None,
) -> dict[str, object]:
    if projection is None:
        return dict(document)
    return {key: value for key, value in document.items() if projection.get(key) == 1}


class FakeCursor:
    def __init__(self, documents: Sequence[Mapping[str, object]]) -> None:
        self.documents = [dict(document) for document in documents]
        self.sort_spec: list[tuple[str, int]] = []
        self.limit_value: int | None = None

    def sort(self, key_or_list: Sequence[tuple[str, int]]) -> FakeCursor:
        self.sort_spec = list(key_or_list)
        for key, direction in reversed(self.sort_spec):
            self.documents.sort(key=lambda document: document[key], reverse=direction < 0)
        return self

    def limit(self, limit: int) -> FakeCursor:
        self.limit_value = limit
        self.documents = self.documents[:limit]
        return self

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        return iter(self.documents)


class FakeCollection:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, object]] = {}
        self.calls: list[tuple[str, Mapping[str, object]]] = []
        self.indexes: list[IndexModel] = []
        self.failure: Exception | None = None
        self.last_cursor: FakeCursor | None = None

    def _fail(self) -> None:
        if self.failure is not None:
            raise self.failure

    def create_indexes(self, indexes: list[IndexModel]) -> list[str]:
        self._fail()
        self.indexes.extend(indexes)
        return [str(index.document["name"]) for index in indexes]

    def find_one(
        self,
        filter: Mapping[str, object],
        projection: Mapping[str, int] | None = None,
        *,
        session: object | None = None,
    ) -> Mapping[str, object] | None:
        self._fail()
        self.calls.append(("find_one", dict(filter)))
        for document in self.documents.values():
            if _matches(document, filter):
                return _project(document, projection)
        return None

    def find(
        self,
        filter: Mapping[str, object],
        projection: Mapping[str, int] | None = None,
    ) -> FakeCursor:
        self._fail()
        self.calls.append(("find", dict(filter)))
        selected = [
            _project(document, projection) for document in self.documents.values() if _matches(document, filter)
        ]
        self.last_cursor = FakeCursor(selected)
        return self.last_cursor

    def insert_one(
        self,
        document: Mapping[str, object],
        *,
        session: object | None = None,
    ) -> object:
        self._fail()
        self.calls.append(("insert_one", dict(document)))
        identifier = cast(str, document["_id"])
        if identifier in self.documents:
            raise DuplicateKeyError("duplicate private detail")
        unique_fields = ("tenant_id", "owner_subject", "session_id", "message_id")
        for existing in self.documents.values():
            comparable = [field for field in unique_fields if field in document]
            if comparable and all(existing.get(field) == document.get(field) for field in comparable):
                raise DuplicateKeyError("duplicate private detail")
        self.documents[identifier] = dict(document)
        return object()

    def find_one_and_update(
        self,
        filter: Mapping[str, object],
        update: Mapping[str, object],
        *,
        projection: Mapping[str, int],
        return_document: bool,
        session: object | None = None,
    ) -> Mapping[str, object] | None:
        self._fail()
        self.calls.append(("find_one_and_update", dict(filter)))
        for document in self.documents.values():
            if _matches(document, filter):
                document.update(cast(Mapping[str, object], update["$set"]))
                return _project(document, projection)
        return None

    def update_one(
        self,
        filter: Mapping[str, object],
        update: Mapping[str, object],
        *,
        session: object | None = None,
    ) -> FakeResult:
        self._fail()
        self.calls.append(("update_one", dict(filter)))
        for document in self.documents.values():
            if _matches(document, filter):
                document.update(cast(Mapping[str, object], update["$set"]))
                return FakeResult(matched_count=1)
        return FakeResult()

    def delete_one(
        self,
        filter: Mapping[str, object],
        *,
        session: object | None = None,
    ) -> object:
        self._fail()
        self.calls.append(("delete_one", dict(filter)))
        for identifier, document in tuple(self.documents.items()):
            if _matches(document, filter):
                del self.documents[identifier]
                break
        return object()

    def delete_many(
        self,
        filter: Mapping[str, object],
        *,
        session: object | None = None,
    ) -> object:
        self._fail()
        self.calls.append(("delete_many", dict(filter)))
        for identifier, document in tuple(self.documents.items()):
            if _matches(document, filter):
                del self.documents[identifier]
        return object()


class FakeDatabase:
    def __init__(self, existing: Sequence[str] = ()) -> None:
        self.collections = {name: FakeCollection() for name in existing}
        self.create_calls: list[tuple[str, Mapping[str, object]]] = []
        self.commands: list[Mapping[str, object]] = []
        self.failure: Exception | None = None

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection())

    def list_collection_names(self) -> list[str]:
        if self.failure is not None:
            raise self.failure
        return list(self.collections)

    def create_collection(self, name: str, **kwargs: object) -> FakeCollection:
        self.create_calls.append((name, kwargs))
        collection = FakeCollection()
        self.collections[name] = collection
        return collection

    def command(self, command: Mapping[str, object]) -> Mapping[str, object]:
        self.commands.append(dict(command))
        return {"ok": 1}


class ImmediateTransactions:
    def __init__(self) -> None:
        self.calls = 0
        self.failure: Exception | None = None

    def run(self, operation: Callable[[object], _T]) -> _T:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return operation(object())


class MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self) -> None:
        self.now += timedelta(microseconds=1)


def scope(
    tenant_id: str = "tenant-a",
    owner_subject: str = "user:42",
    session_id: str = _SESSION_ID,
) -> ChatSessionScope:
    return ChatSessionScope(tenant_id, owner_subject, session_id)


def repository() -> tuple[MongoChatRepository, FakeCollection, FakeCollection, ImmediateTransactions, MutableClock]:
    sessions = FakeCollection()
    messages = FakeCollection()
    transactions = ImmediateTransactions()
    clock = MutableClock()
    repo = MongoChatRepository(
        cast(Any, sessions),
        cast(Any, messages),
        transactions,
        clock=clock,
    )
    return repo, sessions, messages, transactions, clock


def new_message(message_id: str = _MESSAGE_1, content: str = "hello") -> NewChatMessage:
    return NewChatMessage(
        message_id=message_id,
        role=ChatMessageRole.USER,
        content=content,
        attributes=(("trace.id", "trace-1"),),
    )


def assert_full_scope(query: Mapping[str, object], expected: ChatSessionScope) -> None:
    assert query["tenant_id"] == expected.tenant_id
    assert query["owner_subject"] == expected.owner_subject
    assert query["session_id"] == expected.session_id


def test_schema_manager_installs_strict_validators_and_required_indexes() -> None:
    database = FakeDatabase()
    manager = MongoChatSchemaManager(cast(Any, database))

    assert database.create_calls == []
    manager.ensure_schema()

    assert [name for name, _options in database.create_calls] == [
        "chat_sessions_v1",
        "chat_messages_v1",
    ]
    for _name, options in database.create_calls:
        assert options["validationLevel"] == "strict"
        assert options["validationAction"] == "error"
        json_schema = cast(Mapping[str, object], options["validator"])["$jsonSchema"]
        assert cast(Mapping[str, object], json_schema)["additionalProperties"] is False

    session_indexes = {str(index.document["name"]): index.document for index in database["chat_sessions_v1"].indexes}
    message_indexes = {str(index.document["name"]): index.document for index in database["chat_messages_v1"].indexes}
    assert session_indexes["uq_chat_session_scope"]["unique"] is True
    assert message_indexes["uq_chat_message_idempotency"]["unique"] is True
    assert list(cast(Mapping[str, int], message_indexes["ix_chat_message_page"]["key"]).items()) == [
        ("tenant_id", 1),
        ("owner_subject", 1),
        ("session_id", 1),
        ("created_at", 1),
        ("message_id", 1),
    ]


def test_existing_collections_are_upgraded_with_collmod() -> None:
    database = FakeDatabase(("chat_sessions_v1", "chat_messages_v1"))
    manager = MongoChatSchemaManager(cast(Any, database))

    manager.ensure_schema()

    assert database.create_calls == []
    assert [command["collMod"] for command in database.commands] == [
        "chat_sessions_v1",
        "chat_messages_v1",
    ]
    assert all(command["validationLevel"] == "strict" for command in database.commands)


def test_repository_builder_installs_schema_before_exposing_repository() -> None:
    database = FakeDatabase()
    manager = MongoChatSchemaManager(cast(Any, database))
    transactions = ImmediateTransactions()

    repository = manager.build_repository(transactions)

    assert repository.durable is True
    assert [name for name, _options in database.create_calls] == [
        "chat_sessions_v1",
        "chat_messages_v1",
    ]


def test_schema_failure_is_sanitized() -> None:
    database = FakeDatabase()
    database.failure = RuntimeError("mongodb://root:secret@private-host")
    manager = MongoChatSchemaManager(cast(Any, database))

    with pytest.raises(DependencyError) as raised:
        manager.ensure_schema()

    assert str(raised.value) == "Chat history store is unavailable"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_session_crud_always_uses_complete_scope_and_cross_tenant_is_not_found() -> None:
    repo, sessions, _messages, _transactions, clock = repository()
    owned = scope()

    created = repo.create_session(NewChatSession(owned, "Support"))
    assert created.scope == owned
    assert created.title == "Support"
    assert repo.create_session(NewChatSession(owned, "Support")) == created

    clock.advance()
    updated = repo.update_session_title(owned, "Returns")
    assert updated.title == "Returns"
    assert updated.updated_at > updated.created_at
    assert repo.get_session(owned) == updated

    foreign = scope(tenant_id="tenant-b")
    with pytest.raises(AppError) as raised:
        repo.get_session(foreign)
    assert raised.value.code is ErrorCode.RESOURCE_NOT_FOUND
    assert raised.value.message == "Chat resource not found"

    for operation, query in sessions.calls:
        if operation != "insert_one":
            assert {"tenant_id", "owner_subject", "session_id"}.issubset(query)


def test_same_session_identity_with_different_create_payload_conflicts() -> None:
    repo, _sessions, _messages, _transactions, _clock = repository()
    repo.create_session(NewChatSession(scope(), "Title one"))

    with pytest.raises(AppError) as raised:
        repo.create_session(NewChatSession(scope(), "Title two"))

    assert raised.value.code is ErrorCode.RESOURCE_CONFLICT


def test_session_cursor_lists_only_the_exact_owner_scope() -> None:
    repo, sessions, _messages, _transactions, clock = repository()
    owner = ChatOwnerScope("tenant-a", "user:42")
    identifiers = (
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
        "00000000-0000-4000-8000-000000000003",
    )
    for session_id in identifiers:
        repo.create_session(NewChatSession(scope(session_id=session_id), session_id))
        clock.advance()
    repo.create_session(NewChatSession(scope(tenant_id="tenant-b"), "foreign"))

    first = repo.list_sessions(owner, limit=2)
    second = repo.list_sessions(owner, limit=2, cursor=first.next_cursor)

    assert [item.scope.session_id for item in first.items] == [identifiers[2], identifiers[1]]
    assert first.next_cursor is not None
    assert [item.scope.session_id for item in second.items] == [identifiers[0]]
    assert second.next_cursor is None
    find_queries = [query for operation, query in sessions.calls if operation == "find"]
    assert find_queries[0]["tenant_id"] == owner.tenant_id
    assert find_queries[0]["owner_subject"] == owner.owner_subject
    assert "session_id" not in find_queries[0]
    assert "$or" in find_queries[1]


def test_message_cursor_cannot_be_reused_as_a_session_cursor() -> None:
    repo, _sessions, _messages, _transactions, clock = repository()
    owned = scope()
    repo.create_session(NewChatSession(owned))
    clock.advance()
    repo.append_message(owned, new_message(_MESSAGE_1))
    clock.advance()
    repo.append_message(owned, new_message(_MESSAGE_2))
    message_cursor = repo.list_messages(owned, limit=1).next_cursor

    with pytest.raises(ValueError, match="session cursor"):
        repo.list_sessions(owned.owner_scope, cursor=message_cursor)


def test_append_is_transactional_and_message_id_is_idempotent() -> None:
    repo, sessions, messages, transactions, clock = repository()
    owned = scope()
    repo.create_session(NewChatSession(owned))
    clock.advance()
    draft = new_message()

    first = repo.append_message(owned, draft)
    repeated = repo.append_message(owned, draft)

    assert repeated == first
    assert transactions.calls == 2
    assert len(messages.documents) == 1
    assert_full_scope(cast(Mapping[str, object], next(iter(messages.documents.values()))), owned)
    assert any(operation == "update_one" for operation, _query in sessions.calls)

    with pytest.raises(AppError) as raised:
        repo.append_message(owned, new_message(content="different payload"))
    assert raised.value.code is ErrorCode.RESOURCE_CONFLICT


def test_same_message_id_is_isolated_by_complete_ownership_scope() -> None:
    repo, _sessions, messages, _transactions, clock = repository()
    first = scope()
    second = scope(tenant_id="tenant-b")
    repo.create_session(NewChatSession(first))
    repo.create_session(NewChatSession(second))

    repo.append_message(first, new_message())
    clock.advance()
    repo.append_message(second, new_message())

    assert len(messages.documents) == 2
    assert repo.get_message(first, _MESSAGE_1).scope == first
    assert repo.get_message(second, _MESSAGE_1).scope == second


def test_append_to_missing_or_foreign_session_returns_same_not_found_error() -> None:
    repo, _sessions, _messages, _transactions, _clock = repository()
    repo.create_session(NewChatSession(scope()))

    errors: list[AppError] = []
    for inaccessible in (scope(tenant_id="tenant-b"), scope(owner_subject="user:other")):
        with pytest.raises(AppError) as raised:
            repo.append_message(inaccessible, new_message())
        errors.append(raised.value)

    assert [(error.code, error.message) for error in errors] == [
        (ErrorCode.RESOURCE_NOT_FOUND, "Chat resource not found"),
        (ErrorCode.RESOURCE_NOT_FOUND, "Chat resource not found"),
    ]


def test_message_cursor_paginates_stably_with_exact_scope_filter() -> None:
    repo, _sessions, messages, _transactions, clock = repository()
    owned = scope()
    repo.create_session(NewChatSession(owned))
    for message_id in (_MESSAGE_1, _MESSAGE_2, _MESSAGE_3):
        clock.advance()
        repo.append_message(owned, new_message(message_id, message_id))

    first = repo.list_messages(owned, limit=2)
    second = repo.list_messages(owned, limit=2, cursor=first.next_cursor)

    assert [message.message_id for message in first.items] == [_MESSAGE_1, _MESSAGE_2]
    assert first.next_cursor is not None
    assert [message.message_id for message in second.items] == [_MESSAGE_3]
    assert second.next_cursor is None
    find_queries = [query for operation, query in messages.calls if operation == "find"]
    assert len(find_queries) == 2
    assert_full_scope(find_queries[0], owned)
    assert_full_scope(find_queries[1], owned)
    assert "$or" in find_queries[1]
    assert messages.last_cursor is not None
    assert messages.last_cursor.limit_value == 3


@pytest.mark.parametrize("cursor", ["", "not base64!", "e30", "A" * 513])
def test_invalid_cursor_is_rejected_before_message_query(cursor: str) -> None:
    repo, _sessions, messages, _transactions, _clock = repository()
    repo.create_session(NewChatSession(scope()))

    with pytest.raises(ValueError, match="cursor"):
        repo.list_messages(scope(), cursor=cursor)

    assert not any(operation == "find" for operation, _query in messages.calls)


def test_delete_operations_do_not_disclose_existence_and_use_full_scope() -> None:
    repo, sessions, messages, _transactions, _clock = repository()
    owned = scope()
    repo.create_session(NewChatSession(owned))
    repo.append_message(owned, new_message())

    assert repo.delete_message(scope(tenant_id="tenant-b"), _MESSAGE_1) is None
    assert repo.delete_session(scope(owner_subject="user:other")) is None
    assert repo.get_message(owned, _MESSAGE_1).message_id == _MESSAGE_1

    assert repo.delete_session(owned) is None
    assert sessions.documents == {}
    assert messages.documents == {}
    for operation, query in sessions.calls + messages.calls:
        if operation in {"delete_one", "delete_many", "update_one"}:
            assert {"tenant_id", "owner_subject", "session_id"}.issubset(query)


def test_malformed_database_record_and_driver_error_fail_closed_without_details() -> None:
    repo, sessions, _messages, _transactions, _clock = repository()
    repo.create_session(NewChatSession(scope()))
    stored = next(iter(sessions.documents.values()))
    stored["schema_version"] = 2

    with pytest.raises(DependencyError) as malformed:
        repo.get_session(scope())
    assert str(malformed.value) == "Chat history store is unavailable"

    sessions.failure = RuntimeError("mongodb://root:secret@private-host")
    with pytest.raises(DependencyError) as failed:
        repo.get_session(scope())
    assert str(failed.value) == "Chat history store is unavailable"
    assert "secret" not in str(failed.value)
    assert failed.value.__cause__ is None


def test_malformed_message_record_is_sanitized() -> None:
    repo, _sessions, messages, _transactions, _clock = repository()
    repo.create_session(NewChatSession(scope()))
    repo.append_message(scope(), new_message())
    stored = next(iter(messages.documents.values()))
    stored["role"] = "administrator"

    with pytest.raises(DependencyError) as raised:
        repo.get_message(scope(), _MESSAGE_1)

    assert str(raised.value) == "Chat history store is unavailable"
    assert raised.value.__cause__ is None


def test_transaction_failure_is_sanitized() -> None:
    repo, _sessions, _messages, transactions, _clock = repository()
    repo.create_session(NewChatSession(scope()))
    transactions.failure = RuntimeError("private replica topology")

    with pytest.raises(DependencyError) as raised:
        repo.append_message(scope(), new_message())

    assert str(raised.value) == "Chat history store is unavailable"
    assert "topology" not in str(raised.value)
