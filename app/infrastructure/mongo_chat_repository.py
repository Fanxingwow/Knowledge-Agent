from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar, cast

from pymongo import ASCENDING, DESCENDING, IndexModel, ReturnDocument
from pymongo.errors import CollectionInvalid, DuplicateKeyError

from app.core.errors import AppError, DependencyError, ErrorCode
from app.services.chat_history import (
    ChatMessagePage,
    ChatMessageRole,
    ChatMessageSnapshot,
    ChatOwnerScope,
    ChatSessionPage,
    ChatSessionScope,
    ChatSessionSnapshot,
    NewChatMessage,
    NewChatSession,
)

MongoDocument = dict[str, Any]
_T = TypeVar("_T")
_COLLECTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_UNAVAILABLE_MESSAGE = "Chat history store is unavailable"
_SESSION_FIELDS = frozenset(
    {
        "_id",
        "schema_version",
        "tenant_id",
        "owner_subject",
        "session_id",
        "title",
        "created_at",
        "updated_at",
    }
)
_MESSAGE_FIELDS = frozenset(
    {
        "_id",
        "schema_version",
        "tenant_id",
        "owner_subject",
        "session_id",
        "message_id",
        "role",
        "content",
        "attributes",
        "created_at",
    }
)


class MongoCursor(Protocol):
    def sort(self, key_or_list: Sequence[tuple[str, int]]) -> MongoCursor: ...

    def limit(self, limit: int) -> MongoCursor: ...

    def __iter__(self) -> Iterator[Mapping[str, object]]: ...


class MongoCollection(Protocol):
    def create_indexes(self, indexes: list[IndexModel]) -> list[str]: ...

    def find_one(
        self,
        filter: Mapping[str, object],
        projection: Mapping[str, int] | None = None,
        *,
        session: object | None = None,
    ) -> Mapping[str, object] | None: ...

    def find(
        self,
        filter: Mapping[str, object],
        projection: Mapping[str, int] | None = None,
    ) -> MongoCursor: ...

    def insert_one(
        self,
        document: Mapping[str, object],
        *,
        session: object | None = None,
    ) -> object: ...

    def find_one_and_update(
        self,
        filter: Mapping[str, object],
        update: Mapping[str, object],
        *,
        projection: Mapping[str, int],
        return_document: bool,
        session: object | None = None,
    ) -> Mapping[str, object] | None: ...

    def update_one(
        self,
        filter: Mapping[str, object],
        update: Mapping[str, object],
        *,
        session: object | None = None,
    ) -> object: ...

    def delete_one(
        self,
        filter: Mapping[str, object],
        *,
        session: object | None = None,
    ) -> object: ...

    def delete_many(
        self,
        filter: Mapping[str, object],
        *,
        session: object | None = None,
    ) -> object: ...


class MongoDatabase(Protocol):
    def __getitem__(self, name: str) -> MongoCollection: ...

    def list_collection_names(self) -> list[str]: ...

    def create_collection(self, name: str, **kwargs: object) -> MongoCollection: ...

    def command(self, command: Mapping[str, object]) -> Mapping[str, object]: ...


class MongoChatTransactionRunner(Protocol):
    def run(self, operation: Callable[[object], _T]) -> _T: ...


def _session_schema() -> MongoDocument:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": sorted(_SESSION_FIELDS),
            "additionalProperties": False,
            "properties": {
                "_id": {"bsonType": "string", "minLength": 64, "maxLength": 64},
                "schema_version": {"bsonType": "int", "enum": [1]},
                "tenant_id": {
                    "bsonType": "string",
                    "pattern": "^[A-Za-z0-9._:-]{1,128}$",
                },
                "owner_subject": {"bsonType": "string", "minLength": 1, "maxLength": 256},
                "session_id": {"bsonType": "string", "minLength": 36, "maxLength": 36},
                "title": {"bsonType": ["string", "null"], "maxLength": 200},
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
            },
        }
    }


def _message_schema() -> MongoDocument:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": sorted(_MESSAGE_FIELDS),
            "additionalProperties": False,
            "properties": {
                "_id": {"bsonType": "string", "minLength": 64, "maxLength": 64},
                "schema_version": {"bsonType": "int", "enum": [1]},
                "tenant_id": {
                    "bsonType": "string",
                    "pattern": "^[A-Za-z0-9._:-]{1,128}$",
                },
                "owner_subject": {"bsonType": "string", "minLength": 1, "maxLength": 256},
                "session_id": {"bsonType": "string", "minLength": 36, "maxLength": 36},
                "message_id": {"bsonType": "string", "minLength": 36, "maxLength": 36},
                "role": {
                    "bsonType": "string",
                    "enum": [role.value for role in ChatMessageRole],
                },
                "content": {"bsonType": "string", "minLength": 1, "maxLength": 200_000},
                "attributes": {
                    "bsonType": "object",
                    "maxProperties": 32,
                    "additionalProperties": {"bsonType": "string", "maxLength": 2_000},
                },
                "created_at": {"bsonType": "date"},
            },
        }
    }


def chat_schema_contracts() -> Mapping[str, MongoDocument]:
    """Return fresh validators so callers cannot mutate the runtime contract."""

    return {
        "sessions": _session_schema(),
        "messages": _message_schema(),
    }


def chat_index_contracts() -> Mapping[str, tuple[IndexModel, ...]]:
    return {
        "sessions": (
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("owner_subject", ASCENDING),
                    ("session_id", ASCENDING),
                ],
                unique=True,
                name="uq_chat_session_scope",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("owner_subject", ASCENDING),
                    ("updated_at", DESCENDING),
                    ("session_id", ASCENDING),
                ],
                name="ix_chat_session_owned_updated",
            ),
        ),
        "messages": (
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("owner_subject", ASCENDING),
                    ("session_id", ASCENDING),
                    ("message_id", ASCENDING),
                ],
                unique=True,
                name="uq_chat_message_idempotency",
            ),
            IndexModel(
                [
                    ("tenant_id", ASCENDING),
                    ("owner_subject", ASCENDING),
                    ("session_id", ASCENDING),
                    ("created_at", ASCENDING),
                    ("message_id", ASCENDING),
                ],
                name="ix_chat_message_page",
            ),
        ),
    }


class MongoChatSchemaManager:
    """Explicitly installs strict validators and indexes; construction performs no I/O."""

    def __init__(
        self,
        database: MongoDatabase,
        *,
        sessions_collection: str = "chat_sessions_v1",
        messages_collection: str = "chat_messages_v1",
    ) -> None:
        for name in (sessions_collection, messages_collection):
            if not isinstance(name, str) or _COLLECTION_NAME.fullmatch(name) is None:
                raise ValueError("invalid chat collection name")
        if sessions_collection == messages_collection:
            raise ValueError("chat collections must be distinct")
        self._database = database
        self.sessions_collection = sessions_collection
        self.messages_collection = messages_collection

    @staticmethod
    def _options(validator: Mapping[str, object]) -> MongoDocument:
        return {
            "validator": dict(validator),
            "validationLevel": "strict",
            "validationAction": "error",
        }

    def _ensure_collection(self, name: str, validator: Mapping[str, object], existing: set[str]) -> None:
        options = self._options(validator)
        if name not in existing:
            try:
                self._database.create_collection(name, **options)
                return
            except CollectionInvalid:
                pass
        self._database.command({"collMod": name, **options})

    def ensure_schema(self) -> None:
        schemas = chat_schema_contracts()
        indexes = chat_index_contracts()
        try:
            existing = set(self._database.list_collection_names())
            self._ensure_collection(self.sessions_collection, schemas["sessions"], existing)
            self._ensure_collection(self.messages_collection, schemas["messages"], existing)
            self._database[self.sessions_collection].create_indexes(list(indexes["sessions"]))
            self._database[self.messages_collection].create_indexes(list(indexes["messages"]))
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None

    def build_repository(
        self,
        transaction_runner: MongoChatTransactionRunner,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> MongoChatRepository:
        self.ensure_schema()
        return MongoChatRepository(
            self._database[self.sessions_collection],
            self._database[self.messages_collection],
            transaction_runner,
            clock=clock,
        )


def _scope_filter(scope: ChatSessionScope) -> MongoDocument:
    if not isinstance(scope, ChatSessionScope):
        raise ValueError("invalid chat session scope")
    return {
        "tenant_id": scope.tenant_id,
        "owner_subject": scope.owner_subject,
        "session_id": scope.session_id,
    }


def _owner_filter(owner: ChatOwnerScope) -> MongoDocument:
    if not isinstance(owner, ChatOwnerScope):
        raise ValueError("invalid chat owner scope")
    return {
        "tenant_id": owner.tenant_id,
        "owner_subject": owner.owner_subject,
    }


def _resource_not_found() -> AppError:
    return AppError(ErrorCode.RESOURCE_NOT_FOUND, "Chat resource not found", 404)


def _resource_conflict() -> AppError:
    return AppError(ErrorCode.RESOURCE_CONFLICT, "Chat resource conflict", 409)


def _projection(fields: frozenset[str]) -> MongoDocument:
    return {field: 1 for field in fields}


def _utc_timestamp(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("invalid chat timestamp")
    return value.astimezone(UTC)


def _document_id(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _encode_message_cursor(message: ChatMessageSnapshot) -> str:
    payload = json.dumps(
        {
            "id": message.message_id,
            "ts": message.created_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "v": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_message_cursor(cursor: str) -> tuple[datetime, str]:
    if not isinstance(cursor, str) or not 1 <= len(cursor) <= 512:
        raise ValueError("invalid chat history cursor")
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        decoded = json.loads(payload.decode("utf-8"))
        if not isinstance(decoded, dict) or set(decoded) != {"id", "ts", "v"} or decoded["v"] != 1:
            raise ValueError
        message_id = NewChatMessage(
            message_id=decoded["id"],
            role=ChatMessageRole.USER,
            content="cursor",
        ).message_id
        timestamp = datetime.fromisoformat(cast(str, decoded["ts"]))
        timestamp = _utc_timestamp(timestamp)
        canonical = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        if canonical != cursor:
            raise ValueError
    except (KeyError, TypeError, UnicodeError, ValueError, json.JSONDecodeError):
        raise ValueError("invalid chat history cursor") from None
    return timestamp, message_id


def _encode_session_cursor(session: ChatSessionSnapshot) -> str:
    payload = json.dumps(
        {
            "session": session.scope.session_id,
            "ts": session.updated_at.astimezone(UTC).isoformat(timespec="microseconds"),
            "v": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_session_cursor(cursor: str) -> tuple[datetime, str]:
    if not isinstance(cursor, str) or not 1 <= len(cursor) <= 512:
        raise ValueError("invalid chat session cursor")
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        decoded = json.loads(payload.decode("utf-8"))
        if not isinstance(decoded, dict) or set(decoded) != {"session", "ts", "v"} or decoded["v"] != 1:
            raise ValueError
        session_id = ChatSessionScope(
            tenant_id="cursor",
            owner_subject="cursor",
            session_id=decoded["session"],
        ).session_id
        timestamp = _utc_timestamp(datetime.fromisoformat(cast(str, decoded["ts"])))
        canonical = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        if canonical != cursor:
            raise ValueError
    except (KeyError, TypeError, UnicodeError, ValueError, json.JSONDecodeError):
        raise ValueError("invalid chat session cursor") from None
    return timestamp, session_id


class MongoChatRepository:
    """Tenant-owned session and message persistence with atomic cross-collection writes."""

    durable = True

    def __init__(
        self,
        sessions: MongoCollection,
        messages: MongoCollection,
        transaction_runner: MongoChatTransactionRunner,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._sessions = sessions
        self._messages = messages
        self._transactions = transaction_runner
        self._clock = clock or (lambda: datetime.now(UTC))

    def _now(self) -> datetime:
        return _utc_timestamp(self._clock())

    @staticmethod
    def _parse_session(
        document: Mapping[str, object],
        expected: ChatSessionScope | None = None,
    ) -> ChatSessionSnapshot:
        if set(document) != _SESSION_FIELDS or document.get("schema_version") != 1:
            raise ValueError("invalid chat session record")
        scope = ChatSessionScope(
            tenant_id=cast(str, document["tenant_id"]),
            owner_subject=cast(str, document["owner_subject"]),
            session_id=cast(str, document["session_id"]),
        )
        if (expected is not None and scope != expected) or document["_id"] != _document_id(
            scope.tenant_id,
            scope.owner_subject,
            scope.session_id,
        ):
            raise ValueError("out-of-scope chat session record")
        title = document["title"]
        validated = NewChatSession(scope=scope, title=cast(str | None, title))
        created_at = _utc_timestamp(document["created_at"])
        updated_at = _utc_timestamp(document["updated_at"])
        if updated_at < created_at:
            raise ValueError("invalid chat session record")
        return ChatSessionSnapshot(
            scope=scope,
            title=validated.title,
            created_at=created_at,
            updated_at=updated_at,
        )

    @staticmethod
    def _parse_message(document: Mapping[str, object], expected: ChatSessionScope) -> ChatMessageSnapshot:
        if set(document) != _MESSAGE_FIELDS or document.get("schema_version") != 1:
            raise ValueError("invalid chat message record")
        scope = ChatSessionScope(
            tenant_id=cast(str, document["tenant_id"]),
            owner_subject=cast(str, document["owner_subject"]),
            session_id=cast(str, document["session_id"]),
        )
        attributes = document["attributes"]
        if not isinstance(attributes, Mapping):
            raise ValueError("invalid chat message record")
        message = NewChatMessage(
            message_id=cast(str, document["message_id"]),
            role=ChatMessageRole(cast(str, document["role"])),
            content=cast(str, document["content"]),
            attributes=tuple(cast(Mapping[str, str], attributes).items()),
        )
        if scope != expected or document["_id"] != _document_id(
            scope.tenant_id,
            scope.owner_subject,
            scope.session_id,
            message.message_id,
        ):
            raise ValueError("out-of-scope chat message record")
        return ChatMessageSnapshot(
            scope=scope,
            message_id=message.message_id,
            role=message.role,
            content=message.content,
            attributes=message.attributes,
            created_at=_utc_timestamp(document["created_at"]),
        )

    def _find_session(
        self,
        scope: ChatSessionScope,
        *,
        mongo_session: object | None = None,
    ) -> ChatSessionSnapshot | None:
        document = self._sessions.find_one(
            _scope_filter(scope),
            _projection(_SESSION_FIELDS),
            session=mongo_session,
        )
        return None if document is None else self._parse_session(document, scope)

    def _find_message(
        self,
        scope: ChatSessionScope,
        message_id: str,
        *,
        mongo_session: object | None = None,
    ) -> ChatMessageSnapshot | None:
        validated = NewChatMessage(
            message_id=message_id,
            role=ChatMessageRole.USER,
            content="lookup",
        ).message_id
        document = self._messages.find_one(
            {**_scope_filter(scope), "message_id": validated},
            _projection(_MESSAGE_FIELDS),
            session=mongo_session,
        )
        return None if document is None else self._parse_message(document, scope)

    def create_session(self, session: NewChatSession) -> ChatSessionSnapshot:
        if not isinstance(session, NewChatSession):
            raise ValueError("invalid chat session")
        now = self._now()
        scope = session.scope
        document: MongoDocument = {
            "_id": _document_id(scope.tenant_id, scope.owner_subject, scope.session_id),
            "schema_version": 1,
            **_scope_filter(scope),
            "title": session.title,
            "created_at": now,
            "updated_at": now,
        }
        try:
            self._sessions.insert_one(document)
            return self._parse_session(document, scope)
        except DuplicateKeyError:
            try:
                existing = self._find_session(scope)
            except Exception:
                raise DependencyError(_UNAVAILABLE_MESSAGE) from None
            if existing is not None and existing.title == session.title:
                return existing
            raise _resource_conflict() from None
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None

    def get_session(self, scope: ChatSessionScope) -> ChatSessionSnapshot:
        _scope_filter(scope)
        try:
            found = self._find_session(scope)
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None
        if found is None:
            raise _resource_not_found()
        return found

    def list_sessions(
        self,
        owner: ChatOwnerScope,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ChatSessionPage:
        owner_filter = _owner_filter(owner)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("invalid chat session page size")
        position = None if cursor is None else _decode_session_cursor(cursor)
        query: MongoDocument = dict(owner_filter)
        if position is not None:
            timestamp, session_id = position
            query["$or"] = [
                {"updated_at": {"$lt": timestamp}},
                {"updated_at": timestamp, "session_id": {"$gt": session_id}},
            ]
        try:
            mongo_cursor = self._sessions.find(query, _projection(_SESSION_FIELDS))
            documents = tuple(
                mongo_cursor.sort([("updated_at", DESCENDING), ("session_id", ASCENDING)]).limit(limit + 1)
            )
            parsed = tuple(self._parse_session(document) for document in documents)
            if any(item.scope.owner_scope != owner for item in parsed):
                raise ValueError("out-of-scope chat session record")
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None
        items = parsed[:limit]
        next_cursor = _encode_session_cursor(items[-1]) if len(parsed) > limit else None
        return ChatSessionPage(items=items, next_cursor=next_cursor)

    def update_session_title(
        self,
        scope: ChatSessionScope,
        title: str | None,
    ) -> ChatSessionSnapshot:
        validated = NewChatSession(scope=scope, title=title)
        try:
            document = self._sessions.find_one_and_update(
                _scope_filter(scope),
                {"$set": {"title": validated.title, "updated_at": self._now()}},
                projection=_projection(_SESSION_FIELDS),
                return_document=ReturnDocument.AFTER,
            )
            if document is None:
                raise _resource_not_found()
            return self._parse_session(document, scope)
        except AppError:
            raise
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None

    def append_message(
        self,
        scope: ChatSessionScope,
        message: NewChatMessage,
    ) -> ChatMessageSnapshot:
        _scope_filter(scope)
        if not isinstance(message, NewChatMessage):
            raise ValueError("invalid chat message")
        created_at = self._now()
        document: MongoDocument = {
            "_id": _document_id(
                scope.tenant_id,
                scope.owner_subject,
                scope.session_id,
                message.message_id,
            ),
            "schema_version": 1,
            **_scope_filter(scope),
            "message_id": message.message_id,
            "role": message.role.value,
            "content": message.content,
            "attributes": dict(message.attributes),
            "created_at": created_at,
        }

        def operation(mongo_session: object) -> ChatMessageSnapshot:
            if self._find_session(scope, mongo_session=mongo_session) is None:
                raise _resource_not_found()
            self._messages.insert_one(document, session=mongo_session)
            result = self._sessions.update_one(
                _scope_filter(scope),
                {"$set": {"updated_at": created_at}},
                session=mongo_session,
            )
            if getattr(result, "matched_count", None) != 1:
                raise _resource_conflict()
            return self._parse_message(document, scope)

        try:
            return self._transactions.run(operation)
        except DuplicateKeyError:
            try:
                existing = self._find_message(scope, message.message_id)
            except Exception:
                raise DependencyError(_UNAVAILABLE_MESSAGE) from None
            if (
                existing is not None
                and existing.role is message.role
                and existing.content == message.content
                and existing.attributes == message.attributes
            ):
                return existing
            raise _resource_conflict() from None
        except AppError:
            raise
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None

    def get_message(
        self,
        scope: ChatSessionScope,
        message_id: str,
    ) -> ChatMessageSnapshot:
        _scope_filter(scope)
        validated = NewChatMessage(
            message_id=message_id,
            role=ChatMessageRole.USER,
            content="lookup",
        ).message_id
        try:
            found = self._find_message(scope, validated)
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None
        if found is None:
            raise _resource_not_found()
        return found

    def list_messages(
        self,
        scope: ChatSessionScope,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ChatMessagePage:
        scope_filter = _scope_filter(scope)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("invalid chat history page size")
        position = None if cursor is None else _decode_message_cursor(cursor)
        query: MongoDocument = dict(scope_filter)
        if position is not None:
            timestamp, message_id = position
            query["$or"] = [
                {"created_at": {"$gt": timestamp}},
                {"created_at": timestamp, "message_id": {"$gt": message_id}},
            ]
        try:
            if self._find_session(scope) is None:
                raise _resource_not_found()
            mongo_cursor = self._messages.find(query, _projection(_MESSAGE_FIELDS))
            documents = tuple(
                mongo_cursor.sort([("created_at", ASCENDING), ("message_id", ASCENDING)]).limit(limit + 1)
            )
            parsed = tuple(self._parse_message(document, scope) for document in documents)
        except AppError:
            raise
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None
        items = parsed[:limit]
        next_cursor = _encode_message_cursor(items[-1]) if len(parsed) > limit else None
        return ChatMessagePage(items=items, next_cursor=next_cursor)

    def delete_message(self, scope: ChatSessionScope, message_id: str) -> None:
        validated = NewChatMessage(
            message_id=message_id,
            role=ChatMessageRole.USER,
            content="delete",
        ).message_id

        def operation(mongo_session: object) -> None:
            self._messages.delete_one(
                {**_scope_filter(scope), "message_id": validated},
                session=mongo_session,
            )
            self._sessions.update_one(
                _scope_filter(scope),
                {"$set": {"updated_at": self._now()}},
                session=mongo_session,
            )

        try:
            self._transactions.run(operation)
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None

    def delete_session(self, scope: ChatSessionScope) -> None:
        scope_filter = _scope_filter(scope)

        def operation(mongo_session: object) -> None:
            self._messages.delete_many(scope_filter, session=mongo_session)
            self._sessions.delete_one(scope_filter, session=mongo_session)

        try:
            self._transactions.run(operation)
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None
