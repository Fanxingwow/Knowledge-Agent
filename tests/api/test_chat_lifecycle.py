from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import cast

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.errors import DependencyError, ErrorCode
from app.core.settings import (
    ApiSettings,
    AppEnvironment,
    AuthSettings,
    ChatSettings,
    ImportRuntimeSettings,
    LoggingSettings,
    MinioSettings,
    MongoSettings,
    RedisSettings,
    Settings,
)
from app.infrastructure.minio_store import MinioStore
from app.infrastructure.redis_sse_broker import RedisStreamsSseBroker
from app.main import _build_default_session_event_broker, create_app
from app.services.chat_history import (
    ChatMessagePage,
    ChatMessageSnapshot,
    ChatOwnerScope,
    ChatSessionPage,
    ChatSessionScope,
    ChatSessionSnapshot,
    NewChatMessage,
    NewChatSession,
)
from app.services.import_coordinator import ImportCoordinator
from app.services.sse_broker import SessionEventBroker, SseEvent, SseSubscription

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class Store:
    def __init__(self) -> None:
        self.closed = False

    def require_ready(self) -> None:
        return

    def close(self) -> None:
        self.closed = True


class Repository:
    durable = True

    def create_session(self, session: NewChatSession) -> ChatSessionSnapshot:
        return ChatSessionSnapshot(session.scope, session.title, _NOW, _NOW)

    def get_session(self, scope: ChatSessionScope) -> ChatSessionSnapshot:
        return ChatSessionSnapshot(scope, None, _NOW, _NOW)

    def list_sessions(
        self,
        owner: ChatOwnerScope,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ChatSessionPage:
        del owner, limit, cursor
        return ChatSessionPage((), None)

    def update_session_title(
        self,
        scope: ChatSessionScope,
        title: str | None,
    ) -> ChatSessionSnapshot:
        return ChatSessionSnapshot(scope, title, _NOW, _NOW)

    def append_message(
        self,
        scope: ChatSessionScope,
        message: NewChatMessage,
    ) -> ChatMessageSnapshot:
        return ChatMessageSnapshot(
            scope,
            message.message_id,
            message.role,
            message.content,
            message.attributes,
            _NOW,
        )

    def get_message(
        self,
        scope: ChatSessionScope,
        message_id: str,
    ) -> ChatMessageSnapshot:
        raise NotImplementedError

    def list_messages(
        self,
        scope: ChatSessionScope,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ChatMessagePage:
        del scope, limit, cursor
        return ChatMessagePage((), None)

    def delete_message(self, scope: ChatSessionScope, message_id: str) -> None:
        del scope, message_id

    def delete_session(self, scope: ChatSessionScope) -> None:
        del scope


class Coordinator:
    durable = True

    def __init__(self) -> None:
        self.chat_history_repository = Repository()
        self.ready = True
        self.probe_calls = 0
        self.closed = False

    def probe(self) -> None:
        self.probe_calls += 1
        if not self.ready:
            raise RuntimeError("private Mongo detail")

    def close(self) -> None:
        self.closed = True


class Broker:
    def __init__(self, *, distributed: bool = False) -> None:
        self.distributed = distributed
        self.closed = False

    async def publish(
        self,
        scope: ChatSessionScope,
        *,
        event_type: str,
        data: str,
        terminal: bool = False,
    ) -> SseEvent:
        del scope
        return SseEvent(1, event_type, data, terminal=terminal)

    async def subscribe(
        self,
        scope: ChatSessionScope,
        *,
        last_event_id: int | None = None,
    ) -> SseSubscription:
        del scope, last_event_id
        raise NotImplementedError

    async def terminate(
        self,
        scope: ChatSessionScope,
        *,
        event_type: str,
        data: str,
    ) -> SseEvent:
        return await self.publish(scope, event_type=event_type, data=data, terminal=True)

    async def aclose(self) -> None:
        self.closed = True


def settings(*, environment: AppEnvironment = AppEnvironment.TEST) -> Settings:
    return Settings.model_construct(
        api=ApiSettings.model_construct(
            environment=environment,
            import_enabled=False,
            cors_origins_raw="https://console.example.com",
        ),
        import_runtime=ImportRuntimeSettings.model_construct(
            repository="mongo",
            worker_enabled=False,
            poll_interval_seconds=0.05,
            shutdown_grace_seconds=1.0,
        ),
        auth=AuthSettings.model_construct(
            mode="disabled",
            issuer=None,
            audience=None,
            jwks_url=None,
            public_key=None,
            algorithms_raw="RS256",
            dev_tenant_id="tenant-a",
        ),
        logging=LoggingSettings.model_construct(
            console_enabled=False,
            console_level="INFO",
            file_enabled=False,
            file_level="INFO",
            retention="7 days",
            json_output=False,
            diagnose=False,
        ),
        minio=MinioSettings.model_construct(
            readiness_interval_seconds=0.05,
            readiness_probe_timeout_seconds=0.2,
            readiness_ttl_seconds=0.5,
        ),
        chat=ChatSettings.model_construct(
            enabled=True,
            sse_broker="memory",
            max_sessions=20,
            replay_capacity=10,
            max_replay_events=5,
            subscriber_queue_capacity=4,
            max_subscribers_per_session=2,
            heartbeat_seconds=0.1,
            readiness_interval_seconds=0.05,
            readiness_probe_timeout_seconds=0.2,
            readiness_ttl_seconds=0.5,
        ),
        mongo=MongoSettings.model_construct(
            url="mongodb://mongo.invalid:27017",
            health_timeout_seconds=0.2,
            require_replica_set=True,
        ),
    )


def build_test_app(
    coordinator: Coordinator,
    broker: Broker,
    *,
    environment: AppEnvironment = AppEnvironment.TEST,
):
    store = Store()
    broker_builds: list[Settings] = []

    def build_broker(app_settings: Settings) -> SessionEventBroker:
        broker_builds.append(app_settings)
        return cast(SessionEventBroker, broker)

    app = create_app(
        settings=settings(environment=environment),
        minio_builder=lambda _: cast(MinioStore, store),
        import_coordinator_factory=lambda: cast(ImportCoordinator, coordinator),
        session_event_broker_builder=build_broker,
    )
    return app, store, broker_builds


def test_chat_runtime_is_lazy_mounted_ready_and_closed() -> None:
    coordinator = Coordinator()
    broker = Broker()
    app, store, broker_builds = build_test_app(coordinator, broker)

    assert broker_builds == []
    with TestClient(app) as client:
        sessions = client.get("/api/v1/chat/sessions")
        readiness = client.get("/health/ready")
        schema = client.get("/openapi.json").json()

    assert sessions.status_code == 200
    assert sessions.json() == {"items": [], "next_cursor": None}
    assert readiness.status_code == 200
    assert readiness.json()["checks"] == {
        "minio": "ok",
        "chat_history": "ok",
        "chat_events": "ok",
    }
    assert "/api/v1/chat/sessions" in schema["paths"]
    assert broker_builds == [app.state.settings]
    assert coordinator.probe_calls >= 1
    assert coordinator.closed is True
    assert broker.closed is True
    assert store.closed is True


def test_chat_probe_failure_withdraws_api_and_recovers() -> None:
    coordinator = Coordinator()
    coordinator.ready = False
    app, _, _ = build_test_app(coordinator, Broker())

    with TestClient(app) as client:
        unavailable = client.get("/api/v1/chat/sessions")
        assert unavailable.status_code == 503
        assert unavailable.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value
        assert "private Mongo detail" not in unavailable.text
        coordinator.ready = True
        initial_calls = coordinator.probe_calls
        deadline = time.monotonic() + 1.0
        while coordinator.probe_calls == initial_calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.get("/api/v1/chat/sessions").status_code == 200


def test_stale_chat_readiness_fails_closed() -> None:
    app, _, _ = build_test_app(Coordinator(), Broker())

    with TestClient(app) as client:
        assert client.get("/api/v1/chat/sessions").status_code == 200
        app.state.chat_ready_until = time.monotonic() - 1
        response = client.get("/api/v1/chat/sessions")

    assert response.status_code == 503
    assert response.json()["message"] == "Chat runtime is unavailable"


def test_production_rejects_process_local_event_broker() -> None:
    app, _, _ = build_test_app(
        Coordinator(),
        Broker(distributed=False),
        environment=AppEnvironment.PRODUCTION,
    )

    with pytest.raises(DependencyError, match="Chat runtime is not configured"):
        with TestClient(app):
            pass


def test_explicitly_missing_event_broker_aborts_startup_safely() -> None:
    app = create_app(
        settings=settings(),
        minio_builder=lambda _: cast(MinioStore, Store()),
        import_coordinator_factory=lambda: cast(ImportCoordinator, Coordinator()),
        session_event_broker_builder=None,
    )

    with pytest.raises(DependencyError, match="Chat runtime is not configured"):
        with TestClient(app):
            pass


@pytest.mark.asyncio
async def test_default_redis_broker_factory_is_lazy_and_uses_distributed_adapter() -> None:
    configured = settings()
    configured.chat.sse_broker = "redis"
    configured.redis = RedisSettings.model_construct(
        url=SecretStr("rediss://redis.invalid:6379/0"),
        chat_key_hmac_secret=SecretStr("k" * 32),
        key_prefix="knowledge-agent:chat:v1",
        connect_timeout_seconds=2.0,
        socket_timeout_seconds=3.0,
        max_connections=20,
        stream_ttl_seconds=600,
    )

    broker = _build_default_session_event_broker(configured)

    assert isinstance(broker, RedisStreamsSseBroker)
    assert broker.distributed is True
    await broker.aclose()
