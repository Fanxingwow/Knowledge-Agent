from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event, Lock, Thread
from typing import cast
from urllib.parse import urlsplit

import anyio
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes.chat import router as chat_router
from app.api.routes.imports import router as imports_router
from app.api.routes.search import router as search_router
from app.api.schemas.common import ConsoleAuthConfigResponse, HealthResponse
from app.core.admission_middleware import ApiAuthenticationMiddleware, RequestBodyLimitMiddleware
from app.core.errors import AppError, DependencyError, ErrorCode, register_exception_handlers
from app.core.logger import configure_logging, logger
from app.core.metrics import (
    ApplicationMetrics,
    PrometheusMetricsMiddleware,
    build_application_metrics,
    metrics_authorized,
    prometheus_content_type,
    render_metrics,
)
from app.core.request_context import RequestContextMiddleware
from app.core.settings import AppEnvironment, Settings, get_settings
from app.infrastructure.minio_store import MinioStore, build_minio_store
from app.infrastructure.mongo_import_repository import MongoImportCoordinator
from app.infrastructure.mongo_import_store import build_mongo_import_store
from app.infrastructure.readiness import object_storage_is_ready
from app.infrastructure.redis_sse_broker import (
    RedisStreamsBrokerConfig,
    RedisStreamsSseBroker,
)
from app.services.agent_chat import AgentChatBackend, build_agent_chat_backend
from app.services.import_coordinator import ImportCoordinator, InMemoryImportCoordinator
from app.services.import_processor_factory import build_enterprise_import_processor
from app.services.import_registry import ImportTaskRegistry, InMemoryImportTaskRegistry
from app.services.import_runner import ImportProcessor
from app.services.import_worker import ImportWorker, ImportWorkerConfig
from app.services.import_workspace import (
    ImportWorkspace,
    build_import_workspace,
)
from app.services.knowledge_search_runtime import (
    ManagedKnowledgeSearchRuntime,
    build_knowledge_search_runtime,
)
from app.services.sse_broker import InMemorySseBroker, SessionEventBroker
from app.services.tenant_vector_search import CompletedImportVisibilityReader

MinioStoreBuilder = Callable[[Settings], MinioStore]
ImportRegistryFactory = Callable[[], ImportTaskRegistry]
ImportCoordinatorFactory = Callable[[], ImportCoordinator]
ImportProcessorBuilder = Callable[[Settings], ImportProcessor]
ImportWorkspaceBuilder = Callable[[Settings], ImportWorkspace]
KnowledgeSearchRuntimeBuilder = Callable[
    [Settings, CompletedImportVisibilityReader, Callable[[], None]],
    ManagedKnowledgeSearchRuntime,
]
SessionEventBrokerBuilder = Callable[[Settings], SessionEventBroker]
AgentChatBackendBuilder = Callable[[Settings], AgentChatBackend]
_DEFAULT_IMPORT_PROCESSOR_BUILDER = object()
_DEFAULT_KNOWLEDGE_SEARCH_RUNTIME_BUILDER = object()
_DEFAULT_SESSION_EVENT_BROKER_BUILDER = object()
_DEFAULT_AGENT_CHAT_BACKEND_BUILDER = object()
_CONSOLE_DIRECTORY = Path(__file__).resolve().parent / "web_console"
_CONSOLE_ASSETS = frozenset({"console.css", "console.js"})


def _build_default_session_event_broker(settings: Settings) -> SessionEventBroker:
    chat = settings.chat
    if chat.sse_broker == "memory":
        return InMemorySseBroker(
            max_sessions=chat.max_sessions,
            replay_capacity=chat.replay_capacity,
            max_replay_events=chat.max_replay_events,
            subscriber_queue_capacity=chat.subscriber_queue_capacity,
            max_subscribers_per_session=chat.max_subscribers_per_session,
            heartbeat_seconds=chat.heartbeat_seconds,
        )
    redis = settings.redis
    if redis.url is None or redis.chat_key_hmac_secret is None:
        raise DependencyError("Distributed event streaming is not configured")
    try:
        config = RedisStreamsBrokerConfig(
            url=redis.url.get_secret_value(),
            key_hmac_secret=redis.chat_key_hmac_secret.get_secret_value().encode(
                "utf-8",
                errors="strict",
            ),
            key_prefix=redis.key_prefix,
            replay_capacity=chat.replay_capacity,
            max_replay_events=chat.max_replay_events,
            read_batch_size=min(
                chat.subscriber_queue_capacity,
                chat.replay_capacity,
            ),
            max_subscribers_per_session=chat.max_subscribers_per_session,
            heartbeat_seconds=chat.heartbeat_seconds,
            retention_seconds=redis.stream_ttl_seconds,
            connect_timeout_seconds=redis.connect_timeout_seconds,
            socket_timeout_seconds=redis.socket_timeout_seconds,
        )
        return RedisStreamsSseBroker(config)
    except Exception:
        raise DependencyError("Distributed event streaming is not configured") from None


def _build_default_import_coordinator(
    settings: Settings,
    registry_factory: ImportRegistryFactory,
) -> ImportCoordinator:
    if settings.import_runtime.repository == "mongo":
        store = build_mongo_import_store(settings)
        try:
            if settings.search.enabled and settings.chat.enabled:
                return MongoImportCoordinator(
                    store,
                    cleanup_retry_base_seconds=settings.import_runtime.retry_base_seconds,
                    cleanup_retry_max_seconds=settings.import_runtime.retry_max_seconds,
                    terminal_retention_days=settings.mongo.terminal_retention_days,
                    visibility_batch_size=settings.search.max_candidates,
                    chat_enabled=True,
                    chat_sessions_collection=settings.mongo.chat_sessions_collection,
                    chat_messages_collection=settings.mongo.chat_messages_collection,
                )
            if settings.search.enabled:
                return MongoImportCoordinator(
                    store,
                    cleanup_retry_base_seconds=settings.import_runtime.retry_base_seconds,
                    cleanup_retry_max_seconds=settings.import_runtime.retry_max_seconds,
                    terminal_retention_days=settings.mongo.terminal_retention_days,
                    visibility_batch_size=settings.search.max_candidates,
                )
            if settings.chat.enabled:
                return MongoImportCoordinator(
                    store,
                    cleanup_retry_base_seconds=settings.import_runtime.retry_base_seconds,
                    cleanup_retry_max_seconds=settings.import_runtime.retry_max_seconds,
                    terminal_retention_days=settings.mongo.terminal_retention_days,
                    chat_enabled=True,
                    chat_sessions_collection=settings.mongo.chat_sessions_collection,
                    chat_messages_collection=settings.mongo.chat_messages_collection,
                )
            return MongoImportCoordinator(
                store,
                cleanup_retry_base_seconds=settings.import_runtime.retry_base_seconds,
                cleanup_retry_max_seconds=settings.import_runtime.retry_max_seconds,
                terminal_retention_days=settings.mongo.terminal_retention_days,
            )
        except BaseException:
            store.close()
            raise
    return InMemoryImportCoordinator(tasks=registry_factory())


def _mark_storage_unavailable(app: FastAPI) -> None:
    app.state.minio_ready = False
    app.state.minio_ready_until = 0.0
    app.state.minio_store = None


def _mark_storage_ready(app: FastAPI, store: MinioStore) -> None:
    app.state.minio_store = store
    app.state.minio_ready = True
    app.state.minio_ready_until = time.monotonic() + app.state.minio_readiness_ttl_seconds


def _mark_import_unavailable(app: FastAPI) -> None:
    app.state.import_ready = False
    app.state.import_ready_until = 0.0
    checks = {
        "import_registry": "unavailable",
        "import_processor": "unavailable",
        "import_workspace": "unavailable",
    }
    if app.state.settings.import_runtime.worker_enabled:
        checks["import_worker"] = "unavailable"
    app.state.import_checks = checks


def _probe_component(component: object, *, allow_presence: bool = False) -> bool:
    if component is None:
        return False
    probe = getattr(component, "probe", None)
    if callable(probe):
        probe()
        return True
    is_ready = getattr(component, "is_ready", None)
    if callable(is_ready):
        return is_ready() is True
    return allow_presence


async def _bounded_component_probe(
    probe_state: FastAPI | dict[str, _ComponentProbe],
    name: str,
    component: object,
    *,
    timeout_seconds: float,
    allow_presence: bool = False,
) -> bool:
    inflight_by_name = probe_state.state.import_probe_inflight if isinstance(probe_state, FastAPI) else probe_state
    inflight = inflight_by_name.get(name)
    if inflight is not None:
        if not inflight.completion.is_set():
            return False
        inflight_by_name.pop(name, None)

    probe = _ComponentProbe(component, allow_presence=allow_presence)
    inflight_by_name[name] = probe
    try:
        Thread(
            target=probe.run,
            name=f"{name}-readiness-probe",
            daemon=True,
        ).start()
    except Exception:
        if inflight_by_name.get(name) is probe:
            inflight_by_name.pop(name, None)
        return False

    deadline = time.monotonic() + timeout_seconds
    while not probe.completion.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await anyio.sleep(min(0.01, remaining))

    if inflight_by_name.get(name) is probe:
        inflight_by_name.pop(name, None)
    return probe.succeeded


async def _run_import_check(app: FastAPI) -> bool:
    if not app.state.settings.api.import_enabled:
        return True
    if app.state.import_shutting_down:
        _mark_import_unavailable(app)
        return False
    async with app.state.import_probe_lock:
        if app.state.import_shutting_down:
            _mark_import_unavailable(app)
            return False
        timeout = app.state.settings.mongo.health_timeout_seconds
        allow_processor_presence = app.state.settings.api.environment is not AppEnvironment.PRODUCTION
        components: list[tuple[str, object, bool]] = [
            ("import_registry", app.state.import_coordinator, False),
            ("import_processor", app.state.import_processor, allow_processor_presence),
            ("import_workspace", app.state.import_workspace, False),
        ]
        if app.state.settings.import_runtime.worker_enabled:
            components.append(("import_worker", app.state.import_worker, False))
        checks: dict[str, str] = {}
        ready = True
        for name, component, allow_presence in components:
            component_ready = await _bounded_component_probe(
                app.state.import_probe_inflight,
                name,
                component,
                timeout_seconds=timeout,
                allow_presence=allow_presence,
            )
            checks[name] = "ok" if component_ready else "unavailable"
            ready = ready and component_ready
        if app.state.import_shutting_down:
            _mark_import_unavailable(app)
            return False
        app.state.import_checks = checks
        app.state.import_ready = ready
        app.state.import_ready_until = time.monotonic() + app.state.import_readiness_ttl_seconds if ready else 0.0
        return ready


async def _import_probe_loop(app: FastAPI, interval_seconds: float) -> None:
    while True:
        await anyio.sleep(interval_seconds)
        await _run_import_check(app)


def _mark_knowledge_search_unavailable(app: FastAPI) -> None:
    app.state.knowledge_search = None
    app.state.knowledge_search_ready = False
    app.state.knowledge_search_ready_until = 0.0
    app.state.knowledge_search_checks = {"knowledge_search": "unavailable"}


async def _run_knowledge_search_check(app: FastAPI) -> bool:
    if not app.state.settings.search.enabled:
        return True
    if app.state.import_shutting_down:
        _mark_knowledge_search_unavailable(app)
        return False
    async with app.state.knowledge_search_probe_lock:
        if app.state.import_shutting_down:
            _mark_knowledge_search_unavailable(app)
            return False
        ready = await _bounded_component_probe(
            app.state.knowledge_search_probe_inflight,
            "knowledge_search",
            app.state.knowledge_search_runtime,
            timeout_seconds=app.state.settings.search.readiness_probe_timeout_seconds,
        )
        if app.state.import_shutting_down or not ready:
            _mark_knowledge_search_unavailable(app)
            return False
        runtime = app.state.knowledge_search_runtime
        service = getattr(runtime, "service", None)
        if service is None:
            _mark_knowledge_search_unavailable(app)
            return False
        app.state.knowledge_search = service
        app.state.knowledge_search_ready = True
        app.state.knowledge_search_ready_until = time.monotonic() + app.state.settings.search.readiness_ttl_seconds
        app.state.knowledge_search_checks = {"knowledge_search": "ok"}
        return True


async def _knowledge_search_probe_loop(app: FastAPI, interval_seconds: float) -> None:
    while True:
        await anyio.sleep(interval_seconds)
        await _run_knowledge_search_check(app)


def _mark_chat_unavailable(app: FastAPI) -> None:
    app.state.chat_history_repository = None
    app.state.session_event_broker = None
    app.state.chat_ready = False
    app.state.chat_ready_until = 0.0
    app.state.chat_checks = {
        "chat_history": "unavailable",
        "chat_events": "unavailable",
    }


async def _bounded_event_broker_probe(
    app: FastAPI,
    broker: object,
    *,
    timeout_seconds: float,
) -> bool:
    probe = getattr(broker, "probe", None)
    if not callable(probe) or not inspect.iscoroutinefunction(probe):
        return await _bounded_component_probe(
            app.state.chat_probe_inflight,
            "chat_events",
            broker,
            timeout_seconds=timeout_seconds,
            allow_presence=True,
        )
    try:
        with anyio.fail_after(timeout_seconds):
            result = await probe()
        return result is not False
    except Exception:
        return False


async def _run_chat_check(app: FastAPI) -> bool:
    if not app.state.settings.chat.enabled:
        return True
    if app.state.import_shutting_down:
        _mark_chat_unavailable(app)
        return False
    async with app.state.chat_probe_lock:
        if app.state.import_shutting_down:
            _mark_chat_unavailable(app)
            return False
        timeout = app.state.settings.chat.readiness_probe_timeout_seconds
        history_ready = await _bounded_component_probe(
            app.state.chat_probe_inflight,
            "chat_history",
            app.state.import_coordinator,
            timeout_seconds=timeout,
        )
        events_ready = await _bounded_event_broker_probe(
            app,
            app.state.session_event_broker_candidate,
            timeout_seconds=timeout,
        )
        if app.state.import_shutting_down or not history_ready or not events_ready:
            _mark_chat_unavailable(app)
            return False
        repository = app.state.chat_history_repository_candidate
        broker = app.state.session_event_broker_candidate
        if repository is None or broker is None:
            _mark_chat_unavailable(app)
            return False
        app.state.chat_history_repository = repository
        app.state.session_event_broker = broker
        app.state.chat_ready = True
        app.state.chat_ready_until = time.monotonic() + app.state.settings.chat.readiness_ttl_seconds
        app.state.chat_checks = {
            "chat_history": "ok",
            "chat_events": "ok",
        }
        return True


async def _chat_probe_loop(app: FastAPI, interval_seconds: float) -> None:
    while True:
        await anyio.sleep(interval_seconds)
        await _run_chat_check(app)


async def _bounded_close_component(
    component: object,
    *,
    name: str,
    timeout_seconds: float,
) -> None:
    close = getattr(component, "close", None)
    if not callable(close):
        return
    with anyio.move_on_after(max(0.01, timeout_seconds)) as close_scope:
        try:
            await anyio.to_thread.run_sync(
                close,
                abandon_on_cancel=True,
            )
        except Exception as close_error:
            logger.bind(
                component=name,
                error_type=type(close_error).__name__,
            ).warning("Import runtime component close failed")
            return
    if close_scope.cancel_called:
        logger.bind(component=name).warning("Import runtime component close timed out")


async def _bounded_aclose_component(
    component: object,
    *,
    name: str,
    timeout_seconds: float,
) -> None:
    close = getattr(component, "aclose", None)
    if not callable(close):
        return
    with anyio.move_on_after(max(0.01, timeout_seconds)) as close_scope:
        try:
            await close()
        except Exception as close_error:
            logger.bind(
                component=name,
                error_type=type(close_error).__name__,
            ).warning("Async runtime component close failed")
            return
    if close_scope.cancel_called:
        logger.bind(component=name).warning("Async runtime component close timed out")


class _StorageProbe:
    def __init__(self, candidate: MinioStore) -> None:
        self.candidate = candidate
        self.completion = Event()
        self.succeeded = False
        self._state_lock = Lock()
        self._completed = False
        self._shutdown_requested = False

    def run(self) -> None:
        succeeded = False
        try:
            self.candidate.require_ready()
        except Exception:
            succeeded = False
        else:
            succeeded = True
        finally:
            with self._state_lock:
                self.succeeded = succeeded
                self._completed = True
                close_after_probe = self._shutdown_requested
            if close_after_probe:
                try:
                    self.candidate.close()
                except Exception as close_error:
                    logger.bind(
                        component="minio_store",
                        error_type=type(close_error).__name__,
                    ).warning("Object storage close after readiness probe failed")
            self.completion.set()

    def request_shutdown(self) -> bool:
        """Return true when the lifespan owns close; otherwise the probe does."""
        with self._state_lock:
            self._shutdown_requested = True
            return self._completed


class _ComponentProbe:
    """Run one component probe without allowing timed-out probes to stack."""

    def __init__(self, candidate: object, *, allow_presence: bool) -> None:
        self.candidate = candidate
        self.allow_presence = allow_presence
        self.completion = Event()
        self.succeeded = False

    def run(self) -> None:
        try:
            self.succeeded = _probe_component(
                self.candidate,
                allow_presence=self.allow_presence,
            )
        except Exception:
            self.succeeded = False
        finally:
            self.completion.set()


async def _run_storage_check(app: FastAPI) -> bool:
    async with app.state.minio_probe_lock:
        candidate = getattr(app.state, "minio_candidate", None)
        if candidate is None:
            _mark_storage_unavailable(app)
            return False

        inflight = app.state.minio_probe_inflight
        if inflight is not None and not inflight.completion.is_set():
            _mark_storage_unavailable(app)
            return False
        app.state.minio_probe_inflight = None
        probe = _StorageProbe(candidate)
        app.state.minio_probe_inflight = probe
        try:
            Thread(
                target=probe.run,
                name="minio-readiness-probe",
                daemon=True,
            ).start()
        except Exception:
            app.state.minio_probe_inflight = None
            _mark_storage_unavailable(app)
            return False

        completed = await anyio.to_thread.run_sync(
            probe.completion.wait,
            app.state.minio_probe_timeout_seconds,
            abandon_on_cancel=True,
        )
        if not completed:
            _mark_storage_unavailable(app)
            return False

        if app.state.minio_probe_inflight is probe:
            app.state.minio_probe_inflight = None
        ready = probe.succeeded

        if ready:
            _mark_storage_ready(app, candidate)
        else:
            _mark_storage_unavailable(app)
        return ready


async def _storage_probe_loop(app: FastAPI, interval_seconds: float) -> None:
    while True:
        await anyio.sleep(interval_seconds)
        await _run_storage_check(app)


def _readiness_snapshot(
    app: FastAPI,
    settings: Settings,
) -> tuple[bool, dict[str, str]]:
    storage_ready = object_storage_is_ready(app.state)
    checks = {"minio": "ok" if storage_ready else "unavailable"}
    import_ready = True
    if settings.api.import_enabled:
        ready_until = app.state.import_ready_until
        import_ready = (
            app.state.import_ready is True
            and isinstance(ready_until, (int, float))
            and not isinstance(ready_until, bool)
            and math.isfinite(ready_until)
            and time.monotonic() < ready_until
        )
        checks.update(app.state.import_checks)
    search_ready = True
    if settings.search.enabled:
        ready_until = app.state.knowledge_search_ready_until
        search_ready = (
            app.state.knowledge_search_ready is True
            and isinstance(ready_until, (int, float))
            and not isinstance(ready_until, bool)
            and math.isfinite(ready_until)
            and time.monotonic() < ready_until
        )
        checks.update(app.state.knowledge_search_checks)
    chat_ready = True
    if settings.chat.enabled:
        ready_until = app.state.chat_ready_until
        chat_ready = (
            app.state.chat_ready is True
            and isinstance(ready_until, (int, float))
            and not isinstance(ready_until, bool)
            and math.isfinite(ready_until)
            and time.monotonic() < ready_until
        )
        checks.update(app.state.chat_checks)
    return (
        storage_ready and import_ready and search_ready and chat_ready,
        checks,
    )


def create_app(
    settings: Settings | None = None,
    *,
    minio_builder: MinioStoreBuilder | None = None,
    import_registry_factory: ImportRegistryFactory | None = None,
    import_coordinator_factory: ImportCoordinatorFactory | None = None,
    import_processor_builder: ImportProcessorBuilder | None | object = (_DEFAULT_IMPORT_PROCESSOR_BUILDER),
    import_workspace_builder: ImportWorkspaceBuilder | None = None,
    knowledge_search_runtime_builder: KnowledgeSearchRuntimeBuilder | None | object = (
        _DEFAULT_KNOWLEDGE_SEARCH_RUNTIME_BUILDER
    ),
    session_event_broker_builder: SessionEventBrokerBuilder | None | object = (_DEFAULT_SESSION_EVENT_BROKER_BUILDER),
    agent_chat_backend_builder: AgentChatBackendBuilder | None | object = (_DEFAULT_AGENT_CHAT_BACKEND_BUILDER),
) -> FastAPI:
    """Build one API application without import-time dependency I/O."""
    runtime_settings = settings if settings is not None else get_settings()
    storage_builder = minio_builder or build_minio_store
    registry_factory = import_registry_factory or InMemoryImportTaskRegistry
    workspace_builder = import_workspace_builder or build_import_workspace
    default_processor_builder = import_processor_builder is _DEFAULT_IMPORT_PROCESSOR_BUILDER
    if default_processor_builder:
        processor_builder: ImportProcessorBuilder | None = build_enterprise_import_processor
    elif import_processor_builder is None:
        processor_builder = None
    else:
        processor_builder = cast(ImportProcessorBuilder, import_processor_builder)
    default_search_builder = knowledge_search_runtime_builder is _DEFAULT_KNOWLEDGE_SEARCH_RUNTIME_BUILDER
    if default_search_builder:
        search_runtime_builder: KnowledgeSearchRuntimeBuilder | None = build_knowledge_search_runtime
    elif knowledge_search_runtime_builder is None:
        search_runtime_builder = None
    else:
        search_runtime_builder = cast(
            KnowledgeSearchRuntimeBuilder,
            knowledge_search_runtime_builder,
        )
    if session_event_broker_builder is _DEFAULT_SESSION_EVENT_BROKER_BUILDER:
        event_broker_builder: SessionEventBrokerBuilder | None = _build_default_session_event_broker
    elif session_event_broker_builder is None:
        event_broker_builder = None
    else:
        event_broker_builder = cast(
            SessionEventBrokerBuilder,
            session_event_broker_builder,
        )
    if agent_chat_backend_builder is _DEFAULT_AGENT_CHAT_BACKEND_BUILDER:
        agent_backend_builder: AgentChatBackendBuilder | None = cast(
            AgentChatBackendBuilder,
            build_agent_chat_backend,
        )
    elif agent_chat_backend_builder is None:
        agent_backend_builder = None
    else:
        agent_backend_builder = cast(
            AgentChatBackendBuilder,
            agent_chat_backend_builder,
        )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configure_logging(runtime_settings.logging)
        application.state.import_shutting_down = False
        _mark_storage_unavailable(application)
        _mark_import_unavailable(application)
        _mark_knowledge_search_unavailable(application)
        _mark_chat_unavailable(application)
        application.state.minio_probe_lock = anyio.Lock()
        application.state.import_probe_lock = anyio.Lock()
        application.state.import_probe_inflight = {}
        application.state.knowledge_search_probe_lock = anyio.Lock()
        application.state.knowledge_search_probe_inflight = {}
        application.state.chat_probe_lock = anyio.Lock()
        application.state.chat_probe_inflight = {}
        try:
            coordinator_required = (
                runtime_settings.api.import_enabled or runtime_settings.search.enabled or runtime_settings.chat.enabled
            )
            coordinator_configuration_failed = False
            if coordinator_required:
                try:
                    coordinator = (
                        import_coordinator_factory()
                        if import_coordinator_factory is not None
                        else _build_default_import_coordinator(
                            runtime_settings,
                            registry_factory,
                        )
                    )
                    application.state.import_coordinator = coordinator
                except Exception:
                    coordinator_configuration_failed = True
            if coordinator_configuration_failed:
                logger.error("Publication registry configuration is invalid")
                message = (
                    "Import runtime is not configured"
                    if runtime_settings.api.import_enabled
                    else "Knowledge search runtime is not configured"
                )
                raise DependencyError(message)

            search_configuration_failed = False
            if runtime_settings.search.enabled:
                if search_runtime_builder is None:
                    search_configuration_failed = True
                else:
                    coordinator = application.state.import_coordinator
                    visibility = getattr(coordinator, "completed_import_visibility", None)
                    visibility_probe = getattr(coordinator, "probe", None)
                    if not callable(getattr(visibility, "get_completed_versions", None)) or not callable(
                        visibility_probe
                    ):
                        search_configuration_failed = True
                    else:
                        try:
                            search_runtime = search_runtime_builder(
                                runtime_settings,
                                cast(CompletedImportVisibilityReader, visibility),
                                visibility_probe,
                            )
                            application.state.knowledge_search_runtime = search_runtime
                        except Exception:
                            search_configuration_failed = True
                search_runtime = application.state.knowledge_search_runtime
                if (
                    getattr(search_runtime, "tenant_safe", False) is not True
                    or getattr(search_runtime, "authoritative_visibility", False) is not True
                    or not callable(getattr(search_runtime, "probe", None))
                    or not callable(getattr(search_runtime, "close", None))
                    or getattr(search_runtime, "service", None) is None
                    or getattr(application.state.import_coordinator, "durable", False) is not True
                ):
                    search_configuration_failed = True
            if search_configuration_failed:
                logger.error("Knowledge search runtime configuration is invalid")
                raise DependencyError("Knowledge search runtime is not configured")

            chat_configuration_failed = False
            if runtime_settings.chat.enabled:
                repository = getattr(
                    application.state.import_coordinator,
                    "chat_history_repository",
                    None,
                )
                if event_broker_builder is None or repository is None:
                    chat_configuration_failed = True
                else:
                    try:
                        broker = event_broker_builder(runtime_settings)
                        application.state.chat_history_repository_candidate = repository
                        application.state.session_event_broker_candidate = broker
                    except Exception:
                        chat_configuration_failed = True
                if (
                    getattr(repository, "durable", False) is not True
                    or not callable(getattr(repository, "create_session", None))
                    or not callable(getattr(repository, "append_message", None))
                    or not callable(getattr(application.state.session_event_broker_candidate, "subscribe", None))
                    or (
                        runtime_settings.api.environment is AppEnvironment.PRODUCTION
                        and getattr(
                            application.state.session_event_broker_candidate,
                            "distributed",
                            False,
                        )
                        is not True
                    )
                ):
                    chat_configuration_failed = True
            if chat_configuration_failed:
                logger.error("Chat runtime configuration is invalid")
                raise DependencyError("Chat runtime is not configured")

            application.state.agent_chat_backend = None
            if runtime_settings.chat.enabled and runtime_settings.search.enabled and agent_backend_builder is not None:
                try:
                    application.state.agent_chat_backend = agent_backend_builder(runtime_settings)
                except Exception:
                    logger.warning("Agent response backend is not configured; chat history remains available")

            import_configuration_failed = False
            if runtime_settings.api.import_enabled:
                if processor_builder is None:
                    import_configuration_failed = True
                else:
                    try:
                        coordinator = application.state.import_coordinator
                        application.state.import_admissions = coordinator.admissions
                        application.state.import_registry = coordinator.tasks
                        application.state.import_workspace = workspace_builder(runtime_settings)
                        if default_processor_builder and default_search_builder and runtime_settings.search.enabled:
                            shared_runtime = cast(
                                ManagedKnowledgeSearchRuntime,
                                application.state.knowledge_search_runtime,
                            )
                            application.state.import_processor = build_enterprise_import_processor(
                                runtime_settings,
                                embedding_backend_builder=(lambda _settings: shared_runtime.embedding_backend),
                                milvus_runtime_builder=(lambda _settings, _dimension: shared_runtime.milvus_runtime),
                                manage_embedding_backend=False,
                                manage_milvus_runtime=False,
                            )
                        else:
                            application.state.import_processor = processor_builder(runtime_settings)
                    except Exception:
                        import_configuration_failed = True
                if runtime_settings.api.environment is AppEnvironment.PRODUCTION and (
                    getattr(application.state.import_registry, "durable", False) is not True
                    or getattr(application.state.import_coordinator, "durable", False) is not True
                    or getattr(application.state.import_processor, "idempotent_commits", False) is not True
                    or getattr(application.state.import_processor, "fenced_commits", False) is not True
                    or not callable(getattr(application.state.import_processor, "probe", None))
                    or not runtime_settings.import_runtime.worker_enabled
                ):
                    import_configuration_failed = True
            if import_configuration_failed:
                logger.error("Import runtime configuration is invalid")
                raise DependencyError("Import runtime is not configured")

            storage_configuration_failed = False
            try:
                application.state.minio_candidate = storage_builder(runtime_settings)
            except Exception:
                application.state.minio_candidate = None
                storage_configuration_failed = True
            if storage_configuration_failed:
                logger.error("Object storage configuration is invalid")
                raise DependencyError("Object storage is not configured")
            if not await _run_storage_check(application):
                logger.warning("Object storage readiness check failed")
            if runtime_settings.api.import_enabled and runtime_settings.import_runtime.worker_enabled:
                candidate = application.state.minio_candidate
                if candidate is None:
                    raise DependencyError("Import runtime is not configured")
                application.state.import_worker = ImportWorker(
                    registry=application.state.import_registry,
                    admissions=application.state.import_admissions,
                    processor=application.state.import_processor,
                    store=candidate,
                    workspace=application.state.import_workspace,
                    config=ImportWorkerConfig(
                        poll_interval_seconds=runtime_settings.import_runtime.poll_interval_seconds,
                        cleanup_poll_interval_seconds=(runtime_settings.import_runtime.cleanup_poll_interval_seconds),
                        lease_seconds=runtime_settings.import_runtime.lease_seconds,
                        heartbeat_seconds=runtime_settings.import_runtime.heartbeat_seconds,
                        max_attempts=runtime_settings.import_runtime.max_attempts,
                        retry_base_seconds=runtime_settings.import_runtime.retry_base_seconds,
                        retry_max_seconds=runtime_settings.import_runtime.retry_max_seconds,
                    ),
                )
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(
                    _storage_probe_loop,
                    application,
                    runtime_settings.minio.readiness_interval_seconds,
                )
                worker = application.state.import_worker
                if worker is not None:
                    task_group.start_soon(worker.run)
                    await worker.wait_started()
                if runtime_settings.api.import_enabled:
                    await _run_import_check(application)
                    task_group.start_soon(
                        _import_probe_loop,
                        application,
                        application.state.import_probe_interval_seconds,
                    )
                if runtime_settings.search.enabled:
                    await _run_knowledge_search_check(application)
                    task_group.start_soon(
                        _knowledge_search_probe_loop,
                        application,
                        runtime_settings.search.readiness_interval_seconds,
                    )
                if runtime_settings.chat.enabled:
                    await _run_chat_check(application)
                    task_group.start_soon(
                        _chat_probe_loop,
                        application,
                        runtime_settings.chat.readiness_interval_seconds,
                    )
                try:
                    yield
                finally:
                    application.state.import_shutting_down = True
                    _mark_import_unavailable(application)
                    _mark_knowledge_search_unavailable(application)
                    _mark_chat_unavailable(application)
                    if worker is not None:
                        worker.request_stop()
                        with anyio.move_on_after(
                            runtime_settings.import_runtime.shutdown_grace_seconds
                        ) as worker_shutdown_scope:
                            await worker.wait_stopped()
                        if worker_shutdown_scope.cancel_called:
                            logger.bind(grace_seconds=(runtime_settings.import_runtime.shutdown_grace_seconds)).warning(
                                "Import worker grace period expired; abandoning active sync work"
                            )
                    task_group.cancel_scope.cancel()
        finally:
            candidate = application.state.minio_candidate
            inflight = application.state.minio_probe_inflight
            application.state.minio_candidate = None
            _mark_storage_unavailable(application)
            close_in_lifespan = inflight is None or inflight.request_shutdown()
            if candidate is not None and close_in_lifespan:
                await _bounded_close_component(
                    candidate,
                    name="minio_store",
                    timeout_seconds=runtime_settings.import_runtime.shutdown_grace_seconds,
                )
            application.state.minio_probe_lock = None
            application.state.minio_probe_inflight = None
            processor = application.state.import_processor
            search_runtime = application.state.knowledge_search_runtime
            event_broker = application.state.session_event_broker_candidate
            agent_backend = application.state.agent_chat_backend
            coordinator = application.state.import_coordinator
            application.state.import_processor = None
            application.state.import_registry = None
            application.state.import_admissions = None
            application.state.import_coordinator = None
            application.state.import_workspace = None
            application.state.import_worker = None
            application.state.import_probe_lock = None
            application.state.import_probe_inflight = None
            application.state.knowledge_search = None
            application.state.knowledge_search_runtime = None
            application.state.knowledge_search_probe_lock = None
            application.state.knowledge_search_probe_inflight = None
            application.state.chat_history_repository = None
            application.state.chat_history_repository_candidate = None
            application.state.session_event_broker = None
            application.state.session_event_broker_candidate = None
            application.state.agent_chat_backend = None
            application.state.chat_probe_lock = None
            application.state.chat_probe_inflight = None
            await _bounded_aclose_component(
                event_broker,
                name="session_event_broker",
                timeout_seconds=runtime_settings.import_runtime.shutdown_grace_seconds,
            )
            await _bounded_aclose_component(
                agent_backend,
                name="agent_chat_backend",
                timeout_seconds=runtime_settings.import_runtime.shutdown_grace_seconds,
            )
            for name, component in (
                ("import_processor", processor),
                ("knowledge_search_runtime", search_runtime),
                ("import_coordinator", coordinator),
            ):
                await _bounded_close_component(
                    component,
                    name=name,
                    timeout_seconds=runtime_settings.import_runtime.shutdown_grace_seconds,
                )

    application_metrics: ApplicationMetrics | None = None
    if runtime_settings.observability.metrics_enabled:
        application_metrics = build_application_metrics()

    app = FastAPI(
        title="Knowledge Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = runtime_settings
    app.state.application_metrics = application_metrics
    app.state.minio_candidate = None
    app.state.minio_store = None
    app.state.minio_ready = False
    app.state.minio_ready_until = 0.0
    app.state.minio_probe_lock = None
    app.state.minio_probe_inflight = None
    app.state.minio_probe_timeout_seconds = runtime_settings.minio.readiness_probe_timeout_seconds
    app.state.minio_readiness_ttl_seconds = runtime_settings.minio.readiness_ttl_seconds
    app.state.import_registry = None
    app.state.import_admissions = None
    app.state.import_coordinator = None
    app.state.import_processor = None
    app.state.import_workspace = None
    app.state.import_worker = None
    app.state.import_probe_lock = None
    app.state.import_probe_inflight = None
    app.state.import_ready = False
    app.state.import_ready_until = 0.0
    app.state.import_checks = {}
    app.state.import_shutting_down = False
    app.state.import_probe_interval_seconds = max(
        1.0,
        min(10.0, runtime_settings.import_runtime.poll_interval_seconds * 2),
    )
    app.state.import_readiness_ttl_seconds = (
        app.state.import_probe_interval_seconds + runtime_settings.mongo.health_timeout_seconds * 4
    )
    app.state.knowledge_search = None
    app.state.knowledge_search_runtime = None
    app.state.knowledge_search_probe_lock = None
    app.state.knowledge_search_probe_inflight = None
    app.state.knowledge_search_ready = False
    app.state.knowledge_search_ready_until = 0.0
    app.state.knowledge_search_checks = {}
    app.state.chat_history_repository = None
    app.state.chat_history_repository_candidate = None
    app.state.session_event_broker = None
    app.state.session_event_broker_candidate = None
    app.state.agent_chat_backend = None
    app.state.chat_probe_lock = None
    app.state.chat_probe_inflight = None
    app.state.chat_ready = False
    app.state.chat_ready_until = 0.0
    app.state.chat_checks = {}

    if runtime_settings.api.import_enabled:
        request_overhead_bytes = 1024 * 1024 + runtime_settings.api.max_upload_files * 64 * 1024
        app.add_middleware(
            RequestBodyLimitMiddleware,
            max_body_bytes=runtime_settings.api.max_upload_batch_bytes + request_overhead_bytes,
        )
    if runtime_settings.api.import_enabled or runtime_settings.search.enabled or runtime_settings.chat.enabled:
        app.add_middleware(ApiAuthenticationMiddleware, settings=runtime_settings)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Authorization",
            "Content-Type",
            "Idempotency-Key",
            "Last-Event-ID",
            "X-Request-ID",
        ],
        expose_headers=["X-Request-ID"],
    )
    if application_metrics is not None:
        app.add_middleware(
            PrometheusMetricsMiddleware,
            metrics=application_metrics,
            metrics_path=runtime_settings.observability.metrics_path,
        )
    register_exception_handlers(app)
    if runtime_settings.api.import_enabled:
        app.include_router(imports_router, prefix="/api/v1")
    if runtime_settings.search.enabled:
        app.include_router(search_router, prefix="/api/v1")
    if runtime_settings.chat.enabled:
        app.include_router(chat_router, prefix="/api/v1")

    @app.get(
        "/health/live",
        response_model=HealthResponse,
        tags=["health"],
        summary="Process liveness",
    )
    async def liveness() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    async def legacy_health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
        tags=["health"],
        summary="Dependency readiness",
    )
    async def readiness(response: Response) -> HealthResponse:
        ready, checks = _readiness_snapshot(app, runtime_settings)
        if application_metrics is not None:
            application_metrics.record_readiness(checks)
        if ready:
            return HealthResponse(status="ready", checks=checks)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="not_ready",
            checks=checks,
        )

    if application_metrics is not None:
        metrics_token = runtime_settings.observability.metrics_auth_token
        expected_metrics_token = metrics_token.get_secret_value() if metrics_token is not None else None

        @app.get(
            runtime_settings.observability.metrics_path,
            include_in_schema=False,
        )
        async def metrics(request: Request) -> Response:
            if not metrics_authorized(
                request.headers.get("Authorization"),
                expected_metrics_token,
            ):
                return Response(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    headers={"WWW-Authenticate": "Bearer"},
                )
            _, checks = _readiness_snapshot(app, runtime_settings)
            application_metrics.record_readiness(checks)
            return Response(
                content=render_metrics(application_metrics),
                headers={"Content-Type": prometheus_content_type()},
            )

    console_connect_sources = "'self'"
    if runtime_settings.auth.mode == "jwt" and runtime_settings.auth.issuer:
        parsed_issuer = urlsplit(runtime_settings.auth.issuer)
        if (
            parsed_issuer.scheme in {"http", "https"}
            and parsed_issuer.netloc
            and parsed_issuer.username is None
            and parsed_issuer.password is None
        ):
            console_connect_sources += f" {parsed_issuer.scheme}://{parsed_issuer.netloc}"
    console_security_headers = {
        "Cache-Control": "no-store",
        "Content-Security-Policy": (
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
            "form-action 'self'; img-src 'self' data:; script-src 'self'; "
            f"style-src 'self'; connect-src {console_connect_sources}"
        ),
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }

    @app.get("/console", include_in_schema=False)
    @app.get("/console/", include_in_schema=False)
    async def web_console() -> FileResponse:
        return FileResponse(
            _CONSOLE_DIRECTORY / "index.html",
            media_type="text/html; charset=utf-8",
            headers=console_security_headers,
        )

    @app.get(
        "/console/auth-config",
        response_model=ConsoleAuthConfigResponse,
        include_in_schema=False,
    )
    async def console_auth_config(response: Response) -> ConsoleAuthConfigResponse:
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        auth = runtime_settings.auth
        if auth.mode == "jwt":
            return ConsoleAuthConfigResponse(
                mode="oidc",
                issuer=auth.issuer,
                client_id=auth.oidc_client_id,
            )
        return ConsoleAuthConfigResponse(mode="disabled")

    @app.get("/console/assets/{asset_name}", include_in_schema=False)
    async def web_console_asset(asset_name: str) -> FileResponse:
        if asset_name not in _CONSOLE_ASSETS:
            raise AppError(ErrorCode.RESOURCE_NOT_FOUND, "Resource not found", 404)
        media_type = "text/css" if asset_name.endswith(".css") else "text/javascript"
        return FileResponse(
            _CONSOLE_DIRECTORY / asset_name,
            media_type=media_type,
            headers={
                **console_security_headers,
                "Cache-Control": "no-cache, must-revalidate",
            },
        )

    @app.get("/", include_in_schema=False)
    async def service_metadata() -> dict[str, str]:
        return {
            "service": "knowledge-agent",
            "health": "/health/live",
            "readiness": "/health/ready",
            "docs": "/docs",
            "console": "/console",
        }

    return app
