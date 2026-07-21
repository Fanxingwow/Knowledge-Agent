from __future__ import annotations

import time
from collections.abc import Callable
from threading import Event, Lock
from typing import Annotated, cast

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import app.main as main_module
from app.api.dependencies import get_app_settings, get_object_store
from app.core.errors import DependencyError, ErrorCode
from app.core.security import Principal, get_current_principal, require_roles
from app.core.settings import (
    ApiSettings,
    AppEnvironment,
    AuthSettings,
    LoggingSettings,
    MinioSettings,
    Settings,
)
from app.infrastructure.minio_store import MinioStore
from app.main import create_app


class FakeStore:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.failure: Exception | None = None
        self.require_ready_calls = 0
        self.is_ready_calls = 0
        self.check_event = Event()
        self.check_started = Event()
        self.release_checks = Event()
        self.release_checks.set()
        self._active_lock = Lock()
        self.active_checks = 0
        self.max_active_checks = 0
        self.closed = False
        self.closed_event = Event()

    def require_ready(self) -> None:
        self.require_ready_calls += 1
        self.check_started.set()
        with self._active_lock:
            self.active_checks += 1
            self.max_active_checks = max(self.max_active_checks, self.active_checks)
        try:
            if not self.release_checks.wait(timeout=5):
                raise TimeoutError("test probe was not released")
            if self.failure is not None:
                raise self.failure
            if not self.ready:
                raise DependencyError("private dependency detail")
        finally:
            with self._active_lock:
                self.active_checks -= 1
            self.check_event.set()

    def is_ready(self) -> bool:
        self.is_ready_calls += 1
        if self.failure is not None:
            raise self.failure
        return self.ready

    def close(self) -> None:
        self.closed = True
        self.closed_event.set()


def make_settings(
    *,
    tenant_id: str = "tenant-a",
    cors_origins: str = "https://console.example.com",
    auth_mode: str = "disabled",
    auth_issuer: str | None = None,
    auth_audience: str | None = None,
    oidc_client_id: str | None = None,
    auth_jwks_url: str | None = None,
    readiness_interval_seconds: float = 0.05,
    readiness_probe_timeout_seconds: float = 0.2,
    readiness_ttl_seconds: float = 0.5,
) -> Settings:
    return Settings.model_construct(
        api=ApiSettings.model_construct(
            environment=AppEnvironment.TEST,
            cors_origins_raw=cors_origins,
        ),
        auth=AuthSettings.model_construct(
            mode=auth_mode,
            issuer=auth_issuer,
            audience=auth_audience,
            oidc_client_id=oidc_client_id,
            jwks_url=auth_jwks_url,
            public_key=None,
            algorithms_raw="RS256",
            dev_tenant_id=tenant_id,
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
            readiness_interval_seconds=readiness_interval_seconds,
            readiness_probe_timeout_seconds=readiness_probe_timeout_seconds,
            readiness_ttl_seconds=readiness_ttl_seconds,
        ),
    )


def builder_for(store: FakeStore, calls: list[Settings]) -> Callable[[Settings], MinioStore]:
    def build(settings: Settings) -> MinioStore:
        calls.append(settings)
        return cast(MinioStore, store)

    return build


def test_factory_defers_dependency_initialization_until_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings()
    store = FakeStore()
    build_calls: list[Settings] = []
    logging_calls: list[LoggingSettings] = []
    monkeypatch.setattr(main_module, "configure_logging", logging_calls.append)

    app = create_app(settings=settings, minio_builder=builder_for(store, build_calls))

    assert app.state.settings is settings
    assert build_calls == []
    assert logging_calls == []

    with TestClient(app) as client:
        assert build_calls == [settings]
        assert logging_calls == [settings.logging]
        assert store.require_ready_calls >= 1
        assert client.get("/health/live").status_code == 200

    assert app.state.minio_candidate is None
    assert app.state.minio_store is None
    assert app.state.minio_ready is False
    assert app.state.minio_ready_until == 0.0
    assert app.state.minio_probe_lock is None
    assert app.state.minio_probe_inflight is None
    assert store.closed is True


def test_health_endpoints_report_live_and_ready_without_secrets() -> None:
    store = FakeStore()
    app = create_app(settings=make_settings(), minio_builder=builder_for(store, []))

    with TestClient(app) as client:
        live = client.get("/health/live", headers={"X-Request-ID": "health-live"})
        legacy = client.get("/health")
        ready = client.get("/health/ready", headers={"X-Request-ID": "health-ready"})

    assert live.status_code == 200
    assert live.json() == {"status": "ok", "checks": {}}
    assert live.headers["X-Request-ID"] == "health-live"
    assert legacy.status_code == 200
    assert legacy.json() == {"status": "ok", "checks": {}}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "checks": {"minio": "ok"}}
    assert ready.headers["X-Request-ID"] == "health-ready"
    assert store.is_ready_calls == 0


def test_unready_storage_keeps_process_live_and_returns_generic_readiness() -> None:
    store = FakeStore(ready=False)
    app = create_app(settings=make_settings(), minio_builder=builder_for(store, []))

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.json() == {
        "status": "not_ready",
        "checks": {"minio": "unavailable"},
    }
    assert "private dependency detail" not in ready.text


def test_permanent_builder_failure_aborts_startup_with_sanitized_error() -> None:
    def failing_builder(settings: Settings) -> MinioStore:
        del settings
        raise RuntimeError("MINIO_SECRET_KEY=must-not-leak")

    app = create_app(settings=make_settings(), minio_builder=failing_builder)

    with pytest.raises(DependencyError) as captured:
        with TestClient(app):
            pass

    assert str(captured.value) == "Object storage is not configured"
    assert "must-not-leak" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_runtime_storage_exception_is_sanitized_for_health_and_business() -> None:
    store = FakeStore()
    app = create_app(settings=make_settings(), minio_builder=builder_for(store, []))

    @app.get("/test/storage-error")
    def storage_error(
        resolved: Annotated[MinioStore, Depends(get_object_store)],
    ) -> dict[str, bool]:
        return {"resolved": resolved is store}

    with TestClient(app) as client:
        store.failure = RuntimeError("X-Amz-Signature=must-not-leak")
        store.check_event.clear()
        assert store.check_event.wait(timeout=1)
        ready = client.get("/health/ready")
        business = client.get("/test/storage-error")

    assert ready.status_code == 503
    assert business.status_code == 503
    assert business.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    assert "must-not-leak" not in ready.text
    assert "must-not-leak" not in business.text


def test_object_store_dependency_is_only_available_when_ready() -> None:
    store = FakeStore()
    app = create_app(settings=make_settings(), minio_builder=builder_for(store, []))

    @app.get("/test/storage")
    def storage(
        resolved: Annotated[MinioStore, Depends(get_object_store)],
    ) -> dict[str, bool]:
        return {"same": resolved is store}

    with TestClient(app) as client:
        assert client.get("/test/storage").json() == {"same": True}
        store.ready = False
        store.check_event.clear()
        assert store.check_event.wait(timeout=1)
        assert client.get("/health/ready").status_code == 503
        unavailable = client.get("/test/storage")

    assert unavailable.status_code == 503
    assert unavailable.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value


def test_storage_readiness_recovers_in_background() -> None:
    store = FakeStore(ready=False)
    app = create_app(settings=make_settings(), minio_builder=builder_for(store, []))

    @app.get("/test/recovered-storage")
    def recovered_storage(
        resolved: Annotated[MinioStore, Depends(get_object_store)],
    ) -> dict[str, bool]:
        return {"same": resolved is store}

    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 503
        store.ready = True
        store.check_event.clear()
        assert store.check_event.wait(timeout=1)
        assert client.get("/health/ready").status_code == 200
        assert client.get("/test/recovered-storage").json() == {"same": True}


def test_slow_probe_has_hard_deadline_and_never_stacks_workers() -> None:
    store = FakeStore()
    app = create_app(
        settings=make_settings(
            readiness_interval_seconds=0.02,
            readiness_probe_timeout_seconds=0.05,
            readiness_ttl_seconds=0.2,
        ),
        minio_builder=builder_for(store, []),
    )

    with TestClient(app) as client:
        store.check_started.clear()
        store.release_checks.clear()
        assert store.check_started.wait(timeout=1)
        time.sleep(0.09)
        assert client.get("/health/ready").status_code == 503
        time.sleep(0.08)
        assert store.max_active_checks == 1
        store.release_checks.set()


def test_slow_startup_probe_degrades_readiness_but_not_liveness() -> None:
    store = FakeStore()
    store.release_checks.clear()
    app = create_app(
        settings=make_settings(
            readiness_interval_seconds=0.02,
            readiness_probe_timeout_seconds=0.05,
            readiness_ttl_seconds=0.2,
        ),
        minio_builder=builder_for(store, []),
    )

    with TestClient(app) as client:
        assert store.check_started.wait(timeout=1)
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert store.max_active_checks == 1
        store.release_checks.set()


def test_shutdown_transfers_close_ownership_to_blocked_probe() -> None:
    store = FakeStore()
    store.release_checks.clear()
    app = create_app(
        settings=make_settings(
            readiness_interval_seconds=0.02,
            readiness_probe_timeout_seconds=0.05,
            readiness_ttl_seconds=0.2,
        ),
        minio_builder=builder_for(store, []),
    )

    with TestClient(app) as client:
        assert store.check_started.wait(timeout=1)
        assert client.get("/health/ready").status_code == 503

    assert store.closed is False
    store.release_checks.set()
    assert store.closed_event.wait(timeout=1)
    assert store.closed is True


def test_stale_success_cache_fails_closed_without_waiting_for_probe() -> None:
    store = FakeStore()
    app = create_app(settings=make_settings(), minio_builder=builder_for(store, []))

    @app.get("/test/stale-storage")
    def stale_storage(
        resolved: Annotated[MinioStore, Depends(get_object_store)],
    ) -> dict[str, bool]:
        return {"same": resolved is store}

    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        app.state.minio_ready_until = time.monotonic() - 1
        assert client.get("/health/ready").status_code == 503
        business = client.get("/test/stale-storage")

    assert business.status_code == 503
    assert business.json()["code"] == ErrorCode.DEPENDENCY_UNAVAILABLE.value


def test_factory_settings_are_used_by_auth_and_rbac() -> None:
    app = create_app(
        settings=make_settings(tenant_id="tenant-from-factory"),
        minio_builder=builder_for(FakeStore(), []),
    )

    @app.get("/test/principal")
    def principal(
        current: Annotated[Principal, Depends(get_current_principal)],
    ) -> dict[str, object]:
        return {
            "tenant_id": current.tenant_id,
            "roles": sorted(current.roles),
        }

    @app.get("/test/admin")
    def admin(
        current: Annotated[Principal, Depends(require_roles("admin"))],
    ) -> dict[str, str]:
        return {"tenant_id": current.tenant_id}

    with TestClient(app) as client:
        principal_response = client.get("/test/principal")
        admin_response = client.get("/test/admin")

    assert principal_response.json()["tenant_id"] == "tenant-from-factory"
    assert admin_response.json() == {"tenant_id": "tenant-from-factory"}


def test_two_application_instances_do_not_share_settings() -> None:
    first = create_app(
        settings=make_settings(tenant_id="tenant-one"),
        minio_builder=builder_for(FakeStore(), []),
    )
    second = create_app(
        settings=make_settings(tenant_id="tenant-two"),
        minio_builder=builder_for(FakeStore(), []),
    )

    def add_route(app: FastAPI) -> None:
        @app.get("/tenant")
        def tenant(settings: Annotated[Settings, Depends(get_app_settings)]) -> dict[str, str]:
            return {"tenant_id": settings.auth.dev_tenant_id}

    add_route(first)
    add_route(second)

    with TestClient(first) as first_client, TestClient(second) as second_client:
        assert first_client.get("/tenant").json() == {"tenant_id": "tenant-one"}
        assert second_client.get("/tenant").json() == {"tenant_id": "tenant-two"}


def test_missing_application_settings_fail_closed() -> None:
    app = FastAPI()
    main_module.register_exception_handlers(app)

    @app.get("/settings")
    def settings(value: Annotated[Settings, Depends(get_app_settings)]) -> dict[str, str]:
        return {"tenant": value.auth.dev_tenant_id}

    response = TestClient(app, raise_server_exceptions=False).get("/settings")

    assert response.status_code == 500
    assert response.json()["code"] == ErrorCode.INTERNAL_ERROR.value
    assert "settings" not in response.text.lower()


def test_cors_uses_explicit_allowlist() -> None:
    app = create_app(
        settings=make_settings(cors_origins="https://one.example,https://two.example"),
        minio_builder=builder_for(FakeStore(), []),
    )

    with TestClient(app) as client:
        allowed = client.options(
            "/health/live",
            headers={
                "Origin": "https://two.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        denied = client.options(
            "/health/live",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://two.example"
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_allowed_origin_receives_cors_headers_on_unexpected_error() -> None:
    app = create_app(
        settings=make_settings(cors_origins="https://console.example.com"),
        minio_builder=builder_for(FakeStore(), []),
    )

    @app.get("/test/unexpected")
    def unexpected() -> None:
        raise RuntimeError("sensitive-prompt-must-not-leak")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(
            "/test/unexpected",
            headers={
                "Origin": "https://console.example.com",
                "X-Request-ID": "cors-error",
            },
        )

    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == "https://console.example.com"
    assert response.headers["X-Request-ID"] == "cors-error"
    assert response.json()["code"] == ErrorCode.INTERNAL_ERROR.value
    assert "sensitive-prompt" not in response.text


def test_service_metadata_contains_only_public_paths() -> None:
    app = create_app(settings=make_settings(), minio_builder=builder_for(FakeStore(), []))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "knowledge-agent",
        "health": "/health/live",
        "readiness": "/health/ready",
        "docs": "/docs",
        "console": "/console",
    }


def test_web_console_and_allowlisted_assets_are_served_with_security_headers() -> None:
    app = create_app(settings=make_settings(), minio_builder=builder_for(FakeStore(), []))

    with TestClient(app) as client:
        page = client.get("/console")
        script = client.get("/console/assets/console.js?v=20260720.4")
        stylesheet = client.get("/console/assets/console.css?v=20260720.4")
        hidden = client.get("/console/assets/../main.py")
        auth_config = client.get("/console/auth-config")

    assert page.status_code == 200
    assert "企业智能知识库问答" in page.text
    assert "deleteSessionDialog" in page.text
    assert "console.css?v=20260720.4" in page.text
    assert "console.js?v=20260720.4" in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert script.status_code == 200
    assert script.headers["x-content-type-options"] == "nosniff"
    assert script.headers["cache-control"] == "no-cache, must-revalidate"
    assert 'method: "DELETE"' in script.text
    assert "session-delete" in script.text
    assert 'code_challenge_method: "S256"' in script.text
    assert 'grant_type: "authorization_code"' in script.text
    assert "enterprise.console.access_token" in script.text
    assert "Bearer Token" not in page.text
    assert "登录企业账户" in page.text
    assert stylesheet.status_code == 200
    assert "[hidden]" in stylesheet.text
    assert "display: none !important" in stylesheet.text
    assert hidden.status_code == 404
    assert auth_config.status_code == 200
    assert auth_config.json() == {
        "mode": "disabled",
        "issuer": None,
        "client_id": None,
        "scopes": "openid profile email",
    }
    assert auth_config.headers["cache-control"] == "no-store"


def test_console_exposes_public_oidc_metadata_and_allows_issuer_connections() -> None:
    settings = make_settings(
        auth_mode="jwt",
        auth_issuer="https://login.example.com/realms/enterprise",
        auth_audience="enterprise-console",
        oidc_client_id="enterprise-console",
        auth_jwks_url="http://keycloak:8080/realms/enterprise/certs",
    )
    app = create_app(settings=settings, minio_builder=builder_for(FakeStore(), []))

    with TestClient(app) as client:
        page = client.get("/console")
        config = client.get("/console/auth-config")

    assert config.status_code == 200
    assert config.json() == {
        "mode": "oidc",
        "issuer": "https://login.example.com/realms/enterprise",
        "client_id": "enterprise-console",
        "scopes": "openid profile email",
    }
    assert "connect-src 'self' https://login.example.com" in page.headers["content-security-policy"]


def test_import_api_is_not_advertised_when_capability_is_disabled() -> None:
    app = create_app(settings=make_settings(), minio_builder=builder_for(FakeStore(), []))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/imports",
            files={"files": ("guide.md", b"guide", "text/markdown")},
        )
        schema = client.get("/openapi.json").json()

    assert response.status_code == 404
    assert "/api/v1/imports" not in schema["paths"]
