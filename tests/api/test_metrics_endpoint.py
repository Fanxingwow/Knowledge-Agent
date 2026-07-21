from __future__ import annotations

from collections.abc import Callable
from typing import cast

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.settings import (
    ApiSettings,
    AppEnvironment,
    AuthSettings,
    LoggingSettings,
    MinioSettings,
    ObservabilitySettings,
    Settings,
)
from app.infrastructure.minio_store import MinioStore
from app.main import create_app


class ReadyStore:
    def require_ready(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    def close(self) -> None:
        return None


def _settings(*, token: str | None) -> Settings:
    return Settings.model_construct(
        api=ApiSettings.model_construct(
            environment=AppEnvironment.TEST,
            cors_origins_raw="https://console.example.com",
        ),
        auth=AuthSettings.model_construct(
            mode="disabled",
            issuer=None,
            audience=None,
            jwks_url=None,
            public_key=None,
            algorithms_raw="RS256",
            dev_tenant_id="tenant-test",
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
            readiness_interval_seconds=10.0,
            readiness_probe_timeout_seconds=0.2,
            readiness_ttl_seconds=20.0,
        ),
        observability=ObservabilitySettings.model_construct(
            metrics_enabled=True,
            metrics_path="/internal/metrics",
            metrics_auth_token=SecretStr(token) if token is not None else None,
        ),
    )


def _store_builder() -> Callable[[Settings], MinioStore]:
    store = cast(MinioStore, ReadyStore())
    return lambda _settings: store


def test_metrics_endpoint_requires_configured_token_and_exports_safe_metrics() -> None:
    token = "m" * 32
    app = create_app(settings=_settings(token=token), minio_builder=_store_builder())

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/sensitive/tenant-a/resource-123").status_code == 404
        unauthorized = client.get("/internal/metrics")
        authorized = client.get(
            "/internal/metrics",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert unauthorized.status_code == 401
    assert unauthorized.headers["WWW-Authenticate"] == "Bearer"
    assert authorized.status_code == 200
    assert authorized.headers["content-type"].startswith("text/plain")
    assert "knowledge_agent_http_requests_total" in authorized.text
    assert 'route="/health/live"' in authorized.text
    assert 'route="unmatched"' in authorized.text
    assert 'knowledge_agent_dependency_ready{component="minio"} 1.0' in authorized.text
    assert "tenant-a" not in authorized.text
    assert "resource-123" not in authorized.text
    assert 'route="/internal/metrics"' not in authorized.text


def test_metrics_endpoint_can_be_unprotected_outside_production() -> None:
    app = create_app(settings=_settings(token=None), minio_builder=_store_builder())

    with TestClient(app) as client:
        response = client.get("/internal/metrics")

    assert response.status_code == 200
