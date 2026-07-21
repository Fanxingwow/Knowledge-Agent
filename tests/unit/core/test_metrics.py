from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.metrics import (
    PrometheusMetricsMiddleware,
    build_application_metrics,
    metrics_authorized,
    render_metrics,
)


def test_metrics_authorization_uses_an_exact_bearer_token() -> None:
    token = "x" * 32

    assert metrics_authorized(None, None) is True
    assert metrics_authorized(f"Bearer {token}", token) is True
    assert metrics_authorized(None, token) is False
    assert metrics_authorized(token, token) is False
    assert metrics_authorized(f"bearer {token}", token) is False
    assert metrics_authorized(f"Bearer {token}x", token) is False


def test_http_metrics_use_route_templates_and_never_raw_unknown_paths() -> None:
    metrics = build_application_metrics()
    app = FastAPI()
    app.add_middleware(
        PrometheusMetricsMiddleware,
        metrics=metrics,
        metrics_path="/metrics",
    )

    @app.get("/items/{item_id}")
    async def item(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    with TestClient(app) as client:
        assert client.get("/items/private-item-123").status_code == 200
        assert client.get("/not-found/private-tenant-456").status_code == 404

    rendered = render_metrics(metrics).decode("utf-8")

    assert 'route="/items/{item_id}"' in rendered
    assert 'route="unmatched"' in rendered
    assert "private-item-123" not in rendered
    assert "private-tenant-456" not in rendered
    assert 'knowledge_agent_http_requests_total{method="GET",route="/items/{item_id}",status="200"} 1.0' in rendered
    assert 'knowledge_agent_http_requests_in_progress{method="GET"} 0.0' in rendered


def test_metrics_endpoint_path_is_excluded_from_http_measurement() -> None:
    metrics = build_application_metrics()
    app = FastAPI()
    app.add_middleware(
        PrometheusMetricsMiddleware,
        metrics=metrics,
        metrics_path="/metrics",
    )

    @app.get("/metrics")
    async def scrape() -> str:
        return "ok"

    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 200

    assert 'route="/metrics"' not in render_metrics(metrics).decode("utf-8")


def test_dependency_readiness_labels_are_sanitized_and_bounded() -> None:
    metrics = build_application_metrics()

    metrics.record_readiness(
        {
            "mongo": "ok",
            "tenant/secret": "unavailable",  # pragma: allowlist secret
        }
    )
    rendered = render_metrics(metrics).decode("utf-8")

    assert 'knowledge_agent_dependency_ready{component="mongo"} 1.0' in rendered
    assert 'knowledge_agent_dependency_ready{component="unknown"} 0.0' in rendered
    assert "tenant/secret" not in rendered
