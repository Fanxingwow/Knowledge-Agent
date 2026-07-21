from __future__ import annotations

import hmac
import time
from dataclasses import dataclass

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_HTTP_METHOD_FALLBACK = "UNKNOWN"
_ROUTE_FALLBACK = "unmatched"
_ALLOWED_METHODS = frozenset(
    {
        "DELETE",
        "GET",
        "HEAD",
        "OPTIONS",
        "PATCH",
        "POST",
        "PUT",
    }
)
_DEFAULT_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)


@dataclass(frozen=True, slots=True)
class ApplicationMetrics:
    registry: CollectorRegistry
    requests_total: Counter
    request_duration_seconds: Histogram
    requests_in_progress: Gauge
    dependency_ready: Gauge

    def record_readiness(self, checks: dict[str, str]) -> None:
        for component, state in checks.items():
            self.dependency_ready.labels(component=_safe_component(component)).set(
                1.0 if state == "ok" else 0.0
            )


def build_application_metrics() -> ApplicationMetrics:
    registry = CollectorRegistry(auto_describe=True)
    return ApplicationMetrics(
        registry=registry,
        requests_total=Counter(
            "knowledge_agent_http_requests",
            "Completed HTTP requests.",
            labelnames=("method", "route", "status"),
            registry=registry,
        ),
        request_duration_seconds=Histogram(
            "knowledge_agent_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            labelnames=("method", "route"),
            buckets=_DEFAULT_BUCKETS,
            registry=registry,
        ),
        requests_in_progress=Gauge(
            "knowledge_agent_http_requests_in_progress",
            "HTTP requests currently being processed.",
            labelnames=("method",),
            registry=registry,
        ),
        dependency_ready=Gauge(
            "knowledge_agent_dependency_ready",
            "Dependency readiness, where 1 is ready and 0 is unavailable.",
            labelnames=("component",),
            registry=registry,
        ),
    )


class PrometheusMetricsMiddleware:
    """Collect bounded-cardinality HTTP metrics without tenant or user labels."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        metrics: ApplicationMetrics,
        metrics_path: str,
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.metrics_path = metrics_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == self.metrics_path:
            await self.app(scope, receive, send)
            return

        method = _safe_method(scope.get("method"))
        started_at = time.perf_counter()
        status_code = 500
        self.metrics.requests_in_progress.labels(method=method).inc()

        async def observe_response(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                candidate = message.get("status")
                if isinstance(candidate, int):
                    status_code = candidate
            await send(message)

        try:
            await self.app(scope, receive, observe_response)
        finally:
            route = _safe_route(scope)
            self.metrics.requests_in_progress.labels(method=method).dec()
            self.metrics.requests_total.labels(
                method=method,
                route=route,
                status=str(status_code),
            ).inc()
            self.metrics.request_duration_seconds.labels(
                method=method,
                route=route,
            ).observe(max(0.0, time.perf_counter() - started_at))


def metrics_authorized(authorization: str | None, expected_token: str | None) -> bool:
    if expected_token is None:
        return True
    if authorization is None:
        return False
    try:
        supplied = authorization.encode("utf-8", errors="strict")
        expected = f"Bearer {expected_token}".encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(supplied, expected)


def prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST


def render_metrics(metrics: ApplicationMetrics) -> bytes:
    return generate_latest(metrics.registry)


def _safe_method(value: object) -> str:
    if isinstance(value, str):
        normalized = value.upper()
        if normalized in _ALLOWED_METHODS:
            return normalized
    return _HTTP_METHOD_FALLBACK


def _safe_route(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or len(path) > 256
        or any(character in path for character in ("\r", "\n", "\x00"))
    ):
        return _ROUTE_FALLBACK
    return path


def _safe_component(value: str) -> str:
    if (
        1 <= len(value) <= 64
        and value[0].isalnum()
        and all(character.isalnum() or character in "._-" for character in value)
    ):
        return value
    return "unknown"
