import uuid
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, SecretStr, field_validator

from app.core.errors import AppError, ErrorCode, register_exception_handlers
from app.core.logger import configure_logging, logger
from app.core.request_context import RequestContextMiddleware, get_request_id
from app.core.settings import LoggingSettings


class SensitiveDocument(BaseModel):
    document: str

    @field_validator("document")
    @classmethod
    def reject_document(cls, value: str) -> str:
        raise ValueError(f"rejected document body: {value}")


class SensitiveMapping(BaseModel):
    documents: dict[int, str]


def build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise AppError(ErrorCode.RESOURCE_NOT_FOUND, "missing", 404)

    @app.get("/request-id")
    async def request_id() -> dict[str, str]:
        return {"request_id": get_request_id()}

    @app.get("/legacy-missing")
    async def legacy_missing() -> None:
        raise HTTPException(status_code=404, detail="missing")

    @app.get("/legacy-secret")
    async def legacy_secret() -> None:
        raise HTTPException(
            status_code=400,
            detail="internal-document-body",
            headers={"Authorization": "Bearer private-header"},
        )

    @app.get("/validated")
    async def validated(limit: int) -> dict[str, int]:
        return {"limit": limit}

    @app.post("/validated-document")
    async def validated_document(payload: SensitiveDocument) -> dict[str, str]:
        return {"document": payload.document}

    @app.post("/validated-mapping")
    async def validated_mapping(payload: SensitiveMapping) -> dict[str, int]:
        return {"count": len(payload.documents)}

    @app.get("/secret-details")
    async def secret_details() -> None:
        raise AppError(
            ErrorCode.REQUEST_INVALID,
            "invalid request",
            400,
            {"credential": SecretStr("credential-must-not-leak")},
        )

    @app.get("/unexpected")
    async def unexpected() -> None:
        raise RuntimeError("prompt-and-document-body-must-not-leak")

    @app.get("/log")
    async def write_log() -> dict[str, str]:
        logger.info("safe event")
        return {"status": "ok"}

    return app


def test_error_response_contains_request_id() -> None:
    response = TestClient(build_app()).get("/boom", headers={"X-Request-ID": "req-123"})
    assert response.status_code == 404
    assert response.json() == {
        "code": "RESOURCE_NOT_FOUND",
        "message": "missing",
        "request_id": "req-123",
        "details": {},
    }


def test_invalid_request_id_is_replaced() -> None:
    response = TestClient(build_app()).get("/request-id", headers={"X-Request-ID": "bad value with spaces"})
    assert response.status_code == 200
    assert response.json()["request_id"] != "bad value with spaces"
    assert response.headers["X-Request-ID"] == response.json()["request_id"]


def test_http_exception_uses_stable_error_contract() -> None:
    response = TestClient(build_app()).get("/legacy-missing")
    assert response.status_code == 404
    assert response.json()["code"] == "RESOURCE_NOT_FOUND"
    assert "request_id" in response.json()


def test_http_exception_does_not_expose_detail_or_headers() -> None:
    response = TestClient(build_app()).get("/legacy-secret", headers={"X-Request-ID": "req-http"})

    assert response.status_code == 400
    assert response.json() == {
        "code": "REQUEST_INVALID",
        "message": "Request failed",
        "request_id": "req-http",
        "details": {},
    }
    assert response.headers["X-Request-ID"] == "req-http"
    assert "internal-document-body" not in response.text
    assert "private-header" not in response.text


def test_validation_error_omits_rejected_input() -> None:
    rejected_input = "private-prompt-text"
    response = TestClient(build_app()).get(
        "/validated",
        params={"limit": rejected_input},
        headers={"X-Request-ID": "req-validation"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "REQUEST_INVALID"
    assert payload["message"] == "Request validation failed"
    assert payload["request_id"] == "req-validation"
    assert payload["details"]["errors"][0]["location"] == ["query"]
    assert rejected_input not in response.text
    assert response.headers["X-Request-ID"] == "req-validation"


def test_validation_error_omits_sensitive_custom_validator_message() -> None:
    document_body = "private-prompt-and-document-body"
    response = TestClient(build_app()).post(
        "/validated-document",
        json={"document": document_body},
        headers={"X-Request-ID": "req-custom-validation"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "REQUEST_INVALID",
        "message": "Request validation failed",
        "request_id": "req-custom-validation",
        "details": {
            "errors": [
                {
                    "location": ["body"],
                    "message": "Invalid value",
                    "type": "value_error",
                }
            ]
        },
    }
    assert document_body not in response.text
    assert response.headers["X-Request-ID"] == "req-custom-validation"


def test_validation_error_omits_sensitive_dynamic_mapping_key() -> None:
    sensitive_key = "private-prompt-and-document-body"
    response = TestClient(build_app()).post(
        "/validated-mapping",
        json={"documents": {sensitive_key: "value"}},
        headers={"X-Request-ID": "req-mapping-validation"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "REQUEST_INVALID",
        "message": "Request validation failed",
        "request_id": "req-mapping-validation",
        "details": {
            "errors": [
                {
                    "location": ["body"],
                    "message": "Invalid value",
                    "type": "int_parsing",
                }
            ]
        },
    }
    assert sensitive_key not in response.text
    assert response.headers["X-Request-ID"] == "req-mapping-validation"


def test_secret_str_in_app_error_details_is_masked() -> None:
    response = TestClient(build_app(), raise_server_exceptions=False).get(
        "/secret-details",
        headers={"X-Request-ID": "req-secret"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "code": "REQUEST_INVALID",
        "message": "invalid request",
        "request_id": "req-secret",
        "details": {"credential": "**********"},
    }
    assert "credential-must-not-leak" not in response.text
    assert response.headers["X-Request-ID"] == "req-secret"


def test_unhandled_exception_has_stable_response_and_safe_log() -> None:
    messages: list[str] = []
    handler_id = logger.add(messages.append, format="{extra[request_id]}|{message}")
    try:
        response = TestClient(build_app(), raise_server_exceptions=False).get(
            "/unexpected",
            headers={"X-Request-ID": "req-unexpected"},
        )
    finally:
        logger.remove(handler_id)

    assert response.status_code == 500
    assert response.json() == {
        "code": "INTERNAL_ERROR",
        "message": "Internal server error",
        "request_id": "req-unexpected",
        "details": {},
    }
    assert response.headers["X-Request-ID"] == "req-unexpected"
    rendered_logs = "".join(messages)
    assert "Unhandled request error" in rendered_logs
    assert "prompt-and-document-body-must-not-leak" not in rendered_logs
    assert "req-unexpected" in rendered_logs


def test_unhandled_exception_is_contained_before_server_error_middleware() -> None:
    messages: list[str] = []
    handler_id = logger.add(messages.append, format="{extra[request_id]}|{message}")
    try:
        response = TestClient(build_app()).get(
            "/unexpected",
            headers={"X-Request-ID": "req-contained"},
        )
    finally:
        logger.remove(handler_id)

    assert response.status_code == 500
    assert response.json() == {
        "code": "INTERNAL_ERROR",
        "message": "Internal server error",
        "request_id": "req-contained",
        "details": {},
    }
    assert response.headers["X-Request-ID"] == "req-contained"
    rendered_logs = "".join(messages)
    assert rendered_logs.count("Unhandled request error") == 1
    assert "prompt-and-document-body-must-not-leak" not in rendered_logs
    assert "req-contained" in rendered_logs


@pytest.mark.asyncio
async def test_request_context_is_reset_after_response() -> None:
    transport = httpx.ASGITransport(app=build_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/request-id", headers={"X-Request-ID": "req-context"})

    assert response.status_code == 200
    assert response.json()["request_id"] == "req-context"
    assert get_request_id() == ""


@pytest.mark.asyncio
async def test_request_context_is_reset_after_contained_exception() -> None:
    transport = httpx.ASGITransport(app=build_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/unexpected", headers={"X-Request-ID": "req-contained-async"})

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-contained-async"
    assert get_request_id() == ""


def test_request_id_length_limit_generates_uuid() -> None:
    supplied = "a" * 129
    response = TestClient(build_app()).get("/request-id", headers={"X-Request-ID": supplied})

    generated = response.json()["request_id"]
    assert generated != supplied
    assert str(uuid.UUID(generated)) == generated
    assert response.headers["X-Request-ID"] == generated


def test_logging_configuration_is_repeatable_and_preserves_bound_request_id() -> None:
    settings = LoggingSettings.model_construct(
        console_enabled=False,
        console_level="INFO",
        file_enabled=False,
        file_level="INFO",
        retention="7 days",
        json_output=False,
        diagnose=False,
    )
    configure_logging(settings)
    configure_logging(settings)
    messages: list[str] = []
    handler_id = logger.add(messages.append, format="{extra[request_id]}|{message}")
    try:
        logger.bind(request_id="req-bound").info("safe event")
    finally:
        logger.remove(handler_id)

    assert messages == ["req-bound|safe event\n"]


def test_logging_configuration_never_enables_diagnose() -> None:
    settings = LoggingSettings.model_construct(
        console_enabled=True,
        console_level="INFO",
        file_enabled=False,
        file_level="INFO",
        retention="7 days",
        json_output=False,
        diagnose=True,
    )

    with patch.object(logger, "add", return_value=1) as add_handler:
        configure_logging(settings)

    assert add_handler.call_count == 1
    assert add_handler.call_args.kwargs["backtrace"] is False
    assert add_handler.call_args.kwargs["diagnose"] is False
