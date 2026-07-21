import uuid
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.schemas.common import ErrorResponse
from app.core.logger import logger
from app.core.request_context import get_request_id

_VALIDATION_SOURCES = {
    "body": "body",
    "query": "query",
    "path": "path",
    "header": "header",
    "cookie": "cookie",
}


class ErrorCode(StrEnum):
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    REQUEST_INVALID = "REQUEST_INVALID"
    DOCUMENT_TYPE_UNSUPPORTED = "DOCUMENT_TYPE_UNSUPPORTED"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    TOO_MANY_FILES = "TOO_MANY_FILES"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DomainError(AppError):
    pass


class ValidationError(DomainError):
    def __init__(self, code: ErrorCode, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code, message, 422, details)


class DependencyError(AppError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.DEPENDENCY_UNAVAILABLE, message, 503, details)


class RetryableError(DependencyError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        retry_details = {"retryable": True, **(details or {})}
        super().__init__(message, retry_details)


class AuthorizationError(AppError):
    pass


def _resolve_request_id(request: Request) -> str:
    contextual_request_id = get_request_id()
    if contextual_request_id:
        return contextual_request_id
    stored_request_id = getattr(request.state, "request_id", "")
    if isinstance(stored_request_id, str) and stored_request_id:
        return stored_request_id
    generated_request_id = str(uuid.uuid4())
    request.state.request_id = generated_request_id
    return generated_request_id


def _error_response(
    request: Request,
    *,
    code: ErrorCode,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    resolved_request_id = request_id or _resolve_request_id(request)
    payload = ErrorResponse(
        code=code.value,
        message=message,
        request_id=resolved_request_id,
        details=details or {},
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers={"X-Request-ID": resolved_request_id},
    )


def _internal_error_response(request: Request, *, request_id: str | None = None) -> JSONResponse:
    resolved_request_id = request_id or _resolve_request_id(request)
    logger.bind(request_id=resolved_request_id).error("Unhandled request error")
    return _error_response(
        request,
        code=ErrorCode.INTERNAL_ERROR,
        message="Internal server error",
        status_code=500,
        request_id=resolved_request_id,
    )


def _safe_validation_location(location: tuple[str | int, ...]) -> list[str]:
    if not location or not isinstance(location[0], str):
        return ["request"]
    return [_VALIDATION_SOURCES.get(location[0], "request")]


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return _error_response(
            request,
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "location": _safe_validation_location(item["loc"]),
                "message": "Invalid value",
                "type": item["type"],
            }
            for item in exc.errors()
        ]
        return _error_response(
            request,
            code=ErrorCode.REQUEST_INVALID,
            message="Request validation failed",
            status_code=422,
            details={"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {
            401: ErrorCode.AUTHENTICATION_REQUIRED,
            403: ErrorCode.PERMISSION_DENIED,
            404: ErrorCode.RESOURCE_NOT_FOUND,
            409: ErrorCode.RESOURCE_CONFLICT,
        }
        messages = {
            401: "Authentication required",
            403: "Permission denied",
            404: "Resource not found",
            409: "Resource conflict",
        }
        code = codes.get(exc.status_code, ErrorCode.REQUEST_INVALID)
        return _error_response(
            request,
            code=code,
            message=messages.get(exc.status_code, "Request failed"),
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return _internal_error_response(request)
