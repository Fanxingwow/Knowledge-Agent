from __future__ import annotations

import anyio
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import AppError, ErrorCode, _error_response
from app.core.security import authenticate_authorization
from app.core.settings import Settings


class _BodyLimitExceeded(BaseException):
    pass


class ApiAuthenticationMiddleware:
    """Authenticate protected API paths before Starlette parses request bodies."""

    def __init__(self, app: ASGIApp, *, settings: Settings, path_prefix: str = "/api/v1/") -> None:
        self.app = app
        self.settings = settings
        self.path_prefix = path_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(self.path_prefix):
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        try:
            principal = await anyio.to_thread.run_sync(
                authenticate_authorization,
                self.settings,
                request.headers.get("Authorization"),
            )
        except AppError as exc:
            response = _error_response(
                request,
                code=exc.code,
                message=exc.message,
                status_code=exc.status_code,
                details=exc.details,
            )
            await response(scope, receive, send)
            return
        scope.setdefault("state", {})["principal"] = principal
        await self.app(scope, receive, send)


class RequestBodyLimitMiddleware:
    """Count actual ASGI body bytes so chunked requests cannot bypass admission limits."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
        protected_path: str = "/api/v1/imports",
    ) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.protected_path = protected_path

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive=receive)
        response = _error_response(
            request,
            code=ErrorCode.FILE_TOO_LARGE,
            message="Request body exceeds configured size limit",
            status_code=413,
        )
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != self.protected_path
        ):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        raw_length = request.headers.get("Content-Length")
        if raw_length is not None:
            try:
                declared_length = int(raw_length)
            except ValueError:
                declared_length = -1
            if declared_length < 0 or declared_length > self.max_body_bytes:
                await self._reject(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                received += len(body)
                if received > self.max_body_bytes:
                    raise _BodyLimitExceeded
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyLimitExceeded:
            await self._reject(scope, receive, send)
