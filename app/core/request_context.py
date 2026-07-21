import re
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def get_request_id() -> str:
    return _REQUEST_ID.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _VALID_REQUEST_ID.fullmatch(supplied) else str(uuid.uuid4())
        request.state.request_id = request_id
        token = _REQUEST_ID.set(request_id)
        try:
            try:
                response = await call_next(request)
            except Exception:
                from app.core.errors import _internal_error_response

                return _internal_error_response(request, request_id=request_id)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            _REQUEST_ID.reset(token)
