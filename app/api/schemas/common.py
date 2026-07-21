from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, str] = Field(default_factory=dict)


class ConsoleAuthConfigResponse(BaseModel):
    mode: Literal["disabled", "oidc"]
    issuer: str | None = None
    client_id: str | None = None
    scopes: str = "openid profile email"
