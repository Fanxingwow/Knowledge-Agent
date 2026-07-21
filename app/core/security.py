from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from app.api.dependencies import get_app_settings
from app.core.errors import AppError, ErrorCode
from app.core.settings import AppEnvironment, Settings

bearer = HTTPBearer(auto_error=False)
_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _create_jwks_client(jwks_url: str) -> PyJWKClient | None:
    try:
        return PyJWKClient(jwks_url)
    except AppError:
        raise
    except Exception:
        return None


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str
    roles: frozenset[str]


class TokenVerifier:
    def __init__(
        self,
        issuer: str,
        audience: str,
        algorithms: list[str],
        *,
        public_key: str | bytes | None = None,
        jwks_url: str | None = None,
    ) -> None:
        self.issuer = issuer
        self.audience = audience
        self.algorithms = tuple(algorithms)
        self.public_key = public_key
        self.jwks_client = _create_jwks_client(jwks_url) if jwks_url else None
        if jwks_url and self.jwks_client is None:
            raise AppError(
                ErrorCode.AUTHENTICATION_REQUIRED,
                "Invalid access token",
                401,
            )

    def _decode(self, token: str) -> dict[str, object] | None:
        try:
            key = self.jwks_client.get_signing_key_from_jwt(token).key if self.jwks_client else self.public_key
            if key is None:
                return None
            return jwt.decode(
                token,
                key,
                algorithms=list(self.algorithms),
                issuer=self.issuer,
                audience=self.audience,
                # ``nbf`` is optional in RFC 7519 and Keycloak does not emit it
                # for every access token. PyJWT still validates it whenever it
                # is present, so future-dated tokens remain rejected without
                # making the optional claim a hard interoperability requirement.
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except AppError:
            raise
        except Exception:
            return None

    def verify(self, token: str) -> Principal:
        payload = self._decode(token)
        if payload is None:
            raise AppError(
                ErrorCode.AUTHENTICATION_REQUIRED,
                "Invalid access token",
                401,
            )

        raw_subject = payload["sub"]
        if not isinstance(raw_subject, str) or not raw_subject.strip():
            raise AppError(
                ErrorCode.AUTHENTICATION_REQUIRED,
                "Token has an invalid subject",
                401,
            )
        subject = raw_subject.strip()

        tenant_id = _validated_tenant_id(payload.get("tenant_id"))
        if tenant_id is None:
            raise AppError(
                ErrorCode.AUTHENTICATION_REQUIRED,
                "Token has an invalid tenant_id",
                401,
            )

        if "roles" not in payload:
            roles: frozenset[str] = frozenset()
        else:
            raw_roles = payload["roles"]
            if isinstance(raw_roles, str):
                roles = frozenset({raw_roles})
            elif isinstance(raw_roles, list) and all(isinstance(role, str) for role in raw_roles):
                roles = frozenset(raw_roles)
            else:
                raise AppError(
                    ErrorCode.AUTHENTICATION_REQUIRED,
                    "Token has invalid roles",
                    401,
                )

        return Principal(subject, tenant_id, roles)


def _validated_tenant_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value if _TENANT_ID.fullmatch(value) else None


def ensure_roles(principal: Principal, allowed: set[str]) -> Principal:
    if not principal.roles.intersection(allowed):
        raise AppError(ErrorCode.PERMISSION_DENIED, "Insufficient role", 403)
    return principal


@lru_cache(maxsize=16)
def build_verifier(
    issuer: str,
    audience: str,
    algorithms: tuple[str, ...],
    public_key: str | None,
    jwks_url: str | None,
) -> TokenVerifier:
    key_bytes = public_key.encode("utf-8") if public_key else None
    return TokenVerifier(
        issuer,
        audience,
        list(algorithms),
        public_key=key_bytes,
        jwks_url=jwks_url,
    )


def authenticate_credentials(
    settings: Settings,
    credentials: HTTPAuthorizationCredentials | None,
) -> Principal:
    auth = settings.auth
    environment = settings.api.environment

    if auth.mode == "disabled":
        if environment == AppEnvironment.PRODUCTION:
            raise AppError(
                ErrorCode.AUTHENTICATION_REQUIRED,
                "Authentication is required",
                401,
            )
        tenant_id = _validated_tenant_id(auth.dev_tenant_id)
        if tenant_id is None:
            raise AppError(
                ErrorCode.AUTHENTICATION_REQUIRED,
                "Authentication configuration is invalid",
                401,
            )
        return Principal(
            "local-dev",
            tenant_id,
            frozenset({"reader", "editor", "admin"}),
        )

    if auth.mode != "jwt":
        raise AppError(
            ErrorCode.AUTHENTICATION_REQUIRED,
            "Authentication is required",
            401,
        )
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise AppError(
            ErrorCode.AUTHENTICATION_REQUIRED,
            "Bearer token is required",
            401,
        )
    if environment == AppEnvironment.PRODUCTION and not auth.jwks_url:
        raise AppError(
            ErrorCode.AUTHENTICATION_REQUIRED,
            "Authentication configuration is invalid",
            401,
        )

    public_key = None
    if environment != AppEnvironment.PRODUCTION and auth.public_key:
        public_key = auth.public_key.get_secret_value()
    verifier = build_verifier(
        auth.issuer or "",
        auth.audience or "",
        tuple(auth.algorithms),
        public_key,
        auth.jwks_url,
    )
    return verifier.verify(credentials.credentials)


def authenticate_authorization(settings: Settings, authorization: str | None) -> Principal:
    credentials: HTTPAuthorizationCredentials | None = None
    if authorization:
        scheme, separator, token = authorization.partition(" ")
        if separator and token and " " not in token:
            credentials = HTTPAuthorizationCredentials(scheme=scheme, credentials=token)
    return authenticate_credentials(settings, credentials)


def get_current_principal(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer),
    ],
    settings: Annotated[Settings, Depends(get_app_settings)],
    request: Request = None,  # type: ignore[assignment]
) -> Principal:
    request_state = getattr(request, "state", None)
    cached = getattr(request_state, "principal", None)
    if isinstance(cached, Principal):
        return cached
    return authenticate_credentials(settings, credentials)


def require_roles(*roles: str) -> Callable[[Principal], Principal]:
    allowed_roles = set(roles)

    def dependency(
        principal: Annotated[Principal, Depends(get_current_principal)],
    ) -> Principal:
        return ensure_roles(principal, allowed_roles)

    return dependency
