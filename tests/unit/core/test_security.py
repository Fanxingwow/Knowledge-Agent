from __future__ import annotations

import urllib.request
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from typing import Annotated, Any, Never, cast

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from jwt import PyJWKClient
from jwt.algorithms import RSAAlgorithm
from pydantic import SecretStr

from app.api.dependencies import get_app_settings
from app.core.errors import AppError, ErrorCode, register_exception_handlers
from app.core.security import (
    Principal,
    TokenVerifier,
    build_verifier,
    ensure_roles,
    get_current_principal,
    require_roles,
)
from app.core.settings import AppEnvironment, Settings

_INVALID_JWKS_BODY = b"jwks-parser-secret-is-not-json"
_INVALID_JWKS_URL = "ftp://invalid.example/jwks"


@pytest.fixture(scope="module")
def key_pair() -> tuple[bytes, bytes]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def make_token(
    private_key: bytes,
    *,
    algorithm: str = "RS256",
    remove_claims: Iterable[str] = (),
    claims: Mapping[str, object] | None = None,
    headers: dict[str, str] | None = None,
    **overrides: object,
) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": "user-1",
        "tenant_id": "tenant-a",
        "roles": ["reader"],
        "iss": "https://issuer.example",
        "aud": "kb-api",
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(minutes=5),
        **overrides,
    }
    payload.update(claims or {})
    for claim in remove_claims:
        payload.pop(claim, None)
    return jwt.encode(payload, private_key, algorithm=algorithm, headers=headers)


def make_runtime_settings(
    *,
    mode: str = "jwt",
    environment: AppEnvironment = AppEnvironment.DEVELOPMENT,
    public_key: bytes | None = None,
    jwks_url: str | None = None,
    dev_tenant_id: str = "local-dev",
) -> Settings:
    auth = SimpleNamespace(
        mode=mode,
        issuer="https://issuer.example",
        audience="kb-api",
        algorithms=["RS256"],
        public_key=SecretStr(public_key.decode("utf-8")) if public_key else None,
        jwks_url=jwks_url,
        dev_tenant_id=dev_tenant_id,
    )
    api = SimpleNamespace(environment=environment)
    return cast(Settings, SimpleNamespace(auth=auth, api=api))


def make_verifier(public_key: bytes, algorithms: list[str] | None = None) -> TokenVerifier:
    return TokenVerifier(
        "https://issuer.example",
        "kb-api",
        algorithms or ["RS256"],
        public_key=public_key,
    )


def invalid_jwks_response(*_args: object, **_kwargs: object) -> BytesIO:
    return BytesIO(_INVALID_JWKS_BODY)


def test_valid_token_returns_tenant_principal(key_pair: tuple[bytes, bytes]) -> None:
    private_key, public_key = key_pair

    principal = make_verifier(public_key).verify(make_token(private_key))

    assert principal == Principal("user-1", "tenant-a", frozenset({"reader"}))


@pytest.mark.parametrize("tenant_id", ["A0._:-z", "a" * 128])
def test_tenant_id_accepts_the_full_allowed_pattern(key_pair: tuple[bytes, bytes], tenant_id: str) -> None:
    private_key, public_key = key_pair

    principal = make_verifier(public_key).verify(make_token(private_key, tenant_id=tenant_id))

    assert principal.tenant_id == tenant_id


@pytest.mark.parametrize(
    "tenant_id",
    ["", "   ", 'tenant-a" or true', "tenant/a", "a" * 129, 123],
)
def test_invalid_tenant_id_is_rejected(key_pair: tuple[bytes, bytes], tenant_id: object) -> None:
    private_key, public_key = key_pair

    with pytest.raises(AppError) as captured:
        make_verifier(public_key).verify(make_token(private_key, tenant_id=tenant_id))

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert captured.value.status_code == 401


@pytest.mark.parametrize("tenant_id", [" tenant-a ", "tenant-a "])
def test_tenant_id_with_surrounding_whitespace_is_rejected(key_pair: tuple[bytes, bytes], tenant_id: str) -> None:
    private_key, public_key = key_pair

    with pytest.raises(AppError) as captured:
        make_verifier(public_key).verify(make_token(private_key, tenant_id=tenant_id))

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert captured.value.status_code == 401


def test_missing_tenant_id_is_rejected(key_pair: tuple[bytes, bytes]) -> None:
    private_key, public_key = key_pair

    with pytest.raises(AppError) as captured:
        make_verifier(public_key).verify(make_token(private_key, remove_claims=("tenant_id",)))

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED


@pytest.mark.parametrize("claim", ["exp", "iat", "iss", "aud", "sub"])
def test_missing_required_claim_is_rejected(key_pair: tuple[bytes, bytes], claim: str) -> None:
    private_key, public_key = key_pair

    with pytest.raises(AppError) as captured:
        make_verifier(public_key).verify(make_token(private_key, remove_claims=(claim,)))

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert captured.value.status_code == 401


def test_missing_optional_not_before_claim_is_accepted(key_pair: tuple[bytes, bytes]) -> None:
    private_key, public_key = key_pair

    principal = make_verifier(public_key).verify(make_token(private_key, remove_claims=("nbf",)))

    assert principal == Principal("user-1", "tenant-a", frozenset({"reader"}))


@pytest.mark.parametrize("subject", ["", "   ", 7, None, ["user-1"]])
def test_non_string_or_empty_subject_is_rejected(key_pair: tuple[bytes, bytes], subject: object) -> None:
    private_key, public_key = key_pair

    with pytest.raises(AppError) as captured:
        make_verifier(public_key).verify(make_token(private_key, sub=subject))

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED


def test_wrong_signature_is_rejected(key_pair: tuple[bytes, bytes]) -> None:
    _, public_key = key_pair
    other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_private_pem = other_private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )

    with pytest.raises(AppError) as captured:
        make_verifier(public_key).verify(make_token(other_private_pem))

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED


@pytest.mark.parametrize(
    "overrides",
    [
        {"exp": datetime.now(UTC) - timedelta(seconds=1)},
        {"nbf": datetime.now(UTC) + timedelta(minutes=5)},
        {"iss": "https://wrong-issuer.example"},
        {"aud": "wrong-audience"},
    ],
)
def test_time_issuer_and_audience_failures_are_rejected(
    key_pair: tuple[bytes, bytes], overrides: dict[str, object]
) -> None:
    private_key, public_key = key_pair

    with pytest.raises(AppError) as captured:
        make_verifier(public_key).verify(make_token(private_key, claims=overrides))

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED


def test_algorithm_allowlist_is_fixed_when_verifier_is_created(
    key_pair: tuple[bytes, bytes],
) -> None:
    private_key, public_key = key_pair
    algorithms = ["RS256"]
    verifier = make_verifier(public_key, algorithms)
    algorithms.append("RS512")

    with pytest.raises(AppError) as captured:
        verifier.verify(make_token(private_key, algorithm="RS512"))

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED


@pytest.mark.parametrize(
    "roles",
    [
        {"reader": True},
        ["reader", 7],
        [None],
        7,
        None,
    ],
)
def test_roles_with_non_string_values_are_rejected(key_pair: tuple[bytes, bytes], roles: object) -> None:
    private_key, public_key = key_pair

    with pytest.raises(AppError) as captured:
        make_verifier(public_key).verify(make_token(private_key, roles=roles))

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED


def test_string_role_is_accepted(key_pair: tuple[bytes, bytes]) -> None:
    private_key, public_key = key_pair

    principal = make_verifier(public_key).verify(make_token(private_key, roles="reader"))

    assert principal.roles == frozenset({"reader"})


def test_missing_roles_default_to_no_permissions(key_pair: tuple[bytes, bytes]) -> None:
    private_key, public_key = key_pair
    principal = make_verifier(public_key).verify(make_token(private_key, remove_claims=("roles",)))

    with pytest.raises(AppError) as captured:
        ensure_roles(principal, {"reader"})

    assert principal.roles == frozenset()
    assert captured.value.code is ErrorCode.PERMISSION_DENIED
    assert captured.value.status_code == 403


def test_role_check_allows_an_explicitly_allowed_role() -> None:
    principal = Principal("user-1", "tenant-a", frozenset({"reader", "editor"}))

    assert ensure_roles(principal, {"editor", "admin"}) is principal


def test_role_check_rejects_reader_for_editor_action() -> None:
    principal = Principal("user-1", "tenant-a", frozenset({"reader"}))

    with pytest.raises(AppError) as captured:
        ensure_roles(principal, {"editor", "admin"})

    assert captured.value.code is ErrorCode.PERMISSION_DENIED
    assert captured.value.status_code == 403


def test_role_check_defaults_to_deny_when_no_roles_are_allowed() -> None:
    principal = Principal("user-1", "tenant-a", frozenset({"admin"}))

    with pytest.raises(AppError) as captured:
        ensure_roles(principal, set())

    assert captured.value.code is ErrorCode.PERMISSION_DENIED


def test_require_roles_dependency_allows_matching_principal() -> None:
    principal = Principal("user-1", "tenant-a", frozenset({"admin"}))
    dependency = require_roles("editor", "admin")

    assert dependency(principal) is principal


def test_require_roles_dependency_defaults_to_deny() -> None:
    principal = Principal("user-1", "tenant-a", frozenset({"admin"}))
    dependency = require_roles()

    with pytest.raises(AppError) as captured:
        dependency(principal)

    assert captured.value.code is ErrorCode.PERMISSION_DENIED


def test_development_disabled_auth_returns_fixed_local_principal() -> None:
    settings = make_runtime_settings(
        mode="disabled",
        environment=AppEnvironment.DEVELOPMENT,
        dev_tenant_id="tenant-dev",
    )

    principal = get_current_principal(credentials=None, settings=settings)

    assert principal == Principal(
        "local-dev",
        "tenant-dev",
        frozenset({"reader", "editor", "admin"}),
    )


def test_production_disabled_auth_is_rejected_defense_in_depth() -> None:
    settings = make_runtime_settings(
        mode="disabled",
        environment=AppEnvironment.PRODUCTION,
    )

    with pytest.raises(AppError) as captured:
        get_current_principal(credentials=None, settings=settings)

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert captured.value.status_code == 401


def test_disabled_auth_rejects_invalid_development_tenant() -> None:
    settings = make_runtime_settings(
        mode="disabled",
        environment=AppEnvironment.DEVELOPMENT,
        dev_tenant_id="tenant/escape",
    )

    with pytest.raises(AppError) as captured:
        get_current_principal(credentials=None, settings=settings)

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert captured.value.status_code == 401


def test_unknown_auth_mode_is_rejected_defense_in_depth() -> None:
    settings = make_runtime_settings(mode="unexpected")

    with pytest.raises(AppError) as captured:
        get_current_principal(credentials=None, settings=settings)

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert captured.value.status_code == 401


def test_production_static_public_key_is_rejected_defense_in_depth(
    key_pair: tuple[bytes, bytes],
) -> None:
    private_key, public_key = key_pair
    settings = make_runtime_settings(
        environment=AppEnvironment.PRODUCTION,
        public_key=public_key,
    )
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=make_token(private_key),
    )

    with pytest.raises(AppError) as captured:
        get_current_principal(credentials=credentials, settings=settings)

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert captured.value.status_code == 401


def test_development_static_public_key_verifies_bearer_token(
    key_pair: tuple[bytes, bytes],
) -> None:
    private_key, public_key = key_pair
    settings = make_runtime_settings(public_key=public_key)
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=make_token(private_key),
    )

    principal = get_current_principal(credentials=credentials, settings=settings)

    assert principal == Principal("user-1", "tenant-a", frozenset({"reader"}))


def test_development_jwt_without_verification_key_is_rejected(
    key_pair: tuple[bytes, bytes],
) -> None:
    private_key, _ = key_pair
    settings = make_runtime_settings()
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=make_token(private_key),
    )

    with pytest.raises(AppError) as captured:
        get_current_principal(credentials=credentials, settings=settings)

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert captured.value.status_code == 401


def test_production_jwks_verifies_real_rsa_signature(
    key_pair: tuple[bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key, public_key = key_pair
    loaded_public_key = serialization.load_pem_public_key(public_key)
    assert isinstance(loaded_public_key, rsa.RSAPublicKey)
    jwk = RSAAlgorithm.to_jwk(loaded_public_key, as_dict=True)
    assert isinstance(jwk, dict)
    jwk.update({"kid": "key-1", "alg": "RS256", "use": "sig"})

    def fetch_data(_client: PyJWKClient) -> dict[str, object]:
        return {"keys": [jwk]}

    monkeypatch.setattr(PyJWKClient, "fetch_data", fetch_data)
    unrelated_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    unrelated_public_pem = unrelated_private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    settings = make_runtime_settings(
        environment=AppEnvironment.PRODUCTION,
        public_key=unrelated_public_pem,
        jwks_url="https://jwks.example/.well-known/jwks.json",
    )
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=make_token(private_key, headers={"kid": "key-1"}),
    )

    principal = get_current_principal(credentials=credentials, settings=settings)

    assert principal == Principal("user-1", "tenant-a", frozenset({"reader"}))


@pytest.mark.parametrize("construction_path", ["token_verifier", "build_verifier"])
def test_invalid_jwks_scheme_is_mapped_to_safe_auth_error(construction_path: str) -> None:
    with pytest.raises(AppError) as captured:
        if construction_path == "token_verifier":
            TokenVerifier(
                "https://issuer.example",
                "kb-api",
                ["RS256"],
                jwks_url=_INVALID_JWKS_URL,
            )
        else:
            build_verifier(
                "https://issuer.example",
                "kb-api",
                ("RS256",),
                None,
                _INVALID_JWKS_URL,
            )

    error = captured.value
    rendered_error = f"{error!r} {error!s} {error.details!r}"
    assert error.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert error.message == "Invalid access token"
    assert error.status_code == 401
    assert error.__cause__ is None
    assert error.__context__ is None
    assert _INVALID_JWKS_URL not in rendered_error
    assert "only 'http' and 'https'" not in rendered_error


def test_invalid_jwks_json_is_mapped_to_safe_auth_error(
    key_pair: tuple[bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key, _ = key_pair
    token = make_token(private_key, headers={"kid": "key-1"})
    monkeypatch.setattr(urllib.request, "urlopen", invalid_jwks_response)
    verifier = TokenVerifier(
        "https://issuer.example",
        "kb-api",
        ["RS256"],
        jwks_url="https://invalid-json-unit.example/jwks",
    )

    with pytest.raises(AppError) as captured:
        verifier.verify(token)

    error = captured.value
    rendered_error = f"{error!r} {error!s} {error.details!r}"
    assert error.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert error.message == "Invalid access token"
    assert error.status_code == 401
    assert error.__cause__ is None
    assert error.__context__ is None
    assert token not in rendered_error
    assert _INVALID_JWKS_BODY.decode() not in rendered_error


def test_jwks_client_constructor_preserves_application_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = AppError(ErrorCode.RESOURCE_CONFLICT, "constructor sentinel", 409)

    def raise_sentinel(_jwks_url: str) -> Never:
        raise sentinel

    monkeypatch.setattr("app.core.security.PyJWKClient", raise_sentinel)

    with pytest.raises(AppError) as captured:
        TokenVerifier(
            "https://issuer.example",
            "kb-api",
            ["RS256"],
            jwks_url="https://jwks.example/sentinel",
        )

    assert captured.value is sentinel
    assert captured.value.code is ErrorCode.RESOURCE_CONFLICT
    assert captured.value.message == "constructor sentinel"
    assert captured.value.status_code == 409


def test_jwks_signing_key_lookup_preserves_application_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = AppError(ErrorCode.PERMISSION_DENIED, "signing key sentinel", 403)
    verifier = TokenVerifier(
        "https://issuer.example",
        "kb-api",
        ["RS256"],
        jwks_url="https://jwks.example/sentinel",
    )
    assert verifier.jwks_client is not None

    def raise_sentinel(_token: str) -> Never:
        raise sentinel

    monkeypatch.setattr(
        verifier.jwks_client,
        "get_signing_key_from_jwt",
        raise_sentinel,
    )

    with pytest.raises(AppError) as captured:
        verifier.verify("opaque-token")

    assert captured.value is sentinel
    assert captured.value.code is ErrorCode.PERMISSION_DENIED
    assert captured.value.message == "signing key sentinel"
    assert captured.value.status_code == 403


def test_jwt_decode_preserves_application_error(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, "decode sentinel", 503)
    verifier = TokenVerifier(
        "https://issuer.example",
        "kb-api",
        ["RS256"],
        public_key=b"public-key-placeholder",
    )

    def raise_sentinel(*_args: object, **_kwargs: object) -> Never:
        raise sentinel

    monkeypatch.setattr(jwt, "decode", raise_sentinel)

    with pytest.raises(AppError) as captured:
        verifier.verify("opaque-token")

    assert captured.value is sentinel
    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert captured.value.message == "decode sentinel"
    assert captured.value.status_code == 503


def test_missing_credentials_are_rejected_with_stable_error() -> None:
    settings = make_runtime_settings()

    with pytest.raises(AppError) as captured:
        get_current_principal(credentials=None, settings=settings)

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert captured.value.status_code == 401


def test_wrong_authorization_scheme_is_rejected() -> None:
    settings = make_runtime_settings()
    credentials = HTTPAuthorizationCredentials(scheme="Basic", credentials="opaque")

    with pytest.raises(AppError) as captured:
        get_current_principal(credentials=credentials, settings=settings)

    assert captured.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert captured.value.status_code == 401


def test_api_request_without_token_returns_stable_401() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    settings = make_runtime_settings()

    app.dependency_overrides[get_app_settings] = lambda: settings

    @app.get("/protected")
    def protected(
        principal: Annotated[Principal, Depends(get_current_principal)],
    ) -> dict[str, str]:
        return {"subject": principal.subject}

    response = TestClient(app).get("/protected")

    assert response.status_code == 401
    assert response.json()["code"] == ErrorCode.AUTHENTICATION_REQUIRED.value


def test_api_request_with_invalid_jwks_json_returns_safe_401(
    key_pair: tuple[bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key, _ = key_pair
    token = make_token(private_key, headers={"kid": "key-1"})
    monkeypatch.setattr(urllib.request, "urlopen", invalid_jwks_response)
    settings = make_runtime_settings(
        environment=AppEnvironment.PRODUCTION,
        jwks_url="https://invalid-json-api.example/jwks",
    )
    app = FastAPI()
    register_exception_handlers(app)

    app.dependency_overrides[get_app_settings] = lambda: settings

    @app.get("/protected")
    def protected(
        principal: Annotated[Principal, Depends(get_current_principal)],
    ) -> dict[str, str]:
        return {"subject": principal.subject}

    response = TestClient(app, raise_server_exceptions=False).get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert response.json()["code"] == ErrorCode.AUTHENTICATION_REQUIRED.value
    assert response.json()["message"] == "Invalid access token"
    assert token not in response.text
    assert _INVALID_JWKS_BODY.decode() not in response.text


def test_api_request_with_invalid_jwks_scheme_returns_safe_401(
    key_pair: tuple[bytes, bytes],
) -> None:
    private_key, _ = key_pair
    token = make_token(private_key, headers={"kid": "key-1"})
    settings = make_runtime_settings(
        environment=AppEnvironment.PRODUCTION,
        jwks_url=_INVALID_JWKS_URL,
    )
    app = FastAPI()
    register_exception_handlers(app)

    app.dependency_overrides[get_app_settings] = lambda: settings

    @app.get("/protected")
    def protected(
        principal: Annotated[Principal, Depends(get_current_principal)],
    ) -> dict[str, str]:
        return {"subject": principal.subject}

    response = TestClient(app, raise_server_exceptions=False).get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert response.json()["code"] == ErrorCode.AUTHENTICATION_REQUIRED.value
    assert response.json()["message"] == "Invalid access token"
    assert token not in response.text
    assert _INVALID_JWKS_URL not in response.text
    assert "only 'http' and 'https'" not in response.text


def test_jwt_error_does_not_expose_token_or_underlying_exception(
    key_pair: tuple[bytes, bytes],
) -> None:
    private_key, _ = key_pair
    other_public_key = (
        rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        .public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    token = make_token(private_key)

    with pytest.raises(AppError) as captured:
        make_verifier(other_public_key).verify(token)

    error = captured.value
    rendered_error = f"{error!r} {error!s} {error.details!r}"
    assert error.code is ErrorCode.AUTHENTICATION_REQUIRED
    assert error.message == "Invalid access token"
    assert error.details == {}
    assert error.__cause__ is None
    assert error.__context__ is None
    assert token not in rendered_error
    assert "Signature verification failed" not in rendered_error
