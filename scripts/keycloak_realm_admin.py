from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parseaddr
from typing import Any

_REALM_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class ConfigurationError(ValueError):
    pass


def _single_line(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized or "\r" in normalized or "\n" in normalized:
        raise ConfigurationError(f"{field} must be a non-empty single-line value")
    return normalized


def _required(environment: Mapping[str, str], name: str) -> str:
    value = _single_line(environment.get(name, ""), field=name)
    if "change-me" in value.casefold():
        raise ConfigurationError(f"{name} must not use the example placeholder")
    return value


def _boolean(environment: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ConfigurationError(f"{name} must be true or false")


def _integer(
    environment: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = environment.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class RegistrationConfiguration:
    realm: str
    admin_username: str
    admin_password: str
    internal_url: str
    verification_lifespan_seconds: int
    smtp_server: dict[str, str]

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
    ) -> RegistrationConfiguration | None:
        if not _boolean(environment, "KEYCLOAK_REGISTRATION_ENABLED", default=False):
            return None

        realm = _required(environment, "KEYCLOAK_REALM")
        if _REALM_NAME.fullmatch(realm) is None:
            raise ConfigurationError("KEYCLOAK_REALM contains unsupported characters")

        internal_url = _required(environment, "KEYCLOAK_INTERNAL_URL").rstrip("/")
        parsed_url = urllib.parse.urlsplit(internal_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ConfigurationError("KEYCLOAK_INTERNAL_URL must be an HTTP(S) URL")
        if parsed_url.query or parsed_url.fragment or parsed_url.username or parsed_url.password:
            raise ConfigurationError("KEYCLOAK_INTERNAL_URL must not contain credentials, query, or fragment")

        smtp_host = _required(environment, "KEYCLOAK_SMTP_HOST")
        if "://" in smtp_host or any(character.isspace() for character in smtp_host):
            raise ConfigurationError("KEYCLOAK_SMTP_HOST must be a hostname without a URL scheme")
        smtp_port = _integer(
            environment,
            "KEYCLOAK_SMTP_PORT",
            default=465,
            minimum=1,
            maximum=65535,
        )
        smtp_from = _required(environment, "KEYCLOAK_SMTP_FROM")
        parsed_name, parsed_address = parseaddr(smtp_from)
        if parsed_name or parsed_address != smtp_from or "@" not in parsed_address:
            raise ConfigurationError("KEYCLOAK_SMTP_FROM must contain one plain email address")

        smtp_ssl = _boolean(environment, "KEYCLOAK_SMTP_SSL", default=True)
        smtp_starttls = _boolean(environment, "KEYCLOAK_SMTP_STARTTLS", default=False)
        if smtp_ssl and smtp_starttls:
            raise ConfigurationError("KEYCLOAK_SMTP_SSL and KEYCLOAK_SMTP_STARTTLS cannot both be true")
        smtp_auth = _boolean(environment, "KEYCLOAK_SMTP_AUTH", default=True)
        smtp_server = {
            "host": smtp_host,
            "port": str(smtp_port),
            "from": smtp_from,
            "fromDisplayName": environment.get(
                "KEYCLOAK_SMTP_FROM_DISPLAY_NAME",
                "企业智能知识库问答",
            ).strip()
            or "企业智能知识库问答",
            "auth": str(smtp_auth).lower(),
            "ssl": str(smtp_ssl).lower(),
            "starttls": str(smtp_starttls).lower(),
            "allowutf8": str(_boolean(environment, "KEYCLOAK_SMTP_ALLOW_UTF8", default=False)).lower(),
        }
        if smtp_auth:
            smtp_server["user"] = _required(environment, "KEYCLOAK_SMTP_USER")
            smtp_server["password"] = _required(environment, "KEYCLOAK_SMTP_PASSWORD")

        return cls(
            realm=realm,
            admin_username=_required(environment, "KEYCLOAK_ADMIN_USERNAME"),
            admin_password=_required(environment, "KEYCLOAK_ADMIN_PASSWORD"),
            internal_url=internal_url,
            verification_lifespan_seconds=_integer(
                environment,
                "KEYCLOAK_EMAIL_VERIFICATION_LIFESPAN_SECONDS",
                default=900,
                minimum=300,
                maximum=86400,
            ),
            smtp_server=smtp_server,
        )


def registration_realm_updates(configuration: RegistrationConfiguration) -> dict[str, object]:
    return {
        "registrationAllowed": True,
        "registrationEmailAsUsername": True,
        "verifyEmail": True,
        "loginWithEmailAllowed": True,
        "duplicateEmailsAllowed": False,
        "resetPasswordAllowed": True,
        "editUsernameAllowed": False,
        "internationalizationEnabled": True,
        "supportedLocales": ["zh-CN", "en"],
        "defaultLocale": "zh-CN",
        "actionTokenGeneratedByUserLifespan": configuration.verification_lifespan_seconds,
        "smtpServer": configuration.smtp_server,
    }


def _json_request(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, object] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(  # noqa: S310 - caller supplies a validated internal URL
        url,
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - validated internal URL
        raw = response.read()
    if not raw:
        return {}
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise RuntimeError("Keycloak returned a non-object response")
    return decoded


def _admin_token(configuration: RegistrationConfiguration) -> str:
    body = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": configuration.admin_username,
            "password": configuration.admin_password,
        }
    ).encode("ascii")
    request = urllib.request.Request(  # noqa: S310 - base URL was validated above
        f"{configuration.internal_url}/realms/master/protocol/openid-connect/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - validated internal URL
        payload = json.loads(response.read())
    token = payload.get("access_token") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        raise RuntimeError("Keycloak did not return an admin access token")
    return token


def apply_registration_configuration(configuration: RegistrationConfiguration) -> None:
    token = _admin_token(configuration)
    realm_url = f"{configuration.internal_url}/admin/realms/{urllib.parse.quote(configuration.realm, safe='')}"
    current = _json_request(realm_url, token=token)
    current.update(registration_realm_updates(configuration))
    _json_request(realm_url, method="PUT", payload=current, token=token)

    verified = _json_request(realm_url, token=token)
    smtp = verified.get("smtpServer")
    if (
        verified.get("registrationAllowed") is not True
        or verified.get("verifyEmail") is not True
        or not isinstance(smtp, dict)
        or smtp.get("host") != configuration.smtp_server["host"]
        or smtp.get("from") != configuration.smtp_server["from"]
    ):
        raise RuntimeError("Keycloak registration configuration verification failed")


def main() -> int:
    try:
        configuration = RegistrationConfiguration.from_environment(os.environ)
        if configuration is None:
            print(json.dumps({"status": "skipped", "registration_enabled": False}))
            return 0
        apply_registration_configuration(configuration)
        print(
            json.dumps(
                {
                    "status": "configured",
                    "realm": configuration.realm,
                    "registration_enabled": True,
                    "email_verification": True,
                    "smtp_host": configuration.smtp_server["host"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (ConfigurationError, RuntimeError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
