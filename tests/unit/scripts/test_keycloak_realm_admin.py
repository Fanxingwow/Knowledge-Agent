from __future__ import annotations

from collections.abc import Mapping

import pytest

from scripts import keycloak_realm_admin as realm_admin


def valid_environment(**overrides: str) -> dict[str, str]:
    environment = {
        "KEYCLOAK_REGISTRATION_ENABLED": "true",
        "KEYCLOAK_REALM": "enterprise-knowledge",
        "KEYCLOAK_INTERNAL_URL": "http://keycloak:8080",
        "KEYCLOAK_ADMIN_USERNAME": "kc-admin",
        "KEYCLOAK_ADMIN_PASSWORD": "strong-admin-secret",  # pragma: allowlist secret
        "KEYCLOAK_EMAIL_VERIFICATION_LIFESPAN_SECONDS": "900",
        "KEYCLOAK_SMTP_HOST": "smtp.example.com",
        "KEYCLOAK_SMTP_PORT": "465",
        "KEYCLOAK_SMTP_FROM": "no-reply@example.com",
        "KEYCLOAK_SMTP_FROM_DISPLAY_NAME": "Enterprise Knowledge",
        "KEYCLOAK_SMTP_AUTH": "true",
        "KEYCLOAK_SMTP_USER": "no-reply@example.com",
        "KEYCLOAK_SMTP_PASSWORD": "smtp-authorization-code",  # pragma: allowlist secret
        "KEYCLOAK_SMTP_SSL": "true",
        "KEYCLOAK_SMTP_STARTTLS": "false",
        "KEYCLOAK_SMTP_ALLOW_UTF8": "false",
    }
    environment.update(overrides)
    return environment


def test_disabled_registration_is_a_safe_noop() -> None:
    assert realm_admin.RegistrationConfiguration.from_environment({"KEYCLOAK_REGISTRATION_ENABLED": "false"}) is None


def test_registration_configuration_enables_verified_email_only() -> None:
    configuration = realm_admin.RegistrationConfiguration.from_environment(valid_environment())

    assert configuration is not None
    updates = realm_admin.registration_realm_updates(configuration)
    assert updates["registrationAllowed"] is True
    assert updates["registrationEmailAsUsername"] is True
    assert updates["verifyEmail"] is True
    assert updates["duplicateEmailsAllowed"] is False
    assert updates["actionTokenGeneratedByUserLifespan"] == 900
    assert updates["smtpServer"] == {
        "host": "smtp.example.com",
        "port": "465",
        "from": "no-reply@example.com",
        "fromDisplayName": "Enterprise Knowledge",
        "auth": "true",
        "ssl": "true",
        "starttls": "false",
        "allowutf8": "false",
        "user": "no-reply@example.com",
        "password": "smtp-authorization-code",  # pragma: allowlist secret
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"KEYCLOAK_SMTP_PASSWORD": "change-me-smtp-secret"},  # pragma: allowlist secret
            "example placeholder",
        ),
        ({"KEYCLOAK_SMTP_HOST": "https://smtp.example.com"}, "without a URL scheme"),
        (
            {"KEYCLOAK_SMTP_SSL": "true", "KEYCLOAK_SMTP_STARTTLS": "true"},
            "cannot both be true",
        ),
        ({"KEYCLOAK_EMAIL_VERIFICATION_LIFESPAN_SECONDS": "60"}, "between 300 and 86400"),
    ],
)
def test_invalid_or_placeholder_smtp_configuration_is_rejected(
    overrides: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(realm_admin.ConfigurationError, match=message):
        realm_admin.RegistrationConfiguration.from_environment(valid_environment(**overrides))


def test_unauthenticated_smtp_does_not_require_or_store_credentials() -> None:
    environment = valid_environment(KEYCLOAK_SMTP_AUTH="false")
    environment.pop("KEYCLOAK_SMTP_USER")
    environment.pop("KEYCLOAK_SMTP_PASSWORD")

    configuration = realm_admin.RegistrationConfiguration.from_environment(environment)

    assert configuration is not None
    assert "user" not in configuration.smtp_server
    assert "password" not in configuration.smtp_server


def test_apply_updates_existing_realm_without_replacing_unrelated_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = realm_admin.RegistrationConfiguration.from_environment(valid_environment())
    assert configuration is not None
    existing: dict[str, object] = {"realm": configuration.realm, "roles": {"realm": []}}
    stored: dict[str, object] = dict(existing)

    monkeypatch.setattr(realm_admin, "_admin_token", lambda _configuration: "admin-token")

    def fake_json_request(
        _url: str,
        *,
        method: str = "GET",
        payload: Mapping[str, object] | None = None,
        token: str | None = None,
    ) -> dict[str, object]:
        assert token == "admin-token"  # noqa: S105
        if method == "PUT":
            assert payload is not None
            stored.clear()
            stored.update(payload)
            return {}
        return dict(stored)

    monkeypatch.setattr(realm_admin, "_json_request", fake_json_request)

    realm_admin.apply_registration_configuration(configuration)

    assert stored["roles"] == existing["roles"]
    assert stored["registrationAllowed"] is True
    assert stored["verifyEmail"] is True
