from pydantic import SecretStr

from app.core.errors import AppError, ErrorCode


def test_app_error_has_stable_code() -> None:
    error = AppError(ErrorCode.DOCUMENT_TYPE_UNSUPPORTED, "bad type", 415)
    assert error.code == ErrorCode.DOCUMENT_TYPE_UNSUPPORTED
    assert error.status_code == 415
    assert error.details == {}


def test_app_error_keeps_secret_values_wrapped() -> None:
    secret = SecretStr("credential-must-not-leak")
    error = AppError(
        ErrorCode.REQUEST_INVALID,
        "invalid request",
        400,
        {"credential": secret},
    )

    assert error.details == {"credential": secret}
    assert "credential-must-not-leak" not in repr(error.details)
