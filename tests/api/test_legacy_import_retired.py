from fastapi.testclient import TestClient

from app.import_process.api.file_import_service import app


def test_legacy_import_service_is_closed_instead_of_bypassing_unified_security() -> None:
    response = TestClient(app).post("/upload", files={"files": ("x.md", b"x", "text/markdown")})

    assert response.status_code == 410
    assert response.json() == {
        "code": "LEGACY_ENDPOINT_RETIRED",
        "message": "Use the unified /api/v1/imports endpoint",
    }
