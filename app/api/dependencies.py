from __future__ import annotations

from typing import cast

from fastapi import Request

from app.core.errors import AppError, DependencyError, ErrorCode
from app.core.settings import Settings
from app.infrastructure.minio_store import MinioStore
from app.infrastructure.readiness import import_runtime_is_ready, object_storage_is_ready
from app.services.import_admission import ImportAdmissionRepository
from app.services.import_coordinator import ImportCoordinator
from app.services.import_registry import ImportTaskRegistry
from app.services.import_runner import ImportProcessor
from app.services.import_workspace import ImportWorkspace


def get_app_settings(request: Request) -> Settings:
    """Return this application instance's immutable runtime settings."""
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise AppError(
            ErrorCode.INTERNAL_ERROR,
            "Application is not initialized",
            500,
        )
    return settings


async def get_object_store(request: Request) -> MinioStore:
    """Return storage only while the application readiness gate is healthy."""
    store = getattr(request.app.state, "minio_store", None)
    if not object_storage_is_ready(request.app.state) or store is None:
        raise DependencyError("Object storage is unavailable")
    return cast(MinioStore, store)


def get_import_registry(request: Request) -> ImportTaskRegistry:
    registry = getattr(request.app.state, "import_registry", None)
    if registry is None:
        raise DependencyError("Import task registry is unavailable")
    return cast(ImportTaskRegistry, registry)


def get_import_admission_repository(request: Request) -> ImportAdmissionRepository:
    repository = getattr(request.app.state, "import_admissions", None)
    if repository is None:
        raise DependencyError("Import admission repository is unavailable")
    return cast(ImportAdmissionRepository, repository)


def get_import_coordinator(request: Request) -> ImportCoordinator:
    coordinator = getattr(request.app.state, "import_coordinator", None)
    if coordinator is None or not import_runtime_is_ready(request.app.state):
        raise DependencyError("Import coordination is unavailable")
    return cast(ImportCoordinator, coordinator)


def get_import_processor(request: Request) -> ImportProcessor:
    processor = getattr(request.app.state, "import_processor", None)
    if processor is None:
        raise DependencyError("Import processing is unavailable")
    return cast(ImportProcessor, processor)


def get_import_workspace(request: Request) -> ImportWorkspace:
    workspace = getattr(request.app.state, "import_workspace", None)
    if not isinstance(workspace, ImportWorkspace):
        raise DependencyError("Import workspace is unavailable")
    return workspace
