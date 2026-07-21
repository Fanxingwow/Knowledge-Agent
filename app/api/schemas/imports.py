from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.services.import_registry import ImportTaskStatus


class ImportDocumentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: UUID
    original_name: str
    content_type: str
    size: int
    sha256: str


class ImportProgressResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    current_step: str | None
    completed_steps: tuple[str, ...]


class ImportTaskResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: UUID
    status: ImportTaskStatus
    document: ImportDocumentResponse
    progress: ImportProgressResponse
    imported_chunk_count: int | None
    error_code: str | None
    created_at: datetime
    updated_at: datetime


class ImportBatchAcceptedResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    tasks: tuple[ImportTaskResponse, ...]
