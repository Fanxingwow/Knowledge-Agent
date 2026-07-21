from __future__ import annotations

import math
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from app.api.schemas.common import ErrorResponse
from app.api.schemas.search import KnowledgeSearchRequest, KnowledgeSearchResponse
from app.core.errors import DependencyError
from app.core.security import Principal, require_roles
from app.services.knowledge_search import KnowledgeSearchService

router = APIRouter(prefix="/search", tags=["search"])
search_reader = require_roles("reader", "editor", "admin")
_ERROR_RESPONSE: dict[str, Any] = {"model": ErrorResponse}
_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: _ERROR_RESPONSE for code in (401, 403, 422, 500, 503)
}


def get_knowledge_search(request: Request) -> KnowledgeSearchService:
    service = getattr(request.app.state, "knowledge_search", None)
    managed_until = getattr(request.app.state, "knowledge_search_ready_until", None)
    if managed_until is not None and (
        getattr(request.app.state, "knowledge_search_ready", False) is not True
        or isinstance(managed_until, bool)
        or not isinstance(managed_until, (int, float))
        or not math.isfinite(managed_until)
        or time.monotonic() >= managed_until
    ):
        raise DependencyError("Knowledge search is unavailable")
    if not isinstance(service, KnowledgeSearchService):
        raise DependencyError("Knowledge search is unavailable")
    return service


@router.post(
    "",
    response_model=KnowledgeSearchResponse,
    responses=_RESPONSES,
    summary="Search the authenticated tenant's published knowledge",
)
async def search_knowledge(
    payload: KnowledgeSearchRequest,
    principal: Annotated[Principal, Depends(search_reader)],
    service: Annotated[KnowledgeSearchService, Depends(get_knowledge_search)],
) -> KnowledgeSearchResponse:
    result = await service.search(
        tenant_id=principal.tenant_id,
        query_text=payload.query,
        top_k=payload.top_k,
    )
    return KnowledgeSearchResponse.from_service(result)


__all__ = ["get_knowledge_search", "router", "search_knowledge"]
