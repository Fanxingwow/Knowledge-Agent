from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.services.knowledge_search import (
    DEFAULT_SEARCH_TOP_K,
    MAX_QUERY_CHARACTERS,
    MAX_SEARCH_TOP_K,
    KnowledgeSearchResult,
)


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARACTERS)
    top_k: int = Field(default=DEFAULT_SEARCH_TOP_K, ge=1, le=MAX_SEARCH_TOP_K)


class KnowledgeSearchMetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    parent_title: str
    file_title: str
    item_name: str


class KnowledgeSearchHitResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: str = Field(pattern=r"^cit_[0-9a-f]{64}$")
    content: str
    metadata: KnowledgeSearchMetadataResponse
    score: float


class KnowledgeSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hits: tuple[KnowledgeSearchHitResponse, ...]

    @classmethod
    def from_service(cls, result: KnowledgeSearchResult) -> KnowledgeSearchResponse:
        return cls(
            hits=tuple(
                KnowledgeSearchHitResponse(
                    citation_id=hit.citation_id,
                    content=hit.content,
                    metadata=KnowledgeSearchMetadataResponse(
                        title=hit.metadata.title,
                        parent_title=hit.metadata.parent_title,
                        file_title=hit.metadata.file_title,
                        item_name=hit.metadata.item_name,
                    ),
                    score=hit.score,
                )
                for hit in result.hits
            )
        )


__all__ = [
    "KnowledgeSearchHitResponse",
    "KnowledgeSearchMetadataResponse",
    "KnowledgeSearchRequest",
    "KnowledgeSearchResponse",
]
