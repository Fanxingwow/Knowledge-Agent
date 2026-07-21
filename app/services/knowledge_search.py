from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from threading import BoundedSemaphore
from typing import Protocol, TypeVar

import anyio

from app.core.errors import AppError, DependencyError, ErrorCode
from app.services.tenant_vector_search import (
    HybridVectorQuery,
    TenantVectorSearch,
    TenantVectorSearchResult,
    VectorCandidate,
)

MAX_QUERY_CHARACTERS = 2_048
MAX_QUERY_UTF8_BYTES = 4_096
MAX_SEARCH_TOP_K = 20
DEFAULT_SEARCH_TOP_K = 10
_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_UNAVAILABLE_MESSAGE = "Knowledge search is unavailable"
_INVALID_MESSAGE = "Knowledge search request is invalid"
_CITATION_NAMESPACE = b"knowledge-citation-v1\x00"
_ResultT = TypeVar("_ResultT")


class QueryEmbeddingBackend(Protocol):
    """Trusted backend that converts one validated text query to hybrid vectors."""

    def embed_query(self, query_text: str) -> HybridVectorQuery: ...


@dataclass(frozen=True, slots=True)
class KnowledgeSearchMetadata:
    title: str
    parent_title: str
    file_title: str
    item_name: str


@dataclass(frozen=True, slots=True)
class KnowledgeSearchHit:
    """Public search hit; publication evidence intentionally never crosses this boundary."""

    citation_id: str
    content: str
    metadata: KnowledgeSearchMetadata
    score: float


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult:
    hits: tuple[KnowledgeSearchHit, ...]


def _invalid_request() -> AppError:
    return AppError(ErrorCode.REQUEST_INVALID, _INVALID_MESSAGE, 422)


def _validate_tenant_id(value: object) -> str:
    if not isinstance(value, str) or _TENANT_ID.fullmatch(value) is None:
        raise _invalid_request()
    return value


def _is_permitted_query_character(character: str) -> bool:
    # Plain spaces are accepted. Every other separator, control, formatting,
    # surrogate, private-use, and unassigned code point is rejected so hidden
    # direction/line controls cannot reach either the model or backend logs.
    if character == " ":
        return True
    return unicodedata.category(character)[0] not in {"C", "Z"}


def _validate_query_text(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_QUERY_CHARACTERS
    ):
        raise _invalid_request()
    try:
        raw_encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise _invalid_request() from None
    if len(raw_encoded) > MAX_QUERY_UTF8_BYTES:
        raise _invalid_request()
    normalized = unicodedata.normalize("NFC", value).strip(" ")
    if (
        not normalized
        or len(normalized) > MAX_QUERY_CHARACTERS
        or any(not _is_permitted_query_character(character) for character in normalized)
    ):
        raise _invalid_request()
    if len(normalized.encode("utf-8")) > MAX_QUERY_UTF8_BYTES:
        raise _invalid_request()
    return normalized


def _validate_top_k(value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise _invalid_request()
    return value


def _citation_id(tenant_id: str, candidate: VectorCandidate) -> str:
    digest = hashlib.sha256()
    digest.update(_CITATION_NAMESPACE)
    digest.update(tenant_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(candidate.chunk_id.encode("ascii"))
    return f"cit_{digest.hexdigest()}"


def _public_hit(tenant_id: str, candidate: VectorCandidate) -> KnowledgeSearchHit:
    return KnowledgeSearchHit(
        citation_id=_citation_id(tenant_id, candidate),
        content=candidate.content,
        metadata=KnowledgeSearchMetadata(
            title=candidate.title,
            parent_title=candidate.parent_title,
            file_title=candidate.file_title,
            item_name=candidate.item_name,
        ),
        score=float(candidate.score),
    )


class KnowledgeSearchService:
    """Tenant-safe text-query facade over hybrid embedding and vector visibility filtering."""

    def __init__(
        self,
        embedding_backend: QueryEmbeddingBackend,
        vector_search: TenantVectorSearch,
        *,
        timeout_seconds: float = 5.0,
        max_top_k: int = MAX_SEARCH_TOP_K,
        max_concurrent_requests: int = 16,
    ) -> None:
        embed_query = getattr(embedding_backend, "embed_query", None)
        if not callable(embed_query):
            raise ValueError("invalid query embedding backend")
        if not isinstance(vector_search, TenantVectorSearch):
            raise ValueError("invalid tenant vector search")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or not 0.01 <= float(timeout_seconds) <= 60.0
        ):
            raise ValueError("invalid knowledge search timeout")
        if (
            isinstance(max_top_k, bool)
            or not isinstance(max_top_k, int)
            or not 1 <= max_top_k <= MAX_SEARCH_TOP_K
        ):
            raise ValueError("invalid knowledge search result limit")
        if (
            isinstance(max_concurrent_requests, bool)
            or not isinstance(max_concurrent_requests, int)
            or not 1 <= max_concurrent_requests <= 256
        ):
            raise ValueError("invalid knowledge search concurrency limit")
        self._embedding_backend = embedding_backend
        self._vector_search = vector_search
        self._timeout_seconds = float(timeout_seconds)
        self._max_top_k = max_top_k
        self._request_slots = BoundedSemaphore(max_concurrent_requests)

    async def _run_sync(self, operation: Callable[[], _ResultT]) -> _ResultT:
        return await anyio.to_thread.run_sync(operation, abandon_on_cancel=True)

    def _search_sync(
        self,
        *,
        tenant_id: str,
        query_text: str,
        limit: int,
    ) -> TenantVectorSearchResult:
        if not self._request_slots.acquire(blocking=False):
            raise DependencyError(_UNAVAILABLE_MESSAGE)
        try:
            vectors = self._embedding_backend.embed_query(query_text)
            if not isinstance(vectors, HybridVectorQuery):
                raise TypeError("invalid query embedding result")
            result = self._vector_search.search(
                tenant_id=tenant_id,
                query=vectors,
                limit=limit,
            )
            if not isinstance(result, TenantVectorSearchResult):
                raise TypeError("invalid tenant search result")
            return result
        finally:
            self._request_slots.release()

    async def search(
        self,
        *,
        tenant_id: str,
        query_text: str,
        top_k: int = DEFAULT_SEARCH_TOP_K,
    ) -> KnowledgeSearchResult:
        tenant = _validate_tenant_id(tenant_id)
        query = _validate_query_text(query_text)
        limit = _validate_top_k(top_k, maximum=self._max_top_k)
        try:
            with anyio.fail_after(self._timeout_seconds):
                result = await self._run_sync(
                    partial(
                        self._search_sync,
                        tenant_id=tenant,
                        query_text=query,
                        limit=limit,
                    )
                )
        except TimeoutError:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None
        except Exception:
            raise DependencyError(_UNAVAILABLE_MESSAGE) from None
        return KnowledgeSearchResult(
            hits=tuple(_public_hit(tenant, candidate) for candidate in result.hits)
        )


__all__ = [
    "DEFAULT_SEARCH_TOP_K",
    "KnowledgeSearchHit",
    "KnowledgeSearchMetadata",
    "KnowledgeSearchResult",
    "KnowledgeSearchService",
    "MAX_QUERY_CHARACTERS",
    "MAX_QUERY_UTF8_BYTES",
    "MAX_SEARCH_TOP_K",
    "QueryEmbeddingBackend",
]
