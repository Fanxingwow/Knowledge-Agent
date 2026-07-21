from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence

from pymilvus import AnnSearchRequest, WeightedRanker  # type: ignore[import-untyped]

from app.infrastructure.milvus_runtime import (
    MILVUS_CANDIDATE_OUTPUT_FIELDS,
    MilvusHybridSearchClient,
)
from app.services.tenant_vector_search import HybridVectorQuery, VectorCandidate

_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_COLLECTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,254}$")


class MilvusTenantVectorSourceError(RuntimeError):
    """Base error whose text never includes dependency exception details."""


class MilvusTenantVectorUnavailable(MilvusTenantVectorSourceError):
    pass


class MilvusCandidateContractError(MilvusTenantVectorSourceError):
    pass


RequestFactory = Callable[..., object]
RankerFactory = Callable[[float, float], object]


def _request_factory(**kwargs: object) -> object:
    return AnnSearchRequest(**kwargs)


def _ranker_factory(dense_weight: float, sparse_weight: float) -> object:
    return WeightedRanker(dense_weight, sparse_weight, norm_score=True)


def _tenant_expression(tenant_id: str) -> str:
    if not isinstance(tenant_id, str) or _TENANT_ID.fullmatch(tenant_id) is None:
        raise ValueError("invalid tenant identity")
    literal = json.dumps(tenant_id, ensure_ascii=True, separators=(",", ":"))
    return f"tenant_id == {literal}"


def _valid_weight(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


class MilvusTenantVectorSource:
    """Translate typed query vectors and strict Milvus hits into candidates.

    The tenant expression is a performance/defence-in-depth optimization only.
    :class:`TenantVectorSearch` remains the visibility enforcement point.
    """

    def __init__(
        self,
        client: MilvusHybridSearchClient,
        *,
        collection_name: str,
        dense_weight: float = 0.8,
        sparse_weight: float = 0.2,
        max_candidates: int = 500,
        request_factory: RequestFactory = _request_factory,
        ranker_factory: RankerFactory = _ranker_factory,
    ) -> None:
        if (
            not isinstance(collection_name, str)
            or _COLLECTION_NAME.fullmatch(collection_name) is None
        ):
            raise ValueError("invalid Milvus collection name")
        if (
            not _valid_weight(dense_weight)
            or not _valid_weight(sparse_weight)
            or float(dense_weight) + float(sparse_weight) <= 0
        ):
            raise ValueError("invalid Milvus ranker weights")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 1
        ):
            raise ValueError("invalid Milvus candidate limit")
        self._client = client
        self._collection_name = collection_name
        self._dense_weight = float(dense_weight)
        self._sparse_weight = float(sparse_weight)
        self._max_candidates = max_candidates
        self._request_factory = request_factory
        self._ranker_factory = ranker_factory

    def _requests(
        self,
        query: HybridVectorQuery,
        tenant_id: str,
        limit: int,
    ) -> list[object]:
        expression = _tenant_expression(tenant_id)
        try:
            dense_request = self._request_factory(
                data=[list(float(value) for value in query.dense_vector)],
                anns_field="dense_vector",
                param={"metric_type": "COSINE"},
                expr=expression,
                limit=limit,
            )
            sparse_request = self._request_factory(
                data=[{int(index): float(value) for index, value in query.sparse_vector.items()}],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                expr=expression,
                limit=limit,
            )
        except Exception:
            raise MilvusCandidateContractError("Milvus search request is invalid") from None
        return [dense_request, sparse_request]

    @staticmethod
    def _entity(hit: Mapping[str, object]) -> Mapping[str, object]:
        entity = hit.get("entity")
        if not isinstance(entity, Mapping):
            raise MilvusCandidateContractError("Milvus candidate response is invalid")
        return entity

    @staticmethod
    def _required_text(entity: Mapping[str, object], field: str) -> str:
        value = entity.get(field)
        if not isinstance(value, str):
            raise MilvusCandidateContractError("Milvus candidate response is invalid")
        return value

    @classmethod
    def _candidate(cls, hit: object) -> VectorCandidate:
        if not isinstance(hit, Mapping):
            raise MilvusCandidateContractError("Milvus candidate response is invalid")
        entity = cls._entity(hit)
        raw_score = hit.get("distance")
        if (
            not isinstance(raw_score, (int, float))
            or isinstance(raw_score, bool)
            or not math.isfinite(float(raw_score))
        ):
            raise MilvusCandidateContractError("Milvus candidate response is invalid")
        score = float(raw_score)
        entity_chunk_id = entity.get("chunk_id")
        returned_id = hit.get("id")
        if entity_chunk_id is None:
            chunk_id = returned_id
        else:
            chunk_id = entity_chunk_id
        if (
            not isinstance(chunk_id, str)
            or (returned_id is not None and returned_id != chunk_id)
        ):
            raise MilvusCandidateContractError("Milvus candidate response is invalid")
        try:
            return VectorCandidate(
                chunk_id=chunk_id,
                tenant_id=cls._required_text(entity, "tenant_id"),
                document_id=cls._required_text(entity, "document_id"),
                import_version=cls._required_text(entity, "import_version"),
                commit_token=cls._required_text(entity, "commit_token"),
                content_sha256=cls._required_text(entity, "content_sha256"),
                score=score,
                content=cls._required_text(entity, "content"),
                title=cls._required_text(entity, "title"),
                parent_title=cls._required_text(entity, "parent_title"),
                file_title=cls._required_text(entity, "file_title"),
                item_name=cls._required_text(entity, "item_name"),
            )
        except ValueError:
            raise MilvusCandidateContractError("Milvus candidate response is invalid") from None

    @classmethod
    def _candidates(cls, response: object, limit: int) -> tuple[VectorCandidate, ...]:
        if (
            isinstance(response, (str, bytes, bytearray))
            or not isinstance(response, Sequence)
            or len(response) != 1
        ):
            raise MilvusCandidateContractError("Milvus candidate response is invalid")
        hits = response[0]
        if (
            isinstance(hits, (str, bytes, bytearray))
            or not isinstance(hits, Sequence)
            or len(hits) > limit
        ):
            raise MilvusCandidateContractError("Milvus candidate response is invalid")
        return tuple(cls._candidate(hit) for hit in hits)

    def search_candidates(
        self,
        query: HybridVectorQuery,
        *,
        tenant_id: str,
        limit: int,
    ) -> Sequence[VectorCandidate]:
        if not isinstance(query, HybridVectorQuery):
            raise ValueError("invalid vector query")
        _tenant_expression(tenant_id)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= self._max_candidates
        ):
            raise ValueError("invalid Milvus candidate limit")
        requests = self._requests(query, tenant_id, limit)
        try:
            ranker = self._ranker_factory(self._dense_weight, self._sparse_weight)
            response = self._client.hybrid_search(
                collection_name=self._collection_name,
                reqs=requests,
                ranker=ranker,
                limit=limit,
                output_fields=list(MILVUS_CANDIDATE_OUTPUT_FIELDS),
                consistency_level="Strong",
            )
        except MilvusTenantVectorSourceError:
            raise
        except Exception:
            raise MilvusTenantVectorUnavailable("Milvus vector search is unavailable") from None
        return self._candidates(response, limit)
