from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Protocol

from app.core.errors import AppError, DependencyError, ErrorCode

_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_TOKEN = re.compile(r"^[!-~]{1,256}$")
_MAX_VARCHAR_BYTES = 65_535
_UNAVAILABLE_MESSAGE = "Knowledge search is unavailable"


def _canonical_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid vector identity")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("invalid vector identity") from None
    if str(parsed) != value:
        raise ValueError("invalid vector identity")
    return value


def _valid_number(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_tenant_id(value: object) -> str:
    if not isinstance(value, str) or _TENANT_ID.fullmatch(value) is None:
        raise ValueError("invalid tenant identity")
    return value


def _validate_commit_token(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_TOKEN.fullmatch(value) is None:
        raise ValueError("invalid commit token")
    return value


def _validate_metadata(value: object) -> str:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_VARCHAR_BYTES:
        raise ValueError("invalid vector metadata")
    return value


@dataclass(frozen=True, slots=True, init=False)
class HybridVectorQuery:
    """Server-built query vectors.

    The contract intentionally has no raw Milvus expression or client filter.
    Tenant scope is supplied separately from the authenticated request context.
    """

    dense_vector: tuple[float, ...]
    sparse_vector: Mapping[int, float]

    def __init__(
        self,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
    ) -> None:
        if (
            isinstance(dense_vector, (str, bytes, bytearray))
            or not isinstance(dense_vector, Sequence)
            or not dense_vector
            or any(not _valid_number(value) for value in dense_vector)
        ):
            raise ValueError("invalid dense query vector")
        if not isinstance(sparse_vector, Mapping) or not sparse_vector:
            raise ValueError("invalid sparse query vector")
        for index, value in sparse_vector.items():
            if (
                not isinstance(index, Integral)
                or isinstance(index, bool)
                or index < 0
                or not _valid_number(value)
            ):
                raise ValueError("invalid sparse query vector")
        object.__setattr__(
            self,
            "dense_vector",
            tuple(float(value) for value in dense_vector),
        )
        object.__setattr__(
            self,
            "sparse_vector",
            MappingProxyType(
                {int(index): float(value) for index, value in sparse_vector.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class VectorCandidate:
    """One ranked vector candidate with complete publication evidence."""

    chunk_id: str
    tenant_id: str
    document_id: str
    import_version: str
    commit_token: str
    content_sha256: str
    score: float
    content: str
    title: str = ""
    parent_title: str = ""
    file_title: str = ""
    item_name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or _SHA256.fullmatch(self.chunk_id) is None:
            raise ValueError("invalid vector chunk identity")
        _validate_tenant_id(self.tenant_id)
        _canonical_uuid(self.document_id)
        _canonical_uuid(self.import_version)
        _validate_commit_token(self.commit_token)
        if not isinstance(self.content_sha256, str) or _SHA256.fullmatch(self.content_sha256) is None:
            raise ValueError("invalid vector content digest")
        if not _valid_number(self.score):
            raise ValueError("invalid vector score")
        if (
            not isinstance(self.content, str)
            or not self.content
            or len(self.content.encode("utf-8")) > _MAX_VARCHAR_BYTES
        ):
            raise ValueError("invalid vector content")
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.content_sha256:
            raise ValueError("invalid vector content digest")
        _validate_metadata(self.title)
        _validate_metadata(self.parent_title)
        _validate_metadata(self.file_title)
        _validate_metadata(self.item_name)


@dataclass(frozen=True, slots=True)
class CompletedImportVersion:
    """Mongo-backed activation evidence for exactly one immutable version."""

    task_id: str
    tenant_id: str
    document_id: str
    commit_token: str

    def __post_init__(self) -> None:
        _canonical_uuid(self.task_id)
        _validate_tenant_id(self.tenant_id)
        _canonical_uuid(self.document_id)
        _validate_commit_token(self.commit_token)

    @property
    def import_version(self) -> str:
        return self.task_id


class VectorCandidateSource(Protocol):
    """Trusted adapter around vector retrieval; implementations may optimize by tenant.

    The service still validates every returned candidate against both its tenant
    and Mongo activation evidence, so a backend filter can never grant visibility.
    """

    def search_candidates(
        self,
        query: HybridVectorQuery,
        *,
        tenant_id: str,
        limit: int,
    ) -> Sequence[VectorCandidate]: ...


class CompletedImportVisibilityReader(Protocol):
    def get_completed_versions(
        self,
        tenant_id: str,
        import_versions: Sequence[str],
    ) -> Mapping[str, CompletedImportVersion]: ...


@dataclass(frozen=True, slots=True)
class TenantVectorSearchResult:
    hits: tuple[VectorCandidate, ...]
    examined_candidates: int
    rejected_candidates: int
    source_limit: int
    source_exhausted: bool
    candidate_budget_exhausted: bool


class TenantVectorSearch:
    """Bounded retrieval followed by authoritative Mongo publication filtering."""

    def __init__(
        self,
        source: VectorCandidateSource,
        visibility: CompletedImportVisibilityReader,
        *,
        initial_overfetch_factor: int = 3,
        growth_factor: int = 2,
        max_candidates: int = 500,
        max_results: int | None = None,
    ) -> None:
        if (
            isinstance(initial_overfetch_factor, bool)
            or not isinstance(initial_overfetch_factor, int)
            or initial_overfetch_factor < 1
        ):
            raise ValueError("invalid initial overfetch factor")
        if (
            isinstance(growth_factor, bool)
            or not isinstance(growth_factor, int)
            or growth_factor < 2
        ):
            raise ValueError("invalid overfetch growth factor")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 1
        ):
            raise ValueError("invalid vector candidate budget")
        if max_results is None:
            resolved_max_results = min(50, max_candidates)
        elif (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= max_candidates
        ):
            raise ValueError("invalid vector result limit")
        else:
            resolved_max_results = max_results
        self._source = source
        self._visibility = visibility
        self._initial_overfetch_factor = initial_overfetch_factor
        self._growth_factor = growth_factor
        self._max_candidates = max_candidates
        self._max_results = resolved_max_results

    @staticmethod
    def _request_invalid() -> AppError:
        return AppError(
            ErrorCode.REQUEST_INVALID,
            "Vector search request is invalid",
            422,
        )

    def _validate_request(
        self,
        tenant_id: object,
        query: object,
        limit: object,
    ) -> tuple[str, HybridVectorQuery, int]:
        try:
            tenant = _validate_tenant_id(tenant_id)
        except ValueError:
            raise self._request_invalid() from None
        if not isinstance(query, HybridVectorQuery):
            raise self._request_invalid()
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= self._max_results
        ):
            raise self._request_invalid()
        return tenant, query, limit

    @staticmethod
    def _dependency_unavailable() -> DependencyError:
        return DependencyError(_UNAVAILABLE_MESSAGE)

    def _search_source(
        self,
        query: HybridVectorQuery,
        tenant_id: str,
        limit: int,
    ) -> tuple[VectorCandidate, ...]:
        try:
            candidates = self._source.search_candidates(
                query,
                tenant_id=tenant_id,
                limit=limit,
            )
        except Exception:
            raise self._dependency_unavailable() from None
        if (
            isinstance(candidates, (str, bytes, bytearray))
            or not isinstance(candidates, Sequence)
            or len(candidates) > limit
            or any(not isinstance(candidate, VectorCandidate) for candidate in candidates)
        ):
            raise self._dependency_unavailable()
        return tuple(candidates)

    def _load_activations(
        self,
        tenant_id: str,
        import_versions: Sequence[str],
    ) -> dict[str, CompletedImportVersion]:
        try:
            observed = self._visibility.get_completed_versions(tenant_id, import_versions)
        except Exception:
            raise self._dependency_unavailable() from None
        requested = frozenset(import_versions)
        if not isinstance(observed, Mapping):
            raise self._dependency_unavailable()
        activations: dict[str, CompletedImportVersion] = {}
        for version, activation in observed.items():
            if (
                not isinstance(version, str)
                or version not in requested
                or not isinstance(activation, CompletedImportVersion)
                or activation.task_id != version
                or activation.tenant_id != tenant_id
            ):
                raise self._dependency_unavailable()
            activations[version] = activation
        return activations

    @staticmethod
    def _is_visible(candidate: VectorCandidate, activation: CompletedImportVersion | None) -> bool:
        return (
            activation is not None
            and activation.task_id == candidate.import_version
            and activation.tenant_id == candidate.tenant_id
            and activation.document_id == candidate.document_id
            and activation.commit_token == candidate.commit_token
        )

    def search(
        self,
        *,
        tenant_id: str,
        query: HybridVectorQuery,
        limit: int,
    ) -> TenantVectorSearchResult:
        tenant, validated_query, result_limit = self._validate_request(tenant_id, query, limit)
        source_limit = min(
            self._max_candidates,
            max(result_limit, result_limit * self._initial_overfetch_factor),
        )
        hits: list[VectorCandidate] = []
        hit_ids: set[str] = set()
        seen_chunks: dict[str, VectorCandidate] = {}
        activation_cache: dict[str, CompletedImportVersion | None] = {}
        examined = 0
        rejected = 0
        source_exhausted = False
        candidate_budget_reached = False

        while True:
            candidates = self._search_source(validated_query, tenant, source_limit)
            source_exhausted = len(candidates) < source_limit
            newly_scoped: list[VectorCandidate] = []
            versions_to_load: list[str] = []
            pending_versions: set[str] = set()

            for candidate in candidates:
                previous_candidate = seen_chunks.get(candidate.chunk_id)
                if previous_candidate is not None:
                    if previous_candidate != candidate:
                        raise self._dependency_unavailable()
                    continue
                if len(seen_chunks) >= self._max_candidates:
                    candidate_budget_reached = True
                    break
                seen_chunks[candidate.chunk_id] = candidate
                if candidate.tenant_id != tenant:
                    examined += 1
                    rejected += 1
                    continue
                newly_scoped.append(candidate)
                if (
                    candidate.import_version not in activation_cache
                    and candidate.import_version not in pending_versions
                ):
                    versions_to_load.append(candidate.import_version)
                    pending_versions.add(candidate.import_version)

            if versions_to_load:
                loaded = self._load_activations(tenant, versions_to_load)
                for version in versions_to_load:
                    activation_cache[version] = loaded.get(version)

            for candidate in newly_scoped:
                examined += 1
                activation = activation_cache.get(candidate.import_version)
                if not self._is_visible(candidate, activation):
                    rejected += 1
                    continue
                if candidate.chunk_id in hit_ids:
                    continue
                hit_ids.add(candidate.chunk_id)
                hits.append(candidate)
                if len(hits) == result_limit:
                    return TenantVectorSearchResult(
                        hits=tuple(hits),
                        examined_candidates=examined,
                        rejected_candidates=rejected,
                        source_limit=source_limit,
                        source_exhausted=source_exhausted,
                        candidate_budget_exhausted=False,
                    )

            if candidate_budget_reached or source_exhausted or source_limit >= self._max_candidates:
                break
            next_limit = min(self._max_candidates, source_limit * self._growth_factor)
            if next_limit <= source_limit:
                break
            source_limit = next_limit

        return TenantVectorSearchResult(
            hits=tuple(hits),
            examined_candidates=examined,
            rejected_candidates=rejected,
            source_limit=source_limit,
            source_exhausted=source_exhausted,
            candidate_budget_exhausted=(
                len(hits) < result_limit
                and (
                    candidate_budget_reached
                    or (not source_exhausted and source_limit >= self._max_candidates)
                )
            ),
        )
