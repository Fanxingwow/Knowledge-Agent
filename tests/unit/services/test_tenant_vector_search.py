from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

from app.core.errors import AppError, DependencyError, ErrorCode
from app.services.tenant_vector_search import (
    CompletedImportVersion,
    HybridVectorQuery,
    TenantVectorSearch,
    VectorCandidate,
)

_DOCUMENTS = tuple(f"00000000-0000-4000-8000-{index:012d}" for index in range(1, 12))
_VERSIONS = tuple(f"10000000-0000-4000-8000-{index:012d}" for index in range(1, 12))
_QUERY = HybridVectorQuery(dense_vector=(0.1, 0.2), sparse_vector={1: 0.5})
_DIFFERENT_COMMIT_PROOF = "milvus-v1.different"
_CHANGED_COMMIT_PROOF = "milvus-v1.changed"


def candidate(
    index: int,
    *,
    tenant_id: str = "tenant-a",
    commit_token: str | None = None,
) -> VectorCandidate:
    return VectorCandidate(
        chunk_id=f"{index:064x}",
        tenant_id=tenant_id,
        document_id=_DOCUMENTS[index - 1],
        import_version=_VERSIONS[index - 1],
        commit_token=commit_token if commit_token is not None else f"milvus-v1.token-{index}",
        content_sha256=hashlib.sha256(f"chunk {index}".encode()).hexdigest(),
        score=1.0 - index / 100,
        content=f"chunk {index}",
        item_name="item",
    )


def activation(
    item: VectorCandidate,
    *,
    tenant_id: str | None = None,
    document_id: str | None = None,
    commit_token: str | None = None,
) -> CompletedImportVersion:
    return CompletedImportVersion(
        task_id=item.import_version,
        tenant_id=tenant_id or item.tenant_id,
        document_id=document_id or item.document_id,
        commit_token=commit_token or item.commit_token,
    )


class PrefixSource:
    def __init__(self, candidates: Sequence[VectorCandidate]) -> None:
        self.candidates = tuple(candidates)
        self.calls: list[tuple[str, int]] = []

    def search_candidates(
        self,
        query: HybridVectorQuery,
        *,
        tenant_id: str,
        limit: int,
    ) -> Sequence[VectorCandidate]:
        assert query is _QUERY
        self.calls.append((tenant_id, limit))
        return self.candidates[:limit]


class DisjointSource(PrefixSource):
    def __init__(self, batches: Sequence[Sequence[VectorCandidate]]) -> None:
        super().__init__(())
        self.batches = tuple(tuple(batch) for batch in batches)

    def search_candidates(
        self,
        query: HybridVectorQuery,
        *,
        tenant_id: str,
        limit: int,
    ) -> Sequence[VectorCandidate]:
        self.calls.append((tenant_id, limit))
        return self.batches[len(self.calls) - 1]


class Visibility:
    def __init__(self, completed: Mapping[str, CompletedImportVersion]) -> None:
        self.completed = dict(completed)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def get_completed_versions(
        self,
        tenant_id: str,
        import_versions: Sequence[str],
    ) -> Mapping[str, CompletedImportVersion]:
        versions = tuple(import_versions)
        self.calls.append((tenant_id, versions))
        return {version: self.completed[version] for version in versions if version in self.completed}


def test_returns_only_exact_completed_tenant_version_and_commit() -> None:
    allowed = candidate(1)
    wrong_document = candidate(2)
    wrong_commit = candidate(3)
    inactive = candidate(4)
    source = PrefixSource((allowed, wrong_document, wrong_commit, inactive))
    visibility = Visibility(
        {
            allowed.import_version: activation(allowed),
            wrong_document.import_version: activation(
                wrong_document,
                document_id=_DOCUMENTS[7],
            ),
            wrong_commit.import_version: activation(
                wrong_commit,
                commit_token=_DIFFERENT_COMMIT_PROOF,
            ),
        }
    )
    search = TenantVectorSearch(
        source,
        visibility,
        initial_overfetch_factor=4,
        max_candidates=4,
    )

    result = search.search(tenant_id="tenant-a", query=_QUERY, limit=2)

    assert result.hits == (allowed,)
    assert result.examined_candidates == 4
    assert result.rejected_candidates == 3
    assert result.candidate_budget_exhausted is True


def test_cross_tenant_candidates_are_rejected_before_mongo_lookup() -> None:
    foreign = candidate(1, tenant_id="tenant-b")
    allowed = candidate(2)
    source = PrefixSource((foreign, allowed))
    visibility = Visibility({allowed.import_version: activation(allowed)})
    search = TenantVectorSearch(
        source,
        visibility,
        initial_overfetch_factor=1,
        max_candidates=2,
    )

    result = search.search(tenant_id="tenant-a", query=_QUERY, limit=2)

    assert result.hits == (allowed,)
    assert visibility.calls == [("tenant-a", (allowed.import_version,))]
    assert result.rejected_candidates == 1


def test_bounded_overfetch_expands_until_enough_visible_hits() -> None:
    candidates = tuple(candidate(index) for index in range(1, 9))
    visible = {
        candidates[3].import_version: activation(candidates[3]),
        candidates[4].import_version: activation(candidates[4]),
    }
    source = PrefixSource(candidates)
    visibility = Visibility(visible)
    search = TenantVectorSearch(
        source,
        visibility,
        initial_overfetch_factor=2,
        growth_factor=2,
        max_candidates=8,
    )

    result = search.search(tenant_id="tenant-a", query=_QUERY, limit=2)

    assert result.hits == (candidates[3], candidates[4])
    assert source.calls == [("tenant-a", 4), ("tenant-a", 8)]
    assert result.source_limit == 8
    assert result.candidate_budget_exhausted is False
    looked_up = [version for _, versions in visibility.calls for version in versions]
    assert looked_up == list(candidate.import_version for candidate in candidates)
    assert len(looked_up) == len(set(looked_up))


def test_candidate_budget_is_never_exceeded() -> None:
    candidates = tuple(candidate(index) for index in range(1, 7))
    source = PrefixSource(candidates)
    visibility = Visibility({})
    search = TenantVectorSearch(
        source,
        visibility,
        initial_overfetch_factor=1,
        growth_factor=2,
        max_candidates=5,
    )

    result = search.search(tenant_id="tenant-a", query=_QUERY, limit=2)

    assert source.calls == [("tenant-a", 2), ("tenant-a", 4), ("tenant-a", 5)]
    assert max(limit for _, limit in source.calls) == 5
    assert result.hits == ()
    assert result.examined_candidates == 5
    assert result.rejected_candidates == 5
    assert result.candidate_budget_exhausted is True


def test_total_unique_candidates_are_bounded_even_if_source_is_not_prefix_stable() -> None:
    items = tuple(candidate(index) for index in range(1, 7))
    source = DisjointSource((items[:2], items[2:5]))
    search = TenantVectorSearch(
        source,
        Visibility({}),
        initial_overfetch_factor=1,
        growth_factor=2,
        max_candidates=3,
    )

    result = search.search(tenant_id="tenant-a", query=_QUERY, limit=2)

    assert source.calls == [("tenant-a", 2), ("tenant-a", 3)]
    assert result.examined_candidates == 3
    assert result.rejected_candidates == 3
    assert result.candidate_budget_exhausted is True


def test_source_exhaustion_stops_expansion() -> None:
    item = candidate(1)
    source = PrefixSource((item,))
    visibility = Visibility({})
    search = TenantVectorSearch(source, visibility, initial_overfetch_factor=2)

    result = search.search(tenant_id="tenant-a", query=_QUERY, limit=2)

    assert source.calls == [("tenant-a", 4)]
    assert result.source_exhausted is True
    assert result.candidate_budget_exhausted is False


@pytest.mark.parametrize("tenant_id", ["", "../tenant", "tenant a", "tenant\nother"])
def test_tenant_is_mandatory_and_strictly_validated(tenant_id: str) -> None:
    search = TenantVectorSearch(PrefixSource(()), Visibility({}))

    with pytest.raises(AppError) as raised:
        search.search(tenant_id=tenant_id, query=_QUERY, limit=1)

    assert raised.value.code is ErrorCode.REQUEST_INVALID
    assert raised.value.message == "Vector search request is invalid"


def test_query_contract_does_not_accept_a_raw_client_filter() -> None:
    with pytest.raises(TypeError):
        HybridVectorQuery(  # type: ignore[call-arg]
            dense_vector=(0.1,),
            sparse_vector={1: 0.2},
            filter='tenant_id == "tenant-b"',
        )


def test_query_vectors_are_deeply_normalized_and_immutable() -> None:
    dense = [0.1, 0.2]
    sparse = {1: 0.3}

    query = HybridVectorQuery(dense, sparse)
    dense[0] = 9.9
    sparse[1] = 8.8

    assert query.dense_vector == (0.1, 0.2)
    assert query.sparse_vector == {1: 0.3}
    assert isinstance(query.dense_vector, tuple)
    with pytest.raises(TypeError):
        query.sparse_vector[1] = 7.7  # type: ignore[index]


@pytest.mark.parametrize("commit_token", ["", "token with spaces", "token\nsecret"])
def test_candidate_requires_a_safe_commit_token(commit_token: str) -> None:
    with pytest.raises(ValueError, match="commit token"):
        candidate(1, commit_token=commit_token)


def test_candidate_content_obeys_milvus_utf8_byte_limit() -> None:
    with pytest.raises(ValueError, match="content"):
        replace(
            candidate(1),
            content="中" * 21_846,
            content_sha256="a" * 64,
        )


def test_candidate_source_contract_violation_fails_closed() -> None:
    class InvalidSource:
        def search_candidates(
            self,
            query: HybridVectorQuery,
            *,
            tenant_id: str,
            limit: int,
        ) -> Sequence[VectorCandidate]:
            return [object()]  # type: ignore[list-item]

    search = TenantVectorSearch(InvalidSource(), Visibility({}))

    with pytest.raises(DependencyError) as raised:
        search.search(tenant_id="tenant-a", query=_QUERY, limit=1)

    assert str(raised.value) == "Knowledge search is unavailable"


@pytest.mark.parametrize("dependency", ["source", "visibility"])
def test_dependency_errors_are_sanitized(dependency: str) -> None:
    leaked_detail = "mongodb://root:credential@private-host"
    item = candidate(1)

    class FailingSource(PrefixSource):
        def search_candidates(
            self,
            query: HybridVectorQuery,
            *,
            tenant_id: str,
            limit: int,
        ) -> Sequence[VectorCandidate]:
            raise RuntimeError(leaked_detail)

    class FailingVisibility(Visibility):
        def get_completed_versions(
            self,
            tenant_id: str,
            import_versions: Sequence[str],
        ) -> Mapping[str, CompletedImportVersion]:
            raise RuntimeError(leaked_detail)

    source = FailingSource((item,)) if dependency == "source" else PrefixSource((item,))
    visibility = FailingVisibility({}) if dependency == "visibility" else Visibility({})
    search = TenantVectorSearch(source, visibility, initial_overfetch_factor=1)

    with pytest.raises(DependencyError) as raised:
        search.search(tenant_id="tenant-a", query=_QUERY, limit=1)

    assert str(raised.value) == "Knowledge search is unavailable"
    assert leaked_detail not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("drift", ["token", "content", "metadata", "score"])
def test_any_drift_for_same_chunk_fails_closed(drift: str) -> None:
    first = candidate(1)
    if drift == "token":
        conflicting = replace(first, commit_token=_CHANGED_COMMIT_PROOF)
    elif drift == "content":
        changed_content = "changed content"
        conflicting = replace(
            first,
            content=changed_content,
            content_sha256=hashlib.sha256(changed_content.encode()).hexdigest(),
        )
    elif drift == "metadata":
        conflicting = replace(first, title="changed title")
    else:
        conflicting = replace(first, score=0.01)
    search = TenantVectorSearch(
        PrefixSource((first, conflicting)),
        Visibility({}),
        initial_overfetch_factor=2,
    )

    with pytest.raises(DependencyError, match="Knowledge search is unavailable"):
        search.search(tenant_id="tenant-a", query=_QUERY, limit=1)


def test_candidate_content_must_match_its_digest() -> None:
    with pytest.raises(ValueError, match="content digest"):
        replace(candidate(1), content="tampered content")
