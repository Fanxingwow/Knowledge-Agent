from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from app.core.errors import DependencyError, ErrorCode
from app.import_process.agent.nodes.node_hybrid_embedding import (
    BgeM3ModelConfig,
    HybridEmbedding,
    LazyLocalBgeM3Backend,
)
from app.infrastructure.query_embedding_adapter import HybridQueryEmbeddingAdapter


class FakeBackend:
    def __init__(self, *, dense_dimension: object = 3) -> None:
        self._dense_dimension = dense_dimension
        self.calls: list[tuple[str, ...]] = []
        self.result: object = (
            HybridEmbedding(dense=(0.1, 0.2, 0.3), sparse={9: 0.4, 2: 0.7}),
        )
        self.failure: BaseException | None = None
        self.dimension_failure: Exception | None = None
        self.probe_calls = 0
        self.close_calls = 0

    @property
    def dense_dimension(self) -> int:
        if self.dimension_failure is not None:
            raise self.dimension_failure
        return cast(int, self._dense_dimension)

    def embed(self, texts: Sequence[str]) -> Sequence[HybridEmbedding]:
        self.calls.append(tuple(texts))
        if self.failure is not None:
            raise self.failure
        return cast(Sequence[HybridEmbedding], self.result)

    def probe(self) -> None:
        self.probe_calls += 1
        if self.failure is not None:
            raise self.failure

    def close(self) -> None:
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure


def test_single_query_is_a_one_element_batch_and_output_is_deeply_normalized() -> None:
    dense = [0.1, 0.2, 0.3]
    sparse = {9: 0.4, 2: 0.7}
    backend = FakeBackend()
    backend.result = (
        HybridEmbedding(
            dense=cast(tuple[float, ...], dense),
            sparse=sparse,
        ),
    )
    adapter = HybridQueryEmbeddingAdapter(backend)

    result = adapter.embed_query("one query")
    dense[0] = 9.9
    sparse[2] = 8.8

    assert backend.calls == [("one query",)]
    assert adapter.dense_dimension == 3
    assert result.dense_vector == (0.1, 0.2, 0.3)
    assert result.sparse_vector == {2: 0.7, 9: 0.4}
    with pytest.raises(TypeError):
        result.sparse_vector[2] = 1.0  # type: ignore[index]


@pytest.mark.parametrize(
    "result",
    [
        (),
        (
            HybridEmbedding(dense=(0.1, 0.2, 0.3), sparse={1: 0.5}),
            HybridEmbedding(dense=(0.4, 0.5, 0.6), sparse={2: 0.5}),
        ),
        "not-a-batch",
        (object(),),
    ],
)
def test_output_must_contain_exactly_one_hybrid_embedding(result: object) -> None:
    backend = FakeBackend()
    backend.result = result
    adapter = HybridQueryEmbeddingAdapter(backend)

    with pytest.raises(DependencyError) as raised:
        adapter.embed_query("query")

    assert raised.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert raised.value.message == "Query embedding is unavailable"
    assert raised.value.details == {}
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    "dense",
    [
        (),
        (0.1, 0.2),
        (0.1, 0.2, 0.3, 0.4),
        (0.1, float("nan"), 0.3),
        (0.1, float("inf"), 0.3),
        (0.1, True, 0.3),
        (0.1, "0.2", 0.3),
    ],
)
def test_dense_vector_dimension_and_finite_values_are_strict(dense: object) -> None:
    backend = FakeBackend()
    backend.result = (
        HybridEmbedding(
            dense=cast(tuple[float, ...], dense),
            sparse={1: 0.5},
        ),
    )
    adapter = HybridQueryEmbeddingAdapter(backend)

    with pytest.raises(DependencyError, match="Query embedding is unavailable"):
        adapter.embed_query("query")


@pytest.mark.parametrize(
    "sparse",
    [
        {},
        {-1: 0.5},
        {6: 0.5},
        {True: 0.5},
        {1.0: 0.5},
        {"1": 0.5},
        {1: float("nan")},
        {1: float("inf")},
        {1: True},
        {1: 0.0},
        {1: -0.5},
        {1: 0.5, 2: 0.4, 3: 0.3},
    ],
)
def test_sparse_indices_cardinality_and_finite_positive_values_are_strict(
    sparse: object,
) -> None:
    backend = FakeBackend()
    backend.result = (
        HybridEmbedding(
            dense=(0.1, 0.2, 0.3),
            sparse=cast(Mapping[int, float], sparse),
        ),
    )
    adapter = HybridQueryEmbeddingAdapter(
        backend,
        max_sparse_index=5,
        max_sparse_nonzero=2,
    )

    with pytest.raises(DependencyError, match="Query embedding is unavailable"):
        adapter.embed_query("query")


@pytest.mark.parametrize("dense_dimension", [True, 0, -1, 32_769, "3"])
def test_declared_dense_dimension_is_validated_at_construction(
    dense_dimension: object,
) -> None:
    backend = FakeBackend(dense_dimension=dense_dimension)

    with pytest.raises(ValueError, match="dimension"):
        HybridQueryEmbeddingAdapter(backend)

    assert backend.calls == []


def test_dimension_lookup_failure_is_sanitized_at_construction() -> None:
    leaked_detail = "secret model declaration"
    backend = FakeBackend()
    backend.dimension_failure = RuntimeError(leaked_detail)

    with pytest.raises(DependencyError) as raised:
        HybridQueryEmbeddingAdapter(backend)

    assert raised.value.message == "Query embedding is unavailable"
    assert leaked_detail not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("max_sparse_index", "max_sparse_nonzero"),
    [
        (True, 1),
        (0, 1),
        (2_147_483_648, 1),
        (1, True),
        (1, 0),
        (1, 65_537),
    ],
)
def test_sparse_policy_limits_are_strict(
    max_sparse_index: object,
    max_sparse_nonzero: object,
) -> None:
    with pytest.raises(ValueError, match="limit"):
        HybridQueryEmbeddingAdapter(
            FakeBackend(),
            max_sparse_index=max_sparse_index,  # type: ignore[arg-type]
            max_sparse_nonzero=max_sparse_nonzero,  # type: ignore[arg-type]
        )


def test_invalid_query_is_rejected_before_backend_access() -> None:
    backend = FakeBackend()
    adapter = HybridQueryEmbeddingAdapter(backend)

    for query in ("", None, 1):
        with pytest.raises(ValueError, match="input"):
            adapter.embed_query(query)  # type: ignore[arg-type]

    assert backend.calls == []


def test_backend_failures_are_sanitized_without_exception_chaining() -> None:
    leaked_detail = "C:/private/model?credential=must-not-leak"
    backend = FakeBackend()
    backend.failure = RuntimeError(leaked_detail)
    adapter = HybridQueryEmbeddingAdapter(backend)

    with pytest.raises(DependencyError) as raised:
        adapter.embed_query("query")

    rendered = f"{raised.value!r} {raised.value!s} {raised.value.details!r}"
    assert raised.value.message == "Query embedding is unavailable"
    assert raised.value.details == {}
    assert raised.value.__cause__ is None
    assert leaked_detail not in rendered


def test_backend_timeout_keeps_type_but_is_sanitized_for_upper_deadline_layer() -> None:
    leaked_detail = "timeout contacting secret accelerator host"
    backend = FakeBackend()
    backend.failure = TimeoutError(leaked_detail)
    adapter = HybridQueryEmbeddingAdapter(backend)

    with pytest.raises(TimeoutError) as raised:
        adapter.embed_query("query")

    assert str(raised.value) == "Query embedding timed out"
    assert raised.value.__cause__ is None
    assert leaked_detail not in str(raised.value)


def test_cancellation_propagates_unchanged_to_upper_layer() -> None:
    cancellation = asyncio.CancelledError("cancel inference")
    backend = FakeBackend()
    backend.failure = cancellation
    adapter = HybridQueryEmbeddingAdapter(backend)

    with pytest.raises(asyncio.CancelledError) as raised:
        adapter.embed_query("query")

    assert raised.value is cancellation


def test_probe_and_close_delegate_explicitly_and_sanitize_failures() -> None:
    backend = FakeBackend()
    adapter = HybridQueryEmbeddingAdapter(backend)

    adapter.probe()
    adapter.close()

    assert backend.probe_calls == 1
    assert backend.close_calls == 1
    backend.failure = RuntimeError("secret device state")
    with pytest.raises(DependencyError, match="Query embedding is unavailable"):
        adapter.probe()
    with pytest.raises(DependencyError, match="Query embedding is unavailable"):
        adapter.close()


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert_vecs: bool,
    ) -> object:
        self.calls.append(sentences)
        return {
            "dense_vecs": [[0.1, 0.2, 0.3]],
            "lexical_weights": [{"4": 0.6}],
        }


def test_adapter_wraps_lazy_local_bge_without_eager_model_loading(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    encoder = FakeEncoder()
    factory_calls: list[BgeM3ModelConfig] = []

    def factory(config: BgeM3ModelConfig) -> FakeEncoder:
        factory_calls.append(config)
        return encoder

    backend = LazyLocalBgeM3Backend(
        model_dir,
        dense_dimension=3,
        model_factory=factory,
    )
    adapter = HybridQueryEmbeddingAdapter(backend)
    assert factory_calls == []

    result = adapter.embed_query("tenant query")

    assert len(factory_calls) == 1
    assert encoder.calls == [["tenant query"]]
    assert result.dense_vector == (0.1, 0.2, 0.3)
    assert result.sparse_vector == {4: 0.6}
