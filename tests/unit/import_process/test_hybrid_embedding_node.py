from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import Event, Thread

import pytest

from app.core.errors import AppError, DependencyError, ErrorCode
from app.import_process.agent.nodes.node_hybrid_embedding import (
    BgeM3ModelConfig,
    EmbeddingBackendOutputError,
    EmbeddingBackendUnavailable,
    HybridEmbedding,
    HybridEmbeddingNode,
    HybridEmbeddingPolicy,
    LazyLocalBgeM3Backend,
)
from app.import_process.agent.nodes.safe_node_contracts import (
    EMBEDDED_CHUNK_FIELDS,
    SafeNodeContractError,
)
from app.import_process.agent.state import ImportGraphState, create_default_state


def _chunks(count: int = 5) -> list[dict[str, object]]:
    return [
        {
            "content": f"chunk-{index}",
            "title": "Section",
            "parent_title": "Manual",
            "file_title": "Guide",
            "item_name": "Product",
        }
        for index in range(count)
    ]


def _state(count: int = 5) -> ImportGraphState:
    return create_default_state(chunks=_chunks(count))


class FakeBackend:
    dense_dimension = 3

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.probe_calls = 0
        self.close_calls = 0
        self.failure: BaseException | None = None
        self.result: Sequence[HybridEmbedding] | None = None

    def embed(self, texts: Sequence[str]) -> Sequence[HybridEmbedding]:
        self.calls.append(tuple(texts))
        if self.failure is not None:
            raise self.failure
        if self.result is not None:
            return self.result
        return tuple(
            HybridEmbedding(
                dense=(float(index), 0.2, 0.3),
                sparse={index + 1: 0.5},
            )
            for index, _ in enumerate(texts)
        )

    def probe(self) -> None:
        self.probe_calls += 1
        if self.failure is not None:
            raise self.failure

    def close(self) -> None:
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure


def test_hybrid_embedding_node_batches_and_emits_only_fixed_fields() -> None:
    backend = FakeBackend()
    node = HybridEmbeddingNode(backend, policy=HybridEmbeddingPolicy(batch_size=2))
    state = _state()
    before = copy.deepcopy(state)

    update = node(state)

    assert state == before
    assert backend.calls == [
        ("chunk-0", "chunk-1"),
        ("chunk-2", "chunk-3"),
        ("chunk-4",),
    ]
    chunks = update["chunks"]
    assert isinstance(chunks, list)
    assert len(chunks) == 5
    assert all(set(chunk) == EMBEDDED_CHUNK_FIELDS for chunk in chunks)
    assert chunks[0]["dense_vector"] == [0.0, 0.2, 0.3]
    assert chunks[0]["sparse_vector"] == {1: 0.5}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda chunks: chunks[0].update({"tenant_id": "smuggled"}),
        lambda chunks: chunks[0].update({"content": ""}),
        lambda chunks: chunks[0].update({"content": "x" * 7_000}),
    ],
)
def test_embedding_input_is_validated_before_backend_access(
    mutate: Callable[[list[dict[str, object]]], None],
) -> None:
    backend = FakeBackend()
    chunks = _chunks(1)
    mutate(chunks)
    state = create_default_state(chunks=chunks)

    with pytest.raises(SafeNodeContractError):
        HybridEmbeddingNode(backend)(state)

    assert backend.calls == []


def test_embedding_total_input_bytes_are_bounded_before_backend_access() -> None:
    backend = FakeBackend()
    node = HybridEmbeddingNode(
        backend,
        policy=HybridEmbeddingPolicy(max_total_input_bytes=5),
    )

    with pytest.raises(SafeNodeContractError, match="byte limit"):
        node(_state(1))

    assert backend.calls == []


@pytest.mark.parametrize(
    "result",
    [
        (),
        (HybridEmbedding(dense=(0.1, 0.2), sparse={1: 0.5}),),
        (HybridEmbedding(dense=(0.1, float("nan"), 0.3), sparse={1: 0.5}),),
        (HybridEmbedding(dense=(0.1, 0.2, 0.3), sparse={-1: 0.5}),),
        (HybridEmbedding(dense=(0.1, 0.2, 0.3), sparse={1: float("inf")}),),
        (HybridEmbedding(dense=(0.1, 0.2, 0.3), sparse={1: 0.0}),),
    ],
)
def test_invalid_backend_vectors_fail_closed_with_sanitized_error(
    result: Sequence[HybridEmbedding],
) -> None:
    backend = FakeBackend()
    backend.result = result

    with pytest.raises(AppError) as captured:
        HybridEmbeddingNode(backend)(_state(1))

    assert captured.value.code is ErrorCode.INTERNAL_ERROR
    assert str(captured.value) == "Embedding output validation failed"


def test_backend_dependency_error_is_sanitized_and_cancellation_propagates() -> None:
    backend = FakeBackend()
    backend.failure = RuntimeError("secret model path and device")
    with pytest.raises(DependencyError) as captured:
        HybridEmbeddingNode(backend)(_state(1))
    assert captured.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert "secret" not in str(captured.value)

    backend.failure = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        HybridEmbeddingNode(backend)(_state(1))


def test_embedding_probe_and_close_are_explicit_and_fail_loudly() -> None:
    backend = FakeBackend()
    node = HybridEmbeddingNode(backend)
    assert backend.probe_calls == backend.close_calls == 0

    node.probe()
    node.close()

    assert backend.probe_calls == backend.close_calls == 1
    backend.failure = RuntimeError("secret runtime detail")
    with pytest.raises(DependencyError) as captured:
        node.probe()
    assert "secret" not in str(captured.value)


class ArrayLike:
    def __init__(self, value: object) -> None:
        self._value = value

    def tolist(self) -> object:
        return self._value


class FakeEncoder:
    def __init__(self, raw: object) -> None:
        self.raw = raw
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.close_calls = 0
        self.failure: Exception | None = None

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
        self.calls.append(
            (
                sentences,
                {
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "return_dense": return_dense,
                    "return_sparse": return_sparse,
                    "return_colbert_vecs": return_colbert_vecs,
                },
            )
        )
        if self.failure is not None:
            raise self.failure
        return self.raw

    def close(self) -> None:
        self.close_calls += 1


class BlockingEncoder(FakeEncoder):
    def __init__(self, raw: object) -> None:
        super().__init__(raw)
        self.entered = Event()
        self.release = Event()

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
        if sentences != ["health"]:
            self.entered.set()
            if not self.release.wait(timeout=2):
                raise TimeoutError("test inference was not released")
        return super().encode(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
        )


def _raw_vectors(count: int = 2) -> Mapping[str, object]:
    return {
        "dense_vecs": ArrayLike([[0.1 + index, 0.2, 0.3] for index in range(count)]),
        "lexical_weights": [{"9": 0.4, "2": 0.7 + index} for index in range(count)],
        "colbert_vecs": None,
    }


def test_local_bge_backend_is_lazy_reuses_model_and_normalizes_output(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    encoder = FakeEncoder(_raw_vectors())
    configs: list[BgeM3ModelConfig] = []

    def factory(config: BgeM3ModelConfig) -> FakeEncoder:
        configs.append(config)
        return encoder

    backend = LazyLocalBgeM3Backend(
        model_dir,
        dense_dimension=3,
        model_batch_size=8,
        max_tokens=512,
        model_factory=factory,
    )
    assert configs == []

    vectors = backend.embed(("one", "two"))
    backend.embed(("one", "two"))

    assert len(configs) == 1
    assert configs[0].model_path == model_dir.resolve()
    assert encoder.calls[0] == (
        ["one", "two"],
        {
            "batch_size": 2,
            "max_length": 512,
            "return_dense": True,
            "return_sparse": True,
            "return_colbert_vecs": False,
        },
    )
    assert vectors == (
        HybridEmbedding(dense=(0.1, 0.2, 0.3), sparse={2: 0.7, 9: 0.4}),
        HybridEmbedding(dense=(1.1, 0.2, 0.3), sparse={2: 1.7, 9: 0.4}),
    )


def test_local_bge_construction_does_not_import_or_load_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls = 0

    def forbidden_import(name: str) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError(name)

    monkeypatch.setattr(
        "app.import_process.agent.nodes.node_hybrid_embedding.importlib.import_module",
        forbidden_import,
    )

    LazyLocalBgeM3Backend(model_dir, dense_dimension=3)

    assert calls == 0


def test_local_bge_probe_loads_validates_and_close_is_terminal(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    encoder = FakeEncoder(_raw_vectors(1))
    backend = LazyLocalBgeM3Backend(
        model_dir,
        dense_dimension=3,
        model_factory=lambda config: encoder,
    )

    backend.probe()
    backend.probe()
    backend.close()
    backend.close()

    assert encoder.calls[0][0] == ["health"]
    assert len(encoder.calls) == 1
    assert encoder.close_calls == 1
    with pytest.raises(EmbeddingBackendUnavailable, match="closed"):
        backend.embed(("after-close",))


def test_local_bge_probe_does_not_block_behind_active_inference_after_warmup(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    encoder = BlockingEncoder(_raw_vectors(1))
    backend = LazyLocalBgeM3Backend(
        model_dir,
        dense_dimension=3,
        model_factory=lambda config: encoder,
    )
    backend.probe()

    inference = Thread(target=lambda: backend.embed(("busy",)), daemon=True)
    inference.start()
    assert encoder.entered.wait(timeout=1)

    readiness = Thread(target=backend.probe, daemon=True)
    readiness.start()
    readiness.join(timeout=0.2)

    try:
        assert not readiness.is_alive()
    finally:
        encoder.release.set()
        inference.join(timeout=2)
        readiness.join(timeout=2)
        backend.close()


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"dense_vecs": [[0.1, 0.2]], "lexical_weights": [{"1": 0.5}]},
        {"dense_vecs": [[0.1, 0.2, float("nan")]], "lexical_weights": [{"1": 0.5}]},
        {"dense_vecs": [[0.1, 0.2, 0.3]], "lexical_weights": [{}]},
        {"dense_vecs": [[0.1, 0.2, 0.3]], "lexical_weights": [{"01": 0.5}]},
        {"dense_vecs": [[0.1, 0.2, 0.3]], "lexical_weights": [{"1": -0.5}]},
    ],
)
def test_local_bge_rejects_invalid_dense_and_sparse_shapes(
    tmp_path: Path,
    raw: object,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    encoder = FakeEncoder(raw)
    backend = LazyLocalBgeM3Backend(
        model_dir,
        dense_dimension=3,
        model_factory=lambda config: encoder,
    )

    with pytest.raises(EmbeddingBackendOutputError, match="validation failed"):
        backend.embed(("one",))


def test_local_bge_load_and_inference_failures_do_not_leak_details(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "secret-model-path"
    backend = LazyLocalBgeM3Backend(
        missing,
        dense_dimension=3,
        model_factory=lambda config: FakeEncoder(_raw_vectors(1)),
    )
    with pytest.raises(EmbeddingBackendUnavailable) as captured:
        backend.embed(("one",))
    assert "secret-model-path" not in str(captured.value)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    encoder = FakeEncoder(_raw_vectors(1))
    encoder.failure = RuntimeError("secret CUDA allocation detail")
    backend = LazyLocalBgeM3Backend(
        model_dir,
        dense_dimension=3,
        model_factory=lambda config: encoder,
    )
    with pytest.raises(EmbeddingBackendUnavailable) as captured:
        backend.embed(("one",))
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize(
    "model_path",
    ["relative/model", ""],
)
def test_local_bge_requires_an_explicit_local_absolute_path(model_path: str) -> None:
    with pytest.raises(ValueError, match="absolute"):
        LazyLocalBgeM3Backend(model_path)
