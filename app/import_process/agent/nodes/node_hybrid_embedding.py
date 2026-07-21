from __future__ import annotations

import importlib
import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Protocol, cast

from app.core.errors import AppError, DependencyError, ErrorCode
from app.import_process.agent.nodes.safe_node_contracts import (
    EmbeddedChunk,
    SafeNodeContractError,
    TextChunk,
    require_text_chunks,
)
from app.import_process.agent.state import ImportGraphState

_DEVICE = re.compile(r"^(?:cpu|mps|cuda(?::[0-9]+)?)$")
_SPARSE_STRING_INDEX = re.compile(r"^(?:0|[1-9][0-9]*)$")


class EmbeddingBackendUnavailable(RuntimeError):
    """The local embedding runtime could not complete an operation."""


class EmbeddingBackendOutputError(RuntimeError):
    """The embedding runtime returned data outside the fixed vector contract."""


@dataclass(frozen=True, slots=True)
class HybridEmbedding:
    dense: tuple[float, ...]
    sparse: Mapping[int, float]


class HybridEmbeddingBackend(Protocol):
    @property
    def dense_dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> Sequence[HybridEmbedding]: ...

    def probe(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BgeM3ModelConfig:
    model_path: Path
    device: str
    use_fp16: bool
    model_batch_size: int
    max_tokens: int


class BgeM3Encoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert_vecs: bool,
    ) -> object: ...


class _ListConvertible(Protocol):
    def tolist(self) -> object: ...


BgeM3ModelFactory = Callable[[BgeM3ModelConfig], BgeM3Encoder]


def _default_bge_m3_factory(config: BgeM3ModelConfig) -> BgeM3Encoder:
    module = importlib.import_module("FlagEmbedding.inference.embedder.encoder_only.m3")
    model_type = getattr(module, "M3Embedder", None)
    if not callable(model_type):
        raise RuntimeError("BGE-M3 implementation is unavailable")
    model = model_type(
        str(config.model_path),
        normalize_embeddings=True,
        use_fp16=config.use_fp16,
        devices=config.device,
        trust_remote_code=False,
        batch_size=config.model_batch_size,
        passage_max_length=config.max_tokens,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return cast(BgeM3Encoder, model)


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return cast(Sequence[object], value)
    converter = getattr(value, "tolist", None)
    if callable(converter):
        converted = cast(_ListConvertible, value).tolist()
        if isinstance(converted, Sequence) and not isinstance(
            converted,
            (str, bytes, bytearray),
        ):
            return cast(Sequence[object], converted)
    raise EmbeddingBackendOutputError("Embedding output validation failed")


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EmbeddingBackendOutputError("Embedding output validation failed")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise EmbeddingBackendOutputError("Embedding output validation failed")
    return normalized


def _dense_vector(value: object, *, dimension: int) -> tuple[float, ...]:
    raw = _as_sequence(value)
    if len(raw) != dimension:
        raise EmbeddingBackendOutputError("Embedding output validation failed")
    return tuple(_finite_number(element) for element in raw)


def _sparse_index(value: object, *, maximum: int) -> int:
    if isinstance(value, bool):
        raise EmbeddingBackendOutputError("Embedding output validation failed")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and _SPARSE_STRING_INDEX.fullmatch(value) is not None:
        normalized = int(value)
    else:
        raise EmbeddingBackendOutputError("Embedding output validation failed")
    if normalized < 0 or normalized > maximum:
        raise EmbeddingBackendOutputError("Embedding output validation failed")
    return normalized


def _sparse_vector(
    value: object,
    *,
    maximum_index: int,
    maximum_nonzero: int,
) -> dict[int, float]:
    if not isinstance(value, Mapping) or not value or len(value) > maximum_nonzero:
        raise EmbeddingBackendOutputError("Embedding output validation failed")
    normalized: dict[int, float] = {}
    for raw_index, raw_weight in value.items():
        index = _sparse_index(raw_index, maximum=maximum_index)
        weight = _finite_number(raw_weight)
        if weight <= 0.0 or index in normalized:
            raise EmbeddingBackendOutputError("Embedding output validation failed")
        normalized[index] = weight
    return dict(sorted(normalized.items()))


def _normalize_vector(
    value: object,
    *,
    dense_dimension: int,
    maximum_sparse_index: int,
    maximum_sparse_nonzero: int,
) -> HybridEmbedding:
    if not isinstance(value, HybridEmbedding):
        raise EmbeddingBackendOutputError("Embedding output validation failed")
    return HybridEmbedding(
        dense=_dense_vector(value.dense, dimension=dense_dimension),
        sparse=_sparse_vector(
            value.sparse,
            maximum_index=maximum_sparse_index,
            maximum_nonzero=maximum_sparse_nonzero,
        ),
    )


class LazyLocalBgeM3Backend:
    """Thread-safe BGE-M3 adapter that loads only from a local path on first use."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        dense_dimension: int = 1_024,
        device: str = "cpu",
        use_fp16: bool = False,
        model_batch_size: int = 32,
        max_tokens: int = 8_192,
        max_sparse_index: int = 2_147_483_647,
        max_sparse_nonzero: int = 8_192,
        model_factory: BgeM3ModelFactory | None = None,
    ) -> None:
        path = Path(model_path)
        integer_limits = (
            dense_dimension,
            model_batch_size,
            max_tokens,
            max_sparse_index,
            max_sparse_nonzero,
        )
        if not path.is_absolute() or not str(path) or "\x00" in str(path):
            raise ValueError("BGE-M3 model path must be absolute")
        if _DEVICE.fullmatch(device) is None:
            raise ValueError("invalid BGE-M3 device")
        if not isinstance(use_fp16, bool):
            raise ValueError("invalid BGE-M3 precision")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integer_limits):
            raise ValueError("invalid BGE-M3 limit")
        self._config = BgeM3ModelConfig(
            model_path=path,
            device=device,
            use_fp16=use_fp16,
            model_batch_size=model_batch_size,
            max_tokens=max_tokens,
        )
        self._dense_dimension = dense_dimension
        self._max_sparse_index = max_sparse_index
        self._max_sparse_nonzero = max_sparse_nonzero
        self._factory = model_factory or _default_bge_m3_factory
        self._model: BgeM3Encoder | None = None
        self._probe_ready = threading.Event()
        self._closed = False
        self._lock = threading.RLock()

    @property
    def dense_dimension(self) -> int:
        return self._dense_dimension

    def _model_locked(self) -> BgeM3Encoder:
        if self._closed:
            raise EmbeddingBackendUnavailable("Embedding backend is closed")
        if self._model is not None:
            return self._model
        try:
            resolved = self._config.model_path.resolve(strict=True)
            if not resolved.is_dir():
                raise OSError("model path is not a directory")
            config = BgeM3ModelConfig(
                model_path=resolved,
                device=self._config.device,
                use_fp16=self._config.use_fp16,
                model_batch_size=self._config.model_batch_size,
                max_tokens=self._config.max_tokens,
            )
            self._model = self._factory(config)
        except Exception:
            raise EmbeddingBackendUnavailable("BGE-M3 embedding backend is unavailable") from None
        return self._model

    def embed(self, texts: Sequence[str]) -> Sequence[HybridEmbedding]:
        if (
            isinstance(texts, (str, bytes, bytearray))
            or not isinstance(texts, Sequence)
            or not texts
            or any(not isinstance(text, str) or not text for text in texts)
        ):
            raise ValueError("invalid BGE-M3 input batch")
        with self._lock:
            model = self._model_locked()
            try:
                raw = model.encode(
                    list(texts),
                    batch_size=min(self._config.model_batch_size, len(texts)),
                    max_length=self._config.max_tokens,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
            except Exception:
                raise EmbeddingBackendUnavailable("BGE-M3 embedding inference failed") from None
        if not isinstance(raw, Mapping):
            raise EmbeddingBackendOutputError("Embedding output validation failed")
        dense_rows = _as_sequence(raw.get("dense_vecs"))
        sparse_rows = _as_sequence(raw.get("lexical_weights"))
        if len(dense_rows) != len(texts) or len(sparse_rows) != len(texts):
            raise EmbeddingBackendOutputError("Embedding output validation failed")
        return tuple(
            HybridEmbedding(
                dense=_dense_vector(dense, dimension=self._dense_dimension),
                sparse=_sparse_vector(
                    sparse,
                    maximum_index=self._max_sparse_index,
                    maximum_nonzero=self._max_sparse_nonzero,
                ),
            )
            for dense, sparse in zip(dense_rows, sparse_rows, strict=True)
        )

    def probe(self) -> None:
        # A successfully initialized backend remains ready while an inference
        # owns the model lock. Blocking here makes normal CPU work look like a
        # dependency outage and can cause orchestrators to remove healthy
        # instances from service.
        if self._probe_ready.is_set():
            return
        with self._lock:
            if self._closed:
                raise EmbeddingBackendUnavailable("Embedding backend is closed")
            if self._probe_ready.is_set():
                return
            self.embed(("health",))
            self._probe_ready.set()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._probe_ready.clear()
            model = self._model
            self._model = None
            closer = getattr(model, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    raise EmbeddingBackendUnavailable("BGE-M3 embedding backend could not close") from None


@dataclass(frozen=True, slots=True)
class HybridEmbeddingPolicy:
    batch_size: int = 32
    max_chunks: int = 10_000
    max_chunk_characters: int = 1_600
    max_chunk_bytes: int = 6_400
    max_total_input_bytes: int = 32_000_000
    max_sparse_index: int = 2_147_483_647
    max_sparse_nonzero: int = 8_192

    def __post_init__(self) -> None:
        limits = (
            self.batch_size,
            self.max_chunks,
            self.max_chunk_characters,
            self.max_chunk_bytes,
            self.max_total_input_bytes,
            self.max_sparse_index,
            self.max_sparse_nonzero,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in limits):
            raise ValueError("invalid hybrid embedding policy")
        if self.batch_size > self.max_chunks:
            raise ValueError("embedding batch exceeds chunk limit")


def _embedding_failure(exc: Exception) -> AppError:
    if isinstance(exc, EmbeddingBackendOutputError):
        return AppError(
            ErrorCode.INTERNAL_ERROR,
            "Embedding output validation failed",
            500,
        )
    return DependencyError("Embedding backend is unavailable")


class HybridEmbeddingNode:
    """Batch a fixed chunk schema through an injected hybrid embedding backend."""

    def __init__(
        self,
        backend: HybridEmbeddingBackend,
        *,
        policy: HybridEmbeddingPolicy | None = None,
    ) -> None:
        self._backend = backend
        self._policy = policy or HybridEmbeddingPolicy()
        dimension = backend.dense_dimension
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("invalid embedding dimension")
        self._dense_dimension = dimension

    def _call_backend(self, texts: Sequence[str]) -> tuple[HybridEmbedding, ...]:
        try:
            raw_vectors = self._backend.embed(texts)
            if (
                isinstance(raw_vectors, (str, bytes, bytearray))
                or not isinstance(raw_vectors, Sequence)
                or len(raw_vectors) != len(texts)
            ):
                raise EmbeddingBackendOutputError("Embedding output validation failed")
            return tuple(
                _normalize_vector(
                    vector,
                    dense_dimension=self._dense_dimension,
                    maximum_sparse_index=self._policy.max_sparse_index,
                    maximum_sparse_nonzero=self._policy.max_sparse_nonzero,
                )
                for vector in raw_vectors
            )
        except Exception as exc:
            raise _embedding_failure(exc) from None

    def __call__(self, state: ImportGraphState) -> Mapping[str, object]:
        chunks = require_text_chunks(
            state.get("chunks"),
            max_chunks=self._policy.max_chunks,
            max_content_characters=self._policy.max_chunk_characters,
            max_content_bytes=self._policy.max_chunk_bytes,
        )
        total_bytes = sum(len(chunk["content"].encode("utf-8")) for chunk in chunks)
        if total_bytes > self._policy.max_total_input_bytes:
            raise SafeNodeContractError("embedding input byte limit exceeded")

        output: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self._policy.batch_size):
            batch = chunks[start : start + self._policy.batch_size]
            vectors = self._call_backend(tuple(chunk["content"] for chunk in batch))
            for chunk, vector in zip(batch, vectors, strict=True):
                output.append(_embedded_chunk(chunk, vector))
        return {"chunks": output}

    def probe(self) -> None:
        try:
            self._backend.probe()
        except Exception as exc:
            raise _embedding_failure(exc) from None

    def close(self) -> None:
        try:
            self._backend.close()
        except Exception as exc:
            raise _embedding_failure(exc) from None


def _embedded_chunk(chunk: TextChunk, vector: HybridEmbedding) -> EmbeddedChunk:
    return {
        "content": chunk["content"],
        "title": chunk["title"],
        "parent_title": chunk["parent_title"],
        "file_title": chunk["file_title"],
        "item_name": chunk["item_name"],
        "dense_vector": list(vector.dense),
        "sparse_vector": dict(vector.sparse),
    }


__all__ = [
    "BgeM3Encoder",
    "BgeM3ModelConfig",
    "BgeM3ModelFactory",
    "EmbeddingBackendOutputError",
    "EmbeddingBackendUnavailable",
    "HybridEmbedding",
    "HybridEmbeddingBackend",
    "HybridEmbeddingNode",
    "HybridEmbeddingPolicy",
    "LazyLocalBgeM3Backend",
]
