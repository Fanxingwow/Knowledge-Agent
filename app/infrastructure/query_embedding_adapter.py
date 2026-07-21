from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

from app.core.errors import DependencyError
from app.import_process.agent.nodes.node_hybrid_embedding import (
    HybridEmbedding,
    HybridEmbeddingBackend,
)
from app.services.tenant_vector_search import HybridVectorQuery

_UNAVAILABLE_MESSAGE = "Query embedding is unavailable"
_TIMEOUT_MESSAGE = "Query embedding timed out"
_MAX_DENSE_DIMENSION = 32_768
_MAX_SPARSE_INDEX = 2_147_483_647
_MAX_SPARSE_NONZERO = 65_536


def _dependency_unavailable() -> DependencyError:
    return DependencyError(_UNAVAILABLE_MESSAGE)


def _safe_timeout() -> TimeoutError:
    return TimeoutError(_TIMEOUT_MESSAGE)


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("invalid query embedding output")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("invalid query embedding output")
    return normalized


class HybridQueryEmbeddingAdapter:
    """Adapt the batch embedding contract to one strict hybrid query.

    The adapter deliberately owns no timeout. ``KnowledgeSearchService`` keeps
    the end-to-end deadline so model inference and vector retrieval consume one
    shared budget. Backend timeouts retain their type but lose implementation
    details, while cancellation remains a ``BaseException`` and propagates.
    """

    def __init__(
        self,
        backend: HybridEmbeddingBackend,
        *,
        max_sparse_index: int = 2_147_483_647,
        max_sparse_nonzero: int = 8_192,
    ) -> None:
        try:
            lifecycle_methods = tuple(
                getattr(backend, method_name, None)
                for method_name in ("embed", "probe", "close")
            )
        except TimeoutError:
            raise _safe_timeout() from None
        except Exception:
            raise _dependency_unavailable() from None
        if any(not callable(method) for method in lifecycle_methods):
            raise ValueError("invalid hybrid embedding backend")
        if (
            isinstance(max_sparse_index, bool)
            or not isinstance(max_sparse_index, int)
            or not 1 <= max_sparse_index <= _MAX_SPARSE_INDEX
            or isinstance(max_sparse_nonzero, bool)
            or not isinstance(max_sparse_nonzero, int)
            or not 1 <= max_sparse_nonzero <= _MAX_SPARSE_NONZERO
        ):
            raise ValueError("invalid query embedding limit")
        try:
            dense_dimension = backend.dense_dimension
        except TimeoutError:
            raise _safe_timeout() from None
        except Exception:
            raise _dependency_unavailable() from None
        if (
            isinstance(dense_dimension, bool)
            or not isinstance(dense_dimension, int)
            or not 1 <= dense_dimension <= _MAX_DENSE_DIMENSION
        ):
            raise ValueError("invalid query embedding dimension")
        self._backend = backend
        self._dense_dimension = dense_dimension
        self._max_sparse_index = max_sparse_index
        self._max_sparse_nonzero = max_sparse_nonzero

    @property
    def dense_dimension(self) -> int:
        return self._dense_dimension

    def _dense(self, value: object) -> tuple[float, ...]:
        if (
            isinstance(value, (str, bytes, bytearray))
            or not isinstance(value, Sequence)
            or len(value) != self._dense_dimension
        ):
            raise ValueError("invalid query embedding output")
        normalized = tuple(_finite_number(element) for element in value)
        if len(normalized) != self._dense_dimension:
            raise ValueError("invalid query embedding output")
        return normalized

    def _sparse(self, value: object) -> dict[int, float]:
        if (
            not isinstance(value, Mapping)
            or not value
            or len(value) > self._max_sparse_nonzero
        ):
            raise ValueError("invalid query embedding output")
        normalized: dict[int, float] = {}
        for raw_index, raw_weight in value.items():
            if len(normalized) >= self._max_sparse_nonzero:
                raise ValueError("invalid query embedding output")
            if (
                isinstance(raw_index, bool)
                or not isinstance(raw_index, Integral)
                or not 0 <= int(raw_index) <= self._max_sparse_index
            ):
                raise ValueError("invalid query embedding output")
            index = int(raw_index)
            if index in normalized:
                raise ValueError("invalid query embedding output")
            weight = _finite_number(raw_weight)
            if weight <= 0.0:
                raise ValueError("invalid query embedding output")
            normalized[index] = weight
        return dict(sorted(normalized.items()))

    def _normalize(self, result: object) -> HybridVectorQuery:
        if (
            isinstance(result, (str, bytes, bytearray))
            or not isinstance(result, Sequence)
            or len(result) != 1
        ):
            raise ValueError("invalid query embedding output")
        embedding = result[0]
        if not isinstance(embedding, HybridEmbedding):
            raise ValueError("invalid query embedding output")
        return HybridVectorQuery(
            dense_vector=self._dense(embedding.dense),
            sparse_vector=self._sparse(embedding.sparse),
        )

    def embed_query(self, query_text: str) -> HybridVectorQuery:
        if not isinstance(query_text, str) or not query_text:
            raise ValueError("invalid query embedding input")
        try:
            result = self._backend.embed((query_text,))
            return self._normalize(result)
        except TimeoutError:
            raise _safe_timeout() from None
        except Exception:
            raise _dependency_unavailable() from None

    def probe(self) -> None:
        try:
            self._backend.probe()
        except TimeoutError:
            raise _safe_timeout() from None
        except Exception:
            raise _dependency_unavailable() from None

    def close(self) -> None:
        try:
            self._backend.close()
        except TimeoutError:
            raise _safe_timeout() from None
        except Exception:
            raise _dependency_unavailable() from None


__all__ = ["HybridQueryEmbeddingAdapter"]
