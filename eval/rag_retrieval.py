from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, cast

CASE_SCHEMA_VERSION = "rag-retrieval-case.v1"
RESULT_SCHEMA_VERSION = "rag-retrieval-result.v1"
REPORT_SCHEMA_VERSION = "rag-retrieval-report.v1"
ERROR_SCHEMA_VERSION = "rag-retrieval-error.v1"
MAX_RECORDS = 10_000
MAX_JSONL_LINE_BYTES = 1_000_000
MAX_TOP_K = 100

_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_CASE_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_TAG = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CITATION = re.compile(r"^cit_[0-9a-f]{64}$")
_CITATION_NAMESPACE = b"knowledge-citation-v1\x00"
_RecordT = TypeVar("_RecordT")


class EvaluationDataError(ValueError):
    """Raised when an evaluation artifact violates its versioned contract."""


def citation_id(tenant_id: str, chunk_id: str) -> str:
    _validate_tenant(tenant_id)
    _validate_chunk_id(chunk_id)
    digest = hashlib.sha256()
    digest.update(_CITATION_NAMESPACE)
    digest.update(tenant_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(chunk_id.encode("ascii"))
    return f"cit_{digest.hexdigest()}"


def _validate_tenant(value: object) -> str:
    if not isinstance(value, str) or _TENANT_ID.fullmatch(value) is None:
        raise EvaluationDataError("invalid evaluation tenant")
    return value


def _validate_case_id(value: object) -> str:
    if not isinstance(value, str) or _CASE_ID.fullmatch(value) is None:
        raise EvaluationDataError("invalid evaluation case identifier")
    return value


def _validate_uuid(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise EvaluationDataError(f"invalid evaluation {field}")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise EvaluationDataError(f"invalid evaluation {field}") from None
    if str(parsed) != value:
        raise EvaluationDataError(f"invalid evaluation {field}")
    return value


def _validate_chunk_id(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EvaluationDataError("invalid evaluation chunk identifier")
    return value


def _validate_citation(value: object, *, tenant_id: str, chunk_id: str) -> str:
    if not isinstance(value, str) or _CITATION.fullmatch(value) is None or value != citation_id(tenant_id, chunk_id):
        raise EvaluationDataError("invalid evaluation citation identifier")
    return value


def _validate_plain_text(value: object, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EvaluationDataError(f"invalid evaluation {field}")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise EvaluationDataError(f"invalid evaluation {field}") from None
    return value


def _validate_top_k(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_TOP_K:
        raise EvaluationDataError("invalid evaluation top_k")
    return value


def _validate_unit_interval(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise EvaluationDataError(f"invalid evaluation {field}")
    return float(value)


def _exact_mapping(value: object, fields: set[str], *, kind: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields or not all(isinstance(key, str) for key in value):
        raise EvaluationDataError(f"invalid {kind} schema")
    return dict(cast(Mapping[str, object], value))


@dataclass(frozen=True, slots=True)
class RelevantTarget:
    document_id: str
    chunk_id: str
    citation_id: str
    relevance: int

    def __post_init__(self) -> None:
        _validate_uuid(self.document_id, field="document identifier")
        _validate_chunk_id(self.chunk_id)
        if isinstance(self.relevance, bool) or not isinstance(self.relevance, int) or not 1 <= self.relevance <= 3:
            raise EvaluationDataError("invalid evaluation relevance gain")

    def validate_citation(self, tenant_id: str) -> None:
        _validate_citation(self.citation_id, tenant_id=tenant_id, chunk_id=self.chunk_id)

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.document_id, self.chunk_id, self.citation_id

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "citation_id": self.citation_id,
            "relevance": self.relevance,
        }

    @classmethod
    def from_dict(cls, value: object, *, tenant_id: str) -> RelevantTarget:
        document = _exact_mapping(
            value,
            {"document_id", "chunk_id", "citation_id", "relevance"},
            kind="relevant target",
        )
        target = cls(
            document_id=cast(str, document["document_id"]),
            chunk_id=cast(str, document["chunk_id"]),
            citation_id=cast(str, document["citation_id"]),
            relevance=cast(int, document["relevance"]),
        )
        target.validate_citation(tenant_id)
        return target


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    case_id: str
    tenant_id: str
    query: str
    top_k: int
    answerable: bool
    relevant: tuple[RelevantTarget, ...]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_case_id(self.case_id)
        _validate_tenant(self.tenant_id)
        _validate_plain_text(self.query, field="query", maximum=2_048)
        _validate_top_k(self.top_k)
        if not isinstance(self.answerable, bool):
            raise EvaluationDataError("invalid evaluation answerable flag")
        if not isinstance(self.relevant, tuple) or not all(
            isinstance(target, RelevantTarget) for target in self.relevant
        ):
            raise EvaluationDataError("invalid evaluation relevant targets")
        if self.answerable != bool(self.relevant):
            raise EvaluationDataError("answerability and relevant targets disagree")
        identities = [target.identity for target in self.relevant]
        if len(set(identities)) != len(identities):
            raise EvaluationDataError("duplicate evaluation relevant target")
        citations = [target.citation_id for target in self.relevant]
        chunks = [target.chunk_id for target in self.relevant]
        if len(set(citations)) != len(citations) or len(set(chunks)) != len(chunks):
            raise EvaluationDataError("ambiguous evaluation relevant target")
        for target in self.relevant:
            target.validate_citation(self.tenant_id)
        if not isinstance(self.tags, tuple) or len(self.tags) > 32:
            raise EvaluationDataError("invalid evaluation tags")
        if len(set(self.tags)) != len(self.tags) or any(
            not isinstance(tag, str) or _TAG.fullmatch(tag) is None for tag in self.tags
        ):
            raise EvaluationDataError("invalid evaluation tags")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CASE_SCHEMA_VERSION,
            "case_id": self.case_id,
            "tenant_id": self.tenant_id,
            "query": self.query,
            "top_k": self.top_k,
            "answerable": self.answerable,
            "relevant": [target.to_dict() for target in self.relevant],
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, value: object) -> RetrievalCase:
        document = _exact_mapping(
            value,
            {
                "schema_version",
                "case_id",
                "tenant_id",
                "query",
                "top_k",
                "answerable",
                "relevant",
                "tags",
            },
            kind="retrieval case",
        )
        if document["schema_version"] != CASE_SCHEMA_VERSION:
            raise EvaluationDataError("unsupported retrieval case schema")
        tenant_id = _validate_tenant(document["tenant_id"])
        raw_relevant = document["relevant"]
        raw_tags = document["tags"]
        if not isinstance(raw_relevant, list) or not isinstance(raw_tags, list):
            raise EvaluationDataError("invalid retrieval case schema")
        return cls(
            case_id=cast(str, document["case_id"]),
            tenant_id=tenant_id,
            query=cast(str, document["query"]),
            top_k=cast(int, document["top_k"]),
            answerable=cast(bool, document["answerable"]),
            relevant=tuple(RelevantTarget.from_dict(target, tenant_id=tenant_id) for target in raw_relevant),
            tags=tuple(cast(list[str], raw_tags)),
        )


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    rank: int
    tenant_id: str
    document_id: str
    chunk_id: str
    citation_id: str
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise EvaluationDataError("invalid retrieval result rank")
        _validate_tenant(self.tenant_id)
        _validate_uuid(self.document_id, field="document identifier")
        _validate_chunk_id(self.chunk_id)
        _validate_citation(self.citation_id, tenant_id=self.tenant_id, chunk_id=self.chunk_id)
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(float(self.score))
        ):
            raise EvaluationDataError("invalid retrieval result score")

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.document_id, self.chunk_id, self.citation_id

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "tenant_id": self.tenant_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "citation_id": self.citation_id,
            "score": float(self.score),
        }

    @classmethod
    def from_dict(cls, value: object) -> RetrievalHit:
        document = _exact_mapping(
            value,
            {"rank", "tenant_id", "document_id", "chunk_id", "citation_id", "score"},
            kind="retrieval hit",
        )
        return cls(
            rank=cast(int, document["rank"]),
            tenant_id=cast(str, document["tenant_id"]),
            document_id=cast(str, document["document_id"]),
            chunk_id=cast(str, document["chunk_id"]),
            citation_id=cast(str, document["citation_id"]),
            score=cast(float, document["score"]),
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    case_id: str
    tenant_id: str
    top_k: int
    hits: tuple[RetrievalHit, ...]

    def __post_init__(self) -> None:
        _validate_case_id(self.case_id)
        _validate_tenant(self.tenant_id)
        _validate_top_k(self.top_k)
        if not isinstance(self.hits, tuple) or not all(isinstance(hit, RetrievalHit) for hit in self.hits):
            raise EvaluationDataError("invalid retrieval result hits")
        if len(self.hits) > self.top_k:
            raise EvaluationDataError("retrieval result exceeds top_k")
        if [hit.rank for hit in self.hits] != list(range(1, len(self.hits) + 1)):
            raise EvaluationDataError("retrieval result ranks are not contiguous")
        identities = [(hit.tenant_id, *hit.identity) for hit in self.hits]
        citations = [hit.citation_id for hit in self.hits]
        if len(set(identities)) != len(identities) or len(set(citations)) != len(citations):
            raise EvaluationDataError("duplicate retrieval result hit")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "case_id": self.case_id,
            "tenant_id": self.tenant_id,
            "top_k": self.top_k,
            "hits": [hit.to_dict() for hit in self.hits],
        }

    @classmethod
    def from_dict(cls, value: object) -> RetrievalResult:
        document = _exact_mapping(
            value,
            {"schema_version", "case_id", "tenant_id", "top_k", "hits"},
            kind="retrieval result",
        )
        if document["schema_version"] != RESULT_SCHEMA_VERSION:
            raise EvaluationDataError("unsupported retrieval result schema")
        raw_hits = document["hits"]
        if not isinstance(raw_hits, list):
            raise EvaluationDataError("invalid retrieval result schema")
        return cls(
            case_id=cast(str, document["case_id"]),
            tenant_id=cast(str, document["tenant_id"]),
            top_k=cast(int, document["top_k"]),
            hits=tuple(RetrievalHit.from_dict(hit) for hit in raw_hits),
        )


class Retriever(Protocol):
    """Offline injection boundary; implementations must not be created by the evaluator."""

    def retrieve(self, case: RetrievalCase, *, limit: int) -> RetrievalResult: ...


class PrecomputedJsonlRetriever:
    def __init__(self, results: Sequence[RetrievalResult]) -> None:
        indexed: dict[str, RetrievalResult] = {}
        for result in results:
            if not isinstance(result, RetrievalResult):
                raise EvaluationDataError("invalid precomputed retrieval result")
            if result.case_id in indexed:
                raise EvaluationDataError("duplicate retrieval result case")
            indexed[result.case_id] = result
        self._results = indexed

    @classmethod
    def from_path(cls, path: Path) -> PrecomputedJsonlRetriever:
        return cls(load_results(path))

    def validate_case_set(self, cases: Sequence[RetrievalCase]) -> None:
        expected = {case.case_id for case in cases}
        if set(self._results) != expected:
            raise EvaluationDataError("case and result identifiers do not match")

    def retrieve(self, case: RetrievalCase, *, limit: int) -> RetrievalResult:
        if limit != case.top_k:
            raise EvaluationDataError("retriever limit does not match case top_k")
        try:
            return self._results[case.case_id]
        except KeyError:
            raise EvaluationDataError("retrieval result is missing") from None


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    answerable: bool
    retrieved_count: int
    relevant_count: int
    relevant_retrieved: int
    recall_at_k: float | None
    reciprocal_rank: float | None
    ndcg_at_k: float | None
    no_answer_correct: bool | None
    tenant_leakage_count: int
    leaked_ranks: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "answerable": self.answerable,
            "retrieved_count": self.retrieved_count,
            "relevant_count": self.relevant_count,
            "relevant_retrieved": self.relevant_retrieved,
            "recall_at_k": self.recall_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "ndcg_at_k": self.ndcg_at_k,
            "no_answer_correct": self.no_answer_correct,
            "tenant_leakage_count": self.tenant_leakage_count,
            "leaked_ranks": list(self.leaked_ranks),
        }


@dataclass(frozen=True, slots=True)
class EvaluationThresholds:
    min_recall_at_k: float = 0.8
    min_mrr: float = 0.8
    min_ndcg_at_k: float = 0.8
    min_no_answer_accuracy: float = 1.0
    require_answerable_cases: bool = True
    require_no_answer_cases: bool = True

    def __post_init__(self) -> None:
        for field, value in (
            ("recall threshold", self.min_recall_at_k),
            ("MRR threshold", self.min_mrr),
            ("nDCG threshold", self.min_ndcg_at_k),
            ("no-answer threshold", self.min_no_answer_accuracy),
        ):
            _validate_unit_interval(value, field=field)
        if not isinstance(self.require_answerable_cases, bool) or not isinstance(self.require_no_answer_cases, bool):
            raise EvaluationDataError("invalid evaluation coverage requirement")

    def to_dict(self) -> dict[str, object]:
        return {
            "min_recall_at_k": self.min_recall_at_k,
            "min_mrr": self.min_mrr,
            "min_ndcg_at_k": self.min_ndcg_at_k,
            "min_no_answer_accuracy": self.min_no_answer_accuracy,
            "require_answerable_cases": self.require_answerable_cases,
            "require_no_answer_cases": self.require_no_answer_cases,
            "max_tenant_leakage_count": 0,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    case_count: int
    answerable_case_count: int
    no_answer_case_count: int
    retrieved_hit_count: int
    recall_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None
    no_answer_accuracy: float | None
    no_answer_false_positive_rate: float | None
    tenant_leakage_count: int
    tenant_leakage_rate: float
    cases: tuple[CaseEvaluation, ...]
    thresholds: EvaluationThresholds
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "case_schema_version": CASE_SCHEMA_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "summary": {
                "case_count": self.case_count,
                "answerable_case_count": self.answerable_case_count,
                "no_answer_case_count": self.no_answer_case_count,
                "retrieved_hit_count": self.retrieved_hit_count,
                "recall_at_k": self.recall_at_k,
                "mrr": self.mrr,
                "ndcg_at_k": self.ndcg_at_k,
                "no_answer_accuracy": self.no_answer_accuracy,
                "no_answer_false_positive_rate": self.no_answer_false_positive_rate,
                "tenant_leakage_count": self.tenant_leakage_count,
                "tenant_leakage_rate": self.tenant_leakage_rate,
            },
            "thresholds": self.thresholds.to_dict(),
            "passed": self.passed,
            "failures": list(self.failures),
            "cases": [case.to_dict() for case in self.cases],
        }


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / len(values)


def _evaluate_case(case: RetrievalCase, result: RetrievalResult) -> CaseEvaluation:
    if result.case_id != case.case_id or result.tenant_id != case.tenant_id or result.top_k != case.top_k:
        raise EvaluationDataError("retrieval result does not match case scope")
    leaked_ranks = tuple(hit.rank for hit in result.hits if hit.tenant_id != case.tenant_id)
    if not case.answerable:
        return CaseEvaluation(
            case_id=case.case_id,
            answerable=False,
            retrieved_count=len(result.hits),
            relevant_count=0,
            relevant_retrieved=0,
            recall_at_k=None,
            reciprocal_rank=None,
            ndcg_at_k=None,
            no_answer_correct=not result.hits,
            tenant_leakage_count=len(leaked_ranks),
            leaked_ranks=leaked_ranks,
        )

    gains = {target.identity: target.relevance for target in case.relevant}
    relevant_hits = [hit for hit in result.hits if hit.tenant_id == case.tenant_id and hit.identity in gains]
    recall = len(relevant_hits) / len(case.relevant)
    reciprocal_rank = 0.0 if not relevant_hits else 1.0 / min(hit.rank for hit in relevant_hits)
    dcg = math.fsum((2 ** gains[hit.identity] - 1) / math.log2(hit.rank + 1) for hit in relevant_hits)
    ideal_gains = sorted(gains.values(), reverse=True)[: case.top_k]
    ideal_dcg = math.fsum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    ndcg = dcg / ideal_dcg
    return CaseEvaluation(
        case_id=case.case_id,
        answerable=True,
        retrieved_count=len(result.hits),
        relevant_count=len(case.relevant),
        relevant_retrieved=len(relevant_hits),
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
        no_answer_correct=None,
        tenant_leakage_count=len(leaked_ranks),
        leaked_ranks=leaked_ranks,
    )


def evaluate(
    cases: Sequence[RetrievalCase],
    retriever: Retriever,
    *,
    thresholds: EvaluationThresholds | None = None,
) -> EvaluationReport:
    case_batch = tuple(cases)
    if not case_batch or len(case_batch) > MAX_RECORDS:
        raise EvaluationDataError("invalid evaluation case count")
    if not all(isinstance(case, RetrievalCase) for case in case_batch):
        raise EvaluationDataError("invalid evaluation case")
    case_ids = [case.case_id for case in case_batch]
    if len(set(case_ids)) != len(case_ids):
        raise EvaluationDataError("duplicate evaluation case identifier")
    retrieve = getattr(retriever, "retrieve", None)
    if not callable(retrieve):
        raise EvaluationDataError("invalid evaluation retriever")
    validate_case_set = getattr(retriever, "validate_case_set", None)
    if callable(validate_case_set):
        validate_case_set(case_batch)
    resolved_thresholds = thresholds or EvaluationThresholds()
    evaluated: list[CaseEvaluation] = []
    for case in case_batch:
        result = retrieve(case, limit=case.top_k)
        if not isinstance(result, RetrievalResult):
            raise EvaluationDataError("retriever returned an invalid result")
        evaluated.append(_evaluate_case(case, result))

    answerable = [item for item in evaluated if item.answerable]
    no_answer = [item for item in evaluated if not item.answerable]
    recall = _mean([cast(float, item.recall_at_k) for item in answerable])
    mrr = _mean([cast(float, item.reciprocal_rank) for item in answerable])
    ndcg = _mean([cast(float, item.ndcg_at_k) for item in answerable])
    no_answer_accuracy = _mean([1.0 if item.no_answer_correct else 0.0 for item in no_answer])
    total_hits = sum(item.retrieved_count for item in evaluated)
    leakage_count = sum(item.tenant_leakage_count for item in evaluated)
    failures: list[str] = []
    if resolved_thresholds.require_answerable_cases and not answerable:
        failures.append("answerable case coverage is required")
    if resolved_thresholds.require_no_answer_cases and not no_answer:
        failures.append("no-answer case coverage is required")
    for metric, value, minimum in (
        ("recall_at_k", recall, resolved_thresholds.min_recall_at_k),
        ("mrr", mrr, resolved_thresholds.min_mrr),
        ("ndcg_at_k", ndcg, resolved_thresholds.min_ndcg_at_k),
        ("no_answer_accuracy", no_answer_accuracy, resolved_thresholds.min_no_answer_accuracy),
    ):
        if value is not None and value < minimum:
            failures.append(f"{metric} is below its threshold")
    if leakage_count:
        failures.append("tenant_leakage_count must be zero")
    return EvaluationReport(
        case_count=len(evaluated),
        answerable_case_count=len(answerable),
        no_answer_case_count=len(no_answer),
        retrieved_hit_count=total_hits,
        recall_at_k=recall,
        mrr=mrr,
        ndcg_at_k=ndcg,
        no_answer_accuracy=no_answer_accuracy,
        no_answer_false_positive_rate=(None if no_answer_accuracy is None else 1.0 - no_answer_accuracy),
        tenant_leakage_count=leakage_count,
        tenant_leakage_rate=0.0 if not total_hits else leakage_count / total_hits,
        cases=tuple(evaluated),
        thresholds=resolved_thresholds,
        passed=not failures,
        failures=tuple(failures),
    )


def _reject_json_constant(_value: str) -> object:
    raise EvaluationDataError("non-finite JSON values are forbidden")


def _load_jsonl(
    path: Path,
    parser: Callable[[object], _RecordT],
) -> tuple[_RecordT, ...]:
    if not isinstance(path, Path) or not path.is_file():
        raise EvaluationDataError("evaluation JSONL file is unavailable")
    records: list[_RecordT] = []
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as source:
            for line_number, line in enumerate(source, start=1):
                if line_number > MAX_RECORDS:
                    raise EvaluationDataError("evaluation JSONL record limit exceeded")
                stripped = line.rstrip("\r\n")
                if not stripped or len(stripped.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
                    raise EvaluationDataError("invalid evaluation JSONL record")
                raw = json.loads(stripped, parse_constant=_reject_json_constant)
                records.append(parser(raw))
    except EvaluationDataError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise EvaluationDataError("invalid evaluation JSONL file") from None
    if not records:
        raise EvaluationDataError("evaluation JSONL file is empty")
    return tuple(records)


def load_cases(path: Path) -> tuple[RetrievalCase, ...]:
    cases = _load_jsonl(path, RetrievalCase.from_dict)
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise EvaluationDataError("duplicate evaluation case identifier")
    return cases


def load_results(path: Path) -> tuple[RetrievalResult, ...]:
    results = _load_jsonl(path, RetrievalResult.from_dict)
    case_ids = [result.case_id for result in results]
    if len(set(case_ids)) != len(case_ids):
        raise EvaluationDataError("duplicate retrieval result case")
    return results


def _write_json(target: str, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if target == "-":
        sys.stdout.write(serialized)
        return
    path = Path(target)
    try:
        path.write_text(serialized, encoding="utf-8", errors="strict", newline="\n")
    except OSError:
        raise EvaluationDataError("evaluation report cannot be written") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate precomputed tenant-safe RAG retrieval results")
    parser.add_argument("--cases", type=Path, required=True, help="Versioned case JSONL")
    parser.add_argument("--results", type=Path, required=True, help="Versioned result JSONL")
    parser.add_argument("--output", default="-", help="JSON report path or '-' for stdout")
    parser.add_argument("--min-recall-at-k", type=float, default=0.8)
    parser.add_argument("--min-mrr", type=float, default=0.8)
    parser.add_argument("--min-ndcg-at-k", type=float, default=0.8)
    parser.add_argument("--min-no-answer-accuracy", type=float, default=1.0)
    parser.add_argument("--allow-missing-answerable-cases", action="store_true")
    parser.add_argument("--allow-missing-no-answer-cases", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        thresholds = EvaluationThresholds(
            min_recall_at_k=arguments.min_recall_at_k,
            min_mrr=arguments.min_mrr,
            min_ndcg_at_k=arguments.min_ndcg_at_k,
            min_no_answer_accuracy=arguments.min_no_answer_accuracy,
            require_answerable_cases=not arguments.allow_missing_answerable_cases,
            require_no_answer_cases=not arguments.allow_missing_no_answer_cases,
        )
        cases = load_cases(arguments.cases)
        retriever = PrecomputedJsonlRetriever.from_path(arguments.results)
        report = evaluate(cases, retriever, thresholds=thresholds)
        _write_json(arguments.output, report.to_dict())
        return 0 if report.passed else 1
    except EvaluationDataError:
        error = {
            "schema_version": ERROR_SCHEMA_VERSION,
            "passed": False,
            "error": "Evaluation input or configuration is invalid",
        }
        try:
            _write_json(arguments.output, error)
        except EvaluationDataError:
            sys.stderr.write(json.dumps(error, sort_keys=True) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
