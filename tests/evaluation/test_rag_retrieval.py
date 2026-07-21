from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest

from eval.rag_retrieval import (
    CASE_SCHEMA_VERSION,
    ERROR_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    EvaluationDataError,
    EvaluationThresholds,
    PrecomputedJsonlRetriever,
    RelevantTarget,
    RetrievalCase,
    RetrievalHit,
    RetrievalResult,
    citation_id,
    evaluate,
    load_cases,
    load_results,
    main,
)

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
DOC_A = "123e4567-e89b-42d3-a456-426614174001"
DOC_B = "123e4567-e89b-42d3-a456-426614174002"
DOC_C = "123e4567-e89b-42d3-a456-426614174003"
CHUNK_A = "a" * 64
CHUNK_B = "b" * 64
CHUNK_C = "c" * 64
FIXTURES = Path(__file__).parents[2] / "eval" / "fixtures"


def _target(
    *,
    tenant_id: str = TENANT_A,
    document_id: str = DOC_A,
    chunk_id: str = CHUNK_A,
    relevance: int = 3,
) -> RelevantTarget:
    return RelevantTarget(
        document_id=document_id,
        chunk_id=chunk_id,
        citation_id=citation_id(tenant_id, chunk_id),
        relevance=relevance,
    )


def _case(
    *,
    case_id: str = "case_one",
    tenant_id: str = TENANT_A,
    top_k: int = 3,
    relevant: tuple[RelevantTarget, ...] | None = None,
) -> RetrievalCase:
    targets = (_target(tenant_id=tenant_id),) if relevant is None else relevant
    return RetrievalCase(
        case_id=case_id,
        tenant_id=tenant_id,
        query="Where is the sample procedure?",
        top_k=top_k,
        answerable=bool(targets),
        relevant=targets,
        tags=("test",),
    )


def _hit(
    *,
    rank: int = 1,
    tenant_id: str = TENANT_A,
    document_id: str = DOC_A,
    chunk_id: str = CHUNK_A,
    score: float = 0.9,
) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        tenant_id=tenant_id,
        document_id=document_id,
        chunk_id=chunk_id,
        citation_id=citation_id(tenant_id, chunk_id),
        score=score,
    )


def _result(case: RetrievalCase, hits: tuple[RetrievalHit, ...]) -> RetrievalResult:
    return RetrievalResult(
        case_id=case.case_id,
        tenant_id=case.tenant_id,
        top_k=case.top_k,
        hits=hits,
    )


def _thresholds(
    *,
    min_recall_at_k: float = 0.0,
    min_mrr: float = 0.0,
    min_ndcg_at_k: float = 0.0,
    min_no_answer_accuracy: float = 0.0,
    require_answerable_cases: bool = True,
    require_no_answer_cases: bool = True,
) -> EvaluationThresholds:
    return EvaluationThresholds(
        min_recall_at_k=min_recall_at_k,
        min_mrr=min_mrr,
        min_ndcg_at_k=min_ndcg_at_k,
        min_no_answer_accuracy=min_no_answer_accuracy,
        require_answerable_cases=require_answerable_cases,
        require_no_answer_cases=require_no_answer_cases,
    )


def _write_jsonl(path: Path, documents: tuple[object, ...]) -> None:
    body = "\n".join(json.dumps(document, sort_keys=True) for document in documents)
    path.write_text(f"{body}\n", encoding="utf-8", newline="\n")


class RecordingRetriever:
    def __init__(self, results: dict[str, RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, case: RetrievalCase, *, limit: int) -> RetrievalResult:
        self.calls.append((case.case_id, limit))
        return self.results[case.case_id]


def test_citation_is_deterministic_and_tenant_bound() -> None:
    first = citation_id(TENANT_A, CHUNK_A)

    assert first == citation_id(TENANT_A, CHUNK_A)
    assert first != citation_id(TENANT_B, CHUNK_A)
    assert first.startswith("cit_")
    assert len(first) == 68


def test_versioned_case_and_result_round_trip_exactly() -> None:
    case = _case()
    result = _result(case, (_hit(),))

    case_payload = case.to_dict()
    result_payload = result.to_dict()
    assert set(case_payload) == {
        "schema_version",
        "case_id",
        "tenant_id",
        "query",
        "top_k",
        "answerable",
        "relevant",
        "tags",
    }
    assert set(result_payload) == {
        "schema_version",
        "case_id",
        "tenant_id",
        "top_k",
        "hits",
    }
    assert case_payload["schema_version"] == CASE_SCHEMA_VERSION
    assert result_payload["schema_version"] == RESULT_SCHEMA_VERSION
    assert RetrievalCase.from_dict(case_payload) == case
    assert RetrievalResult.from_dict(result_payload) == result


@pytest.mark.parametrize("mutation", ["version", "extra", "missing"])
def test_case_schema_rejects_unknown_versions_and_shape(mutation: str) -> None:
    payload = _case().to_dict()
    if mutation == "version":
        payload["schema_version"] = "rag-retrieval-case.v2"
    elif mutation == "extra":
        payload["unexpected"] = True
    else:
        del payload["query"]

    with pytest.raises(EvaluationDataError):
        RetrievalCase.from_dict(payload)


@pytest.mark.parametrize("mutation", ["version", "extra", "missing"])
def test_result_schema_rejects_unknown_versions_and_shape(mutation: str) -> None:
    case = _case()
    payload = _result(case, (_hit(),)).to_dict()
    if mutation == "version":
        payload["schema_version"] = "rag-retrieval-result.v2"
    elif mutation == "extra":
        payload["unexpected"] = True
    else:
        del payload["hits"]

    with pytest.raises(EvaluationDataError):
        RetrievalResult.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "Case One"),
        ("tenant_id", "tenant a"),
        ("query", "bad\nquery"),
        ("top_k", True),
        ("top_k", 0),
        ("top_k", 101),
        ("answerable", 1),
        ("tags", ["UPPERCASE"]),
        ("tags", ["duplicate", "duplicate"]),
    ],
)
def test_case_rejects_invalid_scalar_fields(field: str, value: object) -> None:
    payload = _case().to_dict()
    payload[field] = value

    with pytest.raises(EvaluationDataError):
        RetrievalCase.from_dict(payload)


def test_case_rejects_invalid_gold_identity_and_citation() -> None:
    case_payload = _case().to_dict()
    raw_target = cast(dict[str, object], cast(list[object], case_payload["relevant"])[0])

    for field, value in (
        ("document_id", DOC_A.upper()),
        ("chunk_id", "A" * 64),
        ("citation_id", citation_id(TENANT_B, CHUNK_A)),
        ("relevance", 0),
    ):
        mutated_target = {**raw_target, field: value}
        mutated_case = {**case_payload, "relevant": [mutated_target]}
        with pytest.raises(EvaluationDataError):
            RetrievalCase.from_dict(mutated_case)


def test_case_rejects_answerability_mismatch_and_duplicate_gold() -> None:
    answerable = _case()
    no_answer = _case(case_id="no_answer", relevant=())

    with pytest.raises(EvaluationDataError):
        RetrievalCase.from_dict({**answerable.to_dict(), "answerable": False})
    with pytest.raises(EvaluationDataError):
        RetrievalCase.from_dict({**no_answer.to_dict(), "answerable": True})
    with pytest.raises(EvaluationDataError):
        RetrievalCase(
            case_id="duplicate_gold",
            tenant_id=TENANT_A,
            query="A sample query",
            top_k=3,
            answerable=True,
            relevant=(_target(), _target()),
        )


def test_result_rejects_invalid_ranks_size_duplicates_and_scores() -> None:
    case = _case(top_k=1)

    with pytest.raises(EvaluationDataError):
        _result(case, (_hit(rank=2),))
    with pytest.raises(EvaluationDataError):
        _result(
            case,
            (
                _hit(rank=1),
                _hit(rank=2, document_id=DOC_B, chunk_id=CHUNK_B),
            ),
        )
    with pytest.raises(EvaluationDataError):
        RetrievalResult(
            case_id=case.case_id,
            tenant_id=case.tenant_id,
            top_k=2,
            hits=(
                _hit(rank=1),
                _hit(rank=2, document_id=DOC_B, chunk_id=CHUNK_A),
            ),
        )
    with pytest.raises(EvaluationDataError):
        _hit(score=math.nan)
    with pytest.raises(EvaluationDataError):
        RetrievalHit(
            rank=1,
            tenant_id=TENANT_A,
            document_id=DOC_A,
            chunk_id=CHUNK_A,
            citation_id=citation_id(TENANT_B, CHUNK_A),
            score=0.1,
        )


def test_jsonl_loaders_accept_fixtures_and_reject_bad_files(tmp_path: Path) -> None:
    cases = load_cases(FIXTURES / "rag_retrieval_cases.v1.jsonl")
    results = load_results(FIXTURES / "rag_retrieval_results.v1.jsonl")
    assert len(cases) == len(results) == 3

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(EvaluationDataError):
        load_cases(malformed)

    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n", encoding="utf-8")
    with pytest.raises(EvaluationDataError):
        load_cases(blank)

    duplicate = tmp_path / "duplicate.jsonl"
    payload = _case().to_dict()
    _write_jsonl(duplicate, (payload, payload))
    with pytest.raises(EvaluationDataError):
        load_cases(duplicate)

    non_finite = tmp_path / "non-finite.jsonl"
    result_payload = _result(_case(), (_hit(),)).to_dict()
    cast(dict[str, object], cast(list[object], result_payload["hits"])[0])["score"] = math.nan
    _write_jsonl(non_finite, (result_payload,))
    with pytest.raises(EvaluationDataError):
        load_results(non_finite)


def test_evaluator_injects_case_and_top_k_into_retriever() -> None:
    answerable = _case(case_id="answerable", top_k=2)
    no_answer = _case(case_id="no_answer", tenant_id=TENANT_B, top_k=4, relevant=())
    retriever = RecordingRetriever(
        {
            answerable.case_id: _result(answerable, (_hit(),)),
            no_answer.case_id: _result(no_answer, ()),
        }
    )

    report = evaluate((answerable, no_answer), retriever)

    assert report.passed is True
    assert retriever.calls == [("answerable", 2), ("no_answer", 4)]


def test_metrics_compute_macro_recall_mrr_ndcg_and_no_answer() -> None:
    first = _case(
        case_id="first",
        relevant=(
            _target(document_id=DOC_A, chunk_id=CHUNK_A, relevance=3),
            _target(document_id=DOC_B, chunk_id=CHUNK_B, relevance=1),
        ),
    )
    second = _case(
        case_id="second",
        relevant=(_target(document_id=DOC_C, chunk_id=CHUNK_C, relevance=2),),
    )
    no_answer = _case(case_id="no_answer", tenant_id=TENANT_B, relevant=())
    retriever = RecordingRetriever(
        {
            "first": _result(
                first,
                (
                    _hit(rank=1, document_id=DOC_C, chunk_id=CHUNK_C),
                    _hit(rank=2, document_id=DOC_A, chunk_id=CHUNK_A),
                ),
            ),
            "second": _result(
                second,
                (_hit(rank=1, document_id=DOC_C, chunk_id=CHUNK_C),),
            ),
            "no_answer": _result(no_answer, ()),
        }
    )

    report = evaluate((first, second, no_answer), retriever, thresholds=_thresholds())

    first_ndcg = (7 / math.log2(3)) / (7 + 1 / math.log2(3))
    assert report.passed is True
    assert report.recall_at_k == pytest.approx(0.75)
    assert report.mrr == pytest.approx(0.75)
    assert report.ndcg_at_k == pytest.approx((first_ndcg + 1) / 2)
    assert report.no_answer_accuracy == pytest.approx(1.0)
    assert report.no_answer_false_positive_rate == pytest.approx(0.0)
    assert report.retrieved_hit_count == 3


def test_no_answer_false_positive_fails_threshold() -> None:
    answerable = _case(case_id="answerable")
    no_answer = _case(case_id="no_answer", relevant=())
    retriever = RecordingRetriever(
        {
            "answerable": _result(answerable, (_hit(),)),
            "no_answer": _result(
                no_answer,
                (_hit(document_id=DOC_B, chunk_id=CHUNK_B),),
            ),
        }
    )

    report = evaluate(
        (answerable, no_answer),
        retriever,
        thresholds=_thresholds(min_no_answer_accuracy=1.0),
    )

    assert report.passed is False
    assert report.no_answer_accuracy == 0.0
    assert report.no_answer_false_positive_rate == 1.0
    assert report.failures == ("no_answer_accuracy is below its threshold",)


def test_tenant_leakage_is_always_a_hard_failure() -> None:
    no_answer = _case(case_id="no_answer", relevant=())
    leaked = _hit(tenant_id=TENANT_B, document_id=DOC_B, chunk_id=CHUNK_B)
    retriever = RecordingRetriever({"no_answer": _result(no_answer, (leaked,))})

    report = evaluate(
        (no_answer,),
        retriever,
        thresholds=_thresholds(require_answerable_cases=False),
    )

    assert report.passed is False
    assert report.tenant_leakage_count == 1
    assert report.tenant_leakage_rate == 1.0
    assert report.cases[0].leaked_ranks == (1,)
    assert "tenant_leakage_count must be zero" in report.failures


@pytest.mark.parametrize("mismatch", ["case_id", "tenant_id", "top_k"])
def test_evaluator_rejects_result_scope_mismatch(mismatch: str) -> None:
    case = _case()
    values: dict[str, object] = {
        "case_id": case.case_id,
        "tenant_id": case.tenant_id,
        "top_k": case.top_k,
    }
    values[mismatch] = {
        "case_id": "other_case",
        "tenant_id": TENANT_B,
        "top_k": 2,
    }[mismatch]
    returned = RetrievalResult(
        case_id=cast(str, values["case_id"]),
        tenant_id=cast(str, values["tenant_id"]),
        top_k=cast(int, values["top_k"]),
        hits=(),
    )

    with pytest.raises(EvaluationDataError, match="scope"):
        evaluate(
            (case,),
            RecordingRetriever({case.case_id: returned}),
            thresholds=_thresholds(require_no_answer_cases=False),
        )


def test_precomputed_retriever_requires_exact_case_set() -> None:
    case = _case()
    extra = _case(case_id="extra")

    for results in (
        (),
        (_result(case, (_hit(),)), _result(extra, (_hit(),))),
    ):
        with pytest.raises(EvaluationDataError, match="identifiers"):
            evaluate(
                (case,),
                PrecomputedJsonlRetriever(results),
                thresholds=_thresholds(require_no_answer_cases=False),
            )

    with pytest.raises(EvaluationDataError, match="duplicate"):
        PrecomputedJsonlRetriever((_result(case, ()), _result(case, ())))


@pytest.mark.parametrize("coverage", ["answerable", "no_answer"])
def test_required_case_coverage_can_be_explicitly_relaxed(coverage: str) -> None:
    if coverage == "answerable":
        case = _case()
        result = _result(case, (_hit(),))
        relaxed = _thresholds(require_no_answer_cases=False)
        expected_failure = "no-answer case coverage is required"
    else:
        case = _case(relevant=())
        result = _result(case, ())
        relaxed = _thresholds(require_answerable_cases=False)
        expected_failure = "answerable case coverage is required"
    retriever = RecordingRetriever({case.case_id: result})

    required = evaluate((case,), retriever, thresholds=_thresholds())
    optional = evaluate((case,), retriever, thresholds=relaxed)

    assert required.passed is False
    assert expected_failure in required.failures
    assert optional.passed is True


def test_cli_passes_sample_baseline_and_writes_ci_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases_path = FIXTURES / "rag_retrieval_cases.v1.jsonl"
    results_path = FIXTURES / "rag_retrieval_results.v1.jsonl"

    exit_code = main(["--cases", str(cases_path), "--results", str(results_path)])
    stdout_report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert stdout_report["schema_version"] == REPORT_SCHEMA_VERSION
    assert stdout_report["passed"] is True
    assert stdout_report["summary"]["tenant_leakage_count"] == 0

    output_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--results",
            str(results_path),
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True


def test_cli_returns_one_when_threshold_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases_path = FIXTURES / "rag_retrieval_cases.v1.jsonl"
    results = list(load_results(FIXTURES / "rag_retrieval_results.v1.jsonl"))
    no_answer = load_cases(cases_path)[-1]
    results[-1] = _result(
        no_answer,
        (_hit(tenant_id=TENANT_B, document_id=DOC_B, chunk_id=CHUNK_B),),
    )
    failing_results = tmp_path / "failing-results.jsonl"
    _write_jsonl(failing_results, tuple(result.to_dict() for result in results))

    exit_code = main(["--cases", str(cases_path), "--results", str(failing_results)])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["passed"] is False
    assert "no_answer_accuracy is below its threshold" in report["failures"]


def test_cli_returns_two_and_generic_json_for_invalid_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_cases = tmp_path / "private-invalid-path.jsonl"
    invalid_cases.write_text("{}\n", encoding="utf-8")

    exit_code = main(
        [
            "--cases",
            str(invalid_cases),
            "--results",
            str(FIXTURES / "rag_retrieval_results.v1.jsonl"),
        ]
    )
    error = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert error == {
        "schema_version": ERROR_SCHEMA_VERSION,
        "passed": False,
        "error": "Evaluation input or configuration is invalid",
    }
    assert "private-invalid-path" not in json.dumps(error)
