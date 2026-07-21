from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pytest

from app.core.errors import DependencyError
from app.infrastructure.mongo_import_visibility import MongoCompletedImportVisibilityReader

_VERSION_1 = "10000000-0000-4000-8000-000000000001"
_VERSION_2 = "10000000-0000-4000-8000-000000000002"
_DOCUMENT_1 = "00000000-0000-4000-8000-000000000001"
_COMMIT_PROOF = "milvus-v1.commit"
_INVALID_COMMIT_PROOF = "token with spaces"


class FakeCollection:
    def __init__(self, documents: Sequence[Mapping[str, object]] = ()) -> None:
        self.documents = tuple(documents)
        self.calls: list[tuple[Mapping[str, object], Mapping[str, int]]] = []

    def find(
        self,
        filter: Mapping[str, object],
        projection: Mapping[str, int],
    ) -> Iterable[Mapping[str, object]]:
        self.calls.append((filter, projection))
        return self.documents


def completed_document(
    *,
    task_id: str = _VERSION_1,
    tenant_id: str = "tenant-a",
    document_id: str = _DOCUMENT_1,
    commit_token: str | None = None,
) -> dict[str, object]:
    return {
        "_id": task_id,
        "tenant_id": tenant_id,
        "document_id": document_id,
        "status": "completed",
        "commit_token": commit_token if commit_token is not None else _COMMIT_PROOF,
    }


def test_reads_only_exact_tenant_completed_task_versions() -> None:
    collection = FakeCollection((completed_document(),))
    reader = MongoCompletedImportVisibilityReader(collection)

    result = reader.get_completed_versions(
        "tenant-a",
        (_VERSION_1, _VERSION_1, _VERSION_2),
    )

    assert tuple(result) == (_VERSION_1,)
    activation = result[_VERSION_1]
    assert activation.task_id == _VERSION_1
    assert activation.import_version == _VERSION_1
    assert activation.tenant_id == "tenant-a"
    assert activation.document_id == _DOCUMENT_1
    assert activation.commit_token == _COMMIT_PROOF
    assert collection.calls == [
        (
            {
                "_id": {"$in": [_VERSION_1, _VERSION_2]},
                "tenant_id": "tenant-a",
                "status": "completed",
            },
            {
                "_id": 1,
                "tenant_id": 1,
                "document_id": 1,
                "status": 1,
                "commit_token": 1,
            },
        )
    ]


@pytest.mark.parametrize(
    "document",
    [
        completed_document(tenant_id="tenant-b"),
        {**completed_document(), "status": "processing"},
        {key: value for key, value in completed_document().items() if key != "commit_token"},
        completed_document(commit_token=_INVALID_COMMIT_PROOF),
        completed_document(task_id=_VERSION_2),
    ],
)
def test_out_of_scope_or_malformed_mongo_records_fail_closed(
    document: Mapping[str, object],
) -> None:
    reader = MongoCompletedImportVisibilityReader(FakeCollection((document,)))

    with pytest.raises(DependencyError) as raised:
        reader.get_completed_versions("tenant-a", (_VERSION_1,))

    assert str(raised.value) == "Knowledge visibility registry is unavailable"


def test_empty_versions_do_not_touch_mongo() -> None:
    collection = FakeCollection()
    reader = MongoCompletedImportVisibilityReader(collection)

    assert reader.get_completed_versions("tenant-a", ()) == {}
    assert collection.calls == []


def test_batch_limit_is_exposed_for_safe_runtime_alignment() -> None:
    reader = MongoCompletedImportVisibilityReader(FakeCollection(), max_batch_size=123)

    assert reader.max_batch_size == 123


@pytest.mark.parametrize("tenant_id", ["", "tenant a", "../tenant"])
def test_invalid_tenant_is_rejected_before_mongo(tenant_id: str) -> None:
    collection = FakeCollection()
    reader = MongoCompletedImportVisibilityReader(collection)

    with pytest.raises(ValueError, match="tenant"):
        reader.get_completed_versions(tenant_id, (_VERSION_1,))

    assert collection.calls == []


def test_version_batch_is_bounded_before_mongo() -> None:
    collection = FakeCollection()
    reader = MongoCompletedImportVisibilityReader(collection, max_batch_size=1)

    with pytest.raises(ValueError, match="too many"):
        reader.get_completed_versions("tenant-a", (_VERSION_1, _VERSION_2))

    assert collection.calls == []


def test_mongo_errors_are_sanitized() -> None:
    leaked_detail = "mongodb://root:credential@private-host"  # pragma: allowlist secret

    class FailingCollection(FakeCollection):
        def find(
            self,
            filter: Mapping[str, object],
            projection: Mapping[str, int],
        ) -> Iterable[Mapping[str, object]]:
            raise RuntimeError(leaked_detail)

    reader = MongoCompletedImportVisibilityReader(FailingCollection())

    with pytest.raises(DependencyError) as raised:
        reader.get_completed_versions("tenant-a", (_VERSION_1,))

    assert str(raised.value) == "Knowledge visibility registry is unavailable"
    assert leaked_detail not in str(raised.value)
    assert raised.value.__cause__ is None
