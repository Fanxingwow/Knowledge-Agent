from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from app.infrastructure.milvus_import_index import (
    CommitFenceError,
    ImportVersionConflict,
    IndexWriteUncertain,
    MilvusChunk,
    MilvusCommitReceipt,
    MilvusImportIndex,
)

_DOCUMENT_ID = "22222222-2222-4222-8222-222222222222"
_IMPORT_VERSION = "11111111-1111-4111-8111-111111111111"
_REQUIRED_SCOPE = {"tenant_id", "document_id", "import_version", "pipeline_version"}
_REQUIRED_RECORD_FIELDS = {
    "chunk_id",
    "tenant_id",
    "document_id",
    "import_version",
    "pipeline_version",
    "ordinal",
    "content_sha256",
    "record_sha256",
    "manifest_sha256",
    "commit_token",
}
_CLAUSE = re.compile(r'([a-z_]+) == ("(?:[^"\\]|\\.)*")')


def _parse_filter(expression: str) -> dict[str, str]:
    clauses = {name: json.loads(literal) for name, literal in _CLAUSE.findall(expression)}
    assert set(clauses) == _REQUIRED_SCOPE
    assert expression.count(" and ") == 3
    assert "item_name" not in expression
    return clauses


class FakeMilvusClient:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.query_calls: list[dict[str, object]] = []
        self.upsert_calls: list[tuple[dict[str, object], ...]] = []
        self.delete_calls: list[str] = []
        self.partial_then_raise: int | None = None
        self.raise_after_all = False
        self.delete_then_raise = False
        self.forge_query_record: dict[str, object] | None = None

    def query(
        self,
        *,
        collection_name: str,
        filter: str,
        output_fields: Sequence[str],
        limit: int,
        consistency_level: str,
    ) -> Sequence[Mapping[str, object]]:
        scope = _parse_filter(filter)
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "filter": filter,
                "output_fields": tuple(output_fields),
                "limit": limit,
                "consistency_level": consistency_level,
            }
        )
        if self.forge_query_record is not None:
            return [dict(self.forge_query_record)]
        return [
            {field: record[field] for field in output_fields}
            for record in self.records.values()
            if all(record.get(field) == value for field, value in scope.items())
        ][:limit]

    def upsert(
        self,
        *,
        collection_name: str,
        data: Sequence[Mapping[str, object]],
    ) -> object:
        assert collection_name == "import_chunks_v1"
        copied = tuple(dict(record) for record in data)
        assert copied
        scope = {
            (
                record["tenant_id"],
                record["document_id"],
                record["import_version"],
                record["pipeline_version"],
            )
            for record in copied
        }
        assert len(scope) == 1
        assert all(_REQUIRED_RECORD_FIELDS <= set(record) for record in copied)
        self.upsert_calls.append(copied)
        count = self.partial_then_raise
        if count is not None:
            self.partial_then_raise = None
            for record in copied[:count]:
                self.records[str(record["chunk_id"])] = record
            raise TimeoutError("unknown write result with secret endpoint")
        for record in copied:
            self.records[str(record["chunk_id"])] = record
        if self.raise_after_all:
            self.raise_after_all = False
            raise TimeoutError("acknowledgement lost")
        return {"upsert_count": len(copied)}

    def delete(self, *, collection_name: str, filter: str) -> object:
        assert collection_name == "import_chunks_v1"
        scope = _parse_filter(filter)
        self.delete_calls.append(filter)
        doomed = [
            chunk_id
            for chunk_id, record in self.records.items()
            if all(record.get(field) == value for field, value in scope.items())
        ]
        for chunk_id in doomed:
            del self.records[chunk_id]
        if self.delete_then_raise:
            self.delete_then_raise = False
            raise TimeoutError("delete acknowledgement lost")
        return {"delete_count": len(doomed)}


@dataclass(slots=True)
class FakeFence:
    task_id: str = _IMPORT_VERSION
    document_id: str = _DOCUMENT_ID
    tenant_id: str = "tenant-a"
    import_version: str = _IMPORT_VERSION
    attempt_count: int = 1
    fence_id: str = "fence-attempt-1"
    current: bool = True
    stale_on_assertion: int | None = None
    assertions: int = 0

    def assert_current(self) -> None:
        self.assertions += 1
        if not self.current or (
            self.stale_on_assertion is not None
            and self.assertions >= self.stale_on_assertion
        ):
            raise RuntimeError("stale lease includes secret token")


def _chunks(count: int = 2) -> list[MilvusChunk]:
    return [
        MilvusChunk(
            content=f"tenant-safe content {ordinal}",
            title=f"Title {ordinal}",
            parent_title="Parent",
            file_title="Manual",
            item_name="Product",
            dense_vector=[0.1 + ordinal, 0.2 + ordinal],
            sparse_vector={ordinal: 0.5},
        )
        for ordinal in range(count)
    ]


def _index(client: FakeMilvusClient, *, batch_size: int = 8) -> MilvusImportIndex:
    return MilvusImportIndex(
        client,
        collection_name="import_chunks_v1",
        pipeline_version="bge-m3.v1",
        batch_size=batch_size,
    )


def _stage(
    index: MilvusImportIndex,
    fence: FakeFence,
    *,
    chunks: Sequence[MilvusChunk] | None = None,
) -> MilvusCommitReceipt:
    return index.stage(
        tenant_id=fence.tenant_id,
        document_id=fence.document_id,
        import_version=fence.import_version,
        chunks=chunks or _chunks(),
        fence=fence,
    )


def test_records_have_deterministic_primary_keys_and_verified_receipt() -> None:
    client = FakeMilvusClient()
    index = _index(client)
    fence = FakeFence()

    receipt = _stage(index, fence)

    assert receipt.task_id == _IMPORT_VERSION
    assert receipt.import_version == _IMPORT_VERSION
    assert receipt.tenant_id == "tenant-a"
    assert receipt.document_id == _DOCUMENT_ID
    assert receipt.pipeline_version == "bge-m3.v1"
    assert receipt.imported_chunk_count == 2
    assert re.fullmatch(r"[0-9a-f]{64}", receipt.manifest_sha256)
    assert re.fullmatch(r"milvus-v1\.[0-9a-f]{64}", receipt.commit_token)
    assert len(client.records) == 2
    assert all(re.fullmatch(r"[0-9a-f]{64}", chunk_id) for chunk_id in client.records)
    assert all(record["tenant_id"] == "tenant-a" for record in client.records.values())
    assert all(record["document_id"] == _DOCUMENT_ID for record in client.records.values())
    assert all(record["import_version"] == _IMPORT_VERSION for record in client.records.values())
    assert all(record["commit_token"] == receipt.commit_token for record in client.records.values())
    assert all(call["consistency_level"] == "Strong" for call in client.query_calls)
    assert fence.assertions >= 6


def test_retry_is_idempotent_and_commit_token_is_stable_across_attempts() -> None:
    client = FakeMilvusClient()
    index = _index(client)

    first = _stage(index, FakeFence())
    first_upsert_count = len(client.upsert_calls)
    second = _stage(
        index,
        FakeFence(attempt_count=2, fence_id="fence-attempt-2"),
    )

    assert len(client.records) == 2
    assert len(client.upsert_calls) == first_upsert_count
    assert second.commit_token == first.commit_token
    assert second.manifest_sha256 == first.manifest_sha256
    assert second.attempt_count == 2
    assert second.fence_id == "fence-attempt-2"


def test_each_upsert_batch_is_fenced_and_contains_one_exact_scope() -> None:
    client = FakeMilvusClient()
    fence = FakeFence()

    receipt = _stage(_index(client, batch_size=2), fence, chunks=_chunks(5))

    assert receipt.imported_chunk_count == 5
    assert [len(batch) for batch in client.upsert_calls] == [2, 2, 1]
    assert fence.assertions >= 10


def test_cross_tenant_records_and_delete_are_strictly_scoped() -> None:
    client = FakeMilvusClient()
    index = _index(client)
    tenant_a = FakeFence(tenant_id="tenant-a", fence_id="fence-a")
    tenant_b = FakeFence(tenant_id="tenant-b", fence_id="fence-b")

    receipt_a = _stage(index, tenant_a)
    receipt_b = _stage(index, tenant_b)

    assert receipt_a.commit_token != receipt_b.commit_token
    assert len(client.records) == 4
    deleted = index.delete_staging(
        tenant_id=tenant_a.tenant_id,
        document_id=tenant_a.document_id,
        import_version=tenant_a.import_version,
        fence=tenant_a,
    )
    assert deleted == 2
    assert len(client.records) == 2
    assert {record["tenant_id"] for record in client.records.values()} == {"tenant-b"}
    assert len(client.delete_calls) == 1
    assert _parse_filter(client.delete_calls[0])["tenant_id"] == "tenant-a"


@pytest.mark.parametrize("tenant_id", [".tenant-a", "_tenant-a", ":tenant-a", "-tenant-a"])
def test_tenant_contract_matches_auth_and_durable_registry(tenant_id: str) -> None:
    client = FakeMilvusClient()
    fence = FakeFence(tenant_id=tenant_id)

    receipt = _stage(_index(client), fence)

    assert receipt.tenant_id == tenant_id
    assert {record["tenant_id"] for record in client.records.values()} == {tenant_id}


def test_unknown_delete_result_is_reconciled_by_exact_scope() -> None:
    client = FakeMilvusClient()
    index = _index(client)
    fence = FakeFence()
    _stage(index, fence)
    client.delete_then_raise = True

    deleted = index.delete_staging(
        tenant_id=fence.tenant_id,
        document_id=fence.document_id,
        import_version=fence.import_version,
        fence=fence,
    )

    assert deleted == 2
    assert client.records == {}


def test_partial_unknown_write_is_recovered_without_duplicates_on_retry() -> None:
    client = FakeMilvusClient()
    client.partial_then_raise = 1
    index = _index(client)

    with pytest.raises(IndexWriteUncertain, match="outcome is uncertain") as captured:
        _stage(index, FakeFence(), chunks=_chunks(3))

    assert "secret" not in str(captured.value)
    assert len(client.records) == 1
    receipt = _stage(
        index,
        FakeFence(attempt_count=2, fence_id="fence-attempt-2"),
        chunks=_chunks(3),
    )
    assert receipt.imported_chunk_count == 3
    assert len(client.records) == 3
    assert len(set(client.records)) == 3
    assert len(client.upsert_calls[1]) == 2


def test_lost_ack_is_reconciled_in_the_same_attempt() -> None:
    client = FakeMilvusClient()
    client.raise_after_all = True

    receipt = _stage(_index(client), FakeFence(), chunks=_chunks(3))

    assert receipt.imported_chunk_count == 3
    assert len(client.records) == 3


def test_stale_fence_is_rejected_before_any_milvus_operation() -> None:
    client = FakeMilvusClient()
    fence = FakeFence(current=False)

    with pytest.raises(CommitFenceError, match="no longer current") as captured:
        _stage(_index(client), fence)

    assert "secret" not in str(captured.value)
    assert client.query_calls == []
    assert client.upsert_calls == []


def test_fence_that_expires_after_write_never_receives_a_commit_receipt() -> None:
    client = FakeMilvusClient()
    # Assertions 1-4 cover entry and the pre-write read; assertion 5 is the
    # mandatory post-upsert fence check.
    stale = FakeFence(stale_on_assertion=5)

    with pytest.raises(CommitFenceError, match="no longer current"):
        _stage(_index(client), stale)

    assert len(client.records) == 2
    recovered = _stage(
        _index(client),
        FakeFence(attempt_count=2, fence_id="replacement-fence"),
    )
    assert recovered.imported_chunk_count == 2


def test_different_payload_for_immutable_version_fails_closed() -> None:
    client = FakeMilvusClient()
    index = _index(client)
    _stage(index, FakeFence())

    changed = _chunks()
    changed[0] = MilvusChunk(
        content="different content",
        dense_vector=[0.1, 0.2],
        sparse_vector={0: 0.5},
    )
    with pytest.raises(ImportVersionConflict, match="different chunks"):
        _stage(index, FakeFence(attempt_count=2, fence_id="fence-2"), chunks=changed)

    assert len(client.records) == 2


def test_vector_or_metadata_change_for_same_content_is_a_manifest_conflict() -> None:
    client = FakeMilvusClient()
    index = _index(client)
    original = _chunks()
    _stage(index, FakeFence(), chunks=original)

    changed = list(original)
    changed[0] = MilvusChunk(
        content=original[0].content,
        title="Changed title",
        dense_vector=[9.1, 9.2],
        sparse_vector={9: 0.9},
    )
    with pytest.raises(ImportVersionConflict, match="different chunks"):
        _stage(index, FakeFence(attempt_count=2, fence_id="fence-2"), chunks=changed)

    assert len(client.upsert_calls) == 1


def test_forged_cross_tenant_query_result_is_rejected() -> None:
    client = FakeMilvusClient()
    index = _index(client)
    _stage(index, FakeFence())
    forged = dict(next(iter(client.records.values())))
    forged["tenant_id"] = "tenant-b"
    client.forge_query_record = forged

    with pytest.raises(ImportVersionConflict):
        _stage(index, FakeFence(attempt_count=2, fence_id="fence-2"))


def test_forged_commit_token_is_rejected_on_strong_readback() -> None:
    client = FakeMilvusClient()
    index = _index(client)
    _stage(index, FakeFence())
    forged = dict(next(iter(client.records.values())))
    forged["commit_token"] = "milvus-v1." + "0" * 64
    client.forge_query_record = forged

    with pytest.raises(ImportVersionConflict):
        _stage(index, FakeFence(attempt_count=2, fence_id="fence-2"))


@pytest.mark.parametrize(
    "tenant_id",
    [
        'tenant-a" or true',
        "tenant-a\n or import_version != \"\"",
        "",
        "../tenant-a",
    ],
)
def test_malicious_tenant_expression_is_rejected_before_query(tenant_id: str) -> None:
    client = FakeMilvusClient()
    fence = FakeFence(tenant_id=tenant_id)

    with pytest.raises(ValueError, match="tenant identity"):
        _stage(_index(client), fence)

    assert client.query_calls == []
    assert client.upsert_calls == []


@pytest.mark.parametrize(
    ("collection_name", "pipeline_version"),
    [
        ('chunks; drop collection "prod"', "bge-m3.v1"),
        ("import_chunks_v1", 'v1" or true'),
    ],
)
def test_collection_and_pipeline_expression_fragments_are_rejected(
    collection_name: str,
    pipeline_version: str,
) -> None:
    with pytest.raises(ValueError):
        MilvusImportIndex(
            FakeMilvusClient(),
            collection_name=collection_name,
            pipeline_version=pipeline_version,
        )


def test_fence_identity_must_match_all_staging_identity_fields() -> None:
    client = FakeMilvusClient()
    fence = FakeFence(task_id="33333333-3333-4333-8333-333333333333")

    with pytest.raises(CommitFenceError, match="does not match"):
        _stage(_index(client), fence)

    assert client.query_calls == []


@pytest.mark.parametrize(
    "chunk",
    [
        MilvusChunk(content="", dense_vector=[0.1], sparse_vector={0: 0.2}),
        MilvusChunk(content="valid", dense_vector=[float("nan")], sparse_vector={0: 0.2}),
        MilvusChunk(content="valid", dense_vector=[0.1], sparse_vector={-1: 0.2}),
    ],
)
def test_invalid_chunk_data_is_rejected_before_external_calls(chunk: MilvusChunk) -> None:
    client = FakeMilvusClient()

    with pytest.raises(ValueError):
        _stage(_index(client), FakeFence(), chunks=[chunk])

    assert client.query_calls == []
    assert client.upsert_calls == []


def test_utf8_content_and_metadata_respect_milvus_varchar_byte_limit() -> None:
    valid = "中" * 21_845
    receipt = _stage(
        _index(FakeMilvusClient()),
        FakeFence(),
        chunks=[MilvusChunk(content=valid, dense_vector=[0.1], sparse_vector={0: 0.2})],
    )
    assert receipt.imported_chunk_count == 1

    for chunk in (
        MilvusChunk(content=valid + "中", dense_vector=[0.1], sparse_vector={0: 0.2}),
        MilvusChunk(
            content="valid",
            title=valid + "中",
            dense_vector=[0.1],
            sparse_vector={0: 0.2},
        ),
    ):
        with pytest.raises(ValueError):
            _stage(
                _index(FakeMilvusClient()),
                FakeFence(),
                chunks=[chunk],
            )
