from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

_TENANT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_PIPELINE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_COLLECTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,254}$")
_FENCE_ID = re.compile(r"^[!-~]{1,256}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_VARCHAR_BYTES = 65_535

_READ_FIELDS = (
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
    "content",
)


class MilvusImportIndexError(RuntimeError):
    """Base error for a tenant-scoped staging operation."""


class CommitFenceError(MilvusImportIndexError):
    """The caller could not prove it still owns the import attempt."""


class ImportVersionConflict(MilvusImportIndexError):
    """An immutable import version already contains different data."""


class IndexWriteUncertain(MilvusImportIndexError):
    """A dependency failed and the exact post-write state could not be proven."""


class IndexVerificationError(MilvusImportIndexError):
    """Read-after-write verification did not match the staged manifest."""


class CommitFence(Protocol):
    """Read-only proof that a worker still owns one durable task attempt.

    Implementations must perform a strongly consistent check in
    :meth:`assert_current` and raise if the durable lease, attempt, or activation
    generation no longer matches.  The raw task lease token is deliberately not
    exposed to the index layer.
    """

    @property
    def task_id(self) -> str: ...

    @property
    def document_id(self) -> str: ...

    @property
    def tenant_id(self) -> str: ...

    @property
    def import_version(self) -> str: ...

    @property
    def attempt_count(self) -> int: ...

    @property
    def fence_id(self) -> str: ...

    def assert_current(self) -> None: ...


class MilvusImportClient(Protocol):
    """The small synchronous Milvus surface required by the staging writer."""

    def query(
        self,
        *,
        collection_name: str,
        filter: str,
        output_fields: list[str],
        limit: int,
        consistency_level: str,
    ) -> Sequence[Mapping[str, object]]: ...

    def upsert(
        self,
        *,
        collection_name: str,
        data: list[dict[str, object]],
    ) -> object: ...

    def delete(self, *, collection_name: str, filter: str) -> object: ...


@dataclass(frozen=True, slots=True)
class ImportIndexIdentity:
    """The complete immutable scope of one staged import version."""

    tenant_id: str
    document_id: str
    import_version: str
    pipeline_version: str

    @property
    def task_id(self) -> str:
        return self.import_version


@dataclass(frozen=True, slots=True)
class MilvusChunk:
    """A fixed-field chunk accepted by the staging writer.

    Dynamic caller-supplied field names are intentionally unsupported.  This
    prevents graph output from smuggling identity fields or expression fragments
    into a Milvus entity.
    """

    content: str
    dense_vector: Sequence[float]
    sparse_vector: Mapping[int, float]
    title: str = ""
    parent_title: str = ""
    file_title: str = ""
    item_name: str = ""


@dataclass(frozen=True, slots=True)
class MilvusCommitReceipt:
    """Stable proof that the exact staging manifest was read back."""

    task_id: str
    document_id: str
    tenant_id: str
    import_version: str
    pipeline_version: str
    imported_chunk_count: int
    manifest_sha256: str
    commit_token: str
    fence_id: str
    attempt_count: int


@dataclass(frozen=True, slots=True)
class _FenceEvidence:
    fence_id: str
    attempt_count: int


def _canonical_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid import identifier")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("invalid import identifier") from None
    if str(parsed) != value:
        raise ValueError("invalid import identifier")
    return value


def _validate_identity(identity: ImportIndexIdentity) -> None:
    if not isinstance(identity.tenant_id, str) or _TENANT_ID.fullmatch(identity.tenant_id) is None:
        raise ValueError("invalid tenant identity")
    _canonical_uuid(identity.document_id)
    _canonical_uuid(identity.import_version)
    if (
        not isinstance(identity.pipeline_version, str)
        or _PIPELINE_VERSION.fullmatch(identity.pipeline_version) is None
    ):
        raise ValueError("invalid pipeline version")


def _json_string(value: str) -> str:
    """Encode a validated literal without accepting raw expression fragments."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _scope_filter(identity: ImportIndexIdentity) -> str:
    _validate_identity(identity)
    return " and ".join(
        (
            f"tenant_id == {_json_string(identity.tenant_id)}",
            f"document_id == {_json_string(identity.document_id)}",
            f"import_version == {_json_string(identity.import_version)}",
            f"pipeline_version == {_json_string(identity.pipeline_version)}",
        )
    )


def _valid_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _normalize_dense(vector: Sequence[float]) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes, bytearray)) or not isinstance(vector, Sequence):
        raise ValueError("invalid dense vector")
    normalized = tuple(float(value) for value in vector if _valid_number(value))
    if not normalized or len(normalized) != len(vector):
        raise ValueError("invalid dense vector")
    return normalized


def _normalize_sparse(vector: Mapping[int, float]) -> dict[int, float]:
    if not isinstance(vector, Mapping) or not vector:
        raise ValueError("invalid sparse vector")
    normalized: dict[int, float] = {}
    for raw_index, raw_value in vector.items():
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
            or not _valid_number(raw_value)
        ):
            raise ValueError("invalid sparse vector")
        normalized[raw_index] = float(raw_value)
    return normalized


def _metadata_text(value: object) -> str:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_VARCHAR_BYTES:
        raise ValueError("invalid chunk metadata")
    return value


def _content_digest(content: str) -> str:
    if not isinstance(content, str) or not content:
        raise ValueError("invalid chunk content")
    encoded = content.encode("utf-8")
    if len(encoded) > _MAX_VARCHAR_BYTES:
        raise ValueError("invalid chunk content")
    return hashlib.sha256(encoded).hexdigest()


def _chunk_id(identity: ImportIndexIdentity, ordinal: int, content_sha256: str) -> str:
    material = "\x00".join(
        (
            "milvus-import-chunk-v1",
            identity.tenant_id,
            identity.document_id,
            identity.import_version,
            identity.pipeline_version,
            str(ordinal),
            content_sha256,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _manifest_digest(records: Sequence[Mapping[str, object]]) -> str:
    manifest = [
        {
            "chunk_id": record["chunk_id"],
            "content_sha256": record["content_sha256"],
            "record_sha256": record["record_sha256"],
            "ordinal": record["ordinal"],
        }
        for record in records
    ]
    encoded = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_digest(record: Mapping[str, object]) -> str:
    """Hash the complete caller-controlled payload before it crosses the client."""

    payload = {
        "content": record["content"],
        "dense_vector": record["dense_vector"],
        "file_title": record["file_title"],
        "item_name": record["item_name"],
        "parent_title": record["parent_title"],
        "sparse_vector": record["sparse_vector"],
        "title": record["title"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _commit_token(identity: ImportIndexIdentity, manifest_sha256: str, count: int) -> str:
    material = "\x00".join(
        (
            "milvus-staging-commit-v1",
            identity.tenant_id,
            identity.document_id,
            identity.import_version,
            identity.pipeline_version,
            str(count),
            manifest_sha256,
        )
    )
    return f"milvus-v1.{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


class MilvusImportIndex:
    """Tenant-safe, retry-idempotent writer for immutable import staging data.

    This class proves only the Milvus staging commit.  Publication/activation is
    a separate durable transaction and must happen before callers advertise the
    import processor as supporting fully idempotent commits.
    """

    def __init__(
        self,
        client: MilvusImportClient,
        *,
        collection_name: str,
        pipeline_version: str,
        batch_size: int = 128,
        max_chunks: int = 10_000,
    ) -> None:
        if not isinstance(collection_name, str) or _COLLECTION_NAME.fullmatch(collection_name) is None:
            raise ValueError("invalid Milvus collection name")
        if not isinstance(pipeline_version, str) or _PIPELINE_VERSION.fullmatch(pipeline_version) is None:
            raise ValueError("invalid pipeline version")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("invalid Milvus batch size")
        if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks <= 0:
            raise ValueError("invalid Milvus chunk limit")
        if batch_size > max_chunks:
            raise ValueError("Milvus batch size exceeds chunk limit")
        self._client = client
        self._collection_name = collection_name
        self._pipeline_version = pipeline_version
        self._batch_size = batch_size
        self._max_chunks = max_chunks

    @property
    def pipeline_version(self) -> str:
        return self._pipeline_version

    def identity(
        self,
        *,
        tenant_id: str,
        document_id: str,
        import_version: str,
    ) -> ImportIndexIdentity:
        identity = ImportIndexIdentity(
            tenant_id=tenant_id,
            document_id=document_id,
            import_version=import_version,
            pipeline_version=self._pipeline_version,
        )
        _validate_identity(identity)
        return identity

    @staticmethod
    def _assert_fence(
        identity: ImportIndexIdentity,
        fence: CommitFence,
    ) -> _FenceEvidence:
        expected_identity = (
            identity.task_id,
            identity.document_id,
            identity.tenant_id,
            identity.import_version,
        )

        def evidence() -> tuple[tuple[object, object, object, object], object, object]:
            return (
                (
                    fence.task_id,
                    fence.document_id,
                    fence.tenant_id,
                    fence.import_version,
                ),
                fence.attempt_count,
                fence.fence_id,
            )

        try:
            before_identity, before_attempt, before_fence_id = evidence()
        except Exception:
            raise CommitFenceError("Import commit fence is invalid") from None
        if (
            before_identity != expected_identity
            or identity.task_id != identity.import_version
            or isinstance(before_attempt, bool)
            or not isinstance(before_attempt, int)
            or before_attempt <= 0
            or not isinstance(before_fence_id, str)
            or _FENCE_ID.fullmatch(before_fence_id) is None
        ):
            raise CommitFenceError("Import commit fence does not match the staging scope")
        try:
            fence.assert_current()
        except Exception:
            raise CommitFenceError("Import commit fence is no longer current") from None
        try:
            after_identity, after_attempt, after_fence_id = evidence()
        except Exception:
            raise CommitFenceError("Import commit fence is invalid") from None
        if (
            after_identity != before_identity
            or after_attempt != before_attempt
            or after_fence_id != before_fence_id
        ):
            raise CommitFenceError("Import commit fence changed during verification")
        return _FenceEvidence(
            fence_id=before_fence_id,
            attempt_count=before_attempt,
        )

    def _records(
        self,
        identity: ImportIndexIdentity,
        chunks: Sequence[MilvusChunk],
    ) -> tuple[dict[str, object], ...]:
        if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence):
            raise ValueError("invalid Milvus chunks")
        if not chunks or len(chunks) > self._max_chunks:
            raise ValueError("invalid Milvus chunk count")
        records: list[dict[str, object]] = []
        dense_dimension: int | None = None
        for ordinal, chunk in enumerate(chunks):
            if not isinstance(chunk, MilvusChunk):
                raise ValueError("invalid Milvus chunk")
            content_sha256 = _content_digest(chunk.content)
            dense_vector = _normalize_dense(chunk.dense_vector)
            if dense_dimension is None:
                dense_dimension = len(dense_vector)
            elif len(dense_vector) != dense_dimension:
                raise ValueError("inconsistent dense vector dimensions")
            sparse_vector = _normalize_sparse(chunk.sparse_vector)
            record: dict[str, object] = {
                "chunk_id": _chunk_id(identity, ordinal, content_sha256),
                "tenant_id": identity.tenant_id,
                "document_id": identity.document_id,
                "import_version": identity.import_version,
                "pipeline_version": identity.pipeline_version,
                "ordinal": ordinal,
                "content_sha256": content_sha256,
                "record_sha256": "",
                "manifest_sha256": "",
                "content": chunk.content,
                "title": _metadata_text(chunk.title),
                "parent_title": _metadata_text(chunk.parent_title),
                "file_title": _metadata_text(chunk.file_title),
                "item_name": _metadata_text(chunk.item_name),
                "dense_vector": list(dense_vector),
                "sparse_vector": sparse_vector,
            }
            record["record_sha256"] = _record_digest(record)
            records.append(record)
        manifest_sha256 = _manifest_digest(records)
        commit_token = _commit_token(identity, manifest_sha256, len(records))
        for record in records:
            record["manifest_sha256"] = manifest_sha256
            record["commit_token"] = commit_token
        return tuple(records)

    def _query_scope(
        self,
        identity: ImportIndexIdentity,
        fence: CommitFence,
    ) -> tuple[Mapping[str, object], ...]:
        self._assert_fence(identity, fence)
        try:
            observed = self._client.query(
                collection_name=self._collection_name,
                filter=_scope_filter(identity),
                output_fields=list(_READ_FIELDS),
                limit=self._max_chunks + 1,
                consistency_level="Strong",
            )
        except Exception:
            raise IndexWriteUncertain("Milvus staging state could not be read") from None
        self._assert_fence(identity, fence)
        if isinstance(observed, (str, bytes, bytearray)) or not isinstance(observed, Sequence):
            raise IndexVerificationError("Milvus returned an invalid staging result")
        return tuple(observed)

    @staticmethod
    def _verified_existing(
        identity: ImportIndexIdentity,
        expected: Sequence[Mapping[str, object]],
        observed: Sequence[Mapping[str, object]],
    ) -> frozenset[str]:
        expected_by_id = {str(record["chunk_id"]): record for record in expected}
        found: set[str] = set()
        manifest_sha256 = str(expected[0]["manifest_sha256"])
        for record in observed:
            if not isinstance(record, Mapping):
                raise IndexVerificationError("Milvus returned an invalid staging record")
            chunk_id = record.get("chunk_id")
            if not isinstance(chunk_id, str) or _SHA256.fullmatch(chunk_id) is None:
                raise IndexVerificationError("Milvus returned an invalid staging record")
            if chunk_id in found:
                raise IndexVerificationError("Milvus returned duplicate staging records")
            expected_record = expected_by_id.get(chunk_id)
            if expected_record is None:
                raise ImportVersionConflict("Import version already contains different chunks")
            content = record.get("content")
            if not isinstance(content, str):
                raise IndexVerificationError("Milvus returned an invalid staging record")
            try:
                observed_content_sha256 = _content_digest(content)
            except ValueError:
                raise IndexVerificationError("Milvus returned an invalid staging record") from None
            if (
                record.get("tenant_id") != identity.tenant_id
                or record.get("document_id") != identity.document_id
                or record.get("import_version") != identity.import_version
                or record.get("pipeline_version") != identity.pipeline_version
                or record.get("ordinal") != expected_record["ordinal"]
                or record.get("content_sha256") != expected_record["content_sha256"]
                or record.get("record_sha256") != expected_record["record_sha256"]
                or observed_content_sha256 != expected_record["content_sha256"]
                or record.get("manifest_sha256") != manifest_sha256
                or record.get("commit_token") != expected_record["commit_token"]
            ):
                raise ImportVersionConflict("Import version already contains different chunks")
            found.add(chunk_id)
        return frozenset(found)

    @staticmethod
    def _verified_scope_count(
        identity: ImportIndexIdentity,
        observed: Sequence[Mapping[str, object]],
    ) -> int:
        found: set[str] = set()
        for record in observed:
            if not isinstance(record, Mapping):
                raise IndexVerificationError("Milvus returned an invalid staging record")
            chunk_id = record.get("chunk_id")
            if (
                not isinstance(chunk_id, str)
                or _SHA256.fullmatch(chunk_id) is None
                or chunk_id in found
                or record.get("tenant_id") != identity.tenant_id
                or record.get("document_id") != identity.document_id
                or record.get("import_version") != identity.import_version
                or record.get("pipeline_version") != identity.pipeline_version
            ):
                raise IndexVerificationError("Milvus returned an out-of-scope staging record")
            found.add(chunk_id)
        return len(found)

    @staticmethod
    def _receipt(
        identity: ImportIndexIdentity,
        records: Sequence[Mapping[str, object]],
        fence: _FenceEvidence,
    ) -> MilvusCommitReceipt:
        manifest_sha256 = str(records[0]["manifest_sha256"])
        return MilvusCommitReceipt(
            task_id=identity.task_id,
            document_id=identity.document_id,
            tenant_id=identity.tenant_id,
            import_version=identity.import_version,
            pipeline_version=identity.pipeline_version,
            imported_chunk_count=len(records),
            manifest_sha256=manifest_sha256,
            commit_token=str(records[0]["commit_token"]),
            fence_id=fence.fence_id,
            attempt_count=fence.attempt_count,
        )

    def stage(
        self,
        *,
        tenant_id: str,
        document_id: str,
        import_version: str,
        chunks: Sequence[MilvusChunk],
        fence: CommitFence,
    ) -> MilvusCommitReceipt:
        """Idempotently stage and verify one immutable import manifest."""

        identity = self.identity(
            tenant_id=tenant_id,
            document_id=document_id,
            import_version=import_version,
        )
        records = self._records(identity, chunks)
        self._assert_fence(identity, fence)
        observed = self._query_scope(identity, fence)
        present = self._verified_existing(identity, records, observed)
        missing = [record for record in records if record["chunk_id"] not in present]

        for offset in range(0, len(missing), self._batch_size):
            batch = missing[offset : offset + self._batch_size]
            self._assert_fence(identity, fence)
            try:
                self._client.upsert(
                    collection_name=self._collection_name,
                    data=batch,
                )
            except Exception:
                # A transport error does not prove that Milvus rejected the write.
                # Recover only if a strongly consistent exact read proves success.
                self._assert_fence(identity, fence)
                recovered = self._query_scope(identity, fence)
                recovered_ids = self._verified_existing(identity, records, recovered)
                if len(recovered_ids) == len(records):
                    evidence = self._assert_fence(identity, fence)
                    return self._receipt(identity, records, evidence)
                raise IndexWriteUncertain("Milvus staging write outcome is uncertain") from None
            self._assert_fence(identity, fence)

        committed = self._query_scope(identity, fence)
        committed_ids = self._verified_existing(identity, records, committed)
        if len(committed_ids) != len(records):
            raise IndexVerificationError("Milvus staging manifest is incomplete")
        evidence = self._assert_fence(identity, fence)
        return self._receipt(identity, records, evidence)

    def delete_staging(
        self,
        *,
        tenant_id: str,
        document_id: str,
        import_version: str,
        fence: CommitFence,
    ) -> int:
        """Delete only one exact tenant/document/import/pipeline staging scope."""

        identity = self.identity(
            tenant_id=tenant_id,
            document_id=document_id,
            import_version=import_version,
        )
        observed = self._query_scope(identity, fence)
        count = self._verified_scope_count(identity, observed)
        if count == 0:
            self._assert_fence(identity, fence)
            return 0
        self._assert_fence(identity, fence)
        try:
            self._client.delete(
                collection_name=self._collection_name,
                filter=_scope_filter(identity),
            )
        except Exception:
            self._assert_fence(identity, fence)
            remaining = self._query_scope(identity, fence)
            if self._verified_scope_count(identity, remaining) == 0:
                return count
            raise IndexWriteUncertain("Milvus staging delete outcome is uncertain") from None
        self._assert_fence(identity, fence)
        remaining = self._query_scope(identity, fence)
        if self._verified_scope_count(identity, remaining) != 0:
            raise IndexVerificationError("Milvus staging delete was not verified")
        self._assert_fence(identity, fence)
        return count
