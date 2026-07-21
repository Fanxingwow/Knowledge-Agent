from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from app.import_process.agent.main_graph import (
    EMBED_NODE,
    IMPORT_GRAPH_NODES,
    ImportGraphNodes,
    build_import_graph,
)
from app.import_process.agent.nodes.node_hybrid_embedding import (
    HybridEmbedding,
    HybridEmbeddingNode,
)
from app.import_process.agent.nodes.node_markdown_images import (
    MarkdownImagePolicy,
    MarkdownImageTextNode,
)
from app.import_process.agent.nodes.node_markdown_splitter import (
    MarkdownSplitPolicy,
    MarkdownSplitterNode,
)
from app.import_process.agent.nodes.node_safe_item_name import (
    ItemNameContext,
    SafeItemNameNode,
)
from app.import_process.agent.nodes.node_validated_entry import ValidatedEntryNode
from app.import_process.agent.nodes.safe_node_contracts import (
    EMBEDDED_CHUNK_FIELDS,
    TEXT_CHUNK_FIELDS,
    SafeNodeContractError,
)
from app.import_process.agent.state import ImportGraphState, create_default_state

_TASK_ID = "11111111-1111-4111-8111-111111111111"
_DOCUMENT_ID = "22222222-2222-4222-8222-222222222222"


def _entry_state() -> ImportGraphState:
    return create_default_state(
        task_id=_TASK_ID,
        document_id=_DOCUMENT_ID,
        tenant_id="tenant-a",
        owner_subject="owner-a",
        workspace_dir="C:/private/workspace",
        source_path="C:/private/workspace/source.md",
        source_ref="tenant-a/imports/source@version",
        original_name="Guide.md",
        content_type="text/markdown",
        size=64,
        sha256="a" * 64,
        source_kind="markdown",
        is_md_read_enabled=True,
        is_pdf_read_enabled=False,
        local_dir="C:/private/workspace",
        local_file_path="C:/private/workspace/source.md",
        file_title="Guide",
        md_path="C:/private/workspace/source.md",
        pdf_path="",
    )


def _split_state(markdown: str) -> ImportGraphState:
    return create_default_state(
        file_title="Guide",
        md_content=markdown,
        chunks=[],
        item_name="",
    )


def _text_chunks() -> list[dict[str, object]]:
    return [
        {
            "content": "private body one",
            "title": "Install",
            "parent_title": "Manual",
            "file_title": "Guide",
            "item_name": "",
        },
        {
            "content": "private body two",
            "title": "Usage",
            "parent_title": "Manual",
            "file_title": "Guide",
            "item_name": "",
        },
    ]


def test_validated_entry_is_a_noop_and_does_not_mutate_state() -> None:
    state = _entry_state()
    before = copy.deepcopy(state)

    update = ValidatedEntryNode()(state)

    assert update == {}
    assert state == before


@pytest.mark.parametrize("tenant_id", [".tenant-a", "_tenant-a", ":tenant-a", "-tenant-a"])
def test_validated_entry_matches_the_published_auth_tenant_contract(tenant_id: str) -> None:
    state = _entry_state()
    state["tenant_id"] = tenant_id

    assert ValidatedEntryNode()(state) == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "not-a-task"),
        ("tenant_id", "tenant with spaces"),
        ("sha256", "A" * 64),
        ("size", True),
        ("file_title", "Different"),
        ("source_kind", "pdf"),
        ("local_file_path", "C:/attacker/source.md"),
    ],
)
def test_validated_entry_rejects_invalid_metadata_without_echoing_values(
    field: str,
    value: object,
) -> None:
    state = _entry_state()
    state[field] = value  # type: ignore[literal-required]

    with pytest.raises(SafeNodeContractError) as captured:
        ValidatedEntryNode()(state)

    assert str(value) not in str(captured.value)


def test_validated_entry_rejects_precomputed_derived_data_and_size_limit() -> None:
    state = _entry_state()
    state["chunks"] = _text_chunks()
    with pytest.raises(SafeNodeContractError, match="derived data"):
        ValidatedEntryNode()(state)

    state = _entry_state()
    state["size"] = 65
    with pytest.raises(SafeNodeContractError, match="integrity"):
        ValidatedEntryNode(max_source_bytes=64)(state)


def test_markdown_images_become_alt_text_without_accessing_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("pure image node attempted external I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    markdown = (
        "# Cafe\u0301\r\n"
        "Remote ![dashboard](https://secret.example/a.png) and "
        "<img src='C:/private/image.png' alt='\u56fe \u7247'>.\n"
        "Reference ![logo][asset].\n"
        "[asset]: https://secret.example/logo.png\n"
        "Ordinary [link](https://kept.example).\n"
    )
    state = _split_state(markdown)
    before = copy.deepcopy(state)

    update = MarkdownImageTextNode()(state)

    assert state == before
    content = update["md_content"]
    assert isinstance(content, str)
    assert content.startswith("# Café\n")
    assert "[image: dashboard]" in content
    assert "[image: 图 片]" in content
    assert "[image: logo]" in content
    assert "secret.example" not in content
    assert "C:/private" not in content
    assert "[asset]:" not in content
    assert "https://kept.example" in content


@pytest.mark.parametrize(
    "markdown",
    [
        "bad ![alt](https://example.invalid/unclosed",
        "bad <img src='unclosed'",
        "bad\x00content",
        "bad\u0085content",
        "bad\ufffecontent",
    ],
)
def test_markdown_image_node_rejects_malformed_or_unsafe_text(markdown: str) -> None:
    with pytest.raises(SafeNodeContractError):
        MarkdownImageTextNode()(_split_state(markdown))


def test_markdown_image_count_and_syntax_are_bounded() -> None:
    policy = MarkdownImagePolicy(max_images=1, max_image_syntax_characters=8)
    node = MarkdownImageTextNode(policy)

    with pytest.raises(SafeNodeContractError, match="too many"):
        node(_split_state("![a](x) ![b](y)"))
    with pytest.raises(SafeNodeContractError, match="syntax"):
        node(_split_state("![a](123456789)"))


def test_splitter_is_deterministic_fixed_shape_and_heading_aware() -> None:
    markdown = "preamble\n\n# Manual\nintro\n```md\n# not a heading\n```\n## Install\nstep one step two step three\n"
    policy = MarkdownSplitPolicy(max_chunk_characters=32, max_chunk_bytes=96)
    node = MarkdownSplitterNode(policy)
    state = _split_state(markdown)

    first = node(state)
    second = node(copy.deepcopy(state))

    assert first == second
    chunks = first["chunks"]
    assert isinstance(chunks, list)
    assert all(set(chunk) == TEXT_CHUNK_FIELDS for chunk in chunks)
    assert all(len(chunk["content"]) <= 32 for chunk in chunks)
    assert all(len(chunk["content"].encode("utf-8")) <= 96 for chunk in chunks)
    assert any(chunk["title"] == "Guide" for chunk in chunks)
    assert any(chunk["title"] == "Manual" for chunk in chunks)
    install = [chunk for chunk in chunks if chunk["title"] == "Install"]
    assert install
    assert all(chunk["parent_title"] == "Manual" for chunk in install)
    assert all(chunk["title"] != "not a heading" for chunk in chunks)


def test_splitter_never_breaks_an_extended_unicode_grapheme() -> None:
    family = "👨‍👩‍👧‍👦"
    node = MarkdownSplitterNode(MarkdownSplitPolicy(max_chunk_characters=8, max_chunk_bytes=30))

    update = node(_split_state(f"# H\n{family * 3}"))

    chunks = update["chunks"]
    assert isinstance(chunks, list)
    emoji_chunks = [chunk["content"] for chunk in chunks if family[0] in chunk["content"]]
    assert emoji_chunks == [family, family, family]


def test_splitter_fails_closed_on_document_chunk_and_grapheme_limits() -> None:
    with pytest.raises(SafeNodeContractError, match="chunk limit"):
        MarkdownSplitterNode(
            MarkdownSplitPolicy(
                max_document_characters=20,
                max_document_bytes=80,
                max_chunk_characters=5,
                max_chunk_bytes=20,
                max_chunks=1,
            )
        )(_split_state("one two three"))

    combining_grapheme = "a" + "\u0301" * 8
    with pytest.raises(SafeNodeContractError, match="grapheme"):
        MarkdownSplitterNode(
            MarkdownSplitPolicy(
                max_document_characters=16,
                max_document_bytes=64,
                max_chunk_characters=4,
                max_chunk_bytes=32,
            )
        )(_split_state(combining_grapheme))


def test_item_name_is_derived_locally_and_updates_only_fixed_chunks() -> None:
    state = _split_state("unused")
    state["chunks"] = _text_chunks()
    before = copy.deepcopy(state)

    update = SafeItemNameNode()(state)

    assert state == before
    assert update["item_name"] == "Guide"
    chunks = update["chunks"]
    assert isinstance(chunks, list)
    assert all(set(chunk) == TEXT_CHUNK_FIELDS for chunk in chunks)
    assert {chunk["item_name"] for chunk in chunks} == {"Guide"}


@dataclass
class RecordingRecognizer:
    context: ItemNameContext | None = None

    def __call__(self, context: ItemNameContext) -> str:
        self.context = context
        return "Enterprise Product"


def test_injected_item_recognizer_receives_only_bounded_metadata() -> None:
    recognizer = RecordingRecognizer()
    state = _split_state("unused")
    state["chunks"] = _text_chunks()

    update = SafeItemNameNode(recognizer=recognizer)(state)

    assert update["item_name"] == "Enterprise Product"
    assert recognizer.context == ItemNameContext(
        file_title="Guide",
        titles=("Install", "Usage"),
        parent_titles=("Manual",),
        chunk_count=2,
    )
    assert "private body" not in repr(recognizer.context)


def test_explicit_item_name_and_recognizer_errors_are_not_silently_ignored() -> None:
    state = _split_state("unused")
    state["chunks"] = _text_chunks()
    assert SafeItemNameNode(explicit_name="Pinned Product")(state)["item_name"] == "Pinned Product"

    def broken(context: ItemNameContext) -> str:
        raise RuntimeError("recognizer failed")

    with pytest.raises(RuntimeError, match="recognizer failed"):
        SafeItemNameNode(recognizer=broken)(state)


def test_item_name_node_rejects_dynamic_fields_and_prepopulated_names() -> None:
    state = _split_state("unused")
    chunks = _text_chunks()
    chunks[0]["tenant_id"] = "smuggled"
    state["chunks"] = chunks
    with pytest.raises(SafeNodeContractError, match="shape"):
        SafeItemNameNode()(state)

    state["chunks"] = _text_chunks()
    state["chunks"][0]["item_name"] = "already set"
    with pytest.raises(SafeNodeContractError, match="before recognition"):
        SafeItemNameNode()(state)

    state["chunks"] = _text_chunks()
    state["item_name"] = "already set"
    with pytest.raises(SafeNodeContractError, match="before recognition"):
        SafeItemNameNode()(state)


class _PureBackend:
    dense_dimension = 2

    def embed(self, texts: Sequence[str]) -> Sequence[HybridEmbedding]:
        return tuple(HybridEmbedding(dense=(0.1, 0.2), sparse={1: 0.5}) for _ in texts)

    def probe(self) -> None:
        return

    def close(self) -> None:
        return


def test_pure_nodes_compose_through_the_safe_import_graph_contract() -> None:
    def parse_document(state: ImportGraphState) -> Mapping[str, object]:
        return {"md_content": "# Manual\nUse ![screen](https://untrusted.invalid/a.png)."}

    def commit_index(state: ImportGraphState) -> Mapping[str, object]:
        chunks = state["chunks"]
        assert chunks and set(chunks[0]) == EMBEDDED_CHUNK_FIELDS
        return {
            "commit_receipt": {
                "task_id": state["task_id"],
                "document_id": state["document_id"],
                "tenant_id": state["tenant_id"],
                "imported_chunk_count": len(chunks),
                "commit_token": "pure-test-commit",
            }
        }

    runtime = build_import_graph(
        ImportGraphNodes(
            entry=ValidatedEntryNode(),
            parse_document=parse_document,
            extract_images=MarkdownImageTextNode(),
            split_document=MarkdownSplitterNode(),
            recognize_item=SafeItemNameNode(),
            embed_chunks=HybridEmbeddingNode(_PureBackend()),
            commit_index=commit_index,
        )
    )

    updates = list(runtime.stream(_entry_state()))

    assert {update.node for update in updates} == IMPORT_GRAPH_NODES
    embedded = next(update for update in updates if update.node == EMBED_NODE)
    chunks = embedded.values["chunks"]
    assert isinstance(chunks, list)
    assert set(chunks[0]) == EMBEDDED_CHUNK_FIELDS
    assert "untrusted.invalid" not in chunks[0]["content"]
