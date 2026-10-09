"""list_knowledge_documents / read_knowledge_document (spec §2.1): the
document-level read-only tools.

Reads the bound library's documents straight from the business store — no
Qdrant, no embedding — and answers with an honest status count. Fail-closed
on missing binding / foreign owner, exactly like the retrieval tool.
"""

from __future__ import annotations

import pytest
from deerflow_knowledge.access import ACCESS_DENIED_MESSAGE, NO_KB_GUIDANCE
from deerflow_knowledge.store import KnowledgeStore

from deerflow.tools.builtins.knowledge_documents_tool import _list_documents_impl, _read_document_impl

from .conftest import scope_runtime


async def _seed_kb(store: KnowledgeStore, *, kb_id: str = "kb-l", owner_id: str = "user-1") -> None:
    await store.create_kb(kb_id=kb_id, owner_id=owner_id, name="清单测试库")


async def _add_document(store: KnowledgeStore, doc_id: str, name: str, *, kb_id: str = "kb-l") -> None:
    await store.create_document(doc_id=doc_id, kb_id=kb_id, uploader_id="user-1", name=name, size_bytes=10, storage_path=f"/{doc_id}")


@pytest.mark.asyncio
async def test_lists_every_document_with_honest_counts(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)
    await _add_document(store, "d-1", "架构.md")
    await store.update_document_status("d-1", "ready", chunk_count=3)
    await _add_document(store, "d-2", "手册.pdf")
    await store.update_document_status("d-2", "indexing", progress_percent=40)
    await _add_document(store, "d-3", "坏件.pdf")
    await store.update_document_status("d-3", "failed", error="parse boom")

    result = await _list_documents_impl(scope_runtime(kb_id="kb-l"), store=store)

    assert result["message"] == "共 3 篇文档（就绪 1 · 处理中 1 · 失败 1）。"
    by_id = {document["doc_id"]: document for document in result["documents"]}
    assert len(result["documents"]) == 3
    assert by_id["d-1"] == {"doc_id": "d-1", "name": "架构.md", "status": "ready", "chunk_count": 3}
    assert by_id["d-2"]["status"] == "indexing"
    assert by_id["d-2"]["chunk_count"] is None
    assert by_id["d-3"]["status"] == "failed"


@pytest.mark.asyncio
async def test_message_omits_zero_count_groups(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)
    await _add_document(store, "d-1", "一.md")
    await store.update_document_status("d-1", "ready", chunk_count=1)
    await _add_document(store, "d-2", "二.md")
    await store.update_document_status("d-2", "ready", chunk_count=2)

    result = await _list_documents_impl(scope_runtime(kb_id="kb-l"), store=store)

    assert result["message"] == "共 2 篇文档（就绪 2）。"


@pytest.mark.asyncio
async def test_empty_library_says_so(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)

    result = await _list_documents_impl(scope_runtime(kb_id="kb-l"), store=store)

    assert result == {"documents": [], "message": "当前知识库中还没有文档。"}


@pytest.mark.asyncio
async def test_unbound_runtime_returns_guidance(session_factory) -> None:
    store = KnowledgeStore(session_factory)

    result = await _list_documents_impl(scope_runtime(kb_id=None), store=store)

    assert result == {"documents": [], "message": NO_KB_GUIDANCE}


@pytest.mark.asyncio
async def test_non_owner_is_denied(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)
    await _add_document(store, "d-1", "架构.md")

    result = await _list_documents_impl(scope_runtime(kb_id="kb-l", user_id="user-2"), store=store)

    assert result == {"documents": [], "message": ACCESS_DENIED_MESSAGE}


async def _seed_chunks(store: KnowledgeStore, doc_id: str, indexes_and_texts, *, kb_id: str = "kb-l") -> None:
    await store.insert_chunks(
        [
            {
                "chunk_id": f"{doc_id}#{index:04d}",
                "doc_id": doc_id,
                "kb_id": kb_id,
                "chunk_index": index,
                "text": text,
                "heading_path": ["架构"],
                "page": index + 1,
                "token_count": 10,
            }
            for index, text in indexes_and_texts
        ]
    )


@pytest.mark.asyncio
async def test_read_pages_through_a_document(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)
    await _add_document(store, "d-1", "架构.md")
    await _seed_chunks(store, "d-1", [(i, f"第{i}片正文") for i in range(5)])
    await store.update_document_status("d-1", "ready", chunk_count=5)

    first = await _read_document_impl(scope_runtime(kb_id="kb-l"), doc_id="d-1", limit=2, store=store)
    assert first["doc"] == {"doc_id": "d-1", "name": "架构.md"}
    assert first["total"] == 5
    assert first["offset"] == 0
    assert first["has_more"] is True
    assert first["message"] == "共 5 片；已返回第 1-2 片。"
    assert first["items"] == [
        {"chunk_id": "d-1#0000", "chunk_index": 0, "heading_path": ["架构"], "page": 1, "text": "第0片正文"},
        {"chunk_id": "d-1#0001", "chunk_index": 1, "heading_path": ["架构"], "page": 2, "text": "第1片正文"},
    ]

    last = await _read_document_impl(scope_runtime(kb_id="kb-l"), doc_id="d-1", offset=4, limit=2, store=store)
    assert [item["chunk_index"] for item in last["items"]] == [4]
    assert last["has_more"] is False
    assert last["message"] == "共 5 片；已返回第 5-5 片。"

    beyond = await _read_document_impl(scope_runtime(kb_id="kb-l"), doc_id="d-1", offset=9, limit=2, store=store)
    assert beyond["items"] == []
    assert beyond["has_more"] is False
    assert beyond["message"] == "共 5 片；offset 超出末尾，未返回切片。"


@pytest.mark.asyncio
async def test_read_limit_is_capped_at_fifty(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)
    await _add_document(store, "d-1", "长文.md")
    await _seed_chunks(store, "d-1", [(i, f"第{i}片") for i in range(60)])
    await store.update_document_status("d-1", "ready", chunk_count=60)

    result = await _read_document_impl(scope_runtime(kb_id="kb-l"), doc_id="d-1", limit=999, store=store)

    assert len(result["items"]) == 50
    assert result["has_more"] is True
    assert result["message"] == "共 60 片；已返回第 1-50 片。"


@pytest.mark.asyncio
async def test_read_chunk_window_centers_and_clamps(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)
    await _add_document(store, "d-1", "带洞.md")
    # Live chunks with a hole: index 3 never existed — positions are live-only.
    await _seed_chunks(store, "d-1", [(index, f"第{index}片") for index in (0, 1, 2, 4, 5)])
    await store.update_document_status("d-1", "ready", chunk_count=5)

    middle = await _read_document_impl(scope_runtime(kb_id="kb-l"), chunk_id="d-1#0004", limit=3, store=store)
    assert [item["chunk_index"] for item in middle["items"]] == [2, 4, 5]
    assert middle["items"][1]["chunk_id"] == "d-1#0004"
    assert middle["message"] == "第 4/共 5 片；已返回以其为中心的窗口 3-5（3 条）。"
    assert middle["total"] == 5
    assert middle["offset"] == 2
    assert middle["has_more"] is False  # the window already reaches the last chunk

    left = await _read_document_impl(scope_runtime(kb_id="kb-l"), chunk_id="d-1#0000", limit=4, store=store)
    assert [item["chunk_index"] for item in left["items"]] == [0, 1, 2, 4]
    assert left["message"] == "第 1/共 5 片；已返回以其为中心的窗口 1-4（4 条）。"

    right = await _read_document_impl(scope_runtime(kb_id="kb-l"), chunk_id="d-1#0005", limit=4, store=store)
    assert [item["chunk_index"] for item in right["items"]] == [1, 2, 4, 5]
    assert right["message"] == "第 5/共 5 片；已返回以其为中心的窗口 2-5（4 条）。"
    assert right["has_more"] is False


@pytest.mark.asyncio
async def test_read_rejects_non_ready_documents(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)
    await _add_document(store, "d-1", "解析中.pdf")
    await store.update_document_status("d-1", "indexing", progress_percent=40)
    await _add_document(store, "d-2", "坏件.pdf")
    await store.update_document_status("d-2", "failed", error="parse boom")

    indexing = await _read_document_impl(scope_runtime(kb_id="kb-l"), doc_id="d-1", store=store)
    assert indexing["items"] == []
    assert indexing["message"] == "该文档尚未就绪（当前状态：indexing），暂时无法读取。"

    failed = await _read_document_impl(scope_runtime(kb_id="kb-l"), doc_id="d-2", store=store)
    assert failed["message"] == "该文档尚未就绪（当前状态：failed），暂时无法读取。"


@pytest.mark.asyncio
async def test_read_rejects_missing_and_foreign_targets(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)
    await _add_document(store, "d-1", "架构.md")
    await _seed_chunks(store, "d-1", [(0, "第0片")])
    await store.update_document_status("d-1", "ready", chunk_count=1)
    await store.create_kb(kb_id="kb-x", owner_id="user-1", name="别的库")
    await _add_document(store, "d-x", "他库.md", kb_id="kb-x")
    await _seed_chunks(store, "d-x", [(0, "他库第0片")], kb_id="kb-x")
    await store.update_document_status("d-x", "ready", chunk_count=1)

    runtime = scope_runtime(kb_id="kb-l")

    missing_doc = await _read_document_impl(runtime, doc_id="nope", store=store)
    assert missing_doc["message"] == "该文档不存在或已被删除。"

    foreign_doc = await _read_document_impl(runtime, doc_id="d-x", store=store)
    assert foreign_doc["message"] == "该文档不属于当前绑定的知识库。"

    # A forged chunk id never earns access by resembling the bound doc's prefix.
    forged = await _read_document_impl(runtime, chunk_id="d-1#9999", store=store)
    assert forged["message"] == "该切片不存在或已被删除。"

    foreign_chunk = await _read_document_impl(runtime, chunk_id="d-x#0000", store=store)
    assert foreign_chunk["message"] == "该切片不属于当前绑定的知识库。"

    neither = await _read_document_impl(runtime, store=store)
    assert neither["message"] == "请提供 doc_id（分页读整篇）或 chunk_id（读该片附近的窗口）。"

    await _add_document(store, "d-empty", "空文.md")
    await store.update_document_status("d-empty", "ready", chunk_count=0)
    empty = await _read_document_impl(runtime, doc_id="d-empty", store=store)
    assert empty["items"] == []
    assert empty["message"] == "该文档当前没有切片。"


@pytest.mark.asyncio
async def test_read_truncates_oversized_chunks(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)
    await _add_document(store, "d-1", "大表.md")
    await _seed_chunks(store, "d-1", [(0, "长" * 2100), (1, "短正文")])
    await store.update_document_status("d-1", "ready", chunk_count=2)

    result = await _read_document_impl(scope_runtime(kb_id="kb-l"), doc_id="d-1", store=store)

    assert result["items"][0]["truncated"] is True
    assert len(result["items"][0]["text"]) == 2000
    assert "truncated" not in result["items"][1]
    assert result["items"][1]["text"] == "短正文"


@pytest.mark.asyncio
async def test_read_fails_closed_on_binding_and_ownership(session_factory) -> None:
    store = KnowledgeStore(session_factory)
    await _seed_kb(store)

    unbound = await _read_document_impl(scope_runtime(kb_id=None), doc_id="d-1", store=store)
    assert unbound == {"doc": None, "items": [], "message": NO_KB_GUIDANCE}

    denied = await _read_document_impl(scope_runtime(kb_id="kb-l", user_id="user-2"), doc_id="d-1", store=store)
    assert denied["message"] == ACCESS_DENIED_MESSAGE


def test_registration_surface_imports_without_the_extension() -> None:
    """The host registration surface must expose both tools by name."""
    from deerflow.tools.builtins.knowledge_documents_tool import list_knowledge_documents, read_knowledge_document

    assert list_knowledge_documents.name == "list_knowledge_documents"
    assert read_knowledge_document.name == "read_knowledge_document"
