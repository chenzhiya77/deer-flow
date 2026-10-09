"""Document-level read-only tools (spec §2.1).

``list_knowledge_documents`` enumerates the bound library's documents with
an honest per-status count; ``read_knowledge_document`` reads a document's
chunks — paged by ``doc_id`` or as a window centered on a known
``chunk_id`` — so the model can follow a search hit and answer corpus
questions instead of relying on recall luck. Both read the business store
only (no Qdrant, no embedding) and fail closed on a missing binding or a
foreign owner.

Imports stay function-level on purpose: this module lives in the host and
its registration surface must import even when the knowledge extension is
not installed (the tools only resolve their dependencies when called).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain.tools import tool

from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 50
_MAX_ITEM_CHARS = 2000


def _doc_meta(doc: dict) -> dict:
    return {"doc_id": doc["id"], "name": doc["name"]}


async def _ready_document(store: Any, doc_id: str, kb_id: str) -> tuple[dict | None, dict | None]:
    """Return ``(doc, None)`` for a readable document, else ``(None, refusal)``."""
    doc = await store.get_document(doc_id)
    if doc is None:
        return None, {"doc": None, "items": [], "message": "该文档不存在或已被删除。"}
    if doc["kb_id"] != kb_id:
        return None, {"doc": None, "items": [], "message": "该文档不属于当前绑定的知识库。"}
    if doc["status"] != "ready":
        return None, {"doc": None, "items": [], "message": f"该文档尚未就绪（当前状态：{doc['status']}），暂时无法读取。"}
    return doc, None


def _shape_items(rows: list[dict]) -> list[dict]:
    items = []
    for row in rows:
        text = row["text"]
        item = {
            "chunk_id": row["chunk_id"],
            "chunk_index": row["chunk_index"],
            "heading_path": row["heading_path"],
            "page": row["page"],
            "text": text,
        }
        if len(text) > _MAX_ITEM_CHARS:
            item["text"] = text[:_MAX_ITEM_CHARS]
            item["truncated"] = True
        items.append(item)
    return items


async def _list_documents_impl(runtime: Any, *, store: Any = None) -> dict:
    """Core implementation — testable without the @tool wrapper."""
    from deerflow_knowledge.access import ACCESS_DENIED_MESSAGE, NO_KB_GUIDANCE, can_access, resolve_kb_scope
    from deerflow_knowledge.store import get_knowledge_store

    kb_id, user_id, refusal = resolve_kb_scope(runtime)
    if kb_id is None:
        return {"documents": [], "message": refusal or NO_KB_GUIDANCE}
    store = store or get_knowledge_store()
    if not await can_access(store, user_id, kb_id):
        return {"documents": [], "message": ACCESS_DENIED_MESSAGE}

    rows = await store.list_documents(kb_id)
    documents = [
        {
            "doc_id": row["id"],
            "name": row["name"],
            "status": row["status"],
            "chunk_count": row["chunk_count"],
        }
        for row in rows
    ]
    if not documents:
        return {"documents": [], "message": "当前知识库中还没有文档。"}

    ready = sum(1 for document in documents if document["status"] == "ready")
    failed = sum(1 for document in documents if document["status"] == "failed")
    in_progress = len(documents) - ready - failed
    counts = [label for count, label in ((ready, f"就绪 {ready}"), (in_progress, f"处理中 {in_progress}"), (failed, f"失败 {failed}")) if count]
    return {"documents": documents, "message": f"共 {len(documents)} 篇文档（{' · '.join(counts)}）。"}


async def _read_document_impl(
    runtime: Any,
    *,
    doc_id: str | None = None,
    chunk_id: str | None = None,
    offset: int = 0,
    limit: int = _DEFAULT_LIMIT,
    store: Any = None,
) -> dict:
    """Core implementation — testable without the @tool wrapper.

    ``chunk_id`` takes precedence: the server resolves the chunk row (never
    the id string) to its live position and returns the window around it,
    with the edges clamped and 第 K/共 N 片 stated in the message.
    """
    from deerflow_knowledge.access import ACCESS_DENIED_MESSAGE, NO_KB_GUIDANCE, can_access, resolve_kb_scope
    from deerflow_knowledge.store import get_knowledge_store

    kb_id, user_id, refusal = resolve_kb_scope(runtime)
    if kb_id is None:
        return {"doc": None, "items": [], "message": refusal or NO_KB_GUIDANCE}
    store = store or get_knowledge_store()
    if not await can_access(store, user_id, kb_id):
        return {"doc": None, "items": [], "message": ACCESS_DENIED_MESSAGE}

    width = max(1, min(int(limit), _MAX_LIMIT))
    offset = max(0, int(offset))

    if chunk_id:
        row = await store.get_chunk(chunk_id)
        if row is None:
            return {"doc": None, "items": [], "message": "该切片不存在或已被删除。"}
        if row["kb_id"] != kb_id:
            return {"doc": None, "items": [], "message": "该切片不属于当前绑定的知识库。"}
        doc, refusal_payload = await _ready_document(store, row["doc_id"], kb_id)
        if doc is None:
            return refusal_payload
        total = await store.count_chunks(row["doc_id"])
        if total == 0:
            return {"doc": _doc_meta(doc), "items": [], "total": 0, "offset": 0, "has_more": False, "message": "该文档当前没有切片。"}
        position = (await store.chunk_positions(row["doc_id"], {row["chunk_index"]})).get(row["chunk_index"])
        if position is None:
            return {"doc": None, "items": [], "message": "该切片不存在或已被删除。"}
        window = min(width, total)
        start = max(1, min(position - (window - 1) // 2, total - window + 1))
        end = start + window - 1
        items = _shape_items(await store.list_chunks(row["doc_id"], offset=start - 1, limit=window))
        return {
            "doc": _doc_meta(doc),
            "items": items,
            "total": total,
            "offset": start - 1,
            "has_more": end < total,
            "message": f"第 {position}/共 {total} 片；已返回以其为中心的窗口 {start}-{end}（{len(items)} 条）。",
        }

    if doc_id:
        doc, refusal_payload = await _ready_document(store, doc_id, kb_id)
        if doc is None:
            return refusal_payload
        total = await store.count_chunks(doc_id)
        items = _shape_items(await store.list_chunks(doc_id, offset=offset, limit=width)) if total else []
        if not items:
            message = "该文档当前没有切片。" if total == 0 else f"共 {total} 片；offset 超出末尾，未返回切片。"
            return {"doc": _doc_meta(doc), "items": [], "total": total, "offset": offset, "has_more": False, "message": message}
        return {
            "doc": _doc_meta(doc),
            "items": items,
            "total": total,
            "offset": offset,
            "has_more": offset + len(items) < total,
            "message": f"共 {total} 片；已返回第 {offset + 1}-{offset + len(items)} 片。",
        }

    return {"doc": None, "items": [], "message": "请提供 doc_id（分页读整篇）或 chunk_id（读该片附近的窗口）。"}


@tool(parse_docstring=True)
async def list_knowledge_documents(runtime: Runtime) -> dict:
    """List every document in the knowledge base bound to this conversation, with an honest status count.

    Use this tool when:
    - The question asks what the knowledge base contains — which documents exist (corpus enumeration)
    - You need a document's doc_id to address the same document in a follow-up read

    Skip this tool when:
    - The question needs factual content from the documents — retrieve with knowledge_search instead

    Returns every document as doc_id/name/status/chunk_count plus a status summary (ready / in progress / failed). Missing knowledge-base binding returns guidance instead of listing.

    Args:
        runtime: Tool runtime carrying the admitted knowledge scope in its context.
    """
    return await _list_documents_impl(runtime)


@tool(parse_docstring=True)
async def read_knowledge_document(
    runtime: Runtime,
    doc_id: Annotated[str | None, "Document id (from list_knowledge_documents or a search result) for a paged read."] = None,
    chunk_id: Annotated[str | None, "Chunk id (from a search result); reads the window around that chunk and takes precedence over doc_id."] = None,
    offset: Annotated[int, "Page offset in chunks for the doc_id mode (default 0)."] = 0,
    limit: Annotated[int, "Page/window width in chunks (default 20, capped at 50)."] = _DEFAULT_LIMIT,
) -> dict:
    """Read chunks of one document in the knowledge base bound to this conversation.

    Use this tool when:
    - You need the context around a known chunk (a search hit's chunk_id) — the slices before and after it
    - You need to page through one specific document by doc_id (what else is in this document)

    Skip this tool when:
    - The question needs retrieval across the whole library — use knowledge_search

    Two addressing modes: chunk_id reads the window centered on that chunk
    (edges clamped; the message states 第 K/共 N 片); doc_id + offset/limit
    pages through the document. Oversized chunk text is truncated at 2000
    characters and flagged truncated=true. Missing binding, a foreign owner,
    a document that is not ready yet, or a chunk outside the bound library
    all return an explicit refusal instead of an empty result.

    Args:
        runtime: Tool runtime carrying the admitted knowledge scope in its context.
        doc_id: Document id for the paged mode.
        chunk_id: Chunk id for the centered-window mode (takes precedence).
        offset: Page offset for the doc_id mode.
        limit: Page/window width (default 20, capped at 50).
    """
    return await _read_document_impl(runtime, doc_id=doc_id, chunk_id=chunk_id, offset=offset, limit=limit)
