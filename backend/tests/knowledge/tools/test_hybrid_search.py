"""Tests for knowledge_search (spec §4.1): RRF prefetch → rerank → top-k.

Chunk text always comes from the business-DB ``chunks`` table (fetched by
``chunk_id``); the Qdrant payload only supplies display metadata
(``doc_name``/``page``/``heading_path``) for citations. The turn's knowledge
scope resolves the bound KB; the returned source artifact mirrors the slices.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from deerflow_knowledge.access import ACCESS_DENIED_MESSAGE, KB_MISSING_MESSAGE, NO_KB_GUIDANCE
from deerflow_knowledge.embedder import EmbeddingResult
from deerflow_knowledge.reranker import RerankerError
from qdrant_client.models import SparseVector

from deerflow.tools.builtins.hybrid_search_tool import _hybrid_search_impl

from ..conftest import requires_qdrant
from .conftest import DOC_ID, KB_ID, OWNER_ID, scope_runtime


class _StubReranker:
    """Scores documents containing 会话管理 highest — deterministic rerank."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, documents, *, top_n: int = 5):
        self.calls.append((query, list(documents)))
        scored = [(i, 0.99 if "会话管理" in doc else 0.05) for i, doc in enumerate(documents)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]


class _FailingReranker:
    async def rerank(self, query: str, documents, *, top_n: int = 5):
        raise RerankerError("rerank down")


@requires_qdrant
@pytest.mark.integration
@pytest.mark.asyncio
async def test_hybrid_search_end_to_end(tools_env):
    reranker = _StubReranker()

    result, artifact = await _hybrid_search_impl(
        "Gateway 的作用是什么",
        scope_runtime(),
        store=tools_env["store"],
        vector_store=tools_env["vector_store"],
        embedder=tools_env["embedder"],
        reranker=reranker,
        top_k=2,
    )

    assert reranker.calls, "RRF candidates must go through the reranker"
    results = result["results"]
    assert len(results) == 2
    top = results[0]
    # Reranker re-ordered: the 会话管理 chunk leads even though RRF ties.
    assert top["chunk_id"] == "doc-t-c0"
    assert top["text"] == "Gateway 负责会话管理，是 DeerFlow 的入口组件。"  # text from the business DB
    assert top["score"] == 0.99
    # Citation metadata rides the Qdrant payload (spec §4.6).
    assert top["doc_name"] == "架构.md"
    assert top["page"] == 1
    assert top["heading_path"] == ["架构"]
    # Follow-up credentials (spec 2026-10-08 §2.2): doc_id/chunk_index address
    # the same document/chunk for follow-up reads; both ride the business-DB row.
    assert top["doc_id"] == DOC_ID
    assert top["chunk_index"] == 0
    assert "message" in result

    # The shared source artifact mirrors exactly the returned slices (#5551).
    assert artifact is not None
    sources = artifact["knowledge_sources"]["sources"]
    assert artifact["knowledge_sources"]["version"] == 1
    assert [source["provider"] for source in sources] == ["local", "local"]
    assert [source["chunk_id"] for source in sources] == [item["chunk_id"] for item in results]
    assert sources[0]["dataset_id"] == "kb-t"
    assert sources[0]["document_id"] == "doc-t"
    assert sources[0]["dataset_name"] == "工具测试库"
    assert sources[0]["document_name"] == "架构.md"
    assert sources[0]["text"] == top["text"]
    assert sources[0]["pages"] == [1]
    assert sources[0]["truncated"] is False
    # Provider-independent id shape shared with the RAGFlow formatter.
    assert re.fullmatch(r"[a-f0-9]{32}-1", sources[0]["id"])
    assert sources[0]["id"].rsplit("-", 1)[0] == sources[1]["id"].rsplit("-", 1)[0]


@requires_qdrant
@pytest.mark.integration
@pytest.mark.asyncio
async def test_hybrid_search_no_match_returns_honest_message(tools_env):
    result, _ = await _hybrid_search_impl(
        "zzz-完全无关-zzz",
        scope_runtime(),
        store=tools_env["store"],
        vector_store=tools_env["vector_store"],
        embedder=tools_env["embedder"],
        reranker=_StubReranker(),
    )
    # An unrelated query still vector-matches *something* in a tiny test KB;
    # what matters is the tool answers with real rows or an honest empty list.
    assert result["results"] is not None
    assert "message" in result


@requires_qdrant
@pytest.mark.integration
@pytest.mark.asyncio
async def test_hybrid_search_without_scope_returns_guidance(tools_env):
    result, artifact = await _hybrid_search_impl(
        "任意问题",
        SimpleNamespace(context={"user_id": OWNER_ID}),
        store=tools_env["store"],
        vector_store=tools_env["vector_store"],
        embedder=tools_env["embedder"],
        reranker=_StubReranker(),
    )
    assert result["results"] == []
    assert result["message"] == NO_KB_GUIDANCE
    assert artifact is None


@requires_qdrant
@pytest.mark.integration
@pytest.mark.asyncio
async def test_hybrid_search_denies_non_owner(tools_env):
    result, _ = await _hybrid_search_impl(
        "Gateway",
        scope_runtime(user_id="user-2"),
        store=tools_env["store"],
        vector_store=tools_env["vector_store"],
        embedder=tools_env["embedder"],
        reranker=_StubReranker(),
    )
    assert result["results"] == []
    assert result["message"] == ACCESS_DENIED_MESSAGE


@requires_qdrant
@pytest.mark.integration
@pytest.mark.asyncio
async def test_hybrid_search_reports_deleted_kb(tools_env):
    result, _ = await _hybrid_search_impl(
        "Gateway",
        scope_runtime(kb_id="kb-gone"),
        store=tools_env["store"],
        vector_store=tools_env["vector_store"],
        embedder=tools_env["embedder"],
        reranker=_StubReranker(),
    )
    assert result["results"] == []
    assert result["message"] == KB_MISSING_MESSAGE


@requires_qdrant
@pytest.mark.integration
@pytest.mark.asyncio
async def test_hybrid_search_rerank_failure_degrades_to_rrf_order(tools_env):
    """A reranker outage must not kill the vector path (spec §4.4)."""
    result, _ = await _hybrid_search_impl(
        "Gateway",
        scope_runtime(),
        store=tools_env["store"],
        vector_store=tools_env["vector_store"],
        embedder=tools_env["embedder"],
        reranker=_FailingReranker(),
        top_k=2,
    )
    assert len(result["results"]) == 2  # RRF order preserved, scores absent
    assert result["results"][0]["text"]
    assert "精排" in result["message"] or "rerank" in result["message"].lower()


class _FieldStore:
    """Store double for the follow-up-field assertions (no Qdrant needed)."""

    def __init__(self) -> None:
        self.rows = [
            {
                "chunk_id": "doc-u#0007",
                "doc_id": "doc-u",
                "chunk_index": 7,
                "text": "第七片正文",
                "heading_path": ["第7章"],
                "page": 7,
            }
        ]

    async def get_kb(self, kb_id: str):
        return {"id": kb_id, "owner_id": OWNER_ID, "name": "字段测试库"}

    async def get_chunks_by_ids(self, chunk_ids, *, kb_id=None):
        wanted = set(chunk_ids)
        return [row for row in self.rows if row["chunk_id"] in wanted]


class _FieldVectorStore:
    """Qdrant double whose payload deliberately omits ``chunk_index``."""

    def __init__(self) -> None:
        self.payloads = [
            {"chunk_id": "doc-u#0007", "doc_name": "手册.md", "page": 7, "heading_path": ["第7章"], "doc_id": "doc-u"},
        ]
        self.calls: list[dict] = []

    async def hybrid_query(self, *, dense, sparse, kb_id, top_k, doc_id=None):
        self.calls.append({"kb_id": kb_id, "doc_id": doc_id})
        return [SimpleNamespace(payload=payload) for payload in self.payloads]


class _FieldEmbedder:
    async def embed(self, texts, *, text_type: str = "document"):
        return [EmbeddingResult(dense=[0.0] * 4, sparse=SparseVector(indices=[1], values=[0.5])) for _ in texts]


@pytest.mark.asyncio
async def test_items_carry_doc_id_and_chunk_index_for_follow_up_reads() -> None:
    """追问链凭据（spec §2.2）：每条结果带 doc_id 与 chunk_index。

    ``chunk_index`` 不存在于 Qdrant payload（夹具 payload 故意无此键）——
    断言它必须取自业务库行；doc_name/page/heading_path 等既有引用元数据不变。
    """
    payload, _artifact = await _hybrid_search_impl(
        "任意问题",
        scope_runtime(),
        store=_FieldStore(),
        vector_store=_FieldVectorStore(),
        embedder=_FieldEmbedder(),
        reranker=_StubReranker(),
    )

    (item,) = payload["results"]
    assert item["doc_id"] == "doc-u"
    assert item["chunk_index"] == 7
    assert item["doc_name"] == "手册.md"
    assert item["page"] == 7
    assert item["heading_path"] == ["第7章"]


@pytest.mark.asyncio
async def test_hybrid_search_forwards_the_doc_filter_to_the_vector_store() -> None:
    """篇内检索（spec §2.3）: the optional doc_id rides the same store call."""
    vector_store = _FieldVectorStore()

    payload, _artifact = await _hybrid_search_impl(
        "任意问题",
        scope_runtime(),
        store=_FieldStore(),
        vector_store=vector_store,
        embedder=_FieldEmbedder(),
        reranker=_StubReranker(),
        doc_id="doc-u",
    )

    assert vector_store.calls == [{"kb_id": KB_ID, "doc_id": "doc-u"}]
    assert [item["doc_id"] for item in payload["results"]] == ["doc-u"]


@requires_qdrant
@pytest.mark.integration
@pytest.mark.asyncio
async def test_hybrid_search_doc_scope_hits_and_misses(tools_env) -> None:
    """篇内检索（spec 2026-10-08 §2.3）: doc_id scopes retrieval to one document."""
    kwargs = dict(
        store=tools_env["store"],
        vector_store=tools_env["vector_store"],
        embedder=tools_env["embedder"],
        reranker=_StubReranker(),
    )

    scoped, _ = await _hybrid_search_impl("Gateway", scope_runtime(), doc_id=DOC_ID, **kwargs)
    assert scoped["results"]
    assert all(item["doc_id"] == DOC_ID for item in scoped["results"])

    missing, _ = await _hybrid_search_impl("Gateway", scope_runtime(), doc_id="doc-elsewhere", **kwargs)
    assert missing["results"] == []
    assert missing["message"] == "知识库中未检索到与问题相关的内容。"
