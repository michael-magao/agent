from __future__ import annotations

from pkg.agentic.memory.rag import setup_knowledge_base


def search_sop_knowledge(query: str) -> str:
    """检索 Milvus 知识库中的 SOP 手册和排障指南。"""
    if not query or not str(query).strip():
        return "SOP 检索关键词不能为空。"

    try:
        vector_store = setup_knowledge_base()
        docs = vector_store.similarity_search(str(query), k=1)
    except Exception as exc:
        return f"SOP 知识库暂不可用: {exc}"

    if not docs:
        return "未检索到相关 SOP。"
    return "\n".join([getattr(d, "page_content", str(d)) for d in docs])
