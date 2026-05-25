from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings


class EmptyKnowledgeBase:
    """RAG 后端不可用时的降级实现。"""

    def __init__(self, reason: str = ""):
        self.reason = reason

    def similarity_search(self, query: str, k: int = 1) -> list:
        return []


class SentenceTransformerEmbeddings(Embeddings):
    """将 SentenceTransformer 包装成 LangChain Embeddings 接口。"""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        local_only = os.getenv("AGENT_EMBEDDING_LOCAL_ONLY", "true").lower() not in {"0", "false", "no"}
        self.model = SentenceTransformer(model_name, local_files_only=local_only)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


@lru_cache(maxsize=2)
def _embeddings(model_name: str) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name)


@lru_cache(maxsize=8)
def setup_knowledge_base(
    collection_name: str | None = None,
    host: str | None = None,
    port: str | None = None,
    embedding_model: str | None = None,
):
    """创建知识库连接；缺省或不可用时返回空知识库，避免 Agent 启动即失败。"""
    collection = collection_name or os.getenv("AGENT_RAG_COLLECTION", "demo_connection")
    milvus_host = host or os.getenv("AGENT_MILVUS_HOST", "127.0.0.1")
    milvus_port = port or os.getenv("AGENT_MILVUS_PORT", "19530")
    model_name = embedding_model or os.getenv("AGENT_EMBEDDING_MODEL", "BAAI/bge-m3")

    try:
        from langchain_milvus import Milvus

        return Milvus(
            embedding_function=_embeddings(model_name),
            collection_name=collection,
            connection_args={"host": milvus_host, "port": milvus_port},
        )
    except Exception as exc:
        return EmptyKnowledgeBase(str(exc))
