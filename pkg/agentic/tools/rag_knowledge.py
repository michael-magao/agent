from typing import List

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from sentence_transformers import SentenceTransformer

collection_name = "demo_connection"
model = SentenceTransformer('BAAI/bge-m3') # 该模型输出的维度是1024
class SentenceTransformerEmbeddings(Embeddings):
    """将 SentenceTransformer 包装成 LangChain Embeddings 接口"""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

embeddings = SentenceTransformerEmbeddings(model)

def search_sop_knowledge(query: str):
    """🔴 必须首先调用：检索 Milvus 知识库中的 SOP 手册和排障指南。收到任何告警后，第一步必须调用此工具从知识库中检索相关的标准操作流程和历史案例。参数 query 应该从告警信息中提取关键词，如服务名、错误类型、指标名等。"""
    print("使用 Milvus 向量数据库进行相似度搜索...", query)
    # 直接通过 LangChain 的 Milvus 类连接
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"host": "127.0.0.1", "port": "19530"}
    )

    # 现在可以使用这个方法了
    docs = vector_store.similarity_search(query, k=1) # k表示返回最相似的前2条结果

    # todo 还需要支持rerank

    # todo 还需要混合检索
    # https://help.aliyun.com/zh/milvus/use-cases/full-text-retrieval-by-milvus-bm25-algorithm-and-application-of-hybrid-retrieval-to-rag-system

    # todo 还需要支持返回来源信息，比如文档ID，方便后续追踪

    return "\n".join([d.page_content for d in docs])