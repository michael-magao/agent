from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_milvus import Milvus

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

def setup_knowledge_base():
    """设置知识库（示例）"""

    # 这里可以加载文档
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"host": "127.0.0.1", "port": "19530"}
    )

    return vector_store