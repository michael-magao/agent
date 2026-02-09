from langchain.tools import tool
from langchain_milvus import Milvus
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from typing import List
import json

collection_name = "demo_connection"
model = SentenceTransformer('BAAI/bge-m3') # 该模型输出的维度是1024

# 创建一个包装类，将 SentenceTransformer 适配为 LangChain 的 Embeddings 接口
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

# 创建符合 LangChain 接口的 embedding 对象
embeddings = SentenceTransformerEmbeddings(model)

@tool
def search_sop_knowledge(query: str):
    """🔴 必须首先调用：检索 Milvus 知识库中的 SOP 手册和排障指南。收到任何告警后，第一步必须调用此工具从知识库中检索相关的标准操作流程和历史案例。参数 query 应该从告警信息中提取关键词，如服务名、错误类型、指标名等。"""
    print("使用 Milvus 向量数据库进行相似度搜索...")
    # 直接通过 LangChain 的 Milvus 类连接
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"host": "127.0.0.1", "port": "19530"}
    )

    # 现在可以使用这个方法了
    docs = vector_store.similarity_search(query, k=2)
    return "\n".join([d.page_content for d in docs])

@tool
def get_cluster_info(cluster: str):
    """获取etcd/zk集群的基本信息，如节点列表、状态等。"""
    print("使用集群管理 API 获取集群信息...")
    # 模拟调用集群管理 API
    return f"Cluster {cluster} has 3 nodes: 127.0.0.1:2181,127.0.0.2:2181,127.0.0.3:2181, all healthy."

@tool
def get_log_summary(log_type: str, time_range: str):
    """获取指定时间范围内的日志摘要，如错误日志、访问日志等。"""
    print("使用日志管理系统 API 获取日志摘要...")
    # 模拟调用日志管理系统 API
    return f"Log summary for {log_type} from {time_range}: 5 errors, 20 warnings."

@tool
def get_cluster_metrics(query: str):
    """获取集群的实时监控指标，如 连接数, 请求数，读写延迟。"""
    print("使用 Prometheus API 获取集群监控指标...")
    # 模拟调用 Prometheus API
    return f"System metrics for {query}: CPU usage 75%, Disk usage 60%."

@tool
def get_system_metrics(metric_name: str):
    """获取系统的实时监控指标，如 cpu_usage, disk_usage。"""
    print("使用 Prometheus API 获取系统监控指标...")
    # 模拟调用 Prometheus API
    return f"{metric_name} is currently at 85%"

@tool
def run_ssh_command(action_input: str) -> str:
    """
    执行修复命令：在指定的远程服务器上执行诊断或修复命令。在完成故障分析后，应该尝试调用此工具执行修复操作。
    
    参数
    - action_input 应该是一个 JSON 字符串，格式为：{"host": "127.0.0.1", "command": "ps aux | grep zookeeper"}
    - host: 目标服务器地址，从告警信息中获取的 target_host，例如 "127.0.0.1"
    - command: 要执行的 shell 命令，例如 "ps aux | grep zookeeper" 或 "systemctl restart zookeeper"
    
    执行前应在 Thought 中评估命令的风险。
    """
    # 解析 JSON 字符串
    try:
        if isinstance(action_input, str):
            # 尝试解析 JSON
            params = json.loads(action_input)
        else:
            # 如果已经是字典，直接使用
            params = action_input
    except (json.JSONDecodeError, TypeError, AttributeError):
        return f"错误：无法解析输入参数。请确保输入格式为 JSON 字符串：{{\"host\": \"IP地址\", \"command\": \"命令\"}}。收到的输入：{action_input}"
    
    if not isinstance(params, dict):
        return f"错误：输入参数必须是包含 host 和 command 的 JSON 对象。收到的输入：{action_input}"
    
    host = params.get("host", "")
    command = params.get("command", "")
    
    if not host or not command:
        return f"错误：缺少必要参数。需要 host 和 command 两个参数。收到的参数：{params}"
    
    print(f"使用 SSH 在远程服务器 {host} 上执行命令: {command}")
    # 这里执行真正的 SSH 调用
    return f"Execution result of {command} on {host}: [Success] Command executed successfully."