from pymilvus import Collection
from sentence_transformers import SentenceTransformer
from pymilvus import connections
import re
from typing import List, Dict, Tuple
from collections import defaultdict

# 配置参数
query_text = "Zookeeper连接请求数突增"
collection_name = "demo_connection"
DENSE_WEIGHT = 0.6  # 密集向量检索权重
SPARSE_WEIGHT = 0.4  # 稀疏/关键词检索权重
TOP_K = 10  # 初始检索返回数量
FINAL_K = 3  # 重排序后最终返回数量
USE_RERANKER = True  # 是否使用重排序模型

# 初始化模型
print("正在加载模型...")
embedding_model = SentenceTransformer('BAAI/bge-m3')  # 该模型输出的维度是1024，支持密集和稀疏向量

# 加载重排序模型（可选）
reranker_model = None
if USE_RERANKER:
    try:
        from sentence_transformers import CrossEncoder
        reranker_model = CrossEncoder('BAAI/bge-reranker-base')
        print("重排序模型加载成功")
    except Exception as e:
        print(f"重排序模型加载失败，将跳过重排序: {e}")
        USE_RERANKER = False

# 连接 Milvus
connections.connect(host="127.0.0.1", port="19530")
collection = Collection(collection_name)

# 1. 检查并创建索引（如果不存在）
if not collection.has_index():
    print("索引不存在，正在创建索引...")
    index_params = {
        'metric_type': 'L2',  # 使用 L2 距离（欧氏距离）
        'index_type': 'IVF_FLAT',  # 索引类型
        'params': {'nlist': 1024}  # 聚类中心数量，可根据数据量调整
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print("索引创建成功")

# 2. 加载到内存 (查询前必须执行)
collection.load()

# ========== 混合检索：密集向量检索 + 关键词检索 ==========
print(f"\n开始混合检索，查询: {query_text}")

# 步骤1: 密集向量检索（Dense Retrieval）
print("1. 执行密集向量检索...")
query_dense_vector = embedding_model.encode(query_text, normalize_embeddings=True).tolist()
dense_results = collection.search(
    data=[query_dense_vector],
    anns_field="vector",
    param={'nprobe': 128},
    limit=TOP_K,
    output_fields=["text", "metadata"]
)

# 提取密集检索结果
dense_hits = {}
for hits in dense_results:
    for hit in hits:
        try:
            # 尝试多种方式获取字段
            if hasattr(hit.entity, 'get'):
                text = hit.entity.get('text', '')
                metadata = hit.entity.get('metadata', {})
            else:
                # 使用字典方式访问
                text = hit.entity.get('text', '') if hasattr(hit.entity, 'get') else hit.entity['text']
                try:
                    metadata = hit.entity['metadata']
                except (KeyError, TypeError):
                    metadata = {}
        except (AttributeError, TypeError, KeyError):
            text = ''
            metadata = {}
        
        # 将距离转换为相似度分数（L2距离越小，相似度越高）
        # 使用归一化的相似度：1 / (1 + distance)
        similarity_score = 1 / (1 + hit.distance) if hit.distance >= 0 else 1.0
        dense_hits[hit.id] = {
            'id': hit.id,
            'text': text,
            'metadata': metadata,
            'dense_score': similarity_score,
            'distance': hit.distance
        }

print(f"密集检索返回 {len(dense_hits)} 条结果")

# 步骤2: 关键词检索（作为稀疏检索的替代方案）
print("2. 执行关键词检索...")
def extract_keywords(text: str) -> List[str]:
    """提取关键词（简单实现，可以替换为更复杂的分词）"""
    # 移除标点符号，提取中英文词汇
    words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', text.lower())
    return words

def keyword_match_score(query_keywords: List[str], text: str) -> float:
    """计算关键词匹配分数"""
    text_lower = text.lower()
    matched_count = sum(1 for keyword in query_keywords if keyword in text_lower)
    if not query_keywords:
        return 0.0
    return matched_count / len(query_keywords)

query_keywords = extract_keywords(query_text)
print(f"提取的关键词: {query_keywords}")

# 获取所有文档进行关键词匹配（实际场景中可以使用倒排索引优化）
# 这里为了演示，我们从密集检索结果中进行关键词匹配
sparse_hits = {}
for hit_id, hit_data in dense_hits.items():
    keyword_score = keyword_match_score(query_keywords, hit_data['text'])
    sparse_hits[hit_id] = {
        'id': hit_id,
        'text': hit_data['text'],
        'metadata': hit_data['metadata'],
        'sparse_score': keyword_score
    }

# 如果密集检索结果不足，可以扩大检索范围
if len(dense_hits) < TOP_K:
    # 扩大检索范围获取更多候选
    extended_results = collection.search(
        data=[query_dense_vector],
        anns_field="vector",
        param={'nprobe': 128},
        limit=TOP_K * 2,
        output_fields=["text", "metadata"]
    )
    for hits in extended_results:
        for hit in hits:
            if hit.id not in sparse_hits:
                try:
                    if hasattr(hit.entity, 'get'):
                        text = hit.entity.get('text', '')
                        metadata = hit.entity.get('metadata', {})
                    else:
                        # 使用字典方式访问
                        text = hit.entity['text']
                        try:
                            metadata = hit.entity['metadata']
                        except (KeyError, TypeError):
                            metadata = {}
                except (AttributeError, TypeError, KeyError):
                    text = ''
                    metadata = {}
                keyword_score = keyword_match_score(query_keywords, text)
                if keyword_score > 0:  # 只保留有匹配的
                    sparse_hits[hit.id] = {
                        'id': hit.id,
                        'text': text,
                        'metadata': metadata,
                        'sparse_score': keyword_score
                    }

print(f"关键词检索返回 {len(sparse_hits)} 条结果")

# 步骤3: 合并密集和稀疏检索结果
print("3. 合并检索结果...")
merged_results = defaultdict(lambda: {
    'id': None,
    'text': '',
    'metadata': {},
    'dense_score': 0.0,
    'sparse_score': 0.0,
    'hybrid_score': 0.0
})

# 合并密集检索结果
for hit_id, hit_data in dense_hits.items():
    merged_results[hit_id].update({
        'id': hit_id,
        'text': hit_data['text'],
        'metadata': hit_data['metadata'],
        'dense_score': hit_data['dense_score']
    })

# 合并稀疏检索结果
for hit_id, hit_data in sparse_hits.items():
    if hit_id in merged_results:
        merged_results[hit_id]['sparse_score'] = hit_data['sparse_score']
    else:
        merged_results[hit_id].update({
            'id': hit_id,
            'text': hit_data['text'],
            'metadata': hit_data['metadata'],
            'sparse_score': hit_data['sparse_score']
        })

# 归一化分数并计算混合分数
if merged_results:
    max_dense = max(r['dense_score'] for r in merged_results.values() if r['dense_score'] > 0)
    max_sparse = max(r['sparse_score'] for r in merged_results.values() if r['sparse_score'] > 0)
    
    for hit_id in merged_results:
        result = merged_results[hit_id]
        # 归一化
        norm_dense = result['dense_score'] / max_dense if max_dense > 0 else 0
        norm_sparse = result['sparse_score'] / max_sparse if max_sparse > 0 else 0
        # 加权混合
        result['hybrid_score'] = DENSE_WEIGHT * norm_dense + SPARSE_WEIGHT * norm_sparse

# 按混合分数排序
candidates = sorted(merged_results.values(), key=lambda x: x['hybrid_score'], reverse=True)[:TOP_K]
print(f"混合检索后得到 {len(candidates)} 条候选结果")

# ========== 重排序（Reranking）==========
if USE_RERANKER and reranker_model and candidates:
    print("4. 执行重排序...")
    # 准备重排序的输入对
    pairs = [[query_text, candidate['text']] for candidate in candidates]
    
    # 执行重排序
    rerank_scores = reranker_model.predict(pairs)
    
    # 更新分数
    for i, candidate in enumerate(candidates):
        candidate['rerank_score'] = float(rerank_scores[i])
        # 可以结合混合分数和重排序分数
        candidate['final_score'] = 0.7 * candidate['rerank_score'] + 0.3 * candidate['hybrid_score']
    
    # 按最终分数重新排序
    candidates = sorted(candidates, key=lambda x: x.get('final_score', x['hybrid_score']), reverse=True)
    print("重排序完成")
else:
    print("4. 跳过重排序")
    for candidate in candidates:
        candidate['final_score'] = candidate['hybrid_score']

# 选择最终结果
results = candidates[:FINAL_K]

# ========== 输出最终结果 ==========
print(f"\n最终返回 {len(results)} 条结果:\n")
for i, result in enumerate(results, 1):
    print(f"【结果 {i}】")
    print(f"  ID: {result['id']}")
    print(f"  混合分数: {result['hybrid_score']:.4f} (密集: {result['dense_score']:.4f}, 稀疏: {result['sparse_score']:.4f})")
    if 'rerank_score' in result:
        print(f"  重排序分数: {result['rerank_score']:.4f}")
        print(f"  最终分数: {result['final_score']:.4f}")
    text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
    print(f"  内容: {text_preview}")
    print()