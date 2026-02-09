from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import Collection, utility, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
from pymilvus import connections




# 2、chunk splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # todo 根据实际情况调整
    chunk_overlap=100, # 必须有重叠，防止语义在切割点丢失 # todo 根据实际情况调整
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)
chunks = text_splitter.split_documents(documents)

# 3、TODO 增强语义
# 在每个 Chunk 前面强行加上父级标题。
# 切片原内容： “执行命令：systemctl restart nginx”
# 增强后内容： “【文档：Nginx维护指南 / 章节：故障处理】执行命令：systemctl restart nginx”
# 目的： 帮助向量模型更好地理解这一小段话的背景。

# 4、 定义向量数据库
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024), # 维度根据模型而定
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="metadata", dtype=DataType.JSON)
]
collection_name = "demo_connection"
connections.connect(host="127.0.0.1", port="19530")
# if utility.has_collection(collection_name):
#     utility.drop_collection(collection_name)
#     print(f"已删除旧的 Collection: {collection_name}")

schema = CollectionSchema(fields)
collection = Collection(collection_name, schema)

# 2. 分批插入数据
batch_size = 10 # 每次插入10个切片

# 基于OpenAI的向量化（需要网络）
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=100) # 这里设置每批次大小
# vector_store = Milvus.from_documents(
#     chunks,
#     embeddings,
#     connection_args={"host": "127.0.0.1", "port": "19530"},
#     collection_name="demo_collection"
# )

# 本地处理向量化
model = SentenceTransformer('BAAI/bge-m3') # 该模型输出的维度是1024
for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i+batch_size]
    texts = [chunk.page_content for chunk in batch_chunks]
    metadatas = [chunk.metadata for chunk in batch_chunks]

    # 生成向量
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # 准备数据
    data = [
        {
            "vector": vectors[j],
            "text": texts[j],
            "metadata": metadatas[j]
        }
        for j in range(len(texts))
    ]

    # 插入数据
    res = collection.insert(data=data)
    print(f"Successfully inserted batch {i//batch_size + 1}")

# 6. 刷入磁盘（可选，异步转同步）
collection.flush()

# 7. 创建索引（搜索前必须执行）
index_params = {
    'metric_type': 'L2',  # 使用 L2 距离（欧氏距离）
    'index_type': 'IVF_FLAT',  # 索引类型
    'params': {'nlist': 1024}  # 聚类中心数量，可根据数据量调整
}
collection.create_index(field_name="vector", index_params=index_params)
print("索引创建成功")