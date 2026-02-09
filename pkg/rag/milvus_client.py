from pymilvus import connections, Collection

connections.connect("default", host='localhost', port='19530')

from pymilvus import Collection

# 选择集合
collection_name = "demo_collection"
collection = Collection(collection_name)

# 输入数据


# 指定查询条件
query = {"field_name": {"$in": ["value1", "value2"]}}  # 根据你的数据结构调整查询条件

# 执行查询
results = collection.query(expr=query)

# 打印查询结果
for result in results:
    print(result)