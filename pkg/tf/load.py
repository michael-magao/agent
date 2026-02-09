import torch
import model as tf

input_dim = 100  # 输入维度
model_dim = 64   # 模型维度
num_heads = 4    # 注意力头数
num_layers = 10   # Transformer 层数
output_dim = 100 # 输出维度

model = tf.TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)

# 加载模型权重
model.load_state_dict(torch.load('transformer_model.pth'))

# 设置模型为评估模式
model.eval()

input_data = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)

# 进行预测
with torch.no_grad():
    output = model(input_data)
    predicted = torch.argmax(output, dim=2)

print(predicted)