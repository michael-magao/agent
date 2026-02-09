import torch
import torch.nn as nn

# 模型参数
input_dim = 100  # 输入词汇表大小
model_dim = 64   # 模型维度
num_heads = 4    # 注意力头数
num_layers = 2   # Transformer 层数
output_dim = 100 # 输出类别数

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__() # todo 解析该函数
        self.embedding = nn.Embedding(input_dim, model_dim) # todo 解析该函数
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # 嵌入层
        x = x.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, model_dim)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)  # 转换回 (batch_size, seq_len, model_dim)
        x = self.fc(x)
        return x
