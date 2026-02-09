import torch
import torch.nn as nn

"""Position-wise Feed-Forward Network"""
class FeedForward(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout) # 添加Dropout层以防止过拟合
        self.activation = nn.ReLU() # 使用ReLU作为激活函数 todo 尝试用sigmoid

    def forward(self, x):
        x = self.fc1(x)
        print("x shape after fc1:", x)
        x = self.activation(x) # 这里可以替换为其他激活函数，如GELU
        x = self.dropout(x) # 这里可以加入MOE的门控机制
        x = self.fc2(x)
        return x

input_dim = 10
hidden_dim = 64
output_dim = 3

print("PyTorch 模型结构：")

model_pytorch = FeedForward(input_dim, hidden_dim, output_dim)
print(model_pytorch)


dummy_input = torch.randn(1, input_dim)
print("input", dummy_input)

output_pytorch = model_pytorch(dummy_input)

print("输出张量形状 (Batch x Output_Dim)：", output_pytorch.shape)