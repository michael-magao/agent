import torch
import torch.nn as nn
import math

# todo 可以改用RoPE位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算缩放因子
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数维度用 cos
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# using sin and cos to encode position, the reason is that they can provide a unique representation for each position in the sequence.
