import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, layer, n):
        super(Encoder, self).__init__()

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

"""
编码器层 = 自注意力 + 前馈网络 + 残差/归一化
"""
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # 自注意力机制
        self.feed_forward = feed_forward # 前馈神经网络

    def forward(self, x, mask):
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output) # 残差连接
        x = self.norm1(x)

        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output) # 残差连接
        x = self.norm2(x)

        return x