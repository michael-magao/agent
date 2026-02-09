from torch import nn

from pkg.transformer.decoder import Decoder
from pkg.transformer.encoder import Encoder
from pkg.transformer.position import PositionalEncoding
from pkg.transformer.ffn import FeedForward

# transformer定义
class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, d_model, max_len, num_layers, dropout):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim) # 嵌入层，直接用nn.Embedding
        self.pe = PositionalEncoding(d_model, max_len) # 位置编码，自定义
        self.encoder = Encoder(FeedForward(10, 64, 3), num_layers) # 定义encoder
        self.decoder = Decoder(model_dim, d_model) # 定义decoder

        self.dropout = nn.Dropout(dropout) # 添加Dropout层以防止过拟合 todo 这是干嘛的？

    def forward(self, x):
        x = self.embedding(x)  # 嵌入层
        x = self.pe(x) # 位置编码
        x = self.dropout(x) # dropout

        x = self.encoder(x) # 编码器
        return x