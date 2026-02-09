import numpy as np
from torch.special import softmax

# 注意力机制的实现
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]
    scores = np.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None: # 针对encoder的场景进行处理的
        scores = np.where(mask == 0, -1e9, scores)
    p_attn = softmax(scores, axis=-1)
    if dropout is not None: # todo 理解dropout的作用
        p_attn = dropout(p_attn)
    return np.matmul(p_attn, value), p_attn

# 多头注意力机制
def multi_head_attention(query, key, value, num_heads, mask=None, dropout=None):
    d_model = query.shape[-1]
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    def split_heads(x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)

    query = split_heads(query)
    key = split_heads(key)
    value = split_heads(value)

    if mask is not None:
        mask = mask[:, np.newaxis, :, :]

    attn_output, attn_weights = attention(query, key, value, mask=mask, dropout=dropout)

    def combine_heads(x):
        batch_size = x.shape[0]
        x = x.transpose(0, 2, 1, 3).reshape(batch_size, -1, d_model)
        return x

    attn_output = combine_heads(attn_output)
    return attn_output, attn_weights