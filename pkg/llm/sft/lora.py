import torch
import torch.nn.functional as F


def sft_loss(logits, labels, mask, alpha=1.0):
    """
    计算SFT（Supervised Fine-Tuning）损失。
    参数:
        logits: 模型输出的logits，形状为 (batch_size, seq_len, vocab_size)。
        labels: 真实标签，形状为 (batch_size, seq_len)。
        mask: 掩码，形状为 (batch_size, seq_len)，用于指示哪些位置需要计算损失。
        alpha: 权重参数，控制mask位置和非mask位置的损失权重。
    返回:
        loss: 计算得到的SFT损失。
    """