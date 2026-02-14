import torch
import torch.nn.functional as F
from torch import nn


# ==================== 3. LoRA适配器实现 ====================
class LoRALayer(nn.Module):
    """LoRA适配器层"""
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # LoRA的A矩阵（下投影）
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        # LoRA的B矩阵（上投影）
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5) # 初始化A矩阵
        nn.init.zeros_(self.lora_B.weight) # 初始化B矩阵为0，使得初始时LoRA不影响原始权重

        self.scaling = alpha / rank

    def forward(self, x):
        # 原始前向传播 + LoRA适配
        return self.lora_B(self.lora_A(x)) * self.scaling

def add_lora_to_model(model, lora_config=None):
    """将LoRA层添加到模型的线性层"""
    if lora_config is None:
        lora_config = {
            "rank": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "v_proj"]  # 只对注意力层的部分投影添加LoRA
        }

    for name, module in model.named_modules():
        if any(target in name for target in lora_config["target_modules"]):
            if isinstance(module, nn.Linear):
                # 获取原始层的维度
                in_dim = module.in_features
                out_dim = module.out_features

                # 创建LoRA层
                lora_layer = LoRALayer(
                    in_dim,
                    out_dim,
                    rank=lora_config["rank"],
                    alpha=lora_config["alpha"]
                )

                # 替换原始层的前向传播
                original_forward = module.forward

                def new_forward(x, original=original_forward, lora=lora_layer):
                    return original(x) + lora(x)

                module.forward = new_forward

                # 将LoRA参数设置为可训练
                for param in lora_layer.parameters():
                    param.requires_grad = True

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA添加完成，可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    return model

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