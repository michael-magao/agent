import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# ==================== 2. 模型加载 ====================
def load_model_and_tokenizer(model_path="deepseek-ai/deepseek-llm-7b-chat"):
    """加载预训练模型和分词器"""
    print(f"正在加载模型: {model_path}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    # 冻结基础模型参数（如果使用LoRA）
    for param in model.parameters():
        param.requires_grad = False

    print(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer