# ==================== 6. 推理测试 ====================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_inference(model_path, prompt):
    """测试微调后的模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 加载LoRA权重（如果使用LoRA）
    # 注意：实际使用时需要合并LoRA权重或动态加载

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response