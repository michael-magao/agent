import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

model_id = "meta-llama/Llama-3-8B" # 或本地路径

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # 启用4-bit量化 QLoRa
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

config = LoraConfig( # appter 配置
    r=8,              # 秩大小，越大参数越多（通常 8-32）
    lora_alpha=32,    # 缩放系数
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ], # 针对哪些层进行微调（不同模型名称不同）
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters() # 打印可训练参数百分比

from datasets import load_dataset

dataset = load_dataset("json", data_files="my_data.jsonl", split="train")
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True)

training_args = TrainingArguments(
    output_dir="./lora-llama-result",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100, # 简单测试可以先设小一点
    fp16=True, # 或 bf16
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=lambda data: {"input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]),
                                "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data])}
)

trainer.train()
# 注意：LoRA 只保存增量权重（几十 MB），不保存整个模型
model.save_pretrained("./my-lora-adapter")