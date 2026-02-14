config = {
    "model_path": "deepseek-ai/deepseek-llm-7b-chat",  # 可更换为其他DeepSeek模型
    "data_path": "your_data.jsonl",  # 你的训练数据
    "output_dir": "./checkpoints",
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # 梯度累积（模拟更大batch size）
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "max_length": 512,
    "lora_rank": 8,
    "save_steps": 500
}