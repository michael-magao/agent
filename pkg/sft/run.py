import torch
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM

from pkg.sft.config import config
from pkg.sft.lora import add_lora_to_model
from pkg.sft.model import load_model_and_tokenizer
from pkg.sft.data import FineTuneDataset
from pkg.sft.train import train_epoch
from torch.utils.data import DataLoader


def main():
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if device.type == "cuda":
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"显存可用: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(config["model_path"])
    model.to(device)

    # 2. 添加LoRA适配器
    model = add_lora_to_model(model, {
        "rank": config["lora_rank"],
        "alpha": 16,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    })

    # 3. 准备数据
    print("正在加载和预处理数据...")
    dataset = FineTuneDataset(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        max_length=config["max_length"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )

    # 4. 设置优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=0.01
    )

    total_steps = len(dataloader) * config["epochs"] // config["gradient_accumulation_steps"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps
    )

    # 5. 开始训练
    print(f"开始训练，总步数: {total_steps}")
    for epoch in range(config["epochs"]):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*50}")

        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, device, epoch+1
        )

        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

        # 保存检查点
        if (epoch + 1) % 1 == 0:  # 每轮都保存
            checkpoint_path = f"{config['output_dir']}/epoch_{epoch+1}"

            # 只保存LoRA权重（轻量级）
            lora_state_dict = {
                name: param for name, param in model.named_parameters()
                if param.requires_grad
            }

            torch.save({
                'epoch': epoch,
                'model_state_dict': lora_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, f"{checkpoint_path}_lora.pth")

            print(f"检查点已保存到: {checkpoint_path}_lora.pth")

    # 6. 最终模型保存
    print("\n训练完成！正在保存最终模型...")

    # 保存完整的适配器
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print(f"模型已保存到: {config['output_dir']}")

if __name__ == "__main__":
    # 创建输出目录
    import os
    os.makedirs("./checkpoints", exist_ok=True)

    # 运行主训练
    main()