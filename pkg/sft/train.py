import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """单个训练周期的完整实现"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # 将数据移动到设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )

        # 参数更新
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # 更新进度条
        progress_bar.set_postfix({
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0]
        })

        # 每100步打印一次损失
        if batch_idx % 100 == 0:
            print(f"Step {batch_idx}: loss = {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss