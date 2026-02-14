import json

from torch.utils.data import Dataset

class FineTuneDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据 - 支持多种格式
        with open(data_path, 'r', encoding='utf-8') as f:
            self.examples = []
            for line in f:
                item = json.loads(line.strip())

                # 构建模型输入（对话格式）
                messages = item.get("messages", [])
                if not messages:
                    continue

                # 将对话转换为模型能理解的格式
                text = self.format_conversation(messages)

                # 编码文本
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt"
                )

                # 创建标签（训练时只计算assistant回答部分的损失）
                labels = encoding["input_ids"].clone()
                # 将非assistant部分的标签设为-100（忽略这些位置的损失计算）
                labels = self.mask_non_assistant_tokens(labels, messages)

                self.examples.append({
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": labels.squeeze()
                })

    def format_conversation(self, messages):
        """将消息列表格式化为模型输入的文本"""
        formatted_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted_text += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_text += f"<|assistant|>\n{content}\n"
        formatted_text += "<|endoftext|>"
        return formatted_text

    def mask_non_assistant_tokens(self, labels, messages):
        """只对assistant的回答计算损失"""
        # 这里简化处理：实际需要根据token位置精确定位
        # 更精确的实现需要解析tokenized后的位置信息
        return labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
