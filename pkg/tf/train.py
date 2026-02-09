import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import model as tf

input_sequences = np.random.randint(1, 100, (1000, 10))
target_sequences = np.random.randint(1, 100, (1000, 10))

# 转换为 PyTorch 张量 这一步是干嘛的？
input_tensor = torch.tensor(input_sequences, dtype=torch.long)
target_tensor = torch.tensor(target_sequences, dtype=torch.long)

# 模型参数
input_dim = 100  # 输入词汇表大小
model_dim = 64   # 模型维度
num_heads = 4    # 注意力头数
num_layers = 10   # Transformer 层数
output_dim = 100 # 输出类别数

model = tf.TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)

# 3. 编译模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(input_tensor)
    loss = criterion(outputs.view(-1, output_dim), target_tensor.view(-1))

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 评估模型
model.eval()
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, dim=2)
    accuracy = (predicted.view(-1) == target_tensor.view(-1)).float().mean()
    print(f'Test accuracy: {accuracy.item():.4f}')

# 6. 导出模型
torch.save(model.state_dict(), 'transformer_model.pth')