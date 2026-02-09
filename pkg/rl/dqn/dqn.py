import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim): # DQN需要的输入：环境的状态集合，输出维度：动作集合
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # 使用ReLU激活函数 todo 为什么不使用sigmoid函数
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


