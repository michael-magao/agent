import random

import torch
import math
from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测是否有可用的GPU

steps_done = 0

# 定义经验元组 todo 理解这是干嘛的？？
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# 定义如何在特定的state下选择action的函数
def select_action(state, policy_net, action_size, steps_done, EPS_START, EPS_END, EPS_DECAY):
    # global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if random.random() < eps_threshold:
        return torch.tensor([[random.randrange(action_size)]], device=device, dtype=torch.long)
    else:
        # 利用：选择Q值最大的动作
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)

# 定义优化模型的函数，如何优化模型
def optimize_model(policy_net, target_net, memory, optimizer, batch_size, GAMMA):
    pass