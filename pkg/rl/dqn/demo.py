import gym

env = gym.make("CartPole-v1")
# env.seed(42)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

print(f"状态数: {n_states}, 动作数: {n_actions}")
