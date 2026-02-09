import gym

env = gym.make('CartPole-v1')
env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # 随机选择一个动作
    observation, reward, terminated, done, info = env.step(action)
    print(observation)

env.close()