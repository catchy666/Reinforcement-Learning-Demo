import gym
import numpy as np


class QLearningAgent:
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # 有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, done):
        """
        off-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])  # Q-learning
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 修正q

    # 把Q表格的数据保存到文件中
    def save(self):
        npy_file = 'q_table_ql.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # 从文件中读取数据到 Q表格
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    while True:
        action = agent.sample(obs)  # 根据算法选择一个动作
        next_obs, reward, done, _ = env.step(action)  # 与环境进行一次交互

        # 训练 Q-learning 算法
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        if render:
            env.render()  # 渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        # time.sleep(0.5)
        env.render()
        if done:
            break
    return total_reward


def main():
    # 使用gym创建悬崖环境
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

    # 创建一个agent实例，输入超参数
    agent = QLearningAgent(obs_n=env.observation_space.n,
                           act_n=env.action_space.n,
                           learning_rate=0.1,
                           gamma=0.9,
                           e_greed=0.1)

    # 训练500个episode，打印每个episode的分数
    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, False)
        print(
            "Episode %s: step = %s , reward = %.1f" % (episode, ep_steps, ep_reward))

    agent.save()

    # 全部训练结束，查看算法效果
    test_reward = test_episode(env, agent)
    print('test reward = %.1f' % test_reward)


def print_q_table():
    arr = np.load("q_table_ql.npy")
    print(arr)  # arr的类型为 <class 'numpy.ndarray'>


if __name__ == '__main__':
    # main()
    print_q_table()
