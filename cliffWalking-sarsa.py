#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time

import numpy as np
import gym


class SarsaAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greedy=0.1):
        self.act_n = act_n  # 动作的维度， 有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # 折扣因子，reward的衰减率
        self.epsilon = e_greedy  # 按一定的概率随机选动作
        self.Q = np.zeros((obs_n, act_n))  # 创建一个Q表格

    # 根据输入观察值（这个代码不区分state和observation），采样输出的动作值
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
            action = self.predict(obs)  # 调用函数获得要在该观察值（或状态）条件下要执行的动作
        else:
            action = np.random.choice(self.act_n)  # e_greedy概率直接从动作空间中随机选取一个动作
        return action

    # 根据输入的观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]  # 从Q表中选取状态(或观察值)对应的那一行
        maxQ = np.max(Q_list)  # 获取这一行最大的Q值，可能出现多个相同的最大值

        action_list = np.where(Q_list == maxQ)[0]  # np.where(条件)功能是筛选出满足条件的元素的坐标
        action = np.random.choice(action_list)  # 这里尤其如果最大值出现了多次，随机取一个最大值对应的动作就成
        return action

    # 给环境作用一个动作后，对环境的所有反馈进行学习，也就是用环境反馈的结果来更新Q-table
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """
            on-policy
            obs：交互前的obs, 这里observation和state通用，也就是公式或者伪代码码中的s_t
            action： 本次交互选择的动作， 也就是公式或者伪代码中的a_t
            reward: 本次与环境交互后的奖励,  也就是公式或者伪代码中的r
            next_obs: 本次交互环境返回的下一个状态，也就是s_t+1
            next_action: 根据当前的Q表，针对next_obs会选择的动作，a_t+1
            done: 回合episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 如果到达终止状态， 没有下一个状态了，直接把奖励赋值给target_Q
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]  # 这两行代码直接看伪代码或者公式
        self.Q[obs, action] = predict_Q + self.lr * (target_Q - predict_Q)  # 修正q

    # 把Q表格的数据保存到文件中
    def save(self):
        npy_file = 'q_table_sarsa.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # 从文件中读取数据到 Q表格
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每一个回合episode走了多少step
    total_reward = 0  # 记录一个episode获得总奖励

    obs = env.reset()  # 重置环境，重新开始新的一轮（episode)
    action = agent.sample(obs)  # 根据算法选择一个动作，采用ε-贪婪算法选取动作

    while True:
        next_obs, reward, done, info = env.step(action)  # 与环境进行一次交互，即把动作action作用到环境，并得到环境的反馈
        next_action = agent.sample(next_obs)  # 根据获得的下一个状态，执行ε-贪婪算法后，获得下一个动作

        # 训练Sarsa算法， 更新Q表格
        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs  # 存储上一个观测值（这里状态和观测不区分，正常observation是state的一部分）

        total_reward += reward
        total_steps += 1

        if render:
            env.render()  # 重新画一份效果图
        if done:  # 如果达到了终止状态，则回合结束，跳出该轮循环
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
    env = gym.make("CliffWalking-v0")  # 悬崖边行走游戏，动作空间及其表示为：0 up , 1 right, 2 down, 3 left

    agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greedy=0.1)

    is_render = False

    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        print('Episode %s: steps = %s, reward = %.lf' % (episode, ep_steps, ep_reward))

        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False
    agent.save()

    # 全部训练结束，查看算法效果
    test_reward = test_episode(env, agent)
    print('test reward = %.1f' % test_reward)


def print_q_table():
    arr = np.load("q_table_sarsa.npy")
    print(arr)  # arr的类型为 <class 'numpy.ndarray'>


if __name__ == '__main__':
    # main()
    print_q_table()
