import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

# Load the environment
alg_name = 'Qlearning'
env_id = 'FrozenLake-v0'
env = gym.make(env_id)
render = False  # display the game environment

# ================= Implement Q-Table learning algorithm ===================== #
# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
lr = .85
gamma = .99  # decay factor
num_episodes = 10000
t0 = time.time()

all_episode_reward = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    # The Q-Table learning algorithm
    for j in range(99):
        if render: env.render()
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # Get new state and reward from environment
        s_, r, done, _ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + gamma * np.max(Q[s_, :]) - Q[s, a])
        rAll += r
        s = s_
        if done is True:
            break
    print(
        'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
            i + 1, num_episodes, rAll,
            time.time() - t0
        )
    )
    if i == 0:
        all_episode_reward.append(rAll)
    else:
        all_episode_reward.append(all_episode_reward[-1] * 0.99 + rAll * 0.01)

env.close()
plt.plot(all_episode_reward)
print("Final Q-Table Values:\n %s" % Q)