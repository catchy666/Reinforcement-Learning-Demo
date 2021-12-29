import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

# ####################  hyper parameters  ####################
env_id = 'FrozenLake-v0'
alg_name = 'DQN'
lambd = .99  # decay factor
epsilon = 0.1  # e-Greedy Exploration, the larger the more random
num_episodes = 10000
render = False  # display the game environment


# #################### DQN ##########################

def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a


# Define Q-network q(a,s) that output the rewards of 4 actions by given state, i.e. Action-Value Function.
# encoding for state: 4x4 grid can be represented by one-hot vector with 16 integers.
def get_model(inputs_shape):
    ni = tl.layers.Input(inputs_shape, name='observation')
    nn = tl.layers.Dense(4, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')(ni)
    return tl.models.Model(inputs=ni, outputs=nn, name='Q-Network')


if __name__ == '__main__':
    q_network = get_model([None, 16])
    q_network.train()
    train_weights = q_network.trainable_weights

    optimizer = tf.optimizers.SGD(learning_rate=0.1)
    env = gym.make(env_id)

    t0 = time.time()
    all_episode_reward = []
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()  # observation is state, integer 0 ~ 15
        rAll = 0
        if render: env.render()
        for j in range(99):
            # Choose an action by greedily (with e chance of random action) from the Q-network
            allQ = q_network(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
            a = np.argmax(allQ, 1)

            # e-Greedy Exploration !!! sample random action
            if np.random.rand(1) < epsilon:
                a[0] = env.action_space.sample()

            # Get new state and reward from environment
            s_, r, done, _ = env.step(a[0])

            if render: env.render()
            # Obtain the Q' values by feeding the new state through our network
            Q_ = q_network(np.asarray([to_one_hot(s_, 16)], dtype=np.float32)).numpy()

            # Obtain max Q' and set our target value for chosen action.
            maxQ_ = np.max(Q_)  # in Q-Learning, policy is greedy, so we use "max" to select the next action.
            targetQ = allQ
            targetQ[0, a[0]] = r + lambd * maxQ_

            with tf.GradientTape() as tape:
                _qvalues = q_network(np.asarray([to_one_hot(s, 16)], dtype=np.float32))
                _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
            grad = tape.gradient(_loss, train_weights)
            optimizer.apply_gradients(zip(grad, train_weights))

            rAll += r
            s = s_
            # Reduce chance of random action if an episode is done.
            if done:
                epsilon = 1. / ((i / 50) + 10)  # reduce e, GLIE: Greey in the limit with infinite Exploration
                break
        # Note that, the rewards here with random action
        print('Training  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}'.format(
            i, num_episodes, rAll, time.time() - t0))

        if i == 0:
            all_episode_reward.append(rAll)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.99 + rAll * 0.01)

    plt.plot(all_episode_reward)
