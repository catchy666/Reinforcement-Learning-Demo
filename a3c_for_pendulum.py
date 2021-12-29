import threading
import multiprocessing
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import time

GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()

MAX_EP_STEP = 200
MAX_GLOBAL_EP = 4000  # 2000 4000
GLOBAL_NET_SCOPE = 'Global_Net'
GAMMA = 0.9
UPDATE_GLOBAL_ITER = 100
ENTROPY_BETA = 0.01  # 0.01 0.001

LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
# LR_A = 0.001  # learning rate for actor
# LR_C = 0.002  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]  # bound of output action


class ACNet:
    # This class is to define the global actor-critic and local actor-critics
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                # Get parameters of the actor and critic in global network
                self.a_params, self.c_params = self._build_net(scope)[-2:]

        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')

                with tf.name_scope('c_loss'):
                    # to minimize TD-error
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4
                # distribution of parameters：mu, sigma
                normal_dist = tf.distributions.Normal(mu, sigma)
                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)  # log pi(a)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration:larger entropy means more stochastic actions
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                    # to maximize tf.reduce_mean(self.exp_v)  <=> to minimize tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):  # 把主网络的参数赋予各子网络
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):  # 使用子网络的梯度对主网络参数进行更新
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')

        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        # run by a local: choose action from normal distributions
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})


class Worker:
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC=globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == MAX_EP_STEP - 1 else False
                r /= 10  # normalize reward
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, feed_dict={self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_  # 使用v(s) = r + v(s+1)计算target_v
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target
                    }
                    self.AC.update_global(feed_dict)  # push local gradients to global net
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()  # pull the newest global params to local

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(self.name, "Ep:", GLOBAL_EP, "| Ep_r: %i" % GLOBAL_RUNNING_R[-1])
                    GLOBAL_EP += 1
                    break


if __name__ == '__main__':
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # define two optimizers for actors and critics in local net
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its parameters
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = "W_%i" % i  # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    # Coordinator类用来管理在Session中的多个线程，
    # 使用 tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    start_time = time.time()
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)  # 创建一个线程，并分配其工作
        t.start()  # 开启线程
        worker_threads.append(t)
    COORD.join(worker_threads)  # 把开启的线程加入主线程，等待threads结束

    env.close()
    print("%d线程耗时：" % N_WORKERS, str(time.time() - start_time), 's')

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
