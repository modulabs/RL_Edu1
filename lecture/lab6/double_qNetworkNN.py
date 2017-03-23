import gym
from gym import wrappers
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
if "../../" not in sys.path:
  sys.path.append("../../")
from lib import plotting
import os
import tensorflow as tf


def double_qNetwork(env, n_episodes=3000, gamma=0.99, alpha=0.85, best_enabled=False, log_by_step=False, result_per_episode=100, network_type='LR'):
    nS = env.observation_space.n
    nA = env.action_space.n
    hidden_size = 30
    subRewardList = []
    avgRewardList = []

    def one_hot(x):
        return np.identity(nS)[x:x + 1]

    if best_enabled:
        # record your best-tuned hyperparams here
        env.seed(0)
        np.random.seed(0)
        alpha = 0.003
        gamma = 0.99
        epsilon_decay = 0.95
        e = 1.0

    X = tf.placeholder(shape=[1, nS], dtype=tf.float32)
    Y = tf.placeholder(shape=[1, nA], dtype=tf.float32)

    if network_type == 'NN':
        W1_1 = tf.get_variable("W1_1", shape=[nS, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        Z1_1 = tf.matmul(X, W1_1)
        Z1_1 = tf.nn.tanh(Z1_1)
        W2_1 = tf.get_variable("W2_1", shape=[hidden_size, nA], initializer=tf.contrib.layers.xavier_initializer())
        Qpred_1 = tf.matmul(Z1_1, W2_1)

        W1_2 = tf.get_variable("W1_2", shape=[nS, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        Z1_2 = tf.matmul(X, W1_2)
        Z1_2 = tf.nn.tanh(Z1_2)
        W2_2 = tf.get_variable("W2_2", shape=[hidden_size, nA], initializer=tf.contrib.layers.xavier_initializer())
        Qpred_2 = tf.matmul(Z1_2, W2_2)

    else:  # network_type == 'LR':  (Logistic Regression)
        W_1 = tf.Variable(tf.random_uniform([nS, nA], 0, 0.01))
        W_2 = tf.Variable(tf.random_uniform([nS, nA], 0, 0.01))
        Qpred_1 = tf.matmul(X, W_1)
        Qpred_2 = tf.matmul(X, W_2)

    loss_1 = tf.reduce_sum(tf.square(Y - Qpred_1))
    train_1 = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_1)
    loss_2 = tf.reduce_sum(tf.square(Y - Qpred_2))
    train_2 = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_2)


    init = tf.global_variables_initializer()

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(n_episodes),
        episode_rewards=np.zeros(n_episodes))

    with tf.Session() as sess:
        sess.run(init)

        for i in range(n_episodes):
            s = env.reset()
            total_reward = 0
            e = 1. / ((i / 10) + 10)  # 실행하면서 낮춰주도록 함

            done = False
            count = 0

            Qpred_update = None
            Qpred_other = None
            train = None

            for t in itertools.count():

                # With 0.5 probability
                if np.random.choice(2) == 1:
                    Qpred_update = Qpred_1
                    Qpred_other = Qpred_2
                    train = train_1
                else:
                    Qpred_update = Qpred_2
                    Qpred_other = Qpred_1
                    train = train_2

                Qs = sess.run(Qpred_update, feed_dict={X: one_hot(s)})

                if log_by_step:
                    print(Qs)

                if np.random.rand(1) < e:
                    a = env.action_space.sample()
                else:
                    a = np.argmax(Qs)

                s1, reward, done, _ = env.step(a)

                if log_by_step:
                    print('step %d, curr state : %d, action : %d, next state : %d, reward : %d' % (t, s, a, s1, reward))

                if best_enabled:
                    mod_reward = modify_reward(reward, done)

                    if done:
                        Qs[0, a] = mod_reward
                    else:
                        Qs1 = sess.run(Qpred_other, feed_dict={X: one_hot(s1)})
                        Qs[0, a] = mod_reward + gamma * np.max(Qs1)
                else:
                    if done:
                        Qs[0, a] = reward
                    else:
                        Qs1 = sess.run(Qpred_other, feed_dict={X: one_hot(s1)})
                        Qs[0, a] = reward + gamma * np.max(Qs1)

                sess.run(train, feed_dict={X: one_hot(s), Y: Qs})

                total_reward += reward
                s = s1
                count += 1

                if done:
                    break

            subRewardList.append(total_reward)

            if (i + 1) % result_per_episode == 0:
                avg = sum(subRewardList) / result_per_episode
                avgRewardList.append(avg)
                print(i + 1, ' episode =', total_reward, ', avg =', avg)
                subRewardList = []

            # Update statistics
            stats.episode_rewards[i] += total_reward
            stats.episode_lengths[i] = t

    return stats, subRewardList, avgRewardList


def modify_reward(reward, done):
    # 100.0 being arbitrary scaling factors
    if done and reward == 0:
        return -100.0
    elif done:
        return 100.0
    else:
        return -1.0


def log_episode(i_epi, n_epi):
    if (i_epi + 1) % 100 == 0:
        print("\rEpisode {}/{}.".format(i_epi + 1, n_epi), end="")
        sys.stdout.flush()


def is_solved(stats, target, interval):
    """
    checks if openai's criteria has been met
    """
    # FrozenLake-v0 is considered "solved" when the agent
    # obtains an average reward of at least 0.78 over 100
    # consecutive episodes.
    avg_reward = np.sum(stats.episode_rewards)/len(stats.episode_rewards)
    print("Average reward : {}".format(avg_reward))

    def moving_avg(x, n=100):
        return np.convolve(x, np.ones((n,))/n, mode='valid')

    ma = moving_avg(stats.episode_rewards, interval)
    peaks = np.where(ma > target)[0]
    if len(peaks) > 0:
        print("solved after {} episodes".format(peaks[0]))
        return True
    else:
        print("did not pass the openai criteria")
        return False


if __name__ == "__main__":
    TARGET_AVG_REWARD = 0.78
    TARGET_EPISODE_INTERVAL = 100
    env = gym.make('FrozenLake-v0')
    env = wrappers.Monitor(env, '/tmp/frozenlake-experiment-1', force=True)
    stats, subRewardList, avgRewardList = double_qNetwork(env, best_enabled=True, log_by_step=False, network_type='NN')

    env.close()

    if is_solved(stats, TARGET_AVG_REWARD, TARGET_EPISODE_INTERVAL):
        print('SOLVED!!!')
        #OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
        #gym.upload('/tmp/frozenlake-experiment-1', api_key=OPENAI_API_KEY)

    plt.ylim(0, 1)
    plt.bar(range(len(avgRewardList)), avgRewardList, color='blue', bottom=0)

    plt.show()

'''  [LR]
Average reward : 0.475
did not pass the openai criteria
'''

'''  [NN]
Average reward : 0.43666666666666665
solved after 2802 episodes
SOLVED!!!
'''
