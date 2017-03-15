import gym
from gym import wrappers
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import tensorflow as tf


class Q_network(object):
    """
    approximating Q(s, a) using Feedforward Neural Network
    """
    def __init__(self, env, gamma, alpha):
        super(Q_network, self).__init__()

        self.n_input = env.observation_space.n
        self.n_output = env.action_space.n
        self.alpha = alpha
        self.gamma = gamma

        # initialize tf feedforward neural network
        self.X = tf.placeholder(shape=[1, self.n_input], dtype=tf.float32)
        self.Theta = tf.Variable(tf.random_uniform([self.n_input, self.n_output], 0, 0.01))
        self.Q_pred = tf.matmul(self.X, self.Theta)
        self.Y = tf.placeholder(shape=[1, self.n_output], dtype=tf.float32)

        # kickstart learning Theta
        self.loss = tf.reduce_sum(tf.square(self.Y - self.Q_pred))
        self.train = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.loss)
        # utility function
        self.one_hot = make_one_hot(env.observation_space.n)

    def predict(self, sess, s):
        Qs = sess.run(self.Q_pred, feed_dict={self.X: self.one_hot(s)})
        return Qs

    def update(self, sess, s, y):
        sess.run(self.train, feed_dict={self.X: self.one_hot(s), self.Y: y})


def q_learning(env, n_episodes=2000, gamma=0.99, alpha=0.1):
    """
    using Q learning with Q network under epilson-greedy policy
    """
    reward_per_episode = np.zeros(n_episodes)
    estimator = Q_network(env, gamma, alpha)
    policy = make_epsilon_greedy_policy(
        estimator, estimator.n_output)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(n_episodes):
            s = env.reset()
            done = False
            total_reward = 0
            e = 1.0 / ((i / 10) + 10)
            # useful for debugging
            log_episode(i, n_episodes)

            while not done:
                probs, Qs = policy(sess, s, e)
                a = np.random.choice(np.arange(len(probs)), p=probs)
                next_s, r, done, _ = env.step(a)

                # update our target == Qs
                if done:
                    Qs[0, a] = r
                else:
                    next_Qs = estimator.predict(sess, next_s)
                    Qs[0, a] = r + gamma * np.max(next_Qs)

                # online learning
                estimator.update(sess, s, Qs)

                s = next_s
                total_reward += r

            reward_per_episode[i] = total_reward

        return estimator.Theta, reward_per_episode

def make_epsilon_greedy_policy(estimator, n_output):
    def policy_fn(sess, state, epsilon):
        probs = np.ones(n_output, dtype=float) * epsilon / n_output
        q_values = estimator.predict(sess, state)
        best_action = np.argmax(q_values[0])
        probs[best_action] += (1.0 - epsilon)
        return probs, q_values
    return policy_fn

def make_one_hot(size):
    def one_hot(x):
        return np.identity(size)[x: x+1]
    return one_hot

def log_episode(i_epi, n_epi):
    if (i_epi + 1) % 100 == 0:
        print("\rEpisode {}/{}.".format(i_epi + 1, n_epi), end="")
        sys.stdout.flush()

def visualize(estimator, stats, output_title="output.png"):
    print("Success rate : {}".format(np.sum(stats)/len(stats)))
    print("Final Q-network Values")
    print(estimator)
    plt.figure(figsize=(8,12))
    plt.title("Reward_per_episode")
    plt.plot(stats)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(output_title)

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    env = wrappers.Monitor(env, '/tmp/frozenlake-experiment-qnetwork', force=True)
    Theta, stats = q_learning(env)
    env.close()
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    gym.upload('/tmp/frozenlake-experiment-2', api_key=OPENAI_API_KEY)

    visualize(Theta, stats, "qnetwork_frozen.png")

