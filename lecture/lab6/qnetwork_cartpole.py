import gym
from gym import wrappers
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import tensorflow as tf
import itertools


class Q_network(object):
    """
    approximating Q(s, a) using Feedforward Neural Network
    this is not yet general enough to vary key params like num_hidden_layers
    """
    def __init__(self, env, gamma, alpha):
        super(Q_network, self).__init__()

        self.n_input = env.observation_space.shape[0]
        self.n_output = env.action_space.n
        self.alpha = alpha
        self.gamma = gamma

        # initialize tf feedforward neural network
        self.X = tf.placeholder(shape=[None, self.n_input], dtype=tf.float32)

        # first hidden layer
        self.W = tf.get_variable("W", shape=[self.n_input, self.n_output],
            initializer=tf.contrib.layers.xavier_initializer())

        self.Q_pred = tf.matmul(self.X, self.W)
        self.Y = tf.placeholder(shape=[1, self.n_output], dtype=tf.float32)

        # kickstart learning W
        self.loss = tf.reduce_sum(tf.square(self.Y - self.Q_pred))
        self.train = tf.train.AdamOptimizer(learning_rate=alpha).minimize(self.loss)

    def predict(self, sess, s):
        feed_dict = {self.X: self.preprocess(s)}
        Qs = sess.run(self.Q_pred, feed_dict=feed_dict)
        return Qs

    def update(self, sess, s, y):
        feed_dict = {self.X: self.preprocess(s), self.Y: y}
        sess.run(self.train, feed_dict=feed_dict)

    def preprocess(self, s):
        return np.reshape(s, [1, self.n_input])



def run_agent(env, n_episodes=2000, gamma=0.99, alpha=0.1):
    """
    using Q learning with Q network under epilson-greedy policy
    """
    estimator = Q_network(env, gamma, alpha)
    policy = make_epsilon_greedy_policy(
        estimator, estimator.n_output)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        stats = q_learning(n_episodes, sess, env,
            gamma, policy, estimator, failure_penalty=-100)
        test_estimator(env, estimator, sess)

    return estimator, stats

def q_learning(n_episodes, sess, env, gamma, policy, estimator, failure_penalty=-100):
    reward_per_episode = np.zeros(n_episodes)
    step_history = []
    for i in range(n_episodes):
        s = env.reset()
        done = False
        total_reward = 0
        e = 1.0 / ((i / 10) + 10)
        # useful for debugging
        log_episode(i, n_episodes)

        for t in itertools.count():
            probs, Qs = policy(sess, s, e)
            a = np.random.choice(np.arange(len(probs)), p=probs)
            next_s, r, done, _ = env.step(a)

            # update our target == Qs
            if done:
                Qs[0, a] = failure_penalty
                step_history.append(t)
                break
            else:
                next_Qs = estimator.predict(sess, next_s)
                Qs[0, a] = r + gamma * np.max(next_Qs)

            # online learning
            estimator.update(sess, s, Qs)
            s = next_s
            total_reward += r
        print("Episode: {}  steps: {}".format(i, t))
        # If last 10's avg steps are 500, it's good enough
        if len(step_history) > 10 and np.mean(step_history[-10:]) > 500:
            break

        reward_per_episode[i] = total_reward
    return reward_per_episode

def test_estimator(env, estimator, sess):
    s = env.reset()
    reward_sum = 0
    for t in itertools.count():
        env.render()
        Qs = estimator.predict(sess, estimator.preprocess(s))
        # follow the greedy policy
        a = np.argmax(Qs)
        s, r, done, _ = env.step(a)
        reward_sum += r
        if done:
            print("Test the trained model")
            print("Total rewards: {}".format(reward_sum))
            print("Survived until t = {}".format(t))
            break

def make_epsilon_greedy_policy(estimator, n_output):
    def policy_fn(sess, state, epsilon):
        probs = np.ones(n_output, dtype=float) * epsilon / n_output
        q_values = estimator.predict(sess, state)
        best_action = np.argmax(q_values[0])
        probs[best_action] += (1.0 - epsilon)
        return probs, q_values
    return policy_fn

def log_episode(i_epi, n_epi):
    if (i_epi + 1) % 100 == 0:
        print("\rEpisode {}/{}.".format(i_epi + 1, n_epi), end="")
        sys.stdout.flush()

def visualize(estimator, stats, output_title="output.png"):
    print("Success rate : {}".format(np.sum(stats)/len(stats)))
    plt.figure(figsize=(8,12))
    plt.title("Reward_per_episode")
    plt.plot(stats)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(output_title)
    print("###RUN THIS MANY TIMES TO SEE THE CRAZY VARIANCE###")

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    # env = wrappers.Monitor(env, '/tmp/cartpole-experiment-qnetwork-0', force=True)
    estimator, stats = run_agent(env, n_episodes=500)
    # env.close()
    # OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    # gym.upload('/tmp/cartpole-experiment-qnetwork-0', api_key=OPENAI_API_KEY)

    # visualize(estimator, stats, "qnetwork_cartpole.png")

