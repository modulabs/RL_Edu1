import gym
import sys
import numpy as np
from gym.envs.registration import register
import random
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf


def one_hot(N, x):
    return np.identify(N)[x:x+1]

env = gym.make('FrozenLake-v0')

input_size = env.observation_space.n
output_size = env.action_space.n

## tf.global_variables_initializer. (deprecated)
init = tf.initialize_all_variables()


X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform(shape=[input_size, output_size])) #, minval=0, maxval=1))   # why maxval = 0.01???

discount_factor = 0.99
num_episode = 2000

with tf.Session() as sess:
    sess.run(init)

    for i in range(num_episode):
        s = env.reset()
        e = 1 / ((1/50) + 10)
        rAll = 0
        done = False






        pass

