import numpy as np
import tensorflow as tf
import gym
from dqn import dqn

from collections import deque

def main():
    max_episode = 5000

    #replay_buffer = deque()

    env = gameSetup('CartPole-v0')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    print(env.observation_space.shape)
    print(env.action_space.n)

    with tf.Session() as sess:
        algorithm = dqn(sess, input_size, output_size)
        algorithm.initNetwork()






    #pass


def gameSetup(gameTitle):

    env = gym.make(gameTitle)

    return env





if __name__ == "__main__":
    main()

