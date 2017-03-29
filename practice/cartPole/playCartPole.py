import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import sys

if "../../" not in sys.path:
  sys.path.append("../../")
from lib import gameplay

if "../lecture/lab7" not in sys.path:
  sys.path.append("../lecture/lab7")
from dqn import dqn



class CartPolePlay(gameplay.Gameplay):
    def __init__(self):
        self.algorithm = None
        self.session = None

    def gamePlayMain(self):
        with tf.Session() as sess:
            self.session = sess
            self.gameSetup()
            self.runBotPlay()

    def gameSetup(self):
        env = gym.make('CartPole-v0')
        hyperparams = {}
        hyperparams['input_size'] = env.observation_space.shape[0]
        hyperparams['output_size'] = env.action_space.n

        self.algorithm = dqn(self.session, externalHyperparam=hyperparams)
        self.algorithm.initNetwork()

    def runBotPlay(self, train=True, max_episode = 5000):
        pass


if __name__ == "__main__":
    print("DQN 2013 Game start")
    play = CartPolePlay()
    print("DQN 2013 Game start")
    play.gamePlayMain()

