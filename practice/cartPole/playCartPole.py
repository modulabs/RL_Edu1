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


class CartPolePlay(gameplay.GamePlay):
    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.algorithm = None
        self.session = None
        self.env = None
        self.gameparams = {}

    def gamePlayMain(self):
        with tf.Session() as sess:
            self.session = sess
            self.gameSetup()
            self.trainAgent()
            self.runGamePlay()

    def setAlgorithm(self):
        hyperparams = {}

        if self.algorithm_name == 'dqn_2013':
            #hyperparams['input_size'] = self.env.observation_space.shape[0]
            #hyperparams['output_size'] = self.env.action_space.n
            self.algorithm = dqn(self.session, name='dqn', version='2013', gameparam=self.gameparams, externalHyperparam=hyperparams)
        elif self.algorithm_name == 'dqn_2015':
            #hyperparams['input_size'] = self.env.observation_space.shape[0]
            #hyperparams['output_size'] = self.env.action_space.n
            self.algorithm = dqn(self.session, name='dqn', version='2015', gameparam=self.gameparams, externalHyperparam=hyperparams)
        else:
            pass  # TODO: add more good algorithms!!

    def gameSetup(self):
        self.env = gym.make('CartPole-v0')

        self.gameparams['max_episode'] = 5000
        self.gameparams['input_size'] = self.env.observation_space.shape[0]
        self.gameparams['output_size'] = self.env.action_space.n

        print("gameSetup input_size {}".format(self.gameparams['input_size']))
        print("gameSetup output_size {}".format(self.gameparams['output_size']))

        self.setAlgorithm()
        self.algorithm.initNetwork()

    def getNextAction(self, state, mode):
        action = self.algorithm.getNextAction(state, mode)
        if action is None:
            action = self.env.action_space.sample()

        return action

    def trainAgent(self):
        self.algorithm.initTraining()

        for episode in range(self.gameparams['max_episode']):
            state = self.env.reset()
            done = False
            step_count = 0

            while not done:
                action = self.getNextAction(state, mode='train')
                next_state, reward, done, _ = self.env.step(action)
                reward = self.modifyReward(reward, done)

                self.algorithm.stepTrain(state, action, reward, next_state, done)

                state = next_state
                step_count += 1

                if step_count > 1000:  #if step_count > 10000: # good enough
                    print("Episode: {}, steps: {} [Good Enough]".format(episode, step_count))
                    return

            print("Episode: {}, steps: {}".format(episode, step_count))

            loss = self.algorithm.episodeTrain()
            print(">>>Loss after replay memory: ", loss)



    def modifyReward(self, reward, done):
        if done:
            return -100
        else:
            return reward


    def runGamePlay(self, render_play=True):
        state = self.env.reset()
        reward_sum = 0
        done = False
        while not done:
            if render_play:
                self.env.render()
            action = self.getNextAction(state, mode='play')
            state, reward, done, _ = self.env.step(action)
            reward_sum += reward

        print("Total Score: {}".format(reward_sum))



if __name__ == "__main__":
    print("DQN 2013 Game start")
    play = CartPolePlay('dqn_2013')
    #play = CartPolePlay('dqn_2015')

    print("DQN 2013 Game start")
    play.gamePlayMain()