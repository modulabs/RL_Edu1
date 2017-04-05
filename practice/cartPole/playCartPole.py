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
        self.goodStepCount = 0


    def gamePlayMain(self):
        with tf.Session() as sess:
            self.session = sess
            if self.gameSetup():
                self.trainAgent()
                self.runGamePlay()


    def setAlgorithm(self):
        hyperparams = {}

        if self.algorithm_name == 'dqn_2013':
            #hyperparams['input_size'] = self.env.observation_space.shape[0]
            #hyperparams['output_size'] = self.env.action_space.n
            self.algorithm = dqn(self.session, version='2013', gameparam=self.gameparams, externalHyperparam=hyperparams)
        elif self.algorithm_name == 'dqn_2015':
            #hyperparams['input_size'] = self.env.observation_space.shape[0]
            #hyperparams['output_size'] = self.env.action_space.n
            self.algorithm = dqn(self.session, version='2015', gameparam=self.gameparams, externalHyperparam=hyperparams)
        else:   # TODO: add more good algorithms!!
            self.algorithm = None

        return self.algorithm


    def gameSetup(self):
        self.env = gym.make('CartPole-v0')

        self.gameparams['max_episode'] = 5000
        self.gameparams['input_size'] = self.env.observation_space.shape[0]
        self.gameparams['output_size'] = self.env.action_space.n
        self.gameparams['max_stepPerEdisode'] = 200
        self.gameparams['good_stepPerEdisode'] = 195
        self.gameparams['goodEnoughCount'] = 100

        print("gameSetup input_size {}".format(self.gameparams['input_size']))
        print("gameSetup output_size {}".format(self.gameparams['output_size']))

        if self.setAlgorithm() is None:
            print(">>>> Unknown Algorithm selected!!")
            return False

        self.algorithm.initNetwork()

        return True


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
            self.algorithm.initEpisode()

            while not done:
                action = self.getNextAction(state, mode='train')
                next_state, reward, done, _ = self.env.step(action)
                reward = self.modifyReward(reward, done, step_count+1)

                self.algorithm.stepTrain(state, action, reward, next_state, done)

                state = next_state
                step_count += 1

                if self.isGoodEnough(step_count):
                    print("Episode: {}, steps: {} [Good Enough]".format(episode, step_count))
                    return

            print("Episode: {}, steps: {}".format(episode, step_count))

            loss = self.algorithm.episodeTrain()
            if loss > 0:
                print(">>>Loss after replay memory: ", loss)


    def modifyReward(self, reward, done, step_count):
        if done:
            if step_count < self.gameparams['max_stepPerEdisode']:
                return -100

        return reward


    def isGoodEnough(self, step_count):
        #if step_count > 1000:  # if step_count > 10000: # good enough
        if step_count >= self.gameparams['good_stepPerEdisode']:
            self.goodStepCount += 1

            if self.goodStepCount >= self.gameparams['goodEnoughCount']:
                return True

        else:
            self.goodStepCount = 0

        return False


    def runGamePlay(self, render_play=True):
        print("###################  GAME  START !!!  ########################")
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
        print("###################  GAME  OVER !!!  ########################")



if __name__ == "__main__":

    algorithm = 'dqn_2013'
    #algorithm = 'dqn_2015'
    play = CartPolePlay(algorithm)

    print("CartPole Algorithm {} Game start".format(algorithm))
    play.gamePlayMain()