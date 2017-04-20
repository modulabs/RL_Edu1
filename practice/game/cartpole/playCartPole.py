import gym
import tensorflow as tf

from practice.algorithm.reinforcementLearning.dqn import dqn
from practice.game import gameplay


class CartPolePlay(gameplay.GamePlay):
    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.algorithm = None
        self.session = None
        self.env = None
        self.gameparam = {}
        self.goodStepCount = 0


    def gamePlayMain(self):
        with tf.Session() as sess:
            self.session = sess
            if self.gameSetup():
                self.trainAgent()
                self.runGamePlay()


    def setAlgorithm(self):
        hyperparam = {}

        if self.algorithm_name == 'dqn_2013':
            self.algorithm = dqn(self.session, version='2013', gameparam=self.gameparam, externalHyperparam=hyperparam)
        elif self.algorithm_name == 'dqn_2015':
            self.algorithm = dqn(self.session, version='2015', gameparam=self.gameparam, externalHyperparam=hyperparam)
        else:   # TODO: add more good algorithms!!
            self.algorithm = None

        return self.algorithm


    def gameSetup(self):
        self.env = gym.make('CartPole-v0')

        self.gameparam['max_episode'] = 5000
        self.gameparam['input_size'] = self.env.observation_space.shape[0]
        self.gameparam['output_size'] = self.env.action_space.n
        self.gameparam['max_stepPerEdisode'] = 200
        self.gameparam['good_stepPerEdisode'] = 195
        self.gameparam['goodEnoughCount'] = 100

        print("gameSetup input_size {}".format(self.gameparam['input_size']))
        print("gameSetup output_size {}".format(self.gameparam['output_size']))

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

        for episode in range(self.gameparam['max_episode']):
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

                if step_count >= self.gameparam['max_stepPerEdisode']:
                    break

            if self.isGoodEnough(step_count):
                print("Episode: {}, steps: {} [Good Enough]".format(episode, step_count))
                return
            else:
                print("Episode: {}, steps: {}".format(episode, step_count))

            loss = self.algorithm.episodeTrain()
            if loss > 0:
                print(">>>Loss after replay memory: ", loss)


    def modifyReward(self, reward, done, step_count):
        if done:
            if step_count < self.gameparam['max_stepPerEdisode']:
                return -100

        return reward


    def isGoodEnough(self, step_count):
        #if step_count > 1000:  # if step_count > 10000: # good enough
        if step_count >= self.gameparam['good_stepPerEdisode']:
            self.goodStepCount += 1

            if self.goodStepCount >= self.gameparam['goodEnoughCount']:
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

    #algorithm = 'dqn_2013'
    algorithm = 'dqn_2015'
    play = CartPolePlay(algorithm)

    print("CartPole Algorithm {} Game start".format(algorithm))
    play.gamePlayMain()