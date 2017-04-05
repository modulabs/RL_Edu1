import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import math
import random
from collections import deque

if "../../" not in sys.path:
  sys.path.append("../../")

from lib import algorithm


class dqn(algorithm.Algorithm):
    def __init__(self, session, version='2013', gameparam={}, externalHyperparam={}):
        self.session = session
        self.version = version
        #self.net_name = name
        self.hyperparams = self.setDefaultHyperparam()
        self.hyperparams.update(externalHyperparam)
        self.gameparams = self.setDefaultGameparam()
        self.gameparams.update(gameparam)
        self.network_initialized = False
        self.train_episode_count = 0
        self.replay_buffer = deque()


    def initNetwork(self):
        self.buildNetwork()
        tf.global_variables_initializer().run()
        self.network_initialized = True


    def setDefaultHyperparam(self):
        hyperparams = {}
        #self.hyperparams['hidden_size_1'] = {'dtype':'integer', 'variable':False, 'value':30, 'minvalue':2, 'maxvalue':1000 }
        hyperparams['hidden_layer'] = []
        hyperparams['hidden_layer'].append({'layer_name':'W1',
                                            'size': {'dtype': 'integer', 'variable': False, 'value': 30, 'minvalue': 2,
                                                    'maxvalue': 1000} })
        hyperparams['hidden_layer'].append({'layer_name': 'W2',
                                            'size': {'dtype': 'integer', 'variable': False, 'value': 30, 'minvalue': 2,
                                                     'maxvalue': 1000}})
        hyperparams['learning_rate'] = {'dtype':'log10', 'variable':True, 'value':-3, 'minvalue':-7, 'maxvalue':0 }
        hyperparams['discount_ratio'] = {'dtype':'float', 'variable':True, 'value':0.9, 'minvalue':0, 'maxvalue':1 }

        return hyperparams


    def setDefaultGameparam(self):
        gameparams = {}
        gameparams['input_size'] = 30
        gameparams['output_size'] = 4
        gameparams['continuous_state'] = 'Y'
        gameparams['continuous_action'] = 'N'
        gameparams['replay_size'] = 50000

        return gameparams


    class dqnNetwork():
        def __init__(self, session, version, scopeName="mainDQN", gameparam={}, hyperparam={}):
            self.session = session
            self.version = version
            self.scopeName = scopeName
            self.hyperparams = hyperparam
            self.gameparams = gameparam

        def buildNetwork(self):
            with tf.variable_scope(self.scopeName):
                self._X = tf.placeholder(dtype=tf.float32, shape=[None, self.gameparams['input_size']], name="input_X")

                Lprev = self._X
                size_prev = self.gameparams['input_size']
                for idx in range(len(self.hyperparams['hidden_layer'])):
                    W = tf.get_variable(name=self.hyperparams['hidden_layer'][idx]['layer_name'],
                                        shape=[size_prev, self.hyperparams['hidden_layer'][idx]['size']['value']],
                                        initializer=tf.contrib.layers.xavier_initializer())
                    # L = tf.nn.tanh(tf.matmul(Lprev, W))
                    L = tf.nn.relu(tf.matmul(Lprev, W))  # ReLu is much more effective than tanh in dqn.
                    Lprev = L
                    size_prev = self.hyperparams['hidden_layer'][idx]['size']['value']

                W_last = tf.get_variable(name=(self.scopeName + '_last'),
                                         shape=[size_prev, self.gameparams['output_size']],
                                         initializer=tf.contrib.layers.xavier_initializer())
                self._QPred = tf.matmul(Lprev, W_last)

        def placeHolderX(self):
            return self._X

        def qPrediction(self):
            return self._QPred




    def buildNetwork(self):
        self.mainDQN = self.dqnNetwork(self.session, self.version, scopeName="mainDQN", gameparam=self.gameparams, hyperparam=self.hyperparams)
        self.mainDQN.buildNetwork()
        '''with tf.variable_scope("mainDQN"):
            self._X = tf.placeholder(dtype=tf.float32, shape=[None, self.gameparams['input_size']], name="input_X")

            Lprev = self._X
            size_prev = self.gameparams['input_size']
            for idx in range(len(self.hyperparams['hidden_layer'])):
                W = tf.get_variable( name=self.hyperparams['hidden_layer'][idx]['layer_name'],
                                     shape=[size_prev, self.hyperparams['hidden_layer'][idx]['size']['value']],
                                     initializer=tf.contrib.layers.xavier_initializer())
                #L = tf.nn.tanh(tf.matmul(Lprev, W))
                L = tf.nn.relu(tf.matmul(Lprev, W))   # ReLu is much more effective than tanh in dqn.
                Lprev = L
                size_prev = self.hyperparams['hidden_layer'][idx]['size']['value']

            W_last = tf.get_variable(name=(self.net_name+'_last'),
                                shape=[size_prev, self.gameparams['output_size']],
                                initializer=tf.contrib.layers.xavier_initializer())
            self._QPred = tf.matmul(Lprev, W_last)'''

        self._Y = tf.placeholder(dtype=tf.float32, shape=[None, self.gameparams['output_size']], name="output_Y")

        #self._loss = tf.reduce_mean(tf.square(self._Y-self._QPred))
        self._loss = tf.reduce_mean(tf.square(self._Y - self.mainDQN.qPrediction()))

        learning_rate = math.pow(10, self.hyperparams['learning_rate']['value'])
        print('learning_rate: ', learning_rate)
        self._train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss)


    def predict(self, state):
        #print("input_size {}".format(self.gameparams['input_size']))
        x = np.reshape(state, [1, self.gameparams['input_size']])
        #return self.session.run(self._QPred, feed_dict={self._X: x})
        return self.session.run(self.mainDQN.qPrediction(), feed_dict={self.mainDQN.placeHolderX(): x})


    def update(self, x_stack, y_stack):
        #return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
        return self.session.run([self._loss, self._train], feed_dict={self.mainDQN.placeHolderX(): x_stack, self._Y: y_stack})


    def getNextAction(self, state, mode):
        if mode == 'train':
            if np.random.rand(1) < self.getEpsilon():
                return None

        return np.argmax(self.predict(state))


    def initTraining(self):
        self.train_episode_count = 0


    def initEpisode(self):
        self.train_episode_count += 1


    def stepTrain(self, state, action, reward, next_state, done):
        #self.train_episode_count += 1
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.gameparams['replay_size']:
            self.replay_buffer.popleft()


    def episodeTrain(self):
        if self.train_episode_count % 10 == 1:
            for _ in range(50):
                minibatch = random.sample(self.replay_buffer, 10)
                loss, _ = self.replay_memory_batch(minibatch)

            return loss

        return 0


    def getEpsilon(self):
        return 1. / ((self.train_episode_count / 10) + 1)


    def replay_memory_batch(self, replay_batch):
        x_stack = np.empty(0).reshape(0, self.gameparams['input_size'])
        y_stack = np.empty(0).reshape(0, self.gameparams['output_size'])

        for state, action, reward, next_state, done in replay_batch:
            Q = self.predict(state)

            if done:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + self.hyperparams['discount_ratio']['value'] * np.max(self.predict(next_state))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

        return self.update(x_stack, y_stack)

    def replay_memory_batch_2015(self, replay_batch):
        x_stack = np.empty(0).reshape(0, self.gameparams['input_size'])
        y_stack = np.empty(0).reshape(0, self.gameparams['output_size'])

        for state, action, reward, next_state, done in replay_batch:
            Q = self.predict(state)

            if done:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + self.hyperparams['discount_ratio']['value'] * np.max(self.predict(next_state))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

        return self.update(x_stack, y_stack)
