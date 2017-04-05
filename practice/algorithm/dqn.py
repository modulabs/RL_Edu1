import math
import random
import sys
from collections import deque

import numpy as np
import tensorflow as tf

from practice.algorithm import algorithm


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

        if self.version == '2015':
            self.session.run(self.copy_ops)


    def setDefaultHyperparam(self):
        hyperparams = {}

        hyperparams['hidden_layer'] = []
        hyperparams['hidden_layer'].append({'layer_name':'W1',
                                            'size': {'dtype': 'integer', 'variable': False, 'value': 20, 'minvalue': 2,
                                                    'maxvalue': 1000} })
        hyperparams['hidden_layer'].append({'layer_name': 'W2',
                                            'size': {'dtype': 'integer', 'variable': False, 'value': 20, 'minvalue': 2,
                                                     'maxvalue': 1000}})
        hyperparams['hidden_layer'].append({'layer_name': 'W3',
                                            'size': {'dtype': 'integer', 'variable': False, 'value': 20, 'minvalue': 2,
                                                     'maxvalue': 1000}})
        hyperparams['learning_rate'] = {'dtype':'log10', 'variable':True, 'value':-2.7, 'minvalue':-7, 'maxvalue':0 }
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
            self.mainDQN = None
            self.targetDQN = None

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


    def buildNetwork(self):
        self.mainDQN = self.dqnNetwork(self.session, self.version, scopeName="mainDQN", gameparam=self.gameparams,
                                       hyperparam=self.hyperparams)
        self.mainDQN.buildNetwork()

        if self.version == '2015':
            self.targetDQN = self.dqnNetwork(self.session, self.version, scopeName="targetDQN", gameparam=self.gameparams,
                                           hyperparam=self.hyperparams)
            self.targetDQN.buildNetwork()

        self._Y = tf.placeholder(dtype=tf.float32, shape=[None, self.gameparams['output_size']], name="output_Y")

        #self._loss = tf.reduce_mean(tf.square(self._Y-self._QPred))
        self._loss = tf.reduce_mean(tf.square(self._Y - self.mainDQN._QPred))

        learning_rate = math.pow(10, self.hyperparams['learning_rate']['value'])
        print('learning_rate: ', learning_rate)
        self._train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss)

        if self.version == '2015':
            self.copy_ops = self.get_copy_var_ops(dest_scope_name=self.targetDQN.scopeName, src_scope_name=self.mainDQN.scopeName)


    def predict(self, state, net="main"):
        #print("input_size {}".format(self.gameparams['input_size']))
        x = np.reshape(state, [1, self.gameparams['input_size']])
        #return self.session.run(self._QPred, feed_dict={self._X: x})

        #dqnNetwork = None

        if net=="main":
            dqnNetwork = self.mainDQN
        else:
            dqnNetwork = self.targetDQN

        return self.session.run(dqnNetwork._QPred, feed_dict={dqnNetwork._X: x})


    def update(self, x_stack, y_stack):
        #return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
        return self.session.run([self._loss, self._train], feed_dict={self.mainDQN._X: x_stack, self._Y: y_stack})


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
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.gameparams['replay_size']:
            self.replay_buffer.popleft()


    def episodeTrain(self):
        if self.train_episode_count % 10 == 1:
            for _ in range(50):
                minibatch = random.sample(self.replay_buffer, 10)
                loss, _ = self.replay_memory_batch(minibatch)

            if self.version == '2015':
                self.session.run(self.copy_ops)

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
                if self.version == '2013':
                    Q[0, action] = reward + self.hyperparams['discount_ratio']['value'] * np.max(
                        self.predict(next_state, net="main"))
                else:   # self.version == '2015'
                    Q[0, action] = reward + self.hyperparams['discount_ratio']['value'] * np.max(
                        self.predict(next_state, net="target"))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

        return self.update(x_stack, y_stack)


    def get_copy_var_ops(self, dest_scope_name, src_scope_name):

        # Copy variables src_scope to dest_scope
        op_holder = []

        src_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder
