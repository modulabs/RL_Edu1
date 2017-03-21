import numpy as np
import tensorflow as tf

import gym

env = gym.make('CartPole-v0')


learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

X = tf.placeholder(shape=[1,input_size],dtype=tf.float32)
W1 = tf.get_variable("W1",shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())

Qpred =  tf.matmul(X, W1)

Y = tf.placeholder(shape=[1,output_size],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

num_episodes = 2000
dis = .9

rList = []

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
    e = 1. / ((i/10) + 1)
    rAll = 0
    s = env.reset()

    done = False

    while not done:
        #env.render()
        x = np.reshape(s, [1, input_size])

        Qs = sess.run(Qpred, feed_dict={X:x})

        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else :
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)

        if done :
            Qs[0,a] = -100
        else :
            x1 = np.reshape(s1,[1,input_size])
            Qs1 = sess.run(Qpred, feed_dict={X:x1})
            Qs[0,a] = reward + dis * np.max(Qs1)
        sess.run(train, feed_dict={X:x, Y:Qs})

        rAll += reward
        s = s1

    print(i, ':reward=', rAll )
    rList.append(rAll)

#counts per episode: 26.7965
print('counts per episode:', str(sum(rList) / num_episodes ) )

'''
count = 0

while count < 100:
    s = env.reset()
    count += 1
    done = False
    rAll = 0

    while not done:
        env.render()
        x = np.reshape(s,[1,input_size])
        Qs = sess.run(Qpred, feed_dict={X:x})
        a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        rAll += reward

        if done:
            break;
        else:
            s = s1

    print('Test:',count, ':', rAll)
'''
