import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

input_size = env.observation_space.n
output_size = env.action_space.n

#learning_rate = .01  #성공률 0.103
learning_rate = .1  #성공률 0.485
#learning_rate = .3  #성공률 0.412
#learning_rate = .5  #성공률 0.015

def one_hot(x):
    return np.identity(input_size)[x:x+1]

X = tf.placeholder(shape=[1,input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))

Qpred = tf.matmul(X,W)

Y = tf.placeholder(shape=[1,output_size], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Y - Qpred))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

dis = .99
num_episodes = 2000

rList = []
subRewardList = []
avgRewardList = []


init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)

    for i in range(num_episodes):
        s = env.reset()
        e = 1. / ((i/50) + 10)  #실행하면서 낮춰주도록 함
        rAll = 0
        done = False
        local_loss = []

        count = 0

        while not done:
            Qs = sess.run(Qpred, feed_dict={X:one_hot(s)})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else :
                a = np.argmax(Qs)

            s1, reward, done, _ = env.step(a)

            if done:
                Qs[0,a] = reward
            else:
                Qs1 = sess.run(Qpred, feed_dict={X:one_hot(s1)})
                Qs[0,a] = reward + dis * np.max(Qs1)

            sess.run(train, feed_dict={X:one_hot(s), Y:Qs})

            rAll += reward
            s = s1
            count += 1

        subRewardList.append(rAll)
        rList.append(rAll)

        if (i+1) % 100 == 0 :
            avg = sum(subRewardList)/100
            avgRewardList.append(avg)
            print(i+1,' episode =', rAll, ', avg =', avg)
            subRewardList = []

print('Percent of success:', str(sum(rList) / num_episodes ) )

plt.ylim(0,1)
plt.bar(range(len(avgRewardList)), avgRewardList, color='blue', bottom=0)

plt.show()
