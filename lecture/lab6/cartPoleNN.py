import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt


env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
layer1_size = 10
layer2_size = 10

output_size = env.action_space.n
#learning_rate = 1e-1 # counts per episode: 9.5085
learning_rate = 1e-2 # counts per episode: 95.3945

X = tf.placeholder(shape=[1,input_size],dtype=tf.float32)
#W1 = tf.Variable(tf.random_uniform([input_size,layer1_size],-1,1))
W1 = tf.get_variable("W1",shape=[input_size, layer1_size], initializer=tf.contrib.layers.xavier_initializer())


Z1 = tf.nn.sigmoid(tf.matmul(X,W1))
#Z1 = tf.nn.tanh(tf.matmul(X,W1))

#W2 = tf.Variable(tf.random_uniform([layer1_size, output_size],-1,1))
W2 = tf.get_variable("W2",shape=[layer1_size, layer2_size], initializer=tf.contrib.layers.xavier_initializer())

Z2 = tf.nn.sigmoid(tf.matmul(Z1,W2))
#Z2 = tf.nn.tanh(tf.matmul(Z1,W2))

#W2 = tf.Variable(tf.random_uniform([layer1_size, output_size],-1,1))
W3 = tf.get_variable("W3",shape=[layer2_size, output_size], initializer=tf.contrib.layers.xavier_initializer())


Qpred =  tf.matmul(Z2, W3)

Y = tf.placeholder(shape=[1,output_size],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Y - Qpred))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
#train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


dis = .9
num_episodes = 2000

rList = []
subRewardList = []
avgRewardList = []

init = tf.global_variables_initializer()

#with tf.Session() as sess:
sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
    e = 1. / (i/10 + 1)
    s = env.reset()

    rAll = 0
    done = False

    while not done:
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


    subRewardList.append(rAll)
    rList.append(rAll)

    if (i+1) % 100 == 0 :
        avg = sum(subRewardList)/100
        avgRewardList.append(avg)
        print(i+1, ': avg =', avg)
        subRewardList = []

#layer2: learning rate = 0.01, sigmoid, counts per episode: 36.436
#layer2: learning rate = 0.01, tanh, counts per episode: ?

#layer3: learning rate = 0.01, sigmoid, counts per episode: 38.1825
#layer3: learning rate = 0.01, tanh counts per episode: 84.9365
#layer3: learning rate = 0.001, counts per episode: 96.272

print('counts per episode:', str(sum(rList) / num_episodes ) )

plt.ylim(0,300)
plt.bar(range(len(avgRewardList)), avgRewardList, color='blue', bottom=0)

plt.show()


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







