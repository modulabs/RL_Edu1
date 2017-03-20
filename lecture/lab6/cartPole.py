import gym

env = gym.make('CartPole-v0')

num_episodes = 2000
rAll = 0
rList = []

for i in range(num_episodes) :

    s= env.reset()
    done = False

    while not done:
        #env.render()
        action = env.action_space.sample()
        s, reward, done, info = env.step(action)
        rAll += reward
        if done:
            i += 1
            print(i, ' : reward= ', rAll)
            rList.append(rAll)
            rAll = 0
            env.reset()

#counts per episode: 22.069
print('counts per episode:', str(sum(rList) / num_episodes ) )


