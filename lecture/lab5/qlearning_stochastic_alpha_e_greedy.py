import gym
from gym import wrappers
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def qlearning_alpha_e_greedy(env, n_episodes=2000, gamma=0.99, alpha=0.85):
    nS = env.observation_space.n
    nA = env.action_space.n
    print("Q space initialized: {} x {}".format(nS, nA))

    Q = np.zeros([nS, nA])
    # policy: pi(state) -> prob. distribution of actions
    policy = make_decay_e_greedy_policy(Q, nA)
    reward_per_episode = np.zeros(n_episodes)

    for i in range(n_episodes):
        # useful for debugging
        if (i + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i + 1, n_episodes), end="")
            sys.stdout.flush()

        s = env.reset()
        done = False
        total_reward = 0

        # extremely large variance depending on decaying speed, given n_epi=2000
        e = 1.0 /((i//100) + 1.0)


        while not done:
            # Choose action by decaying e-greedy
            probs = policy(s, e)
            a = np.random.choice(np.arange(nA), p=probs)
            # take a step
            next_s, r, done, _ = env.step(a)

            # backup Q
            td_target = r + gamma * np.max(Q[next_s, :])
            td_delta = td_target - Q[s, a]
            Q[s, a] += alpha * td_delta

            s = next_s
            total_reward += r

        reward_per_episode[i] = total_reward
    return Q, reward_per_episode

def make_decay_e_greedy_policy(Q, nA):
    def policy_fn(state, epsilon):
        # give every action an equal prob of e / n(A)
        A = np.ones(nA, dtype=float) * epsilon / nA
        # random argmax
        m = np.max(Q[state, :])
        max_indices = np.where(Q[state, :]==m)[0]
        best_action = np.random.choice(max_indices)
        # give the best action a bump prob of 1 - e
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def visualize(Q, stats, output_title="output.png"):
    print("Success rate : {}".format(np.sum(stats)/len(stats)))
    print("Final Q-Table Values")
    print(Q)
    plt.figure(figsize=(8,12))
    plt.title("Reward_per_episode")
    plt.plot(stats)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(output_title)

if __name__ == "__main__":

    env = gym.make('FrozenLake-v0')
    env = wrappers.Monitor(env, '/tmp/frozenlake-experiment-1', force=True)
    Q, stats = qlearning_alpha_e_greedy(env, n_episodes=100, gamma=0.5, alpha=0.5)
    env.close()
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    gym.upload('/tmp/frozenlake-experiment-1', api_key=OPENAI_API_KEY)
    visualize(Q, stats, "qlearning_e_greedy.png")

