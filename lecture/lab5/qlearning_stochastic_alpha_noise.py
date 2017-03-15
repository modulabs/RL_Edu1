import gym
from gym import wrappers
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def qlearning_alpha_noise(env, n_episodes=2000, gamma=0.99, alpha=0.85):
    nS = env.observation_space.n
    nA = env.action_space.n

    print("Q space initialized: {} x {}".format(nS, nA))

    Q = np.zeros([nS, nA])
    # policy: pi(state) -> prob. distribution of actions
    policy = make_decay_noisy_policy(Q, nA)
    reward_per_episode = np.zeros(n_episodes)

    for i in range(n_episodes):
        # useful for debugging
        if (i + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i + 1, n_episodes), end="")
            sys.stdout.flush()

        s = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Choose action by noisy probs
            probs = policy(s, i)
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

def make_decay_noisy_policy(Q, nA):
    def policy_fn(state, episode_i):
        noise = np.random.randn(1, nA) / (episode_i + 1)
        # don't manually break ties as being of equal values is unlikely
        dist = Q[state, :]
        best_action = np.argmax(dist + noise)
        # make the policy deterministic as per argmax
        return np.eye(nA, dtype=float)[best_action]
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
    env = wrappers.Monitor(env, '/tmp/frozenlake-experiment-0', force=True)
    Q, stats = qlearning_alpha_noise(env, n_episodes=100)

    env.close()
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    gym.upload('/tmp/frozenlake-experiment-0', api_key=OPENAI_API_KEY)

    visualize(Q, stats, "qlearning_alpha_e_greedy.png")

