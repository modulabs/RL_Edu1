# the same as lab4_v2 except the env = v0, stochastic
import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
from lib import plotting

def qlearning_noise(env, n_episodes=2000, gamma=0.95):
    print("Q space initialized: {} x {}".format(env.nS, env.nA))

    Q = np.zeros([env.nS, env.nA])
    # policy: pi(state) -> prob. distribution of actions
    policy = make_noisy_policy(Q, env.nA)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(n_episodes),
        episode_rewards=np.zeros(n_episodes))

    for i in range(n_episodes):
        # useful for debugging
        if (i + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i + 1, n_episodes), end="")
            sys.stdout.flush()

        s = env.reset()
        done = False
        total_reward = 0

        for t in itertools.count():
            # Choose action by noisy probs
            probs = policy(s, i)
            a = np.random.choice(np.arange(env.nA), p=probs)
            # take a step
            next_s, r, done, _ = env.step(a)

            # backup Q, no alpha
            td_target = r + gamma * np.max(Q[next_s, :])
            Q[s, a] = td_target

            s = next_s
            total_reward += r
            if done:
                break

        # Update statistics
        stats.episode_rewards[i] += total_reward
        stats.episode_lengths[i] = t

    return Q, stats

def make_noisy_policy(Q, nA):
    def policy_fn(state, episode_i):
        noise = np.random.randn(1, env.nA) / (episode_i + 1)
        # don't manually break ties as being of equal values is unlikely
        # Q[state,:] lives on [0, 1]
        dist = Q[state, :]
        best_action = np.argmax(dist + noise)
        # make the policy deterministic as per argmax
        return np.eye(env.nA, dtype=float)[best_action]
    return policy_fn


def visualize(Q, stats, output_title="output.png"):
    success_rate = np.sum(stats.episode_rewards)/len(stats.episode_rewards)
    print("Success rate : {}".format(success_rate))
    print("Final Q-Table Values")
    print(Q)
    plt.figure(figsize=(8,12))
    plt.title("Reward_per_episode")
    plt.plot(stats.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(output_title)

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    Q, stats = qlearning_noise(env)
    visualize(Q, stats, "qlearning_noise.png")

