# the same as lab4_v2 except the env = v0, stochastic
import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
if "../../" not in sys.path:
  sys.path.append("../../")

from lib import plotting

def qlearning_noise(env, n_episodes=2000, gamma=0.95):
    nS = env.observation_space.n
    nA = env.action_space.n
    print("Q space initialized: {} x {}".format(nS, nA))

    Q = np.zeros([nS, nA])
    # policy: pi(state) -> prob. distribution of actions
    policy = make_noisy_policy(Q, nA)

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
            a = np.random.choice(np.arange(nA), p=probs)
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
        noise = np.random.randn(1, nA) / (episode_i + 1)
        # don't manually break ties as being of equal values is unlikely
        # Q[state,:] lives on [0, 1]
        dist = Q[state, :]
        best_action = np.argmax(dist + noise)
        # make the policy deterministic as per argmax
        return np.eye(nA, dtype=float)[best_action]
    return policy_fn

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    Q, stats = qlearning_noise(env)
    avg_reward = np.sum(stats.episode_rewards)/len(stats.episode_rewards)
    print("Average reward : {}".format(avg_reward))


