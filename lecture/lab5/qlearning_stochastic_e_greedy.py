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

def qlearning_e_greedy(env, n_episodes=2000, gamma=0.95):
    nS = env.observation_space.n
    nA = env.action_space.n
    print("Q space initialized: {} x {}".format(nS, nA))

    Q = np.zeros([nS, nA])
    # policy: pi(state) -> prob. distribution of actions
    policy = make_decay_e_greedy_policy(Q, nA)
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

        e = 1.0 /((i/10) + 1.0)

        for t in itertools.count():
            # Choose action by decaying e-greedy
            probs = policy(s, e)
            a = np.random.choice(np.arange(nA), p=probs)
            # take a step
            next_s, r, done, _ = env.step(a)

            # backup Q
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

def make_decay_e_greedy_policy(Q, nA):
    def policy_fn(state, epsilon):
        # give every action an equal prob of e / n(A)
        A = np.ones(nA, dtype=float) * epsilon / nA
        # random argmax to break ties randomly
        m = np.max(Q[state, :])
        max_indices = np.where(Q[state, :]==m)[0]
        best_action = np.random.choice(max_indices)
        # give the best action a bump prob of 1 - e
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    Q, stats = qlearning_e_greedy(env)
    avg_reward = np.sum(stats.episode_rewards)/len(stats.episode_rewards)
    print("Average reward : {}".format(avg_reward))
