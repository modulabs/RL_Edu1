import gym
from gym import wrappers
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
if "../../" not in sys.path:
  sys.path.append("../../")
from lib import plotting
import os

def sarsa(env, n_episodes=2000, gamma=0.99, alpha=0.5, e=0.1, best_enabled=False):
    nS = env.observation_space.n
    nA = env.action_space.n

    if best_enabled:
        # record your best-tuned hyperparams here
        env.seed(0)
        np.random.seed(0)
        alpha = 0.05
        gamma = 0.99
        e = 1.0

    Q = np.zeros([nS, nA])
    policy = make_epsilon_greedy_policy(Q, nA)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(n_episodes),
        episode_rewards=np.zeros(n_episodes))

    for i in range(n_episodes):
        # useful for debuggin
        log_episode(i, n_episodes)

        s = env.reset()
        done = False
        total_reward = 0

        probs = policy(s, e)
        a = np.random.choice(np.arange(nA), p=probs)

        for t in itertools.count():
            # Choose action by decaying e-greedy
            # take a step
            next_s, r, done, _ = env.step(a)

            next_probs = policy(next_s, e)
            next_a = np.random.choice(np.arange(nA), p=next_probs)

            if best_enabled:
                mod_r = modify_reward(r, done)
                td_target = mod_r + gamma * Q[next_s, next_a]
            else:
                td_target = r + gamma * Q[next_s, next_a]

            td_delta = td_target - Q[s, a]
            Q[s, a] += alpha * td_delta

            s = next_s
            a = next_a
            total_reward += r

            if done:
                break

        # Update statistics
        stats.episode_rewards[i] += total_reward
        stats.episode_lengths[i] = t

    return Q, stats


def make_epsilon_greedy_policy(Q, nA):
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



def modify_reward(reward, done):
    # 100.0 being arbitrary scaling factors
    if done and reward == 0:
        return -100.0
    elif done:
        return 100.0
    else:
        return -1.0


def log_episode(i_epi, n_epi):
    if (i_epi + 1) % 100 == 0:
        print("\rEpisode {}/{}.".format(i_epi + 1, n_epi), end="")
        sys.stdout.flush()


def is_solved(stats, target, interval):
    """
    checks if openai's criteria has been met
    """
    # FrozenLake-v0 is considered "solved" when the agent
    # obtains an average reward of at least 0.78 over 100
    # consecutive episodes.
    avg_reward = np.sum(stats.episode_rewards)/len(stats.episode_rewards)
    print("Average reward : {}".format(avg_reward))

    def moving_avg(x, n=100):
        return np.convolve(x, np.ones((n,))/n, mode='valid')

    ma = moving_avg(stats.episode_rewards, interval)
    peaks = np.where(ma > target)[0]
    if len(peaks) > 0:
        print("solved after {} episodes".format(peaks[0]))
        return True
    else:
        print("did not pass the openai criteria")
        return False


if __name__ == "__main__":
    TARGET_AVG_REWARD = 0.78
    TARGET_EPISODE_INTERVAL = 100
    env = gym.make('FrozenLake-v0')
    env = wrappers.Monitor(env, '/tmp/frozenlake-experiment-1', force=True)
    Q, stats = sarsa(env)
    env.close()

    if is_solved(stats, TARGET_AVG_REWARD, TARGET_EPISODE_INTERVAL):
        OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
        gym.upload('/tmp/frozenlake-experiment-1', api_key=OPENAI_API_KEY)
