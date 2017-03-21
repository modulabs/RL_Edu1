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


def double_qlearning(env, n_episodes=2000, gamma=0.99, alpha=0.85, best_enabled=False, log_by_step=False):
    nS = env.observation_space.n
    nA = env.action_space.n

    if best_enabled:
        # record your best-tuned hyperparams here
        env.seed(0)
        np.random.seed(0)
        alpha = 0.13
        gamma = 0.999
        epsilon_decay = 0.95
        e = 1.0

    Q1 = np.zeros([nS, nA])
    Q2 = np.zeros([nS, nA])
    policy = make_decay_e_greedy_policy(Q1, Q2, nA)

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

        if best_enabled:
            e *= epsilon_decay
        else:
            e = 1.0 / ((i / 10) + 1.0)

        q_update = None
        q_other = None

        for t in itertools.count():
            # Choose action by decaying e-greedy
            probs = policy(s, e)
            a = np.random.choice(np.arange(nA), p=probs)
            # take a step
            next_s, r, done, _ = env.step(a)

            # With 0.5 probability
            if np.random.choice(2) == 1:
                q_update = Q1
                q_other = Q2
            else:
                q_update = Q2
                q_other = Q1

            if best_enabled:
                mod_r = modify_reward(r, done)
                td_target = mod_r + gamma * np.max(q_other[next_s, :])
            else:
                td_target = r + gamma * np.max(q_other[next_s, :])

            td_delta = td_target - q_update[s, a]
            q_update[s, a] += alpha * td_delta

            s = next_s
            total_reward += r

            if done:
                break

            if log_by_step:
                print('===============Episode (%d) Iter (%d)===============' % (i, t))
                #print(Q1)
                #print(Q2)
                print(np.concatenate((Q1, Q2), axis=1))
                print('Action : %d, next state : %d, reward : %d' % (a, s, r))

        # Update statistics
        stats.episode_rewards[i] += total_reward
        stats.episode_lengths[i] = t

    return Q1, Q2, stats



# epsilon-soft policy p.109 of Sutton's
def make_decay_e_greedy_policy(Q1, Q2, nA):
    def policy_fn(state, epsilon):
        # give every action an equal prob of e / n(A)
        A = np.ones(nA, dtype=float) * epsilon / nA
        # random argmax
        m = np.max(Q1[state, :]+Q2[state, :])
        max_indices = np.where(Q1[state, :]+Q2[state, :]==m)[0]
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
    Q1, Q2, stats = double_qlearning(env, best_enabled=True)
    #Q1, Q2, stats = double_qlearning(env, n_episodes=10, best_enabled=True, log_by_step=True)
    print(np.concatenate((Q1, Q2), axis=1))
    env.close()

    if is_solved(stats, TARGET_AVG_REWARD, TARGET_EPISODE_INTERVAL):
        print('SOLVED!!!')
        #OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
        #gym.upload('/tmp/frozenlake-experiment-1', api_key=OPENAI_API_KEY)
