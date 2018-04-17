import numpy as np
import gym
import time
from practice.lake_envs import *
from tqdm import tqdm
from matplotlib.pyplot import plot


def learn_Q_QLearning(env, num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function for. Must have nS, nA, and P as
      attributes.
      P[state][action] is tuples with (probability, nextstate, reward, terminal)
    num_episodes: int
      Number of episodes of training.
    gamma: float
      Discount factor. Number in range [0, 1)
    learning_rate: float
      Learning rate. Number in range [0, 1)
    e: float
      Epsilon value used in the epsilon-greedy method.
    decay_rate: float
      Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################

    Q = np.zeros((env.nS, env.nA))
    policy = np.zeros((env.nS, env.nA))
    avarage = []
    for _ in tqdm(range(num_episodes)):
        S = env.reset()
        done = False
        score = 0
        count = 0
        while not done:
            max_a = -1
            max_q = -np.inf
            for action in range(env.nA):
                q = Q[S][action]
                if q > max_q:
                    max_q = q
                    max_a = action
                policy[S][action] = e / env.nA
            policy[S][max_a] = 1 - e + e / env.nA
            # print(policy[S])
            A = np.random.choice(a = range(env.nA), p=policy[S])
            Q_SA = Q[S][A]
            max_Q_ = -np.inf
            S_, reward, done, _ = env.step(A)
            for action in range(env.nA):
                if Q[S_][action] > max_Q_:
                    max_Q_ = Q[S_][action]
            Q[S][A] = Q_SA + lr * (reward + gamma * max_Q_ - Q_SA)
            score += Q[S][A]
            count += 1
            S = S_
        # print(score / count)
        avarage.append(score/count)
    # plot(avarage[:1000])
    return Q


def render_single_Q(env, Q):
    """Renders Q function once on environment. Watch your agent play!

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play Q function on. Must have nS, nA, and P as
        attributes.
      Q: np.array of shape [env.nS x env.nA]
        state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print("Episode reward: %f" % episode_reward)


# Feel free to run your own debug code in main!
def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    Q = learn_Q_QLearning(env)
    render_single_Q(env, Q)


if __name__ == '__main__':
    main()
