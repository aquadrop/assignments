### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from assignment1.lake_envs import *

np.set_printoptions(precision=3)

def policy_evaluation(P, nS, V, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	for i in range(max_iteration):
		delta = 0
		for state in range(nS):
			action = policy[state]
			tuples = P[state][action]
			v = 0
			for tuple_ in tuples:
				v += tuple_[0] * (tuple_[2] + gamma * V[tuple_[1]])
			delta = max(delta, np.abs(v - V[state]))
			V[state] = v
		if delta < tol:
			break
	return V


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	new_policy = np.zeros(nS, dtype='int')
	for state in range(nS):
		max_v = -np.inf
		max_a = -1
		for action in range(nA):
			tuples = P[state][action]
			vv = 0
			for t in tuples:
				vv += t[0] * (t[2] + gamma * value_from_policy[t[1]])
			if vv > max_v:
				max_v = vv
				max_a = action
		new_policy[state] = max_a
	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	for i in range(max_iteration):
		policy_stable = True
		V = policy_evaluation(P, nS, V, policy, gamma=gamma, max_iteration=max_iteration, tol=tol)
		print(V)
		new_policy = policy_improvement(P, nS, nA, V, policy, gamma)
		for state in range(nS):
			old_action = policy[state]
			if old_action != new_policy[state]:
				policy_stable = False
		policy = new_policy
		print(policy)
		if policy_stable:
			break
	return V, policy

def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	delta = 0
	for i in range(max_iteration):
		for state in range(nS):
			# max_a sum_{s',r}p(s',r|s,a)[r + \gammaV(s')]
			max_v = -np.inf
			for action in range(nA):
				candiate = 0
				# sweep across all s' and r
				tuples = P[state][action]
				for tuple_ in tuples:
					candiate += tuple_[0] * (tuple_[2] + gamma * V[tuple_[1]])
				if candiate > max_v:
					max_v = candiate
			# update
			delta = max(delta, np.abs(max_v - V[state]))
			V[state] = max_v
		if delta < tol:
			break
	# update policy
	for state in range(nS):
		max_a = -1
		max_v = -np.inf
		for action in range(nA):
			candiate = 0
			# sweep across all s' and r
			tuples = P[state][action]
			for tuple_ in tuples:
				candiate += tuple_[0] * (tuple_[2] + gamma * V[tuple_[1]])
			if candiate > max_v:
				max_v = candiate
				max_a = action
		policy[state] = max_a
	return V, policy

def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0);
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print("Episode reward: %f" % episode_reward)


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	# env = gym.make("Deterministic-4x4-FrozenLake-v0")
	env = gym.make("Stochastic-4x4-FrozenLake-v0")
	print(env.__doc__)
	# print("Here is an example of state, action, reward, and next state")
	# example(env)
	print(env.P)
	# V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
	render_single(env, p_pi)