import numpy as np
import time
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
	"""
	Evaluate a policy given an environment and a full description of the environment's dynamics.

	Args:
		policy: [S, A] shaped matrix representing the policy.
		env: OpenAI env. env.P represents the transition probabilities of the environment.
		env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
		theta: We stop evaluation once our value function change is less than theta for all states.
		discount_factor: gamma discount factor.
		
		
	Returns:
		Vector of length env.nS representing the value function.
	
		
	"""
	# define set of actions 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
	actions = np.arange(env.nA)
	# initialize V(s) with zeros
	V = np.zeros(env.nS)
	# learning loop
	while True:
		# reset inline result
		v_new = np.zeros(env.nS)
		# loop over all states
		for state in range(env.nS):
			# loop over all actions
			for action in actions:
				# Bellmann Expectation Equation
				v_new[state] += policy[state][action]*(env.P[state][action][0][0]
									*env.P[state][action][0][2] + discount_factor
									*V[env.P[state][action][0][1]])
		# compare delta to parameter theta, if small enough -> break
		if np.sum(np.abs(V-v_new)) < theta:
			break
		# update old value function
		V = v_new
	return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
	"""
	Policy Improvement Algorithm. Iteratively evaluates and improves a policy
	until an optimal policy is found.

	Args:
	env: The OpenAI envrionment.
	policy_eval_fn: Policy Evaluation function that takes 3 arguments:
	policy, env, discount_factor.
	discount_factor: Lambda discount factor.

	Returns:
	A tuple (policy, V). 
	policy is the optimal policy, a matrix of shape [S, A] where each state s
	contains a valid probability distribution over actions.
	V is the value function for the optimal policy.

	"""
	
	# initialize V(s) = 0
	V = np.zeros(env.nS)
	# Start with a random policy
	policy = np.ones([env.nS, env.nA]) / env.nA
	# initialize a comparison policy
	old_policy = np.ones([env.nS, env.nA]) / env.nA
	# define set of actions 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
	actions = np.arange(env.nA)
	# set up state space
	states = np.arange(env.nS)
	# loop until convergence
	while True:
		# Step 1: Do one policy evaluation on initial policy and return the value function
		V = policy_eval_fn(policy,env)
		# loop over all states		
		for state in states:
			# update old policy of state s
			old_policy[state] = policy[state]
			# initialize value candidate list for arg max with respect to actions
			values = []
			# loop over all actions
			for action in actions:
				# generate a list of values for state s
				values.append(env.P[state][action][0][0]*(env.P[state][action][0][2]+discount_factor*
												V[env.P[state][action][0][1]]))
				# reset current state policy to zero
				policy[state] = 0
				# apply argmax in order to get a deterministic policy (for any MDP there exists a deterministic policy that is optimal)
				policy[state][np.argmax(values)] = 1
		# if the old policy and current policy are equal, than according to the policy improvement theorem, the found policy is as good as or better than the old policy -> break the loop
		if np.array_equal(policy,old_policy):
			break
		
	return policy, V	
		
	
