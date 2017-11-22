import numpy as np
import time



def value_iteration(env, theta=0.0001, discount_factor=1.0):
	"""
	Value Iteration Algorithm.

	Args:
		env: OpenAI environment. env.P represents the transition probabilities of the environment.
		theta: Stopping threshold. If the value of all states changes less than theta
		in one iteration we are done.
		discount_factor: lambda time discount factor.
		
	Returns:
		A tuple (policy, V) of the optimal policy and the optimal value function.        
	"""

	# initalize array V with zero
	V = np.zeros(env.nS)
	# initialize polocy array with zeros
	policy = np.zeros([env.nS, env.nA])
	# define set of actions 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
	actions = np.arange(env.nA)
	# set up state space
	states = np.arange(env.nS)
	
	# loop until convergence
	while True:
		# initialize some delta
		delta = 0.0
		# loop over all states
		for state in states:
			# initialize/reset values list
			values = []
			# loop over all actions
			for action in actions:
				# generate a list of values for state s (Bellmann Optimality Equation)
				values.append(env.P[state][action][0][0]*(env.P[state][action][0][2]+
											  discount_factor*V[env.P[state][action][0][1]]))
			# pick greedily the highest value for the new value
			new_value = np.max(values)
			# calculate the delta, which we compare to the parameter theta in order to execute an epsilon convergence
			delta += np.abs(V[state]-new_value)
			# update value of state under consideration with found maximum value
			V[state] = new_value
		# if |old_values - new_values| < theta -> break loop, because convergence criterion is met
		if delta < theta:
			break
	# loop over all states
	for state in states:
		# initialize/reset values list
		values = []
		# loop over all actions
		for action in actions:
			# generate a list of values for state s (Bellmann)
			values.append(env.P[state][action][0][0]*(env.P[state][action][0][2]+
											  discount_factor*V[env.P[state][action][0][1]]))
		# reset policy in state s to zero, because we are acting greedy. The best value function, i.e. largest, is chosen.
		# If the found value is not unique for this state, the first value and the according action is preferred (greedy)
		policy[state] = 0
		policy[state][np.argmax(values)] = 1
		
	return policy, V
