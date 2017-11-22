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
		
		
		T  o  o  o
		o  x  o  o
		o  o  o  o
		o  o  o  T

	Returns:
		Vector of length env.nS representing the value function.
	
		
	"""
	# define action set
	actions = np.arange(0,4)
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
				# inline Bellmann expectation equation
				v_new[state] += policy[state][action]*(env.P[state][action][0][0]
									*env.P[state][action][0][2] + discount_factor
									*V[env.P[state][action][0][1]])
		# compare delta to parameter theta, if small enough -> break
		if np.sum(np.abs(V-v_new)) < theta:
			#print("Random policy")
			#print(np.round(v_new))
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
	V = np.zeros(env.nS)
	# Start with a random policy
	values_candidates = []
	v_max_indices = []
	num_max = []
	# [0.25,0.25,0.25,0.25]
	policy = np.ones([env.nS, env.nA]) / env.nA
	#print(policy)
	V = policy_eval(policy,env)
	states = np.arange(env.nS)
	actions = np.arange(0,4)
	#print(actions)
	#while True:
	#print(V)
	for state in states:
		values_candidates = []
		for action in actions:
			values_candidates.append(V[env.P[state][action][0][1]])
		v_max_indices.append(np.where(values_candidates == np.max(values_candidates)))
		values_candidates = []
		num_max.append(len(v_max_indices[state][0]))
		new_policy = np.zeros([env.nS,env.nA])
			
		#print(v_max_indices[state][0])	
		new_prob = 1/num_max[state]
		#print(new_prob)
		#print(new_prob , policy[state][v_max_indices[state][0]])
		new_policy[state][v_max_indices[state][0]] = new_prob
		policy[state] = new_policy[state][:]
	print(policy)
	
	V = policy_eval(policy,env)
	
		# Implement this!
		#break

	return policy, V

