######################################################################################################
# Import packages
######################################################################################################
import gym
import numpy as np
import matplotlib.pyplot as plt


######################################################################################################
# Global constant settings
######################################################################################################

FALSE = 0
TRUE = 1
LEFT = 0
RIGHT = 1
FOREVER = 0

######################################################################################################
# Definition of functions
######################################################################################################

# Returns a resetted CartPole environment
def setup_CPv0():
	env = gym.make('CartPole-v0')
	env.reset()
	return env

# Creates a vector of default size 4 with random values from -limit to +limit (Can be change for arbitrary length later on)
def create_random_parameters(limit=1, size=4):
	return np.random.rand(size)*(limit*2)-limit


#Starts an RL session on an environment and returns found parameters for a given policy 
def run_CPv0(env, goal_steps, episodes, render=TRUE, limit = 1, console=TRUE):
# Define the parameters, and the rewards
	best_reward = 0
	total_reward = 0
	best_parameters = None
	episode_counter = 0
	reward_history = []
	episode_counter_history = []
	
	# WHEN EPISODES == FOREVER: Simulation will continue until one run is succesful	
	if episodes == FOREVER: #if episodes = 0
		while best_reward < (goal_steps):
			episode_counter +=1
			parameters = create_random_parameters(limit)
			if console:
				print("----------------------------------------------------------------")
				print("Episode {} started".format(episode_counter))
				print("----------------------------------------------------------------")
			observation = env.reset()
			total_reward = 0
			# Start episode
			for t in range(goal_steps):
				if render:
					env.render()
				state = observation # Assuming fully observable environment and no model
				action, parameters = policy(env, state, parameters, 'deterministic') # Select action from policy
				observation, reward, done, info = env.step(action) # Perform action on environment and
				total_reward += reward # add actual reward on total reward
				
				# Print actual information
				if console:
					print("Episode {}, timestep {}.".format(episode_counter, t+1))	
					print("Current observation: {}".format(observation))
					print("Current reward: {}".format(reward))
					print("Current total reward: {}".format(total_reward))
					print_action(action)
					print("----------------------------------------------------------------")
				if done:
					if console:
						print("Episode finished after {} timesteps".format(t+1))
					break

			# Prepare episode history for 
			reward_history.append(total_reward)
			episode_counter_history.append(episode_counter)
			# Discard old parameters and best reward, if new parameters perform better
			if total_reward > best_reward:
				best_reward = total_reward
				best_parameters = parameters

		#Present final results:
		print("Solved in episode {}".format(episode_counter))
		print("Best reward {} and goal steps: {} with parameters: {}".format(best_reward, goal_steps, best_parameters))
		plt.plot(episode_counter_history, reward_history)
		plt.title("Total reward for each episode with random search")
		plt.ylabel("Total reward")
		plt.xlabel("Episode")
		plt.show()

######################################################################################################
# May or may not be adapted later
######################################################################################################
#	else:
#		for i_episode in range(episodes):
#			parameters = create_random_parameters(limit)
#			print("Episode {} started".format(i_episode+1))
#			observation = env.reset()
#			for t in range(goal_steps):
#				if render:
#					env.render()
#				state = observation # Assuming fully observable environment
#				action, parameters = policy(env, state, parameters, 'deterministic')
#				if console:
#					print_action(action)
#				observation, reward, done, info = env.step(action)
#				actual_reward += reward
#				if done:
#					print("Episode finished after {} timesteps".format(t+1))
#					break
#			if actual_reward > best_reward:
#				best_reward = actual_reward
#				best_parameters = parameters

# Selects and runs the policy
def policy(env, state, parameters, pol_type = 'deterministic'):
	if pol_type == 'deterministic':
		action = LEFT if np.matmul(parameters,state) < 0 else RIGHT 
		return action, parameters
	if pol_type == 'random':
		return env.action_space.sample()
	elif pol_type == 'stochastic':
		pass
	else:
		print("Policy type unknown")	

# Small function for more convenience
def print_action(action):
	if action == RIGHT:
		print("Action taken: Going right")
	if action == LEFT:
		print("Action taken: Going left")


### model based agent:

def update_model():
	pass

def predict_next_state():
	pass

def predict_next_reward():
	pass

### Plotting functions

def plot_episodes():
	pass


######################################################################################################
### Parameter settings
######################################################################################################

LIMIT = 4.0
GOAL_STEPS = 200
#EPISODES = 100
EPISODES = FOREVER
RENDER = TRUE
CONSOLE = TRUE


######################################################################################################
### Main starts here
######################################################################################################

env = setup_CPv0()
run_CPv0(env, GOAL_STEPS, EPISODES, RENDER, LIMIT, CONSOLE)


######################################################################################################
# Code dumpster
######################################################################################################
#for t in range(1000):
#	env.render()
#	print(observation)
#	action = env.action_space.sample()
#	observation, reward, done, info = env.step(action)
#	print(info)
#	if done:
#		print("Episodes finished after {} timesteps".format(t+1))
#		break

# observation = env.reset()


#print(env.action_space) #Discrete(2)
#print(env.observation_space) #Box(4,)
#	observation = env.reset()
#	for t in range(100):
#		env.render()
#		print(observation)
#		Insert the Policy and action choosing here and select action instead of sampling from action space
#		action = env.action_space.sample()
#		observation, reward, done, info = env.step(action)
#		if done:
#			print("Episodes finished after {} timesteps".format(t+1))
#			break
