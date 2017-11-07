import numpy as np
import gym
import matplotlib.pyplot as plt



#### FUNCTIONS ############################################

def run_episode(env,parameters):
	observation = env.reset()
	total_reward = 0
	for timesteps in xrange(200):
		if RENDER:
			env.render()
		if np.dot(parameters,observation)<0:
			action=0
		else:
			action=1
		observation, reward, done, info = env.step(action)
		total_reward += reward
		if done:
			break
	return total_reward

def plot_histogram(data):
	plt.figure()
	plt.hist(data, bins=np.unique(data),normed=True)
	plt.xlabel("Episodes required to reach score of 200")
	plt.ylabel("frequency")
	plt.title("Histogram of Search")
	print("Average Episodes needed to reach 200: {}".format(np.mean(data)))
	print("Amount of Runs: {}".format(len(data)))
	plt.show()



	
def random_search(RUNS=1000, RENDER=False, SHOW_PARAMETERS=False):
	results = []
	for _ in xrange(RUNS):
		best_params = None
		best_reward = 0
		counter = 0
		for episode in xrange(10000):
			counter += 1
			parameters = np.random.rand(4)*2-1
			reward = run_episode(env,parameters)
			if reward > best_reward:
				best_reward = reward
				best_params = parameters
				if reward == 200:
					results.append(counter)
					if SHOW_PARAMETERS:
						print("Episodes needed: {:4d}".format(episode))
						print("Best reward: {}".format(reward))
						print("Used parameters: {:.4f} {:.4f} {:.4f} {:.4f}"
								.format(best_params[0] ,best_params[1]
								,best_params[2],best_params[3]))
					break
	plot_histogram(results)
	return counter			







#### STATICS #########################################################
RUNS = 100
RENDER = False
SHOW_PARAMETERS = False

#### MAIN #############################################################

env = gym.make("CartPole-v0")


random_search(RUNS, RENDER, SHOW_PARAMETERS)




results = []


for _ in xrange(RUNS):
	noise_scaling = 0.5
	best_reward = 0
	parameters = np.random.rand(4)*2 -1
	counter = 0
	for _ in xrange(2000):
		counter += 1
		new_params = parameters + (np.random.rand(4)*2-1)*noise_scaling
		reward = run_episode(env,new_params)
		if reward > best_reward:
			best_reward = reward
			parameters = new_params
			if reward == 200:
				results.append(counter)
				break
			



#for episode in xrange(RUNS):
#	best_reward = 0
#	counter = 0
#	reward = 0
#	for timestep in xrange(2000):
#		counter += 1
#		new_params = parameters + (np.random.rand(4)*2-1)*noise_scaling
#		reward = run_episode(env,new_params)
#		#print(reward)
#		if reward > best_reward:
#			best_reward = reward
#			parameters = new_params
#			if reward == 200:
#				results.append(counter)
#				break
print(results)
plt.figure()
plt.hist(results , bins="auto", normed=True)
plt.xlabel("Episodes required to reach score of 200")
plt.ylabel("frequency")
plt.title("Histogram of Search")
print("Average Episodes needed to reach 200: {}".format(np.mean(results)))
print("Amount of Runs: {}".format(RUNS))
plt.show()
