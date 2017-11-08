import numpy as np
import gym
import matplotlib.pyplot as plt



#### FUNCTIONS ############################################




	
def random_search(RUNS=1000, RENDER=False, SHOW_PARAMETERS=False):
	results = []
	for _ in xrange(RUNS):
		best_params = None
		best_reward = 0
		counter = 0
		for episode in xrange(10000):
			counter += 1
			parameters = np.random.rand(4)*2-1
			reward = run_episode(env,parameters,RENDER, check_params=False)
			if reward > best_reward:
				best_reward = reward
				best_params = parameters
				if reward == 200:
					results.append(counter)
					if SHOW_PARAMETERS:
						print("Episodes needed: {:4d}".format(counter))
						print("Best reward: {}".format(reward))
						print("Used parameters: {:.4f} {:.4f} {:.4f} {:.4f}"
								.format(best_params[0] ,best_params[1]
								,best_params[2],best_params[3]))
					break
	plot_histogram(results)
	plot_ecdf(results)	
	return counter			

def run_episode(env,parameters,render=False, check_params=False):
	observation = env.reset()
	total_reward = 0
	for timesteps in xrange(200):
		if render:
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
	plt.hist(data, bins=200,normed=True,color="black")
	plt.xlabel("Episodes required to reach score of 200")
	plt.ylabel("frequency")
	plt.title("Histogram of Search")
	print("Average Episodes needed to reach 200: {}".format(np.mean(data)))
	print("Amount of Runs: {}".format(len(data)))


def plot_ecdf(results):
	plt.figure()
	x = np.sort(results)
	y = np.arange(1,len(results)+1) / float(len(x))
	nrun= x[np.array(np.where(y>=0.5)[0][0])]


	plt.plot(x,y)
	plt.plot([nrun,nrun],[0,0.5])
	plt.title("Empirical Cumulative Distribution Function")
	plt.xlabel("Episodes needed")
	plt.ylabel("ECDF")
	plt.plot([nrun],[0.5],"o", c="black")
	plt.annotate("50% of all runs are successful after\n {} episodes".format(nrun), xy=(nrun,0.5), 						xytext=(nrun+20,0.6), arrowprops=dict(arrowstyle = "->" , facecolor="black"))
	plt.margins(0.02)
	
	



#### STATICS ##########################################################################
# choose amount of total runs for the parameter search
RUNS = 5000
# set true for enabling rendered cartpole, default= False
RENDER = False
# set true to show successful parameters that are found in one successfull run
# default = False
SHOW_PARAMETERS = False
# set noise scaling for noise injection in hill climbing
NOISE_SCALING = 0.5

CHECK_PARAMS = False

#### MAIN #############################################################################

env = gym.make("CartPole-v0")


random_search(RUNS, RENDER, SHOW_PARAMETERS)
plt.show()


'''
results = []
found_weights = []
success = 0

for r in xrange(RUNS):
	best_reward = 0
	parameters = np.random.rand(4)*2 -1
	counter = 0
	for _ in xrange(2000):
		counter += 1
		new_params = parameters + (np.random.rand(4)*2-1)*NOISE_SCALING
		reward = run_episode(env,new_params,render=False)
		if reward > best_reward:
			best_reward = reward
			parameters = new_params
			if reward == 200:
				results.append(counter)
				print("episode {}, good parameters found".format(r))
				#print(parameters)
				found_weights.append(parameters)
				break
			
print(len(found_weights))
print(len(results))

avg_parameters = np.array(np.mean(found_weights,axis=0))
print("Averaged parameters: {}".format(avg_parameters))
for _ in xrange(100):
	reward = run_episode(env, avg_parameters, render=False, check_params = CHECK_PARAMS)
	if(reward == 200):
		print("episode run successful")
		success += 1
print("Sucessfull runs with averaged parameters: {} from {}".format(success, _))


#break
#print(results)
#plt.figure()
#plt.hist(results , bins="auto", normed=True)
#plt.xlabel("Episodes required to reach score of 200")
#plt.ylabel("frequency")
#plt.title("Histogram of Search")
#print("Average Episodes needed to reach 200: {}".format(np.mean(results)))
#print("Amount of Runs: {}".format(RUNS))
#plt.show()

'''
