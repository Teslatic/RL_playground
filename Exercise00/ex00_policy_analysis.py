''' 
Reinforcement Learning Exercise 0
Authors: Nico Ott, Lior Fuks, Hendrik Vloet
Purpose: This program uses the Cart Pole environment of
openAI gym in order to get familiar with topics in 
RL like observations, rewards and policies.
2 policies are implimented and the user can parameterize
it with a small variety of parameters:
	- Random Search
	- Hill Climbing

After  each parameter search, there is a small test run
which serves to validate the found parameters


Also the user can decide if he wants to run the full
analysis of the policies stated above, or just a simple 
demonstration of a randomized cartpole policy, where the actions
are chosen depending on parameters that are chosen randomly

Wheaton's law applies


WARNING: USE PYTHON2 FOR THIS SCRIPT!

08.11.2017 V1.0 inital version
'''



#### CLASSES ####

#####################################
# Class for Random Search
#####################################

class random_search(object):

	def train(self,runs, render, show_parameters):
		print("Performing Random Search...")
		print("Total runs: {}".format(runs))
		results = []
		good_param = []
		for _ in xrange(runs):
			best_params = None
			best_reward = 0
			counter = 0
			for episode in xrange(10000):
				counter += 1
				# draw random parameters 
				parameters = np.random.rand(4)*2-1
				# run one episode with 200 timesteps
				reward = run_episode(env,parameters,render)
				# if the reward is better than the best reward reached
				# so far, update parameters
				if reward > best_reward:
					best_reward = reward
					best_params = parameters
					if reward == 200:
						good_param.append(best_params)
						results.append(counter)
						if show_parameters:
							print("Episodes needed: {:4d}".format(counter))
							print("Best reward: {}".format(reward))
							print("Used parameters: {:.4f} {:.4f} {:.4f} {:.4f}"
									.format(best_params[0] ,best_params[1]
									,best_params[2],best_params[3]))
						break
		#print(len(good_param))
		#avg_params = np.mean(good_param,axis=0)
		# call plotting functions	
		self.plot_histogram(results)
		self.plot_ecdf(results)
		
		return counter, good_param	

	# plots a histogram of the successful runs	
	def plot_histogram(self,data):
		plt.figure()
		plt.hist(data, bins=200,normed=True,color="black")
		plt.xlabel("Episodes required to reach score of 200")
		plt.ylabel("frequency")
		plt.title("Histogram of RANDOM SEARCH")

		print("Amount of successful Runs: {}".format(len(data)))
		print("Average Episodes needed to reach 200: {}".format(np.mean(data)))
		
	# plots an empirical cumulative distribution function of all successful runs
	def plot_ecdf(self,results):
		plt.figure()
		x = np.sort(results)
		y = np.arange(1,len(results)+1) / float(len(x))
		nrun= x[np.array(np.where(y>=0.5)[0][0])]
		plt.plot(x,y)
		plt.plot([nrun,nrun],[0,0.5])
		plt.title("Empirical Cumulative Distribution Function with RANDOM SEARCH")
		plt.xlabel("Episodes needed")
		plt.ylabel("ECDF")
		plt.plot([nrun],[0.5],"o", c="black")
		plt.annotate("50% of all successful training runs are\n successful after {} episodes".format(nrun), xy=(nrun,0.5), 
						xytext=(nrun+(len(x)/100.0),0.6), 
						arrowprops=dict(arrowstyle = "->" , facecolor="black"))
		plt.margins(0.02)

	def test_run(self,env, good_param, render = True, testruns=10):
		list_success = []
		print("amount of parameter set candidates: {}".format(len(good_param)))
		for i in xrange(len(good_param)):
			success = 0
			print("testing parameters {} ...".format(good_param[i]))
			for j in xrange(testruns):
				reward = run_episode(env,good_param[i],render)
				if reward == 200:
					success += 1
				#print("Achieved Reward: {}".format(reward))
			list_success.append(success)
			print("{} of {} testruns successful".format(success,testruns))
			print("... done!")
			#print("success list: {}".format(list_success))
	
		print("overall testing done!")
		array_success =np.array(list_success)
		max_val = np.max(list_success)
		indices = np.array(np.where(array_success == max_val))
		array_good_param = np.array(good_param)

		for idx,val in enumerate(indices):
			best_set = array_good_param[val]			
		#print("Best parameter sets:\n {}\n".format(best_set))

		plt.figure()
		plt.title("Barplot of Parameter Testing with RANDOM SEARCH")
		plt.ylabel("Successes")
		plt.xlabel("Index of the parameter set")
		plt.bar(np.arange(len(list_success)),list_success)

		return best_set


#####################################
# Class for Hill Climbing
#####################################

class hill_climbing(object):

	# hill climbing search with noise injection
	def train(self,runs, render, show_parameters):
		print("Performing Hill Climbing...")
		print("Total runs: {}".format(runs))
		results = []
		good_param = []
		for _ in xrange(runs):
			# initialize parameters at random
			parameters = np.random.rand(4)*2-1
			best_reward = 0
			counter = 0
			for _ in xrange(2000):
				# episode counter
				counter += 1
				# noise injection
				new_params = parameters + (np.random.rand(4)*2-1)*NOISE_SCALING
				reward = run_episode(env,new_params,render)
				# update parameters if reward is better than best reward found so far				
				if reward > best_reward:
					best_reward = reward
					parameters = new_params
					if reward == 200:
						good_param.append(parameters)
						results.append(counter)
						if show_parameters:
							print("Episodes needed: {:4d}".format(counter))
							print("Best reward: {}".format(reward))
							print("Used parameters: {:.4f} {:.4f} {:.4f} {:.4f}"
										.format(parameters[0] ,parameters[1]
										,parameters[2],parameters[3]))
						break
		#avg_params = np.mean(good_param,axis=0)
		self.plot_histogram(results)
		self.plot_ecdf(results)

		return counter, good_param


	def plot_histogram(self,data):
			plt.figure()
			plt.hist(data, bins=200,normed=True,color="black")
			plt.xlabel("Episodes required to reach score of 200")
			plt.ylabel("frequency")
			plt.title("Histogram of HILL CLIMBING")
		
			print("Amount of successful Runs: {}".format(len(data)))
			print("Average Episodes needed to reach 200: {}".format(np.mean(data)))


	def plot_ecdf(self,results):
		plt.figure()
		x = np.sort(results)
		y = np.arange(1,len(results)+1) / float(len(x))
		nrun= x[np.array(np.where(y>=0.5)[0][0])]
		plt.plot(x,y)
		plt.plot([nrun,nrun],[0,0.5])
		plt.title("Empirical Cumulative Distribution Function with HILL CLIMBING")
		plt.xlabel("Episodes needed")
		plt.ylabel("ECDF")
		plt.plot([nrun],[0.5],"o", c="black")
		plt.annotate("50% of all successful training runs are\n successful after {} episodes".format(
					nrun), xy=(nrun,0.5), xytext=(nrun+(len(x)/100.0),0.6), 
					arrowprops=dict(arrowstyle = "->" , facecolor="black"))
		plt.margins(0.02)


	def test_run(self,env, good_param, render = True, testruns=10):
		list_success = []
		print("amount of parameter set candidates: {}".format(len(good_param)))
		for i in xrange(len(good_param)):
			success = 0
			print("testing parameters {} ...".format(good_param[i]))
			for j in xrange(testruns):
				reward = run_episode(env,good_param[i],render)
				if reward == 200:
					success += 1
				#print("Achieved Reward: {}".format(reward))
			list_success.append(success)
			print("{} of {} testruns successful".format(success,testruns))
			print("... done!")
			#print("success list: {}".format(list_success))
	
		print("overall testing done!")
		array_success =np.array(list_success)
		max_val = np.max(list_success)
		indices = np.array(np.where(array_success == max_val))
		array_good_param = np.array(good_param)

		for idx,val in enumerate(indices):
			best_set = array_good_param[val]			
		#print("Best parameter sets:\n {}\n".format(best_set))
		
		plt.figure()
		plt.title("Barplot of Parameter Testing with HILL CLIMBING")
		plt.ylabel("Successes")
		plt.xlabel("Index of the parameter set")
		plt.bar(np.arange(len(list_success)),list_success)
		
		return best_set


### GLOBAL METHODS ###

### basic method to run an episode of 200 timesteps

def run_episode(env,parameters,render=False,ep=None):
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
		if POLICY_ANALYSIS == False:
			print("--------------------------------------------------------------------")
			print("Current Episode: {}".format(ep+1))
			print("Timestep {}".format(timesteps+1))
			print("Observations: {}".format(observation))
			print("Reward: {}".format(total_reward))
			print("--------------------------------------------------------------------")
		if done:
			break
	
	return total_reward

### a simple run for demonstrative purpose
### for every run, a random parameter set is drawn and applied on the cartpole environment
### this is just an application of a pure random policy and does not inclued any learning
def simple_run(runs=20):
	list_reward = []
	env = gym.make("CartPole-v0")
	for _ in xrange(runs):
		#print("-----------------------------------------------------------------------")
		#print("Episode {}".format(_+1))
		# draw random parameters
		parameters = np.random.rand(4)*2-1
		# run episode
		reward = run_episode(env,parameters,True,_)
		list_reward.append(reward)		
		print("Total achieved Reward in episode {}: {}".format(_+1,reward))
		#print("Observations: {:.4f} {:.4f} {:.4f} {:.4f}".format(observation[0],
		#		reward,observation[1], reward,observation[2], reward,observation[3]))
		print("Used parameters: {}".format(parameters))
		print("-----------------------------------------------------------------------")
	plt.figure()
	plt.plot(np.arange(1,runs+1),list_reward)
	plt.bar(np.arange(1,runs+1),list_reward)
	plt.plot(np.arange(1,runs+1),list_reward)
	plt.title("Total reward for each episode with randomized actions")
	plt.xlabel("Episode")
	plt.ylabel("Total Reward")
	plt.show()



#### USER INTERFACE####################################################################


### all values are only for a short demonstration, you can play with them as much as
### you want, but it can increase runtime.

### set this to true, to run the full random search and hill climbing analysis
### if it is false, a simpler version of a random policy will be executed
POLICY_ANALYSIS = False



# choose amount of total runs for the parameter search
RUNS = 100
# set true for enabling rendered cartpole, default= False, WARNING: INCREASES RUNTIME!
RENDER = False
# set true to show successful parameters that are found in one successfull run
# default = False
SHOW_PARAMETERS = False
# set noise scaling for noise injection in hill climbing
# a high noise scaling results in similar performance as random search!
NOISE_SCALING = 0.1

# set amount of test runs for validating found averaged parameters
TESTRUNS = 100

# show plots
SHOW_PLOTS = True

# look for the most successful parameters in random search and hill climbing
TEST_PARAM = True

#### MAIN #############################################################################


import numpy as np
import gym
import matplotlib.pyplot as plt



if POLICY_ANALYSIS:
	env = gym.make("CartPole-v0")
	rnd_search_obj = random_search()
	counter_rnd, good_param_rnd = rnd_search_obj.train(RUNS,RENDER,SHOW_PARAMETERS);
	if TEST_PARAM:
		best_set_rnd = rnd_search_obj.test_run(env, good_param_rnd, False, TESTRUNS)

	### show arbitrarily chosen parameter set from random search
	print("Show some arbitrarily chosen parameter performance from random search")
	print("Used parameters: {}".format(good_param_rnd[0]))
	for _ in xrange(5):
		reward = run_episode(env,good_param_rnd[0], True)
		print("Reward: {}".format(reward))


	hill_climbing_obj = hill_climbing()
	counter_hc, good_param_hc = hill_climbing_obj.train(RUNS,RENDER,SHOW_PARAMETERS);

	### test parameters if flag is set
	if TEST_PARAM:
		best_set_hc = hill_climbing_obj.test_run(env, good_param_hc, False, TESTRUNS)
	
	
	### show arbitrarily chosen parameter set from random search
	print("Show some arbitrarily chosen parameter performance from hill climbing")
	print("Used parameters: {}".format(good_param_hc[0]))
	for _ in xrange(5):
		reward = run_episode(env,good_param_hc[0], True)
		print("Reward: {}".format(reward))

	


	if TEST_PARAM:
		print("Best parameter sets found with random search:\n {}".format(best_set_rnd))
		print("Best parameter sets found with hill climbing:\n {}".format(best_set_hc))
	if SHOW_PLOTS:
		plt.show()
else:
	simple_run()
	






#### CODE DUMPSTER - IGNORE EVERYTHING BELOW -###### 
'''



def test_run(env,avg_params,render=True,testruns=10):
	success = 0
	print("testing parameters...")
	for r in xrange(testruns):
		reward = run_episode(env,avg_params,render=True)
		if reward ==200:
			success += 1		
		print("Achieved reward: {}".format(reward))
	print("{} of {} testruns successful".format(success,testruns))
	print("testing done!")


def test_run(env, good_param, render = True, testruns=10):
	list_success = []
	print("amount of good parameter sets: {}".format(len(good_param)))
	for i in xrange(len(good_param)):
		success = 0
		print("testing parameters {} ...".format(good_param[i]))
		for j in xrange(testruns):
			reward = run_episode(env,good_param[i],render)
			if reward == 200:
				success += 1
			#print("Achieved Reward: {}".format(reward))
		list_success.append(success)
		print("{} of {} testruns successful".format(success,testruns))
		print("... done!")
		print("success list: {}".format(list_success))
	
	print("overall testing done!")
	array_success =np.array(list_success)
	max_val = np.max(list_success)
	indices = np.array(np.where(array_success == max_val))
	array_good_param = np.array(good_param)

	for idx,val in enumerate(indices):
		print("Best parameter sets:\n {}\n".format(array_good_param[val]))
	
	plt.figure()
	plt.title("Barplot of Parameter Testing with HILL CLIMBING")
	plt.ylabel("Successes")
	plt.xlabel("Index of the parameter set")
	plt.bar(np.arange(len(list_success)),list_success)
	


# random search 	
def random_search(RUNS=1000, RENDER=False, SHOW_PARAMETERS=False):
	results = []
	good_param = []
	for _ in xrange(RUNS):
		best_params = None
		best_reward = 0
		counter = 0
		for episode in xrange(10000):
			counter += 1
			parameters = np.random.rand(4)*2-1
			reward = run_episode(env,parameters,RENDER)
			if reward > best_reward:
				best_reward = reward
				best_params = parameters
				if reward == 200:
					good_param.append(best_params)
					results.append(counter)
					if SHOW_PARAMETERS:
						print("Episodes needed: {:4d}".format(counter))
						print("Best reward: {}".format(reward))
						print("Used parameters: {:.4f} {:.4f} {:.4f} {:.4f}"
								.format(best_params[0] ,best_params[1]
								,best_params[2],best_params[3]))
					break
	print(len(good_param))
	avg_params = np.mean(good_param,axis=0)	
	plot_histogram(results)
	plot_ecdf(results)
		
	return counter, avg_params		

# hill climbing search with noise injection
def hill_climbing(runs, render, show_parameters):
	results = []
	good_param = []
	for _ in xrange(runs):
		# initialize parameters at random
		parameters = np.random.rand(4)*2-1
		best_reward = 0
		counter = 0
		for _ in xrange(2000):
			# episode counter
			counter += 1
			# noise injection
			new_params = parameters + (np.random.rand(4)*2-1)*NOISE_SCALING
			reward = run_episode(env,new_params,render)
			if reward > best_reward:
				best_reward = reward
				parameters = new_params
				if reward == 200:
					good_param.append(parameters)
					results.append(counter)
					if show_parameters:
						print("Episodes needed: {:4d}".format(counter))
						print("Best reward: {}".format(reward))
						print("Used parameters: {:.4f} {:.4f} {:.4f} {:.4f}"
									.format(best_params[0] ,best_params[1]
									,best_params[2],best_params[3]))
					break
	avg_params = np.mean(good_param,axis=0)
	plot_histogram(results)
	plot_ecdf(results)

	return counter, avg_params


def plot_histogram(data):
	plt.figure()
	plt.hist(data, bins=200,normed=True,color="black")
	plt.xlabel("Episodes required to reach score of 200")
	plt.ylabel("frequency")
	if HC:
		plt.title("Histogram of HILL CLIMBING")
	if RND:
		plt.title("Histogram of RANDOM SEARCH")

	print("Average Episodes needed to reach 200: {}".format(np.mean(data)))
	print("Amount of Runs: {}".format(len(data)))


def plot_ecdf(results):
	plt.figure()
	x = np.sort(results)
	y = np.arange(1,len(results)+1) / float(len(x))
	nrun= x[np.array(np.where(y>=0.5)[0][0])]
	plt.plot(x,y)
	plt.plot([nrun,nrun],[0,0.5])
	if RND:
		plt.title("Empirical Cumulative Distribution Function with RANDOM SEARCH")
	elif HC:
		plt.title("Empirical Cumulative Distribution Function with HILL CLIMBING")
	plt.xlabel("Episodes needed")
	plt.ylabel("ECDF")
	plt.plot([nrun],[0.5],"o", c="black")
	plt.annotate("50% of all successful training runs are\n successful after {} episodes".format(nrun), xy=(nrun,0.5), 						xytext=(nrun+(len(x)/100.0),0.6), arrowprops=dict(arrowstyle = "->" , facecolor="black"))
	plt.margins(0.02)


if RND:
	print("performing random search...")
	counter,avg_params = random_search(RUNS, RENDER, SHOW_PARAMETERS)
	test_run(env, avg_params, RENDER, TESTRUNS)
if HC:
	print("performing hill climbing...")
	counter, avg_params = hill_climbing(RUNS, RENDER, SHOW_PARAMETERS)
	test_run(env, avg_params, RENDER, TESTRUNS)

plt.show()




counter,avg_params = random_search(RUNS, RENDER, SHOW_PARAMETERS)
test_run(env, avg_params, RENDER, TESTRUNS)

plt.show()



 hill climbing

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
