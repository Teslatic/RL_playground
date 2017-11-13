import numpy as np


#newWorld[i, j] += actionProb[i][j][action] * (actionReward[i][j][action] + discount * world[newPosition[0], newPosition[1]])


# create gridworld
WORLD_SIZE = 5
world = np.zeros((WORLD_SIZE,WORLD_SIZE))

# discount
discount = 0.9

# define actions up,right,down,left
actions = ["U","R","D","L"]

# every action is chosen with uniform probability
action_prob = []
for i in range(0, WORLD_SIZE):
    action_prob.append([])
    for j in range(0, WORLD_SIZE):
        action_prob[i].append(dict({'L':0.25, 'U':0.25, 'R':0.25, 'D':0.25}))




# create rules and rewards
# some memory for last state and next state needed
# also reward of the actions must be listed
next_state = []
action_reward = []

for i in range(0, WORLD_SIZE):
	next_state.append([])
	action_reward.append([])
	for j in range(0, WORLD_SIZE):
		next = dict()
		reward = dict()
		if i == 0:
			next["U"] = [i,j]
			reward["U"] = -1.0
		else:
			next["U"] = [i-1,j]
			reward["U"] = 0.0
		if j == WORLD_SIZE -1:
			next["R"] = [i,j]
			reward["R"] = -1.0
		else:
			next["R"] = [i,j+1]
			reward["R"] = 0.0
		if i == WORLD_SIZE -1:
			next["D"] = [i,j]
			reward["D"] = -1.0
		else:
			next["D"] = [i+1,j]
			reward["D"] = 0.0
		if j == 0:
			next["L"] = [i,j]
			reward["L"] = -1.0
		else:
			next["L"] = [i,j+1]
			reward["L"] = 0.0
		# setting teleport from A to A'
		if [i,j] == [0,1]:
			next["U"] = next["R"] = next["D"] = next["L"] = [4,1]
			reward["U"] = reward["R"] = reward["D"] = reward["L"] = 10.0
		# setting teleport from B to B'
		if [i,j] == [0,3]:
			next["U"] = next["R"] = next["D"] = next["L"] = [2,3]
			reward["U"] = reward["R"] = reward["D"] = reward["L"] = 5.0
	
		next_state[i].append(next)
		action_reward[i].append(reward)

while(1):
	new_world = np.zeros((WORLD_SIZE,WORLD_SIZE))
	for i in range(0,WORLD_SIZE):
		for j in range(0,WORLD_SIZE):
			for action in actions:
				new_position = next_state[i][j][action]
				new_world[i, j] += action_prob[i][j][action] * (action_reward[i][j][action] + discount 					* world[new_position[0], new_position[1]])
					

