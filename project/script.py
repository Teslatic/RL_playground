#!/usr/bin/python3
import gym
import sys
from pendulum import PendulumEnv, angle_normalize
import random
from random import randrange
import numpy as np
from pendulumAgent import DankAgent
import datetime
import time
import matplotlib.pyplot as plt



def print_timestamp(string = ""):
    now = datetime.datetime.now()
    print(string + now.strftime("%Y-%m-%d %H:%M"))



####### INTIALISATION ##########################################################
weights_file = "network.h5"
load_existent_model = False
env = PendulumEnv()
env.cnt += 1
agent = DankAgent([-env.max_torque,env.max_torque])
agent.model.summary()
if load_existent_model:
    agent.load(weights_file)


nepisodes = 0
######## CONSTANTS ############################################################
#def: 50000
MEMORY_SIZE = 80000
memory = []
#def: 30000
TRAINING_EPISODES = 3000
AUTO_SAVER = 50
SHOW_PROGRESS = 10
# def: 2000
TIMESTEPS = 1000

list_episode_reward = []
for ep in range(TRAINING_EPISODES):
    if ep % AUTO_SAVER == 0 and nepisodes != 0:
        print_timestamp("saved")
        agent.save(weights_file)
    state = env.reset()
    state = np.array((state[0],state[2]))
    state = np.round(state,2)
    episode_reward = 0
    for t in range(TIMESTEPS):
        if ep % SHOW_PROGRESS == 0 and ep != 0:
            env.render()
            print("\rRendering episode {}/{}".format(ep,TRAINING_EPISODES),end="")
            sys.stdout.flush()

        action = agent.act(state, True)[0]


        next_state, reward, done , _ = env.step(agent.discrete_actions[action])
        next_state = np.array((next_state[0],next_state[2]))
        next_state = np.round(next_state,2)

        if len(memory) == MEMORY_SIZE:
            memory.pop(0)
        memory.append((state, action, reward, next_state, done))
        batch = random.choice(memory)
        agent.train(batch)
        state = next_state
        episode_reward += reward
    print("Episode {}/{}\t| Step: {}\t| Reward: {:.4f}\t| Loss: {:.4f}\t| epsilon: {:.2f}\t|buffer: {}".format(ep, TRAINING_EPISODES,t,episode_reward,
            agent.history.history['loss'][0],agent.epsilon,len(memory)))

        # print("Episode {}/{}\t| Step: {}\t| Reward: {}\t".format(ep, TRAINING_EPISODES,t,episode_reward))
    agent.epsilon = agent.init_epsilon*np.exp(-agent.eps_decay_rate*ep)+agent.decay_const


    list_episode_reward.append(episode_reward)


    nepisodes += 1
    agent.update_target_model()

plt.figure()
plt.plot(list_episode_reward,label = "DankAgent")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()
