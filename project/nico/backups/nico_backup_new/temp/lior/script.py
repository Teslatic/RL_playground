#!/usr/bin/python3

import os
from os import path
import sys
if "../" not in sys.path:
    sys.path.append("../")

import matplotlib.pyplot as plt
import gym
from pendulum import PendulumEnv, angle_normalize
import random
from random import randrange
import numpy as np
from pendulumAgent import DankAgent
import datetime
import time
from assets.plotter.PolarHeatmapPlotter import PolarHeatmapPlotter




def print_timestamp(string = ""):
    now = datetime.datetime.now()
    print(string + now.strftime("%Y-%m-%d %H:%M"))

class Discretizer():
    def __init__(self,min_val,max_val):
        self.min_val = min_val
        self.max_val = max_val
        self._check_inputs()
        #self._discretize()

    def _check_inputs(self):
        self.step_length = input("Enter desired step length of the action-space that shall be discretized [0..1]: ")
        self.step_length = float(self.step_length)
        assert self.step_length<=env.max_torque, "step length must be between 0 and {}".format(env.max_torque)
        assert self.min_val<self.max_val, "min value has to be greater or equal to the max value"
        assert self.step_length>0, "step length hast to be greater than zero!"

'''    def _discretize(self):
        self.interval = np.array([self.min_val,self.max_val])
        self.ticks = (np.abs(self.min_val) + np.abs(self.max_val)) / self.step_length
        self.discrete_actions = np.linspace(self.min_val,self.max_val,self.ticks)
        print("I discretized your action-space for you:")
        print("|---------------------------------------------------------------|")
        print("|Action Interval {}\t| Ticks: {:.0f}\t| Delta {:.4f}\t|".format(self.interval,self.ticks,np.diff(self.discrete_actions)[-1]))
        print("|---------------------------------------------------------------|")
'''


####### INTIALISATION ##########################################################
weights_file = "network.h5"
exp_directory = 'LiorsCode'
file_index = 0
while path.exists(exp_directory + '%s' % file_index):
    file_index += 1

idx_file_dir = exp_directory + '{}'.format(file_index)
# train_parameters['EXPERIMENT_NAME'] = idx_file_dir
os.makedirs(idx_file_dir)
os.makedirs(idx_file_dir+'/results/policy_plots')
os.makedirs(idx_file_dir+'/results/parameters')
os.makedirs(idx_file_dir+'/results/plots')
print_timestamp('Started experiment {}'.format(idx_file_dir))

env = PendulumEnv()
env.cnt += 1
state = env.reset()
state = state.reshape((1,3))

action_dim = 25
action_high=env.action_space.high[0]
action_low=env.action_space.low[0]
action_sequence = np.linspace(action_low, action_high, action_dim)

agent = DankAgent(env.observation_space.shape[0], action_dim)
agent.model.summary()

nepisodes = 0
######## CONSTANTS ############################################################

def test_run(agent):
    reward_list = []
    for i in range(50):
        episode_reward = 0
        env.seed(i)
        state = env.reset()
        state = state.reshape((1,3))
        for step in range(500):
            #env.render()
            action = agent.act(state, False)
            next_state, reward, done , _ = env.step(action_sequence[action], True)
            next_state = next_state.reshape((1,3))
            state = next_state
            episode_reward += reward
        reward_list.append(episode_reward)
    avg_reward = np.mean(reward_list)
    print("average reward",avg_reward)
    return avg_reward

batch = []
TRAINING_EPISODES = 500
steps = 200
AUTO_SAVER = 25
SHOW_PROGRESS = 25
TEST_PROGRESS = 10
avg_reward_list = []
list_episode_reward = []

for ep in range(TRAINING_EPISODES):
    if ep % AUTO_SAVER == 0 and nepisodes != 0:
        print_timestamp("saved")
        agent.save(weights_file)
    state = env.reset()
    state = state.reshape((1,3))
    episode_reward = 0

    if (ep+1) % TEST_PROGRESS == 0 and ep != 0:
            avg_reward_list.append(test_run(agent))
            print_timestamp('Plotting')
            heat = PolarHeatmapPlotter(2, agent.model, idx_file_dir)
            heat.plot(ep, avg_reward_list[-1])

    for step in range(steps):
        #if ep % SHOW_PROGRESS == 0 and ep != 0:
        if False:
            env.render()
            action = agent.act(state, False)
        else:
            action = agent.act(state, True)

        next_state, reward, done , _ = env.step(action_sequence[action])
        next_state = next_state.reshape((1,3))

        agent.memory_store(state, action, reward, next_state, done)

        if(ep > 15):
            sample_index = np.random.choice(agent.memory_size, size=agent.batch_size)
            batch = agent.memory[sample_index, :]
            agent.train(batch)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon -= 0.00001

        state = next_state
        episode_reward += reward


    if(ep>15):
        print("Episode {}/{}\t| Step: {}\t| Reward: {}\t| epsilon: {:.2f}\t|buffer: {}".format(ep, TRAINING_EPISODES,steps,episode_reward, agent.epsilon, agent.memory_counter ))


    list_episode_reward.append(episode_reward)


    nepisodes += 1

plt.figure()
plt.plot(list_episode_reward,label = "DankAgent")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()

plt.figure()
plt.plot(avg_reward_list,label = "DankAgent")
plt.xlabel("Episode")
plt.ylabel("Average Test Reward")
plt.legend()

plt.show()
