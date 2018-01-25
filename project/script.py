#!/usr/bin/python3
import gym
import sys
from pendulum import PendulumEnv, angle_normalize
import random
from random import randrange
import numpy as np



class Discretizer():
    def __init__(self,min_val,max_val):
        self.min_val = min_val
        self.max_val = max_val
        self._check_inputs()
        self._discretize()

    def _check_inputs(self):
        self.step_length = input("Enter desired step length of the action-space that shall be discretized [0..1]: ")
        self.step_length = float(self.step_length)
        assert self.step_length<=env.max_torque, "step length must be between 0 and {}".format(env.max_torque)
        assert self.min_val<self.max_val, "min value has to be greater or equal to the max value"
        assert self.step_length>0, "step length hast to be greater than zero!"

    def _discretize(self):
        self.interval = np.array([self.min_val,self.max_val])
        self.ticks = (np.abs(self.min_val) + np.abs(self.max_val)) / self.step_length
        self.discrete_actions = np.linspace(self.min_val,self.max_val,self.ticks)
        print("I discretized your action-space for you:")
        print("|---------------------------------------------------------------|")
        print("|Action Interval {}\t| Ticks: {:.0f}\t| Delta {:.4f}\t|".format(self.interval,self.ticks,np.diff(self.discrete_actions)[-1]))
        print("|---------------------------------------------------------------|")



####### INTIALISATION ##########################################################

env = PendulumEnv()
slicer = Discretizer(-env.max_torque,env.max_torque)

######### CONSTANTS ############################################################

SHOW_PROGRESS = 500
EPISODES = 10000
TIMESTEPS = 200

####### GLOBAL VARIABLES #######################################################

total_reward = 0
episode_reward = 0
success = 0

state = env.reset()
for ep in range(1,EPISODES+1):
    #state = env.reset()
    if ep % SHOW_PROGRESS == 0 and ep != 0:
        print("| Episode: {}\t| Reward: {}".format(ep,episode_reward))
    for t in range(TIMESTEPS):
        env.render()
        #next_state, reward, _, _ = env.step(random.choice(slicer.discrete_actions))
        next_state,reward,_,_ = env.step(-2)
        #print("env state [0]",env.state[0])
        #print("normalize function:",angle_normalize(env.state[0]))
        if reward == 1:
            episode_reward += 1
            success += 1
            break
        state = next_state
    total_reward += episode_reward
    episode_reward = 0
print("Total reward over all episodes: {}\t| Success rate: {:.2f}%".format(total_reward,(success/EPISODES)*100))
