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
def _discretize_actions(self):
    self.ticks = (np.abs(self.action_interval[0]) + np.abs(self.action_interval[-1])) / self.step_length
    self.discrete_actions = np.linspace(self.action_interval[0],self.action_interval[-1],self.ticks)
    print("I discretized your action-space for you:")
    print("|---------------------------------------------------------------|")
    print("|Action Interval {}\t| Ticks: {:.0f}\t| Delta {:.4f}\t|".format(self.action_interval,self.ticks,np.diff(self.discrete_actions)[-1]))
    print("|---------------------------------------------------------------|")
    return self.discrete_actions

for ep in range(TRAINING_EPISODES):
    if ep % AUTO_SAVER == 0 and nepisodes != 0:
        print_timestamp("saved")
        agent.save(weights_file)
    state = env.reset()

    for t in range(TIMESTEPS):
        env.render()

        next_state, reward, done , _ = env.step(random.choice(agent.discrete_actions))
        next_state = np.array((next_state[0],next_state[2]))
        # next_state = np.round(next_state,2)
        print(next_state)

        state = next_state
