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

from pendulumAgent import DankAgent, print_timestamp
import argparse
import sys
#sys.path.append('..')
parser = argparse.ArgumentParser()
import random
import datetime



env = PendulumEnv()
env.cnt += 1

test_agent = DankAgent([-env.max_torque,env.max_torque])


parser.add_argument("-w", "--weights", help=".h5 weights_file_name for conv network",
                    default="network.h5")
args = parser.parse_args()
weights_file_name = args.weights
test_agent.load(weights_file_name)
list_episode_reward = []

TESTING_EPISODES = 10

for ep in range(TESTING_EPISODES):
    state = env.reset()
    episode_reward = 0
    for t in range(200):
        env.render()

        action = test_agent.act(state, False)
        print(action)
        next_state, reward, done , _ = env.step(test_agent.discrete_actions[action])


        state = next_state
        episode_reward += reward
