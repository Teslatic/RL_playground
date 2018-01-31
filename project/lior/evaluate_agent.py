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

state = env.reset()
state = state.reshape((1,3))

action_dim = 25
action_high=env.action_space.high[0]
action_low=env.action_space.low[0]
action_sequence = np.linspace(action_low, action_high, action_dim)

test_agent = DankAgent(env.observation_space.shape[0], action_dim)


parser.add_argument("-w", "--weights", help=".h5 weights_file_name for conv network",
                    default="network.h5")
args = parser.parse_args()
weights_file_name = args.weights
test_agent.load("eval_network.h5")
list_episode_reward = []

TESTING_EPISODES = 10

for ep in range(TESTING_EPISODES):

    episode_reward = 0
    for t in range(500):
        env.render()

        action = test_agent.act(state, False)
        #print(test_agent.discrete_actions[action])

        next_state, reward, done , _ = env.step(action_sequence[action])
        print("reward", reward)

        state = next_state
        episode_reward += reward
