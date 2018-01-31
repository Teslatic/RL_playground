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
from keras.models import load_model
from pendulumAgent import DankAgent, print_timestamp
import argparse
import sys
#sys.path.append('..')
parser = argparse.ArgumentParser()
import random
import datetime



env = PendulumEnv()
env.cnt += 1
input_shape = 3
batch_size = 64
test_agent = DankAgent([-env.max_torque,env.max_torque],input_shape,batch_size)
test_agent.model.load_weights('network.h5')

list_episode_reward = []

TESTING_EPISODES = 10

for ep in range(TESTING_EPISODES):
    state = env.reset()
    episode_reward = 0
    for t in range(1000):
        env.render()
        # action = random.choice(test_agent.discrete_actions)
        action = test_agent.act(state, False)
        next_state, reward, done , _ = env.step(test_agent.discrete_actions[action])
        # next_state, reward, done , _ = env.step(action)

        state = next_state
        episode_reward += reward
