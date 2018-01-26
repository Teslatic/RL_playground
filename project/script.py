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



def print_timestamp(string = ""):
    now = datetime.datetime.now()
    print(string + now.strftime("%Y-%m-%d %H:%M"))

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
weights_file = "network.h5"
env = PendulumEnv()
env.cnt += 1
agent = DankAgent([-env.max_torque,env.max_torque])
agent.model.summary()

nepisodes = 0
######## CONSTANTS ############################################################
#
SHOW_PROGRESS = 1
TRAINING_EPISODES = 5000
AUTO_SAVER = 50
#
# state = env.reset()
# state, r, _ , _ = env.step(0)
# time.sleep(4)
# for t in range(20000000):
#
#     state, r, _ , _ = env.step(random.randrange(-2,2))
#     env.render()
#
#
#     print(state,r)
#     time.sleep(0.02)


# TIMESTEPS = 200
# state, reward, _ ,_  = env.step(0.2)
list_episode_reward = []
for ep in range(TRAINING_EPISODES):
    if ep % AUTO_SAVER == 0 and nepisodes != 0:
        print_timestamp("saved")
        agent.save(weights_file)
    state = env.reset()
    episode_reward = 0
    for t in range(200):
        # if ep % SHOW_PROGRESS == 0 and ep != 0:
        #     env.render()

        if random.random() <= agent.epsilon:
            action = randrange(-2,2)
        else:
            action = agent.act(state, False)

        next_state, reward, done , _ = env.step(agent.discrete_actions[action])
        # next_state, reward, done , _ = env.step(2)
        agent.train(state, action, next_state, reward, done)
        state = next_state
        episode_reward += reward
    if ep % SHOW_PROGRESS == 0:
        print("Episode {}/{}\t| Step: {}\t| Reward: {:.4f}\t| Loss: {:.4f}\t| epsilon: {:.2f}".format(ep, TRAINING_EPISODES,t,episode_reward,
            agent.history.history['loss'][0],agent.epsilon))

    agent.epsilon = agent.init_epsilon*np.exp(-agent.eps_decay_rate*ep)+agent.decay_const


    list_episode_reward.append(episode_reward)


    nepisodes += 1
    agent.update_target_model()

# ####### GLOBAL VARIABLES #######################################################
#
# total_reward = 0
# episode_reward = 0
# success = 0
#
# state = env.reset()
# for ep in range(1,EPISODES+1):
#     state = env.reset()
#     if ep % SHOW_PROGRESS == 0 and ep != 0:
#         print("| Episode: {}\t| Reward: {}".format(ep,episode_reward))
#     for t in range(TIMESTEPS):
#         env.render()
#         next_state, reward, _, _ = env.step(random.choice(slicer.discrete_actions))
#         # next_state,reward,_,_ = env.step(-2)
#         print("env state",env.state)
#         #print("normalize function:",angle_normalize(env.state[0]))
#         if reward == 1:
#             episode_reward += 1
#             success += 1
#             break
#         state = next_state
#     total_reward += episode_reward
#     episode_reward = 0
# print("Total reward over all episodes: {}\t| Success rate: {:.2f}%".format(total_reward,(success/EPISODES)*100))
#



def my_reward():
    r = 1 if -0.1 <= angle_normalize(pendulum.state[0]) <= 0.1 else state[0]*1
    print(r)
    return r
