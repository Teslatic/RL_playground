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
TRAINING_EPISODES = 30
AUTO_SAVER = 50
SHOW_PROGRESS = 10
# def: 2000
TIMESTEPS = 1000


angle_range = np.array((-1, 1))
vel_range = np.array((-8, 8))
step_length = 0.1
ticks = 400

angle_ticks = (np.abs(angle_range[0])+np.abs(angle_range[-1])) / step_length

discrete_angles = np.linspace(angle_range[0],angle_range[-1],ticks)

vel_ticks = (np.abs(vel_range[0])+np.abs(vel_range[-1])) / step_length

discrete_vel = np.linspace(vel_range[0],vel_range[-1],ticks)

discretized_state_space = np.round(np.array((discrete_angles,discrete_vel)),2)

tol_a = np.round(np.diff(discretized_state_space[1])[0],4)
tol_v = np.round(np.diff(discretized_state_space[0])[0],4)

def discretizeStateSpace(angle, vel):

    print("passed angle: {}".format(angle))

    angle_candidates = discretized_state_space[0][np.isclose(angle,discretized_state_space[0],rtol=0.1)]
    print("angle candiates {}".format(angle_candidates))
    disc_angle = random.choice(angle_candidates)
    print("chosen angle: {}".format(disc_angle))
    print("error to true angle: {}".format(disc_angle-angle))




    #
    #
    # print("passed vel: {}".format(vel))
    # vel_candidates = discretized_state_space[1][np.isclose(vel,discretized_state_space[1],rtol=0.6)]
    # print("vel candiates {}".format(vel_candidates))
    # disc_velocity = random.choice(vel_candidates)
    # print("chosen vel: {}".format(disc_velocity))
    # print("error to true vel: {}".format(disc_velocity-vel))
    # return disc_angle, disc_velocity
    #




def _discretize_actions(self):
    self.ticks = (np.abs(self.action_interval[0]) + np.abs(self.action_interval[-1])) / self.step_length
    self.discrete_actions = np.linspace(self.action_interval[0],self.action_interval[-1],self.ticks)
    print("I discretized your action-space for you:")
    print("|---------------------------------------------------------------|")
    print("|Action Interval {}\t| Ticks: {:.0f}\t| Delta {:.4f}\t|".format(self.action_interval,self.ticks,np.diff(self.discrete_actions)[-1]))
    print("|---------------------------------------------------------------|")
    return self.discrete_actions

def sample_minibatch(self, batch_size=None):
    if batch_size is None:
        batch_size = self.batch_size
    state      = np.zeros((self.batch_size, self.state_siz*self.hist_len), dtype=np.float32)
    action     = np.zeros((self.batch_size, self.act_num), dtype=np.float32)
    next_state = np.zeros((self.batch_size, self.state_siz*self.hist_len), dtype=np.float32)
    reward     = np.zeros((self.batch_size, 1), dtype=np.float32)
    terminal   = np.zeros((self.batch_size, 1), dtype=np.float32)
    for i in range(batch_size):
        index = np.random.randint(self.bottom, self.bottom + self.size)
        state[i]         = self.states.take(index, axis=0, mode='wrap')
        action[i]        = self.actions.take(index, axis=0, mode='wrap')
        next_state[i]    = self.next_states.take(index, axis=0, mode='wrap')
        reward[i]        = self.rewards.take(index, axis=0, mode='wrap')
        terminal[i]      = self.terminal.take(index, axis=0, mode='wrap')
    return state, action, next_state, reward, terminal



for ep in range(TRAINING_EPISODES):
    if ep % AUTO_SAVER == 0 and nepisodes != 0:
        print_timestamp("saved")
        agent.save(weights_file)
    state = env.reset()
    state = np.array((state[0],state[2]))
    state = np.round(state,2)
    for t in range(TIMESTEPS):
        env.render()


        action =   random.choice(agent.discrete_actions)
        next_state, reward, done , _ = env.step(action, state[1])

        next_state = np.array((next_state[0],next_state[2]))
        next_state = np.round(next_state,2)

        if len(memory) == MEMORY_SIZE:
            memory.pop(0)
        memory.append((state, action, reward, next_state, done))

        batch = random.choice(memory)


        # print("real angle: {}".format(next_state[0]))
        # print("real vel: {}".format(next_state[1]))
        # next_state[0], next_state[1] = discretizeStateSpace(np.round(next_state[0],2),np.round(next_state[1],2))


        # next_state = np.round(next_state,2)
        print(next_state, reward)

        state = next_state
    agent.epsilon = agent.init_epsilon*np.exp(-agent.eps_decay_rate*ep)+agent.decay_const
