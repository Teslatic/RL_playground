#!/usr/bin/python3
import matplotlib.pyplot as plt
import gym
import sys
from pendulum import PendulumEnv, angle_normalize
import random
from random import randrange
import numpy as np
from pendulumAgent import DankAgent
import datetime
import time

####### INTIALISATION ##########################################################
weights_file = "network.h5"
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
