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
from pendulum import PendulumEnv


def print_timestamp(string = ""):
    now = datetime.datetime.now()
    print(string + now.strftime("%Y-%m-%d %H:%M"))

def print_setup():
    print("--------------------------------------------------------------------------------")
    print("| Setup: \t\t\t\t\t\t\t\t\t\t|")
    print("| Memory Size: {}\tBatch Size: {}\t\tDiscount: {}\t\t\t|".format(
    MEMORY_SIZE, BATCH_SIZE , agent.gamma, agent.init_epsilon))
    print("| Start Epsilon: {}\tEpsilon decay rate: {}\tEpsilon constant: {}\t|".format(
    agent.init_epsilon, agent.eps_decay_rate, agent.decay_const))
    print("| Learning rate: {}\tUpdate rate model: {}\tUpdate rate target model: {}\t|".format(
    agent.learning_rate, agent.train_marker, agent.update_target_marker))
    print("| Reward function: {}\t\t\t\t\t\t\t\t|".format(env.reward_fn_string))
    print("--------------------------------------------------------------------------------")

######## CONSTANTS ############################################################
#def: 50000
MEMORY_SIZE = 800#0#0
memory = []
#def: 30000
TRAINING_EPISODES = 3000
AUTO_SAVER = 50
SHOW_PROGRESS = 25
# def: 2000
TIMESTEPS = 500
BATCH_SIZE = 64


####### INTIALISATION ##########################################################
weights_file = "network.h5"
load_existent_model = False
env = PendulumEnv(reward_function = 1)
env.cnt += 1
input_shape = 3
agent = DankAgent([-env.max_torque,env.max_torque], input_shape, BATCH_SIZE)
agent.model.summary()
if load_existent_model:
    agent.load(weights_file)

print_setup()

nepisodes = 0


list_episode_reward = []
list_epsilon = []
for ep in range(TRAINING_EPISODES):
    if ep % AUTO_SAVER == 0 and nepisodes != 0:
        print_timestamp("saved")
        agent.save(weights_file)
    state = env.reset()
    #state = np.array((state[0],state[2]))
    state = np.round(state,2)
    episode_reward = 0

    if ep < 1:
        print("filling memory")
    while len(memory) < MEMORY_SIZE and ep < 1:
        action = agent.act(state, True)[0]

        next_state, reward, done , _ = env.step(agent.discrete_actions[action])
        next_state = np.round(next_state,2)
        memory.append((state, action, reward, next_state, done))

        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            agent.train(batch,memory)

        state = next_state
        #print(len(memory))
    if ep < 1:
        print("memory filled")




    for t in range(TIMESTEPS):
        if ep % SHOW_PROGRESS == 0 and ep != 0 and len(memory) >= MEMORY_SIZE:
            env.render()
            print("\rRendering episode {}/{}".format(ep,TRAINING_EPISODES),end="")
            sys.stdout.flush()

        action = agent.act(state, True)[0]
        #print(state)

        next_state, reward, done , _ = env.step(agent.discrete_actions[action])
        #next_state = np.array((next_state[0],next_state[2]))
        next_state = np.round(next_state,2)



        if len(memory) == MEMORY_SIZE:
            memory.pop(0)

        memory.append(np.array((state, action, reward, next_state, done)))


        if len(memory) >= MEMORY_SIZE:
            batch = random.sample(memory, BATCH_SIZE)

            agent.train(batch,memory)
        else:
            batch = random.choice(memory)


        state = next_state
        episode_reward += reward
    print("| Episode {}/{}\t| Reward: {:.4f}\t| epsilon: {:.2f}\t|".format(ep, TRAINING_EPISODES,episode_reward,
            agent.epsilon))

        # print("Episode {}/{}\t| Step: {}\t| Reward: {}\t".format(ep, TRAINING_EPISODES,t,episode_reward))
    agent.epsilon = agent.init_epsilon*np.exp(-agent.eps_decay_rate*ep)+agent.decay_const


    list_episode_reward.append(episode_reward)
    list_epsilon.append(agent.epsilon)

    nepisodes += 1
    if agent.cnt % agent.update_target_marker == 0:
        agent.update_target_model()

plt.figure()
plt.plot(list_episode_reward,label = "DankAgent")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()

plt.figure()
plt.plot(list_epsilon,label = "DankAgent")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.legend()


plt.show()
