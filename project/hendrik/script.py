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
import pandas as pd
import csv

def print_timestamp(string = ""):
    now = datetime.datetime.now()
    # print(string + now.strftime("%Y-%m-%d %H:%M"))

def print_setup():
    print("---------------------------------------------------------------------------------")
    print("| Setup: \t\t\t\t\t\t\t\t\t|")
    print("| Memory fill:\t\t\t{}\t\tBatch: {}\t|".format(
    MEMORY_FILL, BATCH_SIZE ,  agent.init_epsilon))
    print("| Discount: \t\t\t{}\t\tEpsilon constant: \t{}\t|".format(agent.gamma,agent.decay_const ))
    print("| Start Epsilon:\t\t{}\t\tEpsilon decay rate:\t{}\t|".format(
    agent.init_epsilon+agent.decay_const, agent.eps_decay_rate ))
    print("| Learning rate:\t\t{}\t\tUpdate rate model:\t{}\t|".format(
    agent.learning_rate, agent.train_marker ))
    print("| Reward function:\t\t{}\tTimesteps:\t\t{}\t|".format(env.reward_fn_string, TIMESTEPS))
    print("| Update rate target model \t{}\t\tEpisodes:\t\t{}\t|".format(agent.update_target_marker, TRAINING_EPISODES))
    print("---------------------------------------------------------------------------------")


def test_run(agent):
    test_reward_list = []
    for i in range(TESTS):
        test_episode_reward = 0
        env.seed(i)
        state = env.reset()
        state = state.reshape((1,3))
        for _ in range(500):

            action = agent.act(state, False)[0]
            next_state, reward, done, _ = env.step(agent.discrete_actions[action], True)
            next_state = np.round(next_state,2)

            state = next_state
            test_episode_reward += reward
        test_reward_list.append(test_episode_reward)
    avg_test_reward = np.mean(test_reward_list)
    print("\naverage reward: {}".format(avg_test_reward))
    return avg_test_reward


######## CONSTANTS ############################################################

MEMORY_SIZE = 8000000
MEMORY_FILL = 8000
memory = []
TRAINING_EPISODES = 500
AUTO_SAVER = 50
SHOW_PROGRESS = 25
TIMESTEPS = 500
BATCH_SIZE = [32, 64, 128, 256, 512]
RUNS = 5
TEST_PROGRESS = 10
TESTS = 50

####### INTIALISATION ##########################################################
weights_file = "network.h5"
input_shape = 3
env = PendulumEnv()
agent = DankAgent([-env.max_torque,env.max_torque], input_shape, BATCH_SIZE[0])
agent.model.summary()
agent.model.save_weights('model_init_weights.h5')
agent.target_model.save_weights('target_model_init_weights.h5')


nepisodes = 0
nepisodes = 0
acc_reward = 0
list_acc_reward = [[] for _ in range(RUNS)]
list_episode_reward = [[] for _ in range(RUNS)]

list_epsilon = [[] for _ in range(RUNS)]
list_avg_reward = [[] for _ in range(RUNS)]

list_time = []

print_setup()

for run in range(RUNS):
    # resetting the agent

    env = PendulumEnv()
    agent = DankAgent([-env.max_torque,env.max_torque], input_shape, BATCH_SIZE[run])
    acc_reward = 0
    memory = []
    agent.model.load_weights('model_init_weights.h5')
    agent.target_model.load_weights('target_model_init_weights.h5')
    agent.q_target = np.zeros((BATCH_SIZE[run],int(agent.ticks)))
    agent.t = np.zeros((BATCH_SIZE[run],int(agent.ticks)))
    agent.a = np.zeros((BATCH_SIZE[run],int(agent.ticks)))
    agent.epsilon = 1.0



    start_time = time.time()

    for ep in range(TRAINING_EPISODES):
        if (ep+1) % AUTO_SAVER == 0 and ep != 0:
            print_timestamp("saved")
            str_status = 'saved'
            agent.save(weights_file)
        state = env.reset()
        #state = np.array((state[0],state[2]))
        state = np.round(state,2)
        episode_reward = 0
        if (ep+1) % TEST_PROGRESS == 0 and ep != 0:
            list_avg_reward[run].append(test_run(agent))
        if ep < 1:
            print("filling memory")
        while len(memory) < MEMORY_FILL and ep < 1:
            action = agent.act(state, True)[0]

            next_state, reward, done , _ = env.step(agent.discrete_actions[action], False)
            next_state = np.round(next_state,2)
            memory.append((state, action, reward+10*run, next_state, done))

            if len(memory) > BATCH_SIZE[run]:
                batch = random.sample(memory, BATCH_SIZE[run])
                agent.train(batch,memory)

            state = next_state

        if ep < 1:
            print("memory filled with {} samples".format(MEMORY_FILL))

        for t in range(TIMESTEPS):
            if (ep+1) % SHOW_PROGRESS == 0 and ep != 0 and len(memory) >= MEMORY_FILL:
                env.render()
                sys.stdout.flush()
                # print("\rRendering episode {}/{}".format(ep,TRAINING_EPISODES),end="")
                str_status = 'render'
                sys.stdout.flush()
            else:
                str_status = 'normal'
            action = agent.act(state, True)[0]
            #print(state)

            next_state, reward, done , _ = env.step(agent.discrete_actions[action], False)
            #next_state = np.array((next_state[0],next_state[2]))
            next_state = np.round(next_state,2)



            if len(memory) == MEMORY_SIZE:
                memory.pop(0)

            memory.append(np.array((state, action, reward, next_state, done)))


            if len(memory) >= BATCH_SIZE[run]:
                batch = random.sample(memory, BATCH_SIZE[run])

                agent.train(batch,memory)
            else:
                batch = random.choice(memory)


            state = next_state
            acc_reward += reward

            episode_reward += reward
        sys.stdout.flush()
        print("\r| Run: {} | Episode: {}/{} | Episode Reward: {:.4f} | Acc. Reward: {:.4f} | epsilon: {:.2f} | Status: {} | ".format(run+1,ep+1, TRAINING_EPISODES,episode_reward ,acc_reward,
                agent.epsilon,str_status),end="")
        sys.stdout.flush()

        agent.epsilon = agent.init_epsilon*np.exp(-agent.eps_decay_rate*ep)+agent.decay_const


        list_episode_reward[run].append(episode_reward)
        list_acc_reward[run].append(acc_reward)
        list_epsilon[run].append(agent.epsilon)


    print("\n")
    end_time = time.time()
    time_needed = (end_time - start_time)/60
    list_time.append(time_needed)
    print("Training time: {:.2f} min".format(time_needed))


plt.figure()
for i in range(RUNS):
    plt.plot(list_avg_reward[i], label='{} batch'.format(BATCH_SIZE[i]))
plt.plot(np.mean(list_avg_reward,axis=0), label = 'mean')
plt.plot(np.mean(list_avg_reward,axis=0)+np.std(list_avg_reward), label = 'mean+std. dev.',linestyle = '--', color='pink')
plt.plot(np.mean(list_avg_reward,axis=0)-np.std(list_avg_reward), label = 'mean-std. dev.',linestyle = '--',color='pink')
plt.title("Avg. reward in a intermediate test every {} episodes".format(TEST_PROGRESS))
plt.xlabel("Test")
plt.ylabel("Reward (Vanilla)")
plt.legend()
plt.savefig('avg_test_reward.png')
with open('csv_files/batch_sweep/avg_test_reward.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator="\n")
    writer.writerows(list_avg_reward)

plt.figure()
for i in range(RUNS):
    plt.plot(list_episode_reward[i], label='{} batch'.format(BATCH_SIZE[i]))
plt.title("Rewards per episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.savefig('reward_per_episode.png')
with open('csv_files/batch_sweep/reward_per_episode.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator="\n")
    writer.writerows(list_episode_reward)


plt.figure()
for i in range(RUNS):
    plt.plot(list_acc_reward[i],label = '{} batch'.format(BATCH_SIZE[i]))
plt.plot(np.mean(list_acc_reward,axis=0), label = 'mean')
plt.plot(np.mean(list_acc_reward,axis=0)+np.std(list_acc_reward), label = 'mean+std. dev.',linestyle = '--', color='pink')
plt.plot(np.mean(list_acc_reward,axis=0)-np.std(list_acc_reward), label = 'mean-std. dev.',linestyle = '--',color='pink')
plt.title("Accumulated reward over all episodes")
plt.xlabel("Episode")
plt.ylabel("Accumulated reward")
plt.legend()
plt.savefig('acc_reward.png')
with open('csv_files/batch_sweep/list_acc_reward.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', lineterminator="\n")
    writer.writerows(list_acc_reward)




#plt.show()




### CODE DUMPSTER ##############################################################


    #
    # fig1 = plt.figure(figsize=(10,5))
    # plt.plot(list_epsilon[run])
    # plt.xlabel("Episode")
    # plt.ylabel("Epsilon")
    # fig1.savefig('decaying_epsilon.png')
    #
    # smoothing_window = 10
    # # Plot the episode reward over time
    # fig2 = plt.figure(figsize=(10,5))
    # rewards_smoothed = pd.Series(list_episode_reward[run]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # plt.plot(rewards_smoothed)
    # plt.xlabel("Episode")
    # plt.ylabel("Episode Reward (Smoothed)")
    # plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    # fig2.savefig('reward.png')
    #
    # moothing_window = 10
    # # Plot the episode reward over time
    # fig3 = plt.figure(figsize=(10,5))
    # rewards_smoothed = pd.Series(list_acc_reward[run]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # plt.plot(rewards_smoothed)
    # plt.xlabel("Episode")
    # plt.ylabel("Episode Reward (Smoothed)")
    # plt.title("Accumulated Reward over Time (Smoothed over window size {})".format(smoothing_window))
    # fig3.savefig('acc_reward.png')
