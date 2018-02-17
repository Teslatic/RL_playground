#!/usr/bin/python3

"""
Reinforcement Learning Project
Authors: Nico Ott, Lior Fuks, Hendrik Vloet

The task for the RL project is to balance a pendulum upright.
A modified version of OpenAI's pendulum environment is used  which only uses
binary rewards.

Reward = 1 if the pendulum is in an upright position.
Reward = 0 if not.

The following algorithms have been implemented:
-

19.01.2018 V1.0 inital version
"""

# __version__ = '0.1'
# __author__ = 'Lior Fuks, Nico Ott, Hendrik Vloet'

###############################################################################
# Import packages
###############################################################################


from os import path
import sys
import time

# Import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import itertools

import keras
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.layers import Flatten,Dense, Dropout

if "../" not in sys.path:
    sys.path.append("../")
# from lib.envs.pendulum import PendulumEnv

from assets.helperFunctions.timestamps import print_timestamp

class RL_agent():
    """Basic RL agent class"""

    def __init__(self, env, hyperparameters, *kwargs):
        self._init_basic_parameters(env)
        self._unzip_hyperparameters(hyperparameters)

    def _init_basic_parameters(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render = False
        self.best_reward = 0
        self.total_reward = 0
        self.best_parameters = None
        self.episode_counter = 0

    def _unzip_hyperparameters(self, hyperparameters):
        self.learning_rate = hyperparameters['LEARNING_RATE']
        self.render = hyperparameters['LEARNING_RATE']
        self.batch_size = hyperparameters['BATCH_SIZE']
        self.gamma = hyperparameters['GAMMA']
        self.epsilon = hyperparameters['EPSILON']
        self.epsilon_init = hyperparameters['EPSILON_INIT']
        self.epsilon_decay_rate = hyperparameters['EPS_DECAY_RATE']
        self.const_decay = hyperparameters['CONST_DECAY']
        self.step_length = hyperparameters['STEP_LENGTH']
        # self.training_timesteps = hyperparameters['TIMESTEPS']

    def _initialize_training(self, training_parameters, weights_file):
        """
        This method unzips the hyperparameter set
        """
        self.episode_counter = 0  # Keeps track of current episode
        self.training_episodes = training_parameters['TRAINING_EPISODES']
        self.timesteps = training_parameters['TIMESTEPS']
        self.memory_size = training_parameters['MEMORY_SIZE']
        self.auto_saver = training_parameters['AUTO_SAVER']
        self.show_progress = training_parameters['SHOW_PROGRESS']
        self.save_weights_each = 50
        self.show_progress_each = 100

        # Initialize plotting parameters
        zero_vector = np.zeros(self.training_episodes)
        self.EpisodeStats = namedtuple("Stats",["epis_lengths", "epis_rewards"])
        self.episode = self.EpisodeStats(epis_lengths=zero_vector, epis_rewards=zero_vector)

        # self.plt_reward = []
        # self.plt_episode_counter = []

        self.weights_file = weights_file
        print('TRAINING_EPISODES: {}'.format(self.training_episodes))

    def _initialize_episode(self, ep):
        """
        Every show_progress_each steps the render flag is set to true.
        Every save_weights_each steps the save flag is set to true.
        """
        self.state = self.env.reset()
        self.total_reward = 0
        self.episode_counter += 1 #Increase episode counter
        self.episode_lengths = 0
        self.episode_history = []

        # Flags
        self.render = True if (ep % self.show_progress_each == 0) else False
        self.save = True if (ep % self.save_weights_each == 0) else False

    def _analyze_episode(self, ep):
        # Prepare episode history for plotting
        # self.plt_reward.append(self.total_reward)
        # self.plt_episode_counter.append(ep)
        self.stats.episode_rewards[i_episode] += reward
        self.stats.episode_lengths[i_episode] = t

    def _initialize_timestep(self):

        if self.render:
            self.env.render()

    def _analyze_timestep(self):
        self.total_reward += reward
        self.episode_history.append((state,action,reward))

        if self.save:
            self._save(self.weights_file)  # Saving the models weights_file



    def perform_action(self, env, action):
        """
        The agent performs the given action on the environment and returns:
        observation: state information
        reward: reward by the environment
        done: flag to check if environment has terminated
        info: additional information
        """
        return env.step(action)

    def policy(self):
        """
        The policy which the agent uses
        """
        pass

    def train(self, env):
        """The agent uses his training method on the given environment"""
        pass


    def perceive(self):
        """The observation of the environment"""
        pass


    def unzip_hyp_par(self, hyperparameters):
        """Unzips the hyperparameter set and writes it into single variables"""
        pass

###############################################################################
# Plotting funtions
###############################################################################

    def present(self, smoothing_window=10, noshow=False):
        # Plot the episode length over time
        fig_eps_length = plt.figure(figsize=(10,5))
        plt.plot(self.episode_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Episode Length over Time")
        fig1.savefig('episode_lengths.png')
        if noshow:
            plt.close(fig_eps_length)
        else:
            plt.show(fig_eps_length)

            # Plot the episode reward over time
            fig_eps_rewards = plt.figure(figsize=(10,5))
            rewards_smoothed = pd.Series(self.plt_episode_counter).rolling(smoothing_window, min_periods=smoothing_window).mean()
            plt.plot(rewards_smoothed)
            plt.xlabel("Episode")
            plt.ylabel("Episode Reward (Smoothed)")
            plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
            fig_eps_rewards.savefig('reward.png')
            if noshow:
                plt.close(fig_eps_rewards)
            else:
                plt.show(fig_eps_rewards)


class MBRL_agent(RL_agent):
    """
    An RL agent that uses a model of the environment.

    Args:
        model (object): The model of the MB agent
    """


    def __init__(self, model):
        pass

    def _build_model(self, weights_file):
        """
        Builds a model. Architecture should be defined by input parameters
        """
        pass

    def update_model(self):
        pass


class DQN_agent(MBRL_agent):
    """
    An RL agent that uses DQN as a model of the environment.

    Members:
        model (object): The model of the MB agent
    """


    def __init__(self, model):
        pass

    def _build_model(self, weights_file):
        """
        Builds a DQN model. Architecture should be defined by input parameters
        """
        self.model = Sequential()
        self.model.add(Dense(32,input_shape = (3,),activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(int(self.n_actions)))
        self.model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        self.model.summary() # Prints out a short summary of the architecture

        self.weights_file = weights_file
        if self.weights_file == None:
            pass
        else:
            self._load(self.weights_file)

    def _save(self, file_name):
        print_timestamp()
        self.model.save_weights(file_name)
        print("agent saved weights in file '{}' ".format(file_name))

    def _load(self, file_name):
        self.model.load_weights(file_name)
        print_timestamp()
        print("agent loaded weights from file '{}' ".format(file_name))
        self.update_target_model()

    def update_model(self):
        pass

    def update_target_model(self):
        pass

    def predict_next_state(self):
        pass


    def predict_next_reward(self):
        pass


    def plot_episodes(self):
        pass
