#!/usr/bin/python3

###############################################################################
# Import packages
###############################################################################


from os import path
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import itertools

# Import tensorflow as tf

# if "../" not in sys.path:
#     sys.path.append("../")
from assets.helperFunctions.timestamps import print_timestamp
from assets.helperFunctions.tensor_conversions import *
from assets.helperFunctions.discretize import discretize
from assets.policies.improvement.epsilon_greedy import make_epsilon_greedy_policy

class RL_Agent():
    """Basic RL agent class"""

    def __init__(self, env, hyperparameters):  #*kwargs
        """
        Sets the agents basic parameters according to the environment.
        Unzips the given hyperparameters for the agent.
        """
        self._init_env_parameters(env)
        self._unzip_hyperparameters(hyperparameters)

    def _init_env_parameters(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.state_size = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space
        self.action_high = env.action_space.high[0]
        self.action_low = env.action_space.low[0]
        self.highscore_reward = 0  # best ever achieved reward
        self.best_parameters = None

    def _unzip_hyperparameters(self, hyperparameters):
        """Unzips the hyperparameter set and writes it into single variables"""
        self.learning_rate = hyperparameters['LEARNING_RATE']
        self.render = hyperparameters['LEARNING_RATE']
        self.gamma = hyperparameters['GAMMA']
        self.epsilon = hyperparameters['EPSILON']
        self.epsilon_init = hyperparameters['EPSILON_INIT']
        self.epsilon_decay_rate = hyperparameters['EPS_DECAY_RATE']
        self.const_decay = hyperparameters['CONST_DECAY']
        self.step_length = hyperparameters['STEP_LENGTH']
        self.n_action = hyperparameters['N_ACTION']

    def _initialize_training(self, training_parameters, weights_file):
        """
        This method initialize all members needed for the training.
        """
        self.episode_counter = 0  # Keeps track of current episode
        self.training_episodes = training_parameters['TRAINING_EPISODES']
        self.batch_size = training_parameters['BATCH_SIZE']
        self.training_timesteps = training_parameters['TRAINING_TIMESTEPS']
        self.memory_size = training_parameters['MEMORY_SIZE']
        self.auto_saver = training_parameters['AUTO_SAVER']
        self.save = False
        self.show_progress = training_parameters['SHOW_PROGRESS']
        self.render = False

        # Initialize plotting parameters
        zero_vector = np.zeros(self.training_episodes)
        self.EpisodeStats = namedtuple("Stats",["length", "reward"])
        self.episode_stats = self.EpisodeStats(length=zero_vector, reward=zero_vector)

        self.weights_file = weights_file
        print_timestamp('Start training with {} episodes'.format(self.training_episodes))

    def _initialize_evaluation(self):
        pass

    def _initialize_episode(self, ep):
        """
        Every show_progress_each steps the render flag is set to true.
        Every save_weights_each steps the save flag is set to true.
        """
        self.state = self.env.reset()
        self.episode_counter += 1  # Increase episode counter
        self.episode_length = 0
        self.episode_reward = 0
        self.episode_history = [] # Clearing the history


        """
        TEST-RUN-FLAG!!!
        """
        # Flags
        if ep != 0:  # No flags for the first episode
            if self.show_progress == None:
                self.render = False
            else:
                self.render = True if (ep % self.show_progress == 0) else False

            if self.auto_saver == None:
                self.save = False
            else:
                self.save = True if (ep % self.auto_saver == 0) else False

    def _analyze_episode(self, ep):
        self.episode_stats.reward[ep] = self.episode_reward
        self.episode_stats.length[ep] = self.episode_length

    def _initialize_timestep(self):
        """This method initialize the actual timestep."""
        if self.render:
            self.env.render()

    # def _policy(self, Q, epsilon, nA):
    #      return make_epsilon_greedy_policy(Q, epsilon, nA)

    def _analyze_timestep(self):
        self.episode_reward += self.reward
        self.state = self.next_state
        self.episode_history.append((self.next_state,self.action,self.reward))

        if self.save:
            self._save(self.weights_file)  # Saving the models weights_file

    def _select_action(self, state, policy):
        """
        The agent performs the given action on the environment and returns:
        observation: state information
        reward: reward by the environment
        done: flag to check if environment has terminated
        info: additional information
        """
        return policy(state)  # Returns the action according to the policy

    def _act(self, action):
        """
        The agent performs the given action on the environment and returns:
        observation: state information
        reward: reward by the environment
        done: flag to check if environment has terminated
        info: additional information
        """
        next_state, reward, done, _ = self.env.step(action)
        return convert_vector2tensor(next_state), reward, done

    def train(self, training_parameters, weights_file):
        """The agent uses his training method on the given environment"""
        self._initialize_training(training_parameters, weights_file)
        for ep in range(self.training_episodes):
            self._initialize_episode(ep)
            for t in range(self.training_timesteps):
                self._initialize_timestep()
                # self.Q = self.estimator.predict(state)
                self.policy = make_epsilon_greedy_policy(self.Q, self.epsilon, self.n_action)  # Less efficient than Liors code but no ifs
                self.action = self._select_action(self.state, self.policy)
                self.next_state, self.reward, done, _ = self._perform_action(self.env, self.action)  # Perform action
                self._analyze_timestep()
                self._update_policy()
                if done:
                    break
            self._analyze_episode(ep)
            # reinforce(self, )
###############################################################################
# Plotting funtions
###############################################################################

    def present(self, smoothing_window=10, noshow=False):
        # Plot the episode length over time
        fig_eps_length = plt.figure(figsize=(10,5))
        plt.plot(self.episode_stats.length)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Episode Length over Time")
        fig_eps_length.savefig('episode_lengths.png')
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
