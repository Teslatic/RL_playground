import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import itertools

import keras

from assets.agents.DQN_Agent import DQN_Agent

from assets.helperFunctions.timestamps import print_timestamp
from assets.helperFunctions.discretize import discretize
from assets.policies.reinforce import reinforce
from assets.policies.improvement.epsilon_greedy import *

class Reinforce_Agent(DQN_Agent):
    """
    A model-based RL agent which uses:
    env: The pendulum environment with binary rewards
    policy: This is epsilon-greedy
    model: DQN
    hyperparameters:
    """

    def __init__(self, env, hyperparameters, model):
        """
        Initializing the agent with its hyperparameter set. If a model is necessary (e.g. for prediction) it is also build.
        """
        self._init_env_parameters(env)
        self._unzip_hyperparameters(hyperparameters)
        self.action_space = discretize(self.action_low, self.action_high, hyperparameters["N_ACTION"])
        architecture = self._create_architecture(model)
        self.estimator = self._build_model(architecture)

        EpisodeStats = namedtuple("Stats",["epis_lengths", "epis_rewards"])

    # def select_action(self, state, policy):
    def select_action(self, state):
        """
        The agents selects the next action according to its policy evaluation method (in general epsilon-greedy).
        """
        if False:
            pass
            policy = make_epsilon_greedy_policy(self.estimator, self.epsilon,
                            self.action_space)
            action = np.random.choice(np.arange(len(policy(state))),p=policy(state))
        else:
            action = np.random.random(1)
            return action

    def reinforce(self, policy):
        for t, ep in enumerate(self.episode_history):
            generate_return(ep[t:])
            policy.update(sess,s,a,total_return)

    def generate_return(self, episode_history_from_t):
        """
        Calculates the return
        """
        state_history = episode_history_from_t[0]
        action_history = episode_history_from_t[1]
        reward_history = episode_history_from_t[2]

        total_return = sum(self.gamma**i * reward_history for i,t in enumerate(episode[t:]))
        return state_history

###############################################################################
# Training methods
###############################################################################
# Policy and best policy needed?

    def train(self, training_parameters, weights_file=None):
        """The agent uses his training method on the given environment"""
        self._initialize_training(training_parameters, weights_file)
        for ep in range(self.training_episodes):
            self._initialize_episode(ep)
            for t in range(self.training_timesteps):
                self._initialize_timestep()
                self.Q, self.best_Q = self.estimator.predict(self.state)
                self.action = epsilon_greedy_lior(self.Q, self.epsilon, self.n_action)  # Less efficient than Liors code
                # self.action = self._select_action(self.state, self.policy)
                self.next_state, self.reward, done = self._act(self.action)  # Perform action
                self._analyze_timestep()
                # self.update_policy()
                if done:
                    break
            self._analyze_episode(ep)
            # reinforce(self, )


    def evaluate(self, training_parameters, evaluation_parameters):
        """
        The agent uses its current training state to evaluate how good it performs. The stored data can be analysed with plot functions.
        """
        pass

    def perform(self, evaluation_parameters, weights_file=None):
        self._initialize_evaluation(evaluation_parameters, weight_file)
        for ep in range(self.evaluation_episodes):
            self._initialize_episode(ep)
            for t in range(self.evaluation_timesteps):
                self._initialize_timestep
                self.Q = self.estimator.predict(state)
                self.policy = make_epsilon_greedy_policy(self.Q, self.epsilon, self.n_action)  # Less efficient than Liors code but no ifs
                self.action = self.select_action(self.state)
                self.next_state, self.reward, done = self._act(self.env, self.action)  # Perform action
                self._analyze_timestep()
                if done:
                    break
            self._analyze_episode(ep)
        return None

    def _analyze_timestep(self):
        self.episode_reward += self.reward
        self.episode_history.append((self.next_state,self.action,self.reward))

        if self.save:
            self.estimator.save_weights(self.weights_file_out)  # Saving the models weights_file

###############################################################################
# Code dumpster
###############################################################################

# from keras.models import Sequential
# from keras.optimizers import Adam, RMSprop
# from keras import backend as K
# from keras.layers import Flatten,Dense, Dropout
