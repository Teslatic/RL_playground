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

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import itertools

# Import tensorflow as tf
import keras
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.layers import Flatten,Dense, Dropout

if "../" not in sys.path:
    sys.path.append("../")
# from lib.envs.pendulum import PendulumEnv
from assets.agents.MBRL_Agent import RL_Agent
from assets.helperFunctions.timestamps import print_timestamp
from assets.estimators.NN_estimator import NN_estimator

class DQN_Agent(RL_Agent):
    """
    An model based RL agent that uses DQN as a model of the environment.

    Members:
        model (object): The model of the MB agent
    """

    def __init__(self, env, hyperparameters, model):
        self._init_env_parameters(env)
        self._unzip_hyperparameters(hyperparameters)
        self.action_space = discretize(self.action_low, self.action_high, hyperparameters["N_ACTION"])
        architecture = self._create_architecture(model)
        self.estimator = self._build_model(architecture)

    def _create_architecture(self, model):
        architecture = {
                        'D_IN': self.env.observation_space.shape[0],
                        'D_OUT': self.n_action,
                        'ACTION_SPACE': self.action_space,
                        'ACTIVATION': model["ACTIVATION"],
                        'LOSS': model["LOSS"],
                        'OPTIMIZER': model["OPTIMIZER"],
                        'LEARNING_RATE': model["LEARNING_RATE"],
                        'WEIGHT_FILE': model["WEIGHT_FILE"],
                        'WEIGHT_FILE_OUT': model["WEIGHT_FILE_OUT"]
                        }
        self.weights_file_out = architecture["WEIGHT_FILE_OUT"]
        return architecture

    def _build_model(self, architecture):
        """
        Builds a DQN model. Architecture should be defined by input parameters
        """
        estimator = NN_estimator(architecture)

        weight_file = architecture["WEIGHT_FILE"]
        if weight_file == None:
            pass
        else:
            estimator.load_weights(weight_file)
        return estimator
