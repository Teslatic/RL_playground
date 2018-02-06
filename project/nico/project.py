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
- Policy Gradient REINFORCE
- DQN

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

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.pendulum import PendulumEnv

from assets.agents.Reinforce_Agent import Reinforce_Agent
from assets.estimators.NN_estimator import NN_estimator
import assets.policies.policies
import assets.functions
from assets.helperFunctions.timestamps import print_timestamp
from keras.optimizers import Adam, Nadam


###############################################################################
# Hyperparameter settings
###############################################################################

# MAYBE USED NAMETUPLE INSTEAD HERE
# Agent hyperparameters
hyperparameters =   {
                    'STEP_LENGTH': 0.05,  # ?
                    'GAMMA': 0.9,
                    'LEARNING_RATE': 0.0001,
                    'EPSILON_TYPE': 'exponential',
                    'CONST_DECAY': 0.0,
                    'EPSILON': 1,
                    'EPSILON_INIT': 0.3,
                    'EPS_DECAY_RATE': 0.04,
                    'N_ACTION': 25,
                    'OPTIMIZER': 'NAdam'
                    }

# Training parameters
training_parameters =   {
                        'BATCH_SIZE': 32,
                        'TRAINING_EPISODES': 100,
                        'TRAINING_TIMESTEPS': 200,
                        'MEMORY_SIZE': 80000,
                        'AUTO_SAVER': 50,
                        'SHOW_PROGRESS': None,
                        # 'RENDER': False
                       # learn_counter
                       # memory_counter
                       # self.memory = np.empty([self.memory_size,9])
                        }

# Evaluation parameters
evaluation_parameters = {
                        'EVALUATION_EPISODES': 200,
                        'EVALUATION_TIMESTEPS': 500,
                        'SHOW_PROGRESS': None
                        }

# Model file
model = {
        'MODEL_TYPE': 'DQN',
        'WEIGHT_FILE': None,
        'WEIGHT_FILE_OUT': "estimator.h5",
        'LOAD_EXISTING_MODEL':True,
        'ACTIVATION': 'relu',
        'LOSS': 'mse',
        'OPTIMIZER': Nadam,
        'LEARNING_RATE': 0.0001,
        }


###############################################################################
# Parameter sweeps
###############################################################################

batch_size_sweep = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
gamma_sweep = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
optimizer_sweep = ['NAdam', 'Adam']  # Define other optimizers as well.
reward_function_sweep = ['Vanilla', 'Heuristic1', 'Heuristic2']

###############################################################################
# Main starts here
###############################################################################

print_timestamp('Started main program')
# Create some environments
env = PendulumEnv()  # Setup the environment
# Create some agents
dankAgent = Reinforce_Agent(env, hyperparameters, model)
# memeAgent = DQN_Agent


# Start a training session with a given weight_file (e.g.)
training_weight_file = model["WEIGHT_FILE"]
training_report = dankAgent.train(training_parameters, training_weight_file)

report = dankAgent.perform(model)  # perform with weights
# dankAgent.present() # Plot the results

# Sweeping one parameter
# for sweep_parameter in batch_size_sweep:
#     training_parameters["BATCH_SIZE"] = sweep_parameter
#     print_timestamp('Parameter Sweep: Starting training with batchsize {}'.format(sweep_parameter))
#     dankAgent.train(training_parameters, training_weight_file)  # Train agent
    # report = dankAgent.perform(model)  # perform with weights
    # dankAgent.present() # Plot the results

###############################################################################
# Code dumpster
###############################################################################
