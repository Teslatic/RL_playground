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

###############################################################################
# Import packages
###############################################################################
import os
from os import path
import sys
if "../" not in sys.path:
    sys.path.append("../")
# from lib.envs.pendulum import PendulumEnv  # original

import matplotlib.pyplot as plt
import pickle
import csv


from pendulum import PendulumEnv, angle_normalize  # Overwritten
from assets.agents.Dank_Agent import Dank_Agent
import assets.policies.policies
from assets.helperFunctions.timestamps import print_timestamp
from assets.plotter.DankPlotters import Plotter
from assets.plotter.PolarHeatmapPlotter import PolarHeatmapPlotter
from assets.helperFunctions.FileManager import *


###############################################################################
# Experiment name - Better have no spaces in between
###############################################################################

experiment_name = input("Please enter your experiment name: ")

# experiment_name = 'Multiplot'
description = 'Copy-pasta the hyperparameter set here'
main_path = path.dirname(path.abspath(sys.modules['__main__'].__file__))
all_exp_path = '{}/{}'.format(main_path, 'experiments')
exp_root_dir = create_free_path(all_exp_path, experiment_name)
# description = input("Please enter a short description of the test: ")

###############################################################################
# Hyperparameter settings
###############################################################################

# Agent hyperparameters
hyperparameters = {
                    'D_ACTION': 25,  # Standard 25
                    'GAMMA': 0.9,  # Standard 0.9
                    }

# Test parameters (for showing the agent's progress)
test_parameters = {
                    'TEST_EPISODES': 20, # Standard: 20
                    'TEST_TIMESTEPS': 500, # Standard: 500
                    'RENDER_TEST': None # Standard: None / 10
                    }

# Epsilon parameters
eps_parameters = {
                    'EPSILON_TYPE': 'linear',  # Not used yet
                    'CONST_DECAY': 0.0,
                    'EPSILON': 1, # Standard: 1
                    'EPSILON_MIN': 0.1, # Standard: 0.1
                    'EPSILON_INIT': 0.3,
                    'EPS_DECAY_RATE': 0.00005,  # 0.00001 (if stepwise) 0.002 (if episodic)
                    }

# Training parameters
train_parameters = {
                    'EXPERIMENT_ROOT_DIR': exp_root_dir,
                    'ACTUAL_DIR': exp_root_dir,
                    'DESCRIPTION': description,
                    'TRAINING_EPISODES': 200,  # Standard: 200
                    'TRAINING_TIMESTEPS': 200,  # Standard: 200
                    'BATCH_SIZE': 32,  # Standard: 32
                    'MEMORY_SIZE': 40000,  # Standard: TRAINING_EPISODES*TRAINING_TIMESTEPS*N (with N~1...5)
                    'TAU': 200, # Standard: 200 (Update target model every 200 steps = every episode)
                    'AUTO_SAVER': 50,
                    'SHOW_PROGRESS': None,
                    'STORE_PROGRESS': 10,
                    'TRAINING_FILE': "parameters/network.h5",
                    'TEST_EACH': 5, # Standard: 10
                    'EPSILON_PARAM': eps_parameters,
                    'TEST_PARAMETERS': test_parameters,
                    'REWARD_FNC': 'Vanilla'
                    }

# Model file
model = {
        'MODEL_TYPE': 'DQN',
        'WEIGHT_FILE_IN': None,
        'WEIGHT_FILE_OUT': "estimator.h5",
        'LOAD_EXISTING_MODEL': False,
        'ACTIVATION': 'relu',
        'LOSS': 'mse',
        'OPTIMIZER': 'Nadam',
        'LEARNING_RATE': 0.001, # Like Lior's code
        }


###############################################################################
# Parameter sweeps
###############################################################################

sweeps =    {
            'N_runs': 5,
            'BATCH_SIZE': [8, 16, 32, 64, 256],
            # 'BATCH_SIZE': [8, 16],
            # 'GAMMA': [0, .2, .4, .6, .8, .9, 1],
            'GAMMA': [.1, .3, .5, .7],
            'LEARNING_RATE': [1., .1, .01, .001, .0001],
            'OPTIMIZER': ['Adam','Nadam','SGD', 'RMSprop'],
            'REWARD_FNC': ['Heuristic1', 'Heuristic2', 'Vanilla'],
            # 'TAU': [1, 10, 50, 100, 200, 1000]
            'TAU': [200, 100, 50, 10, 1]
            }

###############################################################################
# Main starts here
###############################################################################

env = PendulumEnv()  # Create some environments
dankAgent = Dank_Agent(env, hyperparameters, model)  # Create some agents

###############################################################################
# SWEEPS
###############################################################################

# BATCH_SIZE
# dankAgent.parameter_sweep('BATCH_SIZE', sweeps['BATCH_SIZE'], train_parameters, sweeps['N_runs'])

# OPTIMIZER
# dankAgent.parameter_sweep('OPTIMIZER', sweeps['OPTIMIZER'], train_parameters, sweeps['N_runs'])

# GAMMA
# dankAgent.parameter_sweep('GAMMA', sweeps['GAMMA'], train_parameters, sweeps['N_runs'])

# TAU
dankAgent.parameter_sweep('TAU', sweeps['TAU'], train_parameters, sweeps['N_runs'])

# LEARNING_RATE
# dankAgent.parameter_sweep('LEARNING_RATE', sweeps['LEARNING_RATE'], train_parameters, sweeps['N_runs'])

# REWARD_FNC
# dankAgent.parameter_sweep('REWARD_FNC', sweeps['REWARD_FNC'], train_parameters, sweeps['N_runs'])

# read_sR = pickle.load(open('{}/report/sweepReport.p'.format(exp_root_dir), 'rb'))
# plotter = Plotter('notused')




###############################################################################
# Code dumpster
###############################################################################

# Start a training session with a given training parameters (e.g.)
# training_report, test_report = dankAgent.learn(train_parameters)
# dankPlotter.create_single_training_report(training_report)
# dankPlotter.create_single_test_report(test_report)


# multiReport = dankAgent.run_n_learning_sessions(5, train_parameters)
# dankPlotter.create_multiple_training_plots(multiReport)

# with open('dict.csv', 'wb') as csv_file:
#     writer = csv.writer(csv_file)
#     for key, value in hyperparameters.items():
#        writer.writerow([key, value])
#
# # To read it back:
#
# with open('dict.csv', 'rb') as csv_file:
#     reader = csv.reader(csv_file)
#     mydict = dict(reader)
#
# print(mydict)
