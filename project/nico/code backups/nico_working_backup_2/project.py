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
from pendulum import PendulumEnv, angle_normalize  # Overwritten

from assets.agents.Dank_Agent import Dank_Agent
import assets.policies.policies
from assets.helperFunctions.timestamps import print_timestamp
from assets.plotter.DankPlotters import Plotter
from assets.plotter.PolarHeatmapPlotter import PolarHeatmapPlotter

import matplotlib.pyplot as plt

###############################################################################
# Hyperparameter settings
###############################################################################

# Agent hyperparameters
hyperparameters = {
                    'D_ACTION': 25,
                    'GAMMA': 0.9,
                    'OPTIMIZER': 'NAdam',  # Not used
                    'LEARNING_RATE': 0.001,  # Not used
                    }

# Test parameters (for showing the agent's progress)
test_parameters = {
                    'TEST_EPISODES': 20, # 20
                    'TEST_TIMESTEPS': 500,
                    'RENDER_TEST': None
                    }

# Epsilon parameters
eps_parameters = {
                    'EPSILON_TYPE': 'linear',  # Not used yet
                    'CONST_DECAY': 0.0,
                    'EPSILON': 1,
                    'EPSILON_MIN': 0.1,
                    'EPSILON_INIT': 0.3,
                    'EPS_DECAY_RATE': 0.002,
                    }

# Training parameters
train_parameters = {
                    'EXPERIMENT_NAME': 'NicosTest',
                    'TRAINING_EPISODES': 500,  # Standard: 200
                    'TRAINING_TIMESTEPS': 300,  # Standard: 200
                    'BATCH_SIZE': 32,  # Standard: 32
                    'MEMORY_SIZE': 24000,  # Standard: 80000
                    'AUTO_SAVER': 50,
                    'SHOW_PROGRESS': None,
                    'STORE_PROGRESS': 10,
                    'TRAINING_FILE': "parameters/network.h5",
                    'TEST_EACH': 10,
                    'EPSILON_PARAM': eps_parameters,
                    'TEST_PARAMETERS': test_parameters
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

batch_size_sweep = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
gamma_sweep = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
optimizer_sweep = ['NAdam', 'Adam']  # Define other optimizers as well.
reward_function_sweep = ['Vanilla', 'Heuristic1', 'Heuristic2']

###############################################################################
# Main starts here
###############################################################################
# print_timestamp('Plotting')
# heat = PolarHeatmapPlotter(2)
# heat.plot()
print_timestamp('Started experiment program')
# main_path = path.dirname(path.abspath(sys.modules['__main__'].__file__))
exp_directory = train_parameters['EXPERIMENT_NAME']
print(exp_directory)
file_index = 0
while path.exists(exp_directory + '%s' % file_index):
    file_index += 1

idx_file_dir = exp_directory + '{}'.format(file_index)
train_parameters['EXPERIMENT_NAME'] = idx_file_dir
os.makedirs(idx_file_dir)
os.makedirs(idx_file_dir+'/results/policy_plots')
os.makedirs(idx_file_dir+'/results/parameters')
os.makedirs(idx_file_dir+'/results/plots')
print_timestamp('Started experiment {}'.format(idx_file_dir))

env = PendulumEnv()  # Create some environments
dankAgent = Dank_Agent(env, hyperparameters, model)  # Create some agents

# Start a training session with a given weight_file (e.g.)
training_report, test_report = dankAgent.learn(train_parameters)

# dankplotter = Plotter
plt.figure()
for idx,val in enumerate(mean_summary):
    plt.plot(val[0], label = '{}'.format(features[idx]))
plt.plot(merged_mean, label ='mean of means')
plt.plot(merged_mean+mean_std[0], label='+', linestyle = '-.')
plt.plot(merged_mean-mean_std[0], label='-', linestyle = '-.')
plt.legend()
plt.show()

# report = dankAgent.perform(model)  # perform with weights
# dankAgent.present() # Plot the results

###############################################################################
# Code dumpster
###############################################################################

# test_parameters = {
#                     'TEST_EPISODES': 200,
#                     'TEST_TIMESTEPS': 500,
#                     'SHOW_EVALUATE': 10
#                     }
