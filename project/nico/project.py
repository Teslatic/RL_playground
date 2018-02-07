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
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.pendulum import PendulumEnv

from assets.agents.Dank_Agent import Dank_Agent
import assets.policies.policies
from assets.helperFunctions.timestamps import print_timestamp
from assets.plotter.DankPlotters import PolarHeatmapPlotter

###############################################################################
# Hyperparameter settings
###############################################################################

# MAYBE USED NAMETUPLE INSTEAD HERE
# Agent hyperparameters
hyperparameters = {
                    'STEP_LENGTH': 0.05,  # ?
                    'GAMMA': 0.9,
                    'LEARNING_RATE': 0.0001,
                    'EPSILON_TYPE': 'exponential',
                    'CONST_DECAY': 0.0,
                    'EPSILON': 1,
                    'EPSILON_INIT': 0.3,
                    'EPS_DECAY_RATE': 0.001,
                    'D_ACTION': 25,
                    'OPTIMIZER': 'NAdam'
                    }

# Evaluation parameters
evaluation_parameters = {
                        'EVALUATION_EPISODES': 20,
                        'EVALUATION_TIMESTEPS': 500,
                        'RENDER_EVALUATE': False
                        }

# Training parameters
training_parameters = {
                        'BATCH_SIZE': 32,  # 32
                        'TRAINING_EPISODES': 200,
                        'TRAINING_TIMESTEPS': 200,
                        'MEMORY_SIZE': 24000, # 80000
                        'AUTO_SAVER': 50,
                        'SHOW_PROGRESS': None,
                        'STORE_PROGRESS': 10,
                        'EVALUATE_EACH': 10,
                        'EVAL_PARAMETERS': evaluation_parameters
                        # 'POLICY': "epsilon_greedy"
                        # 'RENDER': False
                        # learn_counter
                        # memory_counter
                        # self.memory = np.empty([self.memory_size,9])
                       }

test_parameters = {
                        'TEST_EPISODES': 200,
                        'TEST_TIMESTEPS': 500,
                        'SHOW_EVALUATE': 10
                        }

# Model file
model = {
        'MODEL_TYPE': 'DQN',
        'WEIGHT_FILE': None,
        'WEIGHT_FILE_OUT': "estimator.h5",
        'LOAD_EXISTING_MODEL': True,
        'ACTIVATION': 'relu',
        'LOSS': 'mse',
        'OPTIMIZER': 'Nadam',
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
print_timestamp('Plotting')
heat = PolarHeatmapPlotter()
heat.plot()
print_timestamp('Started main program')

env = PendulumEnv()  # Create some environments
dankAgent = Dank_Agent(env, hyperparameters, model)  # Create some agents
# memeAgent = DQN_Agent

# Start a training session with a given weight_file (e.g.)
training_weight_file = model["WEIGHT_FILE"]
training_report = dankAgent.train(training_parameters, training_weight_file)

# report = dankAgent.perform(model)  # perform with weights
# dankAgent.present() # Plot the results

###############################################################################
# Code dumpster
###############################################################################
