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

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.pendulum import PendulumEnv



###############################################################################
# Definition of functions
###############################################################################


def create_random_parameters(limit=1, size=4):
    """
    Creates a vector of with random values from -limit to +limit.
    """
    return np.random.rand(size)*(limit*2)-limit


def print_action(action):
    if action == RIGHT:
        print("Action taken: Going right")
    if action == LEFT:
        print("Action taken: Going left")
