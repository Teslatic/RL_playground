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
import keras
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.layers import Flatten,Dense, Dropout

if "../" not in sys.path:
    sys.path.append("../")
# from lib.envs.pendulum import PendulumEnv
from assets.agents.RL_Agent import RL_Agent
from assets.helperFunctions.timestamps import print_timestamp

class MBRL_Agent(RL_Agent):
    """
    An RL agent that uses a model of the environment.

    Args:
        model (object): The model of the MB agent
    """


    def __init__(self, env, hyperparameters, model, *kwargs):
        pass

    def _build_model(self, weights_file):
        """
        Builds a model. Architecture should be defined by input parameters
        """
        pass

    def update_model(self):
        pass
