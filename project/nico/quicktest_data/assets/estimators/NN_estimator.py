#!/usr/bin/python3

###############################################################################
# Import packages
###############################################################################


from os import path
import sys
import time
import itertools
from collections import defaultdict, namedtuple

import keras
from keras.models import Sequential
from keras.optimizers import Adam, Nadam, RMSprop
from keras import backend as K
from keras.layers import Flatten,Dense, Dropout

from assets.helperFunctions.timestamps import print_timestamp
from assets.helperFunctions.tensor_conversions import *

# Import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.pendulum import PendulumEnv

# EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
###############################################################################
# Policy
###############################################################################


class NN_estimator():
    """
    Neural network function approximator.
    """

    def __init__(self, architecture):
        self.model = self._build_model(architecture)

    def _build_model(self, architecture):
        """
        Builds the Keras model according to the architecture file.
        """
        D_in, D_out, action_space, activation, loss, optimizer, learning_rate = self._unzip_architecture(architecture)
        self.action_space = action_space
        model = Sequential()
        model.add(Dense(16,input_shape = (D_in,),activation = activation))
        model.add(Dense(16, activation = activation))
        model.add(Dense(32, activation = activation))
        model.add(Dense(D_out, activation = 'linear'))
        model.compile(loss = loss, optimizer = optimizer(lr = learning_rate))
        # model.summary() # Prints out a short summary of the architecture
        return model
        # Dropout prevents overfitting # self.model.add(Dropout(0.5))

    def _unzip_architecture(self, architecture):
        D_in = architecture["D_IN"]
        D_out = architecture["D_OUT"]
        action_space = architecture["ACTION_SPACE"]
        activation = architecture["ACTIVATION"]
        loss = architecture["LOSS"]
        if architecture["OPTIMIZER"] == 'Adam':
            optimizer = Adam
        elif architecture["OPTIMIZER"] == 'Nadam':
            optimizer = Nadam
        elif architecture["OPTIMIZER"] == 'RMSprop':
            optimizer = RMSprop
        learning_rate = architecture["LEARNING_RATE"]
        return D_in, D_out, action_space, activation, loss, optimizer, learning_rate

    def load_weights(self, weight_file):
        if weight_file == None:
            print_timestamp("Not loading any weight file")
        else:
            self.model.load_weights(weight_file)
            print_timestamp("Weight file loaded from '{}' ".format(weight_file))
        self.update_target_model

    def save_weights(self, weight_file):
        self.model.save_weights(weight_file)
        # print_timestamp("Weight file save into '{}' ".format(weight_file))

    def predict(self, state):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        state = convert_vector2tensor(state)
        Q = self.model.predict(state)
        Q = convert_tensor2vector(Q)
        return Q

    def update_target_model(self):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.
        """
        self.target_model.set_weights(self.model.get_weights())

###############################################################################
# Code dumpster
###############################################################################

# class BestPolicy(Policy):
#   def __init__(self):
#     Policy.__init__(self)
#     self._associate = self._register_associate()
#
#   def _register_associate(self):
#     tf_vars = tf.trainable_variables()
#     total_vars = len(tf_vars)
#     op_holder = []
#     for idx,var in enumerate(tf_vars[0:total_vars//2]):
#       op_holder.append(tf_vars[idx+total_vars//2].assign((var.value())))
#     return op_holder
#
#   def update(self, sess):
#     for op in self._associate:
#       sess.run(op)

    # self.states_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="states_pl")
    # # The TD target value
    # self.targets_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_pl")
    # # Integer id of which action was selected
    # self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions_pl")
    #
    # batch_size = tf.shape(self.states_pl)[0]
    #
    # self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, activation_fn=tf.nn.relu,
    #   weights_initializer=tf.random_uniform_initializer(0, 0.5))
    # self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn=tf.nn.relu,
    #   weights_initializer=tf.random_uniform_initializer(0, 0.5))
    # self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 3, activation_fn=None,
    #   weights_initializer=tf.random_uniform_initializer(0, 0.5))
    # self.predictions = tf.nn.softmax(self.fc3)     # softmax prediction
    #
    # # Get the predictions for the chosen actions only
    # gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
    # self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
    # self.objective = -tf.log(self.action_predictions)*self.targets_pl
    # self.optimizer = tf.train.AdamOptimizer(0.0001)
    # self.train_op = self.optimizer.minimize(self.objective)


        # def _build_target_model(self):
        #     activation_curve = 'relu'
        #     unit_num = 20
        #     action_dim = self.action_dim
        #     input_eval = Input(shape=(self.D_state,))
        #     l1 = Dense(unit_num, activation=activation_curve)(input_eval)
        #     val_layer = Dense(1)(l1)
        #     val_layer = RepeatVector(action_dim)(val_layer)
        #     val_layer = Reshape(target_shape=(action_dim,), input_shape=(action_dim, 1,))(val_layer)
        #     adv_layer = Dense(action_dim)(l1)
        #     merge_layer = Add()([val_layer, adv_layer])
        #     model = Model(inputs=input_eval, outputs=merge_layer)
        #     model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), )
        #     return model
