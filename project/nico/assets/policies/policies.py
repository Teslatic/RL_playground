#!/usr/bin/python3

###############################################################################
# Import packages
###############################################################################


from os import path
import sys
import time
import itertools
from collections import defaultdict, namedtuple

# Import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.pendulum import PendulumEnv


###############################################################################
# Policy
###############################################################################


EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
VALID_ACTIONS = [0, 1, 2]

class Policy():
  def __init__(self, policy):
    if policy == 'DQN':
        self._build_model()

  def _build_model(self):
    """
    Builds the Tensorflow graph.
    """

    self.states_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="states_pl")
    # The TD target value
    self.targets_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_pl")
    # Integer id of which action was selected
    self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions_pl")

    batch_size = tf.shape(self.states_pl)[0]

    self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, activation_fn=tf.nn.relu,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn=tf.nn.relu,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 3, activation_fn=None,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.predictions = tf.nn.softmax(self.fc3)     # softmax prediction

    # Get the predictions for the chosen actions only
    gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
    self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
    self.objective = -tf.log(self.action_predictions)*self.targets_pl
    self.optimizer = tf.train.AdamOptimizer(0.0001)
    self.train_op = self.optimizer.minimize(self.objective)


  def predict(self, sess, s):
    """
    Args:
      sess: TensorFlow session
      states: array of states for which we want to predict the actions.
    Returns:
      The prediction of the output tensor.
    """
    f_dict = {self.states_pl: [s]}
    p = sess.run(self.predictions, f_dict)[0]
    return np.random.choice(VALID_ACTIONS, p=p), p

  def update(self, sess, s, a, y):
    """
    Updates the weights of the neural network, based on its targets, its
    predictions, its loss and its optimizer.

    Args:
      sess: TensorFlow session.
      states: [current_state] or states of batch
      actions: [current_action] or actions of batch
      targets: [current_target] or targets of batch
    """
    feed_dict = {self.states_pl: [s], self.targets_pl: [y], self.actions_pl:
                    [a]}
    sess.run(self.train_op, feed_dict)
    return

class BestPolicy(Policy):
  def __init__(self):
    Policy.__init__(self)
    self._associate = self._register_associate()

  def _register_associate(self):
    tf_vars = tf.trainable_variables()
    total_vars = len(tf_vars)
    op_holder = []
    for idx,var in enumerate(tf_vars[0:total_vars//2]):
      op_holder.append(tf_vars[idx+total_vars//2].assign((var.value())))
    return op_holder

  def update(self, sess):
    for op in self._associate:
      sess.run(op)
