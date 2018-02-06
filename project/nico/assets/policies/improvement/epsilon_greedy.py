import sys
import numpy as np
import itertools
from collections import defaultdict, namedtuple

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def make_epsilon_greedy_policy(Q, epsilon, nA):
  """
  Creates an epsilon-greedy policy based on a given Q-function and epsilon.

  Args:
    Q: A dictionary that maps from state -> action-values.
      Each value is a numpy array of length nA (see below)
    epsilon: The probability to select a random action . float between 0 and 1.
    nA: Number of actions in the environment.

  Returns:
    A function that takes the observation as an argument and returns
    the probabilities for each action in the form of a numpy array of length nA.
  """

  def policy_fn(observation):
    act_prob = np.ones(nA, dtype=float)* epsilon/nA
    max_action = np.argmax(Q[observation])
    act_prob[max_action] += (1.0-epsilon)
    return act_prob
  return policy_fn

def epsilon_greedy_lior(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
      Q: The Q-values for a given state
      epsilon: The probability to select a random action. float between 0 and 1.
      nA: Number of actions in the environment.

    Returns:
      A function that takes the observation as an argument and returns
      the probabilities for each action in the form of a numpy array of length nA.
    """
    if np.random.random() <= epsilon:
        action = np.random.randint(0, nA)
    else:
        action = np.array(Q)
    return action
