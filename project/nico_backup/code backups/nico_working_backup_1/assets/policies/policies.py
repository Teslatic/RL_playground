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

def epsilon_greedy(Q, epsilon, action_space):
    """
    An epsilon-greedy policy based on given Q-function and epsilon.

    Args:
      Q: The Q-values for a given state
      epsilon: The probability to select a random action. float between 0 and 1.
      nA: Number of actions in the environment.

    Returns:
      The index of the selected action.
    """
    nA = length(action_space)
    if np.random.random() <= epsilon:
        action = np.random.randint(0, nA)
    else:
        action = action_space[np.argmax(Q)]
    return action

def greedy(Q, action_space):
    """
    A greedy policy based on given Q-values.

    Args:
      Q: The Q-values for a all state.

    Returns:
      The value of the selected action.
      The index of the selected action.
    """
    max_idx = np.argmax(Q)
    action = action_space[max_idx]
    return max_idx, action

def greedy_batch(Q):
    """
    A greedy policy based on given Q-values.

    Args:
      Q: A batch of Q-values for a all state.

    Returns:
      The value of the selected action.
      The index of the selected action.
    """
    return np.max(Q, axis=1), np.argmax(Q, axis=1)
