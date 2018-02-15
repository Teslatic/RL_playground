import numpy as np

def discretize(limit_high, limit_low, N_action):
    """
    Method which is used to discretize a continous space.
    Returns a vector with the discretized space.
    """
    return np.linspace(limit_low, limit_high, N_action)
