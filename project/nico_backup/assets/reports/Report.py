import numpy as np
from collections import namedtuple

class TrainingReport():
    """
    This class is used to store the results in a form which can later be easily plotted.
    """
    def __init__(self, training_episodes):
        """
        Creating the data structure which is later given to the plotting function.
        """
        self.training_episodes = training_episodes
        zero_vector = np.zeros(self.training_episodes)
        self.EpisodeStats = namedtuple("Stats", ["length", "reward"])
        self.report = self.EpisodeStats(length=zero_vector, reward=zero_vector)
        self.test_report = self.EpisodeStats(length=zero_vector, reward=zero_vector)

    def add2testreport(self, ep, length, reward):
        self.report[ep](length=length, reward=reward)
