import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pylab import *
from scipy.interpolate import griddata

class Plotter():
    def __init__(self, report):
        """
        histories of every run, could be a dictionary, which maps to the
        according feature?
        e.g. batch sizes have been examined, so we get a dictionary like this:
        report.histories =  {
                            32 : [ [run1],[run2],[run3],[run4],[run5] ],
                            64 : [ [run1],[run2],[run3],[run4],[run5] ],
                            128: [ [run1],[run2],[run3],[run4],[run5] ],
                            256: [ [run1],[run2],[run3],[run4],[run5] ],
                            512: [ [run1],[run2],[run3],[run4],[run5] ]
                            }
        (for the rest i will just assume this structure at hand)
        """
        # number of runs per feature to test
        self.runs = report.run_num
        self.histories = report.histories
        # self.features is a list of features that are tested, e.g. the        batch sizes [32, 64, 128, 256]
        self.features = _get_features()


    # extract features from histories dictionary as a set and converts it
    # to a list which is returned
    def _get_features(self):
        self.features = [_ for _ in self.histories]
        return self.features

    """
    method used to plot the average test rewards from the runs per feature.
    average is calculated along axis 0, e.g. for batch 32:

            Run1    [x11, x12,..., x1N]
                      |
                      V
            Run2    [x21, x22,..., x2N]
                      |
                      V
            Run3    [x31, x32,..., x3N]
                      |
                      V
            Run4    [x41, x42,..., x4N]
                      |
                      V
            Run5    [x51, x52,..., x5N]
                      |
                      v
    List of means:  [µ1,  µ2, ..., µN]

    std. deviation is calculated along axis 0
    Std. dev.    :  [s1,  s2, ..., sN]
    """

###############################################################################
# statistics calculation
###############################################################################

    def calculate_training_statistics(self):
        """

        """
        # Creates a [[][][]] vector with n = len(features) entries
        mean_summary = [[] for _ in range(len(features))]
        merged_hist = []

        # get means according to above illustration, should result in a list
        # of Nfeatures lists, that are of length Nepisodes
        for idx, val in enumerate(self.features):
            mean_summary[idx].append(np.mean(self.histories[idx][val], axis=0))

        # calculate standard deviation of the means
        mean_std = np.std(mean_summary,axis=0)

        # merge all means into one central mean
        merged_mean = np.mean(np.mean(merged_hist,axis=0),axis=0)

    def calculate_evaluation_statistics(self):
        """

        """
        # Creates a [[][][]] vector with n = len(features) entries
        mean_summary = [[] for _ in range(len(features))]
        merged_hist = []

        # get means according to above illustration, should result in a list
        # of Nfeatures lists, that are of length Nepisodes
        for idx, val in enumerate(self.features):
            mean_summary[idx].append(np.mean(self.histories[idx][val], axis=0))

        # calculate standard deviation of the means
        mean_std = np.std(mean_summary,axis=0)

        # merge all means into one central mean
        merged_mean = np.mean(np.mean(merged_hist,axis=0),axis=0)

    def calculate_sweep_statistics(self):
        """

        """
        # Creates a [[][][]] vector with n = len(features) entries
        mean_summary = [[] for _ in range(len(features))]
        merged_hist = []

        # get means according to above illustration, should result in a list
        # of Nfeatures lists, that are of length Nepisodes
        for idx, val in enumerate(self.features):
            mean_summary[idx].append(np.mean(self.histories[idx][val], axis=0))

        # calculate standard deviation of the means
        mean_std = np.std(mean_summary,axis=0)

        # merge all means into one central mean
        merged_mean = np.mean(np.mean(merged_hist,axis=0),axis=0)

###############################################################################
# Plotting funtions
###############################################################################


    def plot_training_report(self):
        """

        """
        plt.figure()
        for idx,val in enumerate(mean_summary):
            plt.plot(val[0], label = '{}'.format(features[idx]))
        plt.plot(merged_mean, label ='mean of means')
        plt.plot(merged_mean+mean_std[0], label='+', linestyle = '-.')
        plt.plot(merged_mean-mean_std[0], label='-', linestyle = '-.')
        plt.legend()
        plt.show()

    def plot_evalutation_report(self):
        """

        """
        plt.figure()
        for idx,val in enumerate(mean_summary):
            plt.plot(val[0], label = '{}'.format(features[idx]))
        plt.plot(merged_mean, label ='mean of means')
        plt.plot(merged_mean+mean_std[0], label='+', linestyle = '-.')
        plt.plot(merged_mean-mean_std[0], label='-', linestyle = '-.')
        plt.legend()
        plt.show()

    def plot_sweep_report(self):
        """

        """
        plt.figure()
        for idx,val in enumerate(mean_summary):
            plt.plot(val[0], label = '{}'.format(features[idx]))
        plt.plot(merged_mean, label ='mean of means')
        plt.plot(merged_mean+mean_std[0], label='+', linestyle = '-.')
        plt.plot(merged_mean-mean_std[0], label='-', linestyle = '-.')
        plt.legend()
        plt.show()

    def present(self, smoothing_window=10, noshow=False):
        # Plot the episode length over time
        fig_eps_length = plt.figure(figsize=(10,5))
        plt.plot(self.episode_stats.length)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Episode Length over Time")
        fig_eps_length.savefig('episode_lengths.png')
        if noshow:
            plt.close(fig_eps_length)
        else:
            plt.show(fig_eps_length)

            # Plot the episode reward over time
            fig_eps_rewards = plt.figure(figsize=(10,5))
            rewards_smoothed = pd.Series(self.plt_episode_counter).rolling(smoothing_window, min_periods=smoothing_window).mean()
            plt.plot(rewards_smoothed)
            plt.xlabel("Episode")
            plt.ylabel("Episode Reward (Smoothed)")
            plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
            fig_eps_rewards.savefig('reward.png')
            if noshow:
                plt.close(fig_eps_rewards)
            else:
                plt.show(fig_eps_rewards)


class PolarHeatmapPlotter():
    def __init__(self, max_r, max_theta, function):
        #create 5000 Random points distributed within the circle radius 100
        max_theta = 2.0 * np.pi
        number_points = 5000
        points = np.random.rand(number_points,2)*[max_r,max_theta]
        #Some function to generate values for these points,
        #this could be values = np.random.rand(number_points)
        # Convert samples to states in vector format (clockwise and counterclck)
        # Predict selected action with estimator and greedy
        values = points[:,0] * np.sin(points[:,1])* np.cos(points[:,1])
        #now we create a grid of values, interpolated from our random sample above
        self.theta = np.linspace(0.0, max_theta, 100)
        self.r = np.linspace(0, max_r, 200)
        grid_r, grid_theta = np.meshgrid(self.r, self.theta)
        self.data = griddata(points, values, (grid_r, grid_theta), method='cubic',fill_value=0)
        # self.uniform_data = np.random.rand(10, 12)
        # plt.imshow(uniform_data, cmap='hot', interpolation='nearest')
        #Create a polar projection
        ax1 = plt.subplot(projection="polar")
        ax1.pcolormesh(self.theta,self.r,self.data.T)

    def update(self):
        pass

    def plot(self):
        plt.show()
        # ax = sns.heatmap(self.uniform_data, annot=True, fmt="d")
