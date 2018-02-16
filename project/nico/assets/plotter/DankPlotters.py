import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pylab import *
from scipy.interpolate import griddata
from assets.policies.policies import greedy_batch
from assets.helperFunctions.timestamps import print_timestamp

import time
import datetime

class Plotter():
    """
    sweepReports: Contain the data which is used for comparing the features which had been swept. This is realized as a dictionary:

    sweepReports =  {
                    32 : [ [run1],[run2],[run3],[run4],[run5] ],
                    64 : [ [run1],[run2],[run3],[run4],[run5] ],
                    128: [ [run1],[run2],[run3],[run4],[run5] ],
                    256: [ [run1],[run2],[run3],[run4],[run5] ],
                    512: [ [run1],[run2],[run3],[run4],[run5] ]
                    }
    This is equivalent to:
    sweepReports =  {
                    32 : multiReport,
                    64 : multiReport,
                    128: multiReport,
                    256: multiReport,
                    512: multiReport
                    }


    multiReports: Contain the data which is used for comparing different runs. This is realized as a list:
    multiReport =   [
                    [[training_report], [test_reports], test_each],  # run 1
                    [[training_report], [test_reports], test_each],  # run 2
                    ...
                    [[training_report], [test_reports], test_each],  # run n
                    ]

    the list [[training_report], [test_reports], test_each] can be imagined as a runReport, even though this object is not defined (yet). It would have been unneccessary work.
    """
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir

###############################################################################
# open funtions
###############################################################################

    def open_sweepReport(self, sweepReport):
        """
        Converts the dictionary to a list of multireports and the feature_vector.
        """
        multireports = []
        feature_vector = []
        for parameter in sweepReport:
            multiReport = sweepReport[parameter]
            multireports.append(multiReport)
            feature_vector.append(parameter)
        return multireports, feature_vector

    def open_multiReport(self, multiReport):
        """
        Converts a multireport into the a list of training_report, test_reports and a test_each_vector.
        """
        training_reports = []
        test_reports = []
        test_each = []
        for report in range(len(multiReport)):
            training_reports.append(multiReport[report][0])
            test_reports.append(multiReport[report][1])
            test_each.append(multiReport[report][2])
        return training_reports, test_reports, test_each


###############################################################################
# save method
###############################################################################

    def _save_plot(self, save_path, name):
        now = datetime.datetime.now()
        file_string_png = save_path + '/plots/' + now.strftime('%Y%m%d_%H%M%S') + name + '.png'
        file_string_png = '{}/plots/png/{}_{}.png'.format(save_path, now.strftime('%Y%m%d_%H%M%S'), name)
        file_string_pdf = '{}/plots/pdf/{}_{}.pdf'.format(save_path, now.strftime('%Y%m%d_%H%M%S'), name)
        plt.savefig(file_string_png)
        plt.savefig(file_string_pdf)

###############################################################################
# statistics calculation
###############################################################################

    def training_statistics(self, multiReport):
        """
        Gets a multiReport and calculates mean and standard deviation of the test_report.
        """
        training_reports, _, _ = self.open_multiReport(multiReport)
        mean_vector = [np.mean(i) for i in zip(*training_reports)]
        std_vector = [np.std(i) for i in zip(*training_reports)]
        return mean_vector, std_vector

    def test_statistics(self, multiReport):
        """
        Gets a multiReport and calculates mean and standard deviation of the test_report.
        """
        _, test_reports, test_each = self.open_multiReport(multiReport)
        mean_vector = [np.mean(i) for i in zip(*test_reports)]
        std_vector = [np.std(i) for i in zip(*test_reports)]
        return mean_vector, std_vector, test_each[0]

    def training_statistics_sweep(self, multireports):
        feature_mean_vector = []
        feature_std_vector = []
        for report in range(len(multireports)):
            mean_vector, std_vector = self.training_statistics(multireports[report])
            feature_mean_vector.append(mean_vector)
            feature_std_vector.append(std_vector)
        meanofmeans = [np.mean(i) for i in zip(*feature_mean_vector)]
        stdofstd = [np.std(i) for i in zip(*feature_std_vector)]
        return meanofmeans, stdofstd

###############################################################################
# Plotting funtions - single runs
###############################################################################

    def plot_training(self, save_path, training_report, run_num=None):
        """
        Make sure that the directory has been created by the FileManager.
        """
        plt.figure()
        x_data = np.arange(len(training_report))
        plt.plot(x_data, training_report,label = "DankAgent")
        plt.xlabel("Training episode")
        plt.ylabel("Total episode reward")
        plt.legend(loc='upper left')

        if run_num is None:
            self._save_plot(save_path, 'training_report')
        else:
            self._save_plot(save_path, 'training_report_run{}'.format(run_num))
        plt.close()

    def plot_test(self, save_path, test_report, run_num=None, testeach=None):
        """
        Make sure that the directory has been created by the FileManager.
        """
        plt.figure()
        if testeach is None:
            plt.plot(test_report, label = "DankAgent")
        else:
            x_data = testeach*(np.arange(len(test_report))+1)
            plt.plot(x_data, test_report, label = "DankAgent")
        plt.xlabel("Test episode")
        plt.ylabel("Average Test Reward")
        plt.legend(loc='upper left')
        if run_num is None:
            self._save_plot(save_path, 'test_report')
        else:
            self._save_plot(save_path, 'test_report_run{}'.format(run_num))
        plt.close()

###############################################################################
# Plotting funtions - multiple runs (with multiReport)
###############################################################################

    def plot_test_multireport(self, multiReport, save_path, name):
        """

        """
        _, test_reports, _ = self.open_multiReport(multiReport)
        mean_vector, std_vector, test_each = self.test_statistics(multiReport)

        # create_plot_test_mean_std

        plt.figure()
        x_data = test_each*np.arange(len(mean_vector)+1)  # +1
        for run in range(len(multiReport)):
            plt.plot(x_data, test_reports[run], label = 'run {}'.format(run))
        plt.plot(x_data, mean_vector, label ='mean reward')
        plt.plot(x_data, np.add(mean_vector, std_vector), label='+STD', linestyle = '-.')
        plt.plot(x_data, np.subtract(mean_vector, std_vector), label='-STD', linestyle = '-.')
        plt.title("Test results for several runs")
        plt.xlabel("Intermediate test after training episode")
        plt.ylabel("Average reward")
        plt.legend(loc='upper left')
        self._save_plot(save_path, name)
        plt.close()

    def plot_training_multireport(self, multiReport, save_path, name):
        """

        """
        training_report, _, _ = self.open_multiReport(multiReport)
        mean_vector, std_vector = self.training_statistics(multiReport)

        plt.figure()
        x_data = np.arange(len(mean_vector))  # +1
        for run in range(len(multiReport)):
            plt.plot(x_data, training_report[run], label = 'run {}'.format(run))
        plt.plot(x_data, mean_vector, label ='mean of means')
        plt.plot(x_data, np.add(mean_vector, std_vector), label='+STD', linestyle = '-.')
        plt.plot(x_data, np.subtract(mean_vector, std_vector), label='-STD', linestyle = '-.')
        plt.title("Training results for several runs")
        plt.xlabel("Training episode")
        plt.ylabel("Episode reward")
        plt.legend(loc='upper left')
        self._save_plot(save_path, name)
        plt.close()

###############################################################################
# Plotting funtions - multiple features (with sweepReport)
###############################################################################

    def plot_training_sweep(self, sweepReport, save_path, name):
        multireports, feature_vector = self.open_sweepReport(sweepReport)
        meanofmeans, stdofstd = self.training_statistics_sweep(multireports)
        self.plot_mean_std_training(meanofmeans, stdofstd, feature_vector, save_path, name)
        self.create_sweep_plot_training(multireports, meanofmeans, stdofstd, feature_vector, save_path, name)

    def plot_mean_std_training(self, mean_vector, std_vector, feature_vector, save_path, name):
        plt.figure()
        x_data = np.arange(len(mean_vector))  # +1
        plt.plot(x_data, mean_vector, label ='mean of means')
        plt.plot(x_data, np.add(mean_vector, std_vector), label='+STD', linestyle = '-.')
        plt.plot(x_data, np.subtract(mean_vector, std_vector), label='-STD', linestyle = '-.')

        plt.title("Training results for several runs")
        plt.xlabel("Training episode")
        plt.ylabel("Episode reward")
        plt.legend(loc='upper left')
        self._save_plot(save_path, name)
        plt.close()

    def create_sweep_plot_training(self, multireports, mean_vector, std_vector, feature_vector, save_path, name):
        plt.figure()
        x_data = np.arange(len(mean_vector))  # +1
        # for feature in range(len(multireports)):
        #     training_report, _, _= self.open_multiReport(multireports[feature])
        #     # print("dimension of x_data: {}".format(len(x_data)))
        #     # print("dimension of y_data: {}".format(len(curves)))
        #     # print(feature_vector[feature])
        #     plt.plot(x_data, training_report, label = 'feature {}'.format(feature_vector[feature]))
        plt.plot(x_data, mean_vector, label ='mean of means')
        plt.plot(x_data, np.add(mean_vector, std_vector), label='+STD', linestyle = '-.')
        plt.plot(x_data, np.subtract(mean_vector, std_vector), label='-STD', linestyle = '-.')

        plt.title("Training results for several runs")
        plt.xlabel("Training episode")
        plt.ylabel("Episode reward")
        plt.legend(loc='upper left')
        self._save_plot(save_path, name)
        plt.close()

###############################################################################
# Not used
###############################################################################

    def calculate_training_statistics(self, multiReport):
        """

        """
        # Creates a [[][][]] vector with n = len(features) entries
        mean_summary = [[] for _ in range(len(multiReport))]
        merged_hist = []

        # get means according to above illustration, should result in a list
        # of Nfeatures lists, that are of length Nepisodes
        for point in range (multiReport[0]):
            for run in range(multiReport):
                mean_vector.append(np.mean())

        for run in enumerate(multiReport):
            mean_summary[idx].append(np.mean(self.histories[idx][val], axis=0))

        # calculate standard deviation of the means
        mean_std = np.std(mean_summary,axis=0)

        # merge all means into one central mean
        merged_mean = np.mean(np.mean(merged_hist,axis=0),axis=0)
        return merged_mean
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

    def create_multiple_training_plots(self, multiReport):
        N_runs = len(multiReport)
        for run in range(N_runs):
            data_train = multiReport[run][0]
            data_test = multiReport[run][1]
            self.plot_training(data_train, run)
            self.plot_test(data_test, run)

    def show_all_plots(self):
        """
        Make sure that the directory has been created by the FileManager.
        """
        plt.show()

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

    # def present(self, smoothing_window=10, noshow=False):
    #     # Plot the episode length over time
    #     fig_eps_length = plt.figure(figsize=(10,5))
    #     plt.plot(self.episode_stats.length)
    #     plt.xlabel("Episode")
    #     plt.ylabel("Episode Length")
    #     plt.title("Episode Length over Time")
    #     fig_eps_length.savefig('episode_lengths.png')
    #     if noshow:
    #         plt.close(fig_eps_length)
    #     else:
    #         plt.show(fig_eps_length)
    #
    #         # Plot the episode reward over time
    #         fig_eps_rewards = plt.figure(figsize=(10,5))
    #         rewards_smoothed = pd.Series(self.plt_episode_counter).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #         plt.plot(rewards_smoothed)
    #         plt.xlabel("Episode")
    #         plt.ylabel("Episode Reward (Smoothed)")
    #         plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    #         fig_eps_rewards.savefig('reward.png')
    #         if noshow:
    #             plt.close(fig_eps_rewards)
    #         else:
    #             plt.show(fig_eps_rewards)
