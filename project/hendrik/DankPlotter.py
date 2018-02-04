




class Plotter():
    def __init__(self, report):
        # number of runs per feature to test
        self.runs = report.run_num
        '''
        histories of every run, coul be a dictionary, which maps to the
        according feature?
        e.g. batch sizes have been examined, so we get a
        dictionary like this:
        report.histories =  {
                            32 : [ [run1],[run2],[run3],[run4],[run5] ],
                            64 : [ [run1],[run2],[run3],[run4],[run5] ],
                            128: [ [run1],[run2],[run3],[run4],[run5] ],
                            256: [ [run1],[run2],[run3],[run4],[run5] ],
                            512: [ [run1],[run2],[run3],[run4],[run5] ]
                            }
        (for the rest i will just assume this structure at hand)
        '''

        self.histories = report.histories

        '''
        self.features is a list of features that are tested, e.g. the
        batch sizes [32, 64, 128, 256]
        '''
        self.features = _get_features()


    # extract features from histories dictionary as a set and converts it
    # to a list which is returned
    def _get_features(self):
        self.featues = [_ for _ in self.histories]
        return self.features

    '''
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

    std. deviation is calculated along axis 1
    Std. dev.    :  [s1,  s2, ..., sN]
    '''

    def plot_avg_reward(self):
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

        plt.figure()
        for idx,val in enumerate(mean_summary):
            plt.plot(val[0], label = '{}'.format(features[idx]))
        plt.plot(merged_mean, label ='mean of means')
        plt.plot(merged_mean+mean_std[0], label='+', linestyle = '-.')
        plt.plot(merged_mean-mean_std[0], label='+', linestyle = '-.')

        plt.legend()
        plt.show()
