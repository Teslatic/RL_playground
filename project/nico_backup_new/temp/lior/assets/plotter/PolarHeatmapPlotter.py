import sys
from os import path
import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pylab import *
from scipy.interpolate import griddata
from assets.policies.policies import greedy_batch


class PolarHeatmapPlotter():
    def __init__(self, max_r, estimator, directory):
        main_path = path.dirname(path.abspath(sys.modules['__main__'].__file__))
        self.dir_path = main_path + '/' + directory
        #create 5000 Random points distributed within the circle radius 2
        self.max_r = max_r
        self.max_theta = 2.0 * np.pi
        N_points = 3600
        points = np.random.rand(N_points, 2)*[self.max_r, self.max_theta]
        # Create grid
        self.theta = np.linspace(0.0, self.max_theta, 360)
        self.r = np.linspace(0, self.max_r, 20)
        grid_r, grid_theta = np.meshgrid(self.r, self.theta)
        action_idx_clw, _ =self. _estimate_policy(points, 'positive', estimator)
        action_idx_cclw, _ = self._estimate_policy(points, 'negative', estimator)

        self.data_clw = griddata(points, action_idx_clw, (grid_r, grid_theta), method='cubic',fill_value=0)
        self.data_cclw = griddata(points, action_idx_cclw, (grid_r, grid_theta), method='cubic',fill_value=0)

    def _estimate_policy(self, points, direction, estimator):
        sine = np.sin(points[:,1])
        cosine = np.cos(points[:,1])
        velocity = points[:, 0]
        states = []
        for idx in range(len(sine)):
            if direction is 'positive':
                states.append([sine[idx], cosine[idx], velocity[idx]])
            if direction is 'negative':
                states.append([sine[idx], cosine[idx], - velocity[idx]])

        # Estimate the q-values and the index of the best action.
        states = np.array(states)
        q_values = estimator.model.predict(states)
        return greedy_batch(q_values)

    def _create_plot(self, theta, r, data, data_cclw, ep, reward, direction):
        """
        Create a polar projection.
        """

        fig_left = plt.subplot(1,2,1, projection="polar")
        fig_left.set_title('Clockwise')
        plt.thetagrids([theta * 15 for theta in range(360//15)])
        fig_left.pcolormesh(theta, r, data.T, cmap='RdBu')

        fig_right = plt.subplot(1,2,2, projection="polar")
        fig_right.set_title('Counterclockwise')
        plt.thetagrids([theta * 15 for theta in range(360//15)])
        fig_right.pcolormesh(theta, r, data_cclw.T, cmap='RdBu')

        plt.tight_layout()
        plt.suptitle('Policy in episode {} with reward {}'.format(ep, reward))

        # fig_right.set_title('Policy ({}) in episode {} with reward {}'.format(direction, ep, reward))
        # plt.plot()
        # fig.set_theta_zero_location("N")

        # plt.thetagrids([theta * 15 for theta in range(360//15)])
        # plt.rgrids([.1 * _ for _ in range(1, 2*self.max_r)])
        return plt.plot()

    def save(self, absolute_path):
        now = datetime.datetime.now()
        file_string = '/{}/results/policy_plots/{}_policy.png'.format(absolute_path,  now.strftime('%Y%m%d_%H%M%S'))
        plt.savefig(file_string)

    def update(self):
        pass

    def plot(self, ep, reward):
        self.fig = self._create_plot(self.theta, self.r, self.data_clw, self.data_cclw, ep, reward, 'clockwise')
        self.save(self.dir_path)
        # self.fig = self._create_plot(self.theta, self.r, self.data_cclw, ep, reward, 'counterclockwise')
        # self.save(self.dir_path, 'counterclockwise')

        # ax = sns.heatmap(self.uniform_data, annot=True, fmt="d")

    def show_plot(self):
        plt.show()
