import numpy as np

import pickle

from assets.agents.DQN_Agent import _DQN_Agent
from assets.helperFunctions.timestamps import print_timestamp
from assets.helperFunctions.discretize import discretize
from assets.policies.policies import epsilon_greedy, greedy, greedy_batch
from assets.plotter.DankPlotters import Plotter
from assets.plotter.PolarHeatmapPlotter import PolarHeatmapPlotter
import assets.helperFunctions.FileManager as fm

class Dank_Agent(_DQN_Agent):
    """
    A model-based RL agent which uses 2 networks:
    estimator model: Tries to estimate the Q-values in a given state
    target model: The actual model, which will select an action
    """

    def __init__(self, env, hyperparameters, model):
        """
        - Initializing parameters according to the environment.
        - Initializing the agents hyperparameter set.
        - Discretizing the action_space for the usage of neural networks.
        - Creating the architecture file which is needed to build the neural
          network.
        - Initializing the estimator and target model.
        Args:
          env: The pendulum environment with binary rewards
          hyperparameters: A set of hyperparameters
          model: Specifications for the neural networks
        """
        # Das ist super, das ist elegant!
        super().__init__(env, hyperparameters, model)
        # Building a target model with the same architecture as the estimator.
        self.target = self._build_model(self.architecture)

###############################################################################
# Update method
###############################################################################

    def update_on_batch(self, batch_memory):
        """

        """
        # Unzip the batch
        batch_state, batch_action, batch_reward, batch_state_next = self.memory.unzip_batch(batch_memory)
        # Prediction time
        q_target = self.estimator.model.predict(batch_state)
        q_next1 = self.estimator.model.predict(batch_state_next)
        q_next2 = self.target.model.predict(batch_state_next)

        # Find the index of the best estimated action
        _, q_next_max_idx = greedy_batch(q_next1)

        batch_index11 = np.arange(self.batch_size, dtype=np.int32)
        q_next_max = q_next2[batch_index11, q_next_max_idx]
        q_target[batch_index11, batch_action] = batch_reward + self.gamma * q_next_max

        self.estimator.model.fit(batch_state, q_target, batch_size=self.batch_size, epochs=1, verbose=0)

    def update_target_model(self):
        """
        Takes the weights of the estimator and stores it into the policy network.
        """
        if self.save:
            self.estimator.save_weights(self.weights_file_out)
        self.target.model.set_weights(self.estimator.model.get_weights())

###############################################################################
# Training method
###############################################################################

    def learn(self, training_parameters, run=None):
        """The agent uses his training method on the given environment"""
        self._initialize_learning(training_parameters)
        for ep in range(self.training_episodes):
            self._initialize_episode(ep)
            for t in range(self.training_timesteps):
                self._initialize_timestep() # Just render
                self.Q = self.estimator.predict(self.state)
                self.action = epsilon_greedy(self.Q, self.epsilon, self.action_space)
                self.next_state, self.reward, done = self._act(self.action, True)
                self._analyze_timestep()
                if self.update:
                    self.update_on_batch(self.memory.get_batch(self.batch_size))
                if done:
                    break
                self._decrease_epsilon()
            self._analyze_episode(ep)
            print_timestamp("Episode {}/{}\t| Reward: {}\t| epsilon: {:.2f}\t".format((ep+1), self.training_episodes, self.episode_reward, self.epsilon))

            if self.update:
                self.update_target_model()

            if self.test:
                test_report = self.run_test(self.test_parameters)
                average_reward = test_report[1].round(2)  # dictionary
                self.average_reward_list.append(average_reward)
                print_timestamp('Test ended with average reward: {}'.format(average_reward))
                print_timestamp('Plotting')
                heat = PolarHeatmapPlotter(2, self.target, self.exp_dir)
                heat.plot(ep, average_reward, run)

        # Plotting time

        self.plotter.plot_training(self.exp_dir, self.reward_list)
        self.plotter.plot_test(self.exp_dir, self.average_reward_list, testeach=self.test_each)

        pickle.dump(self.reward_list, open('{}/report/training_report.p'.format(self.exp_dir), 'wb'))
        pickle.dump(self.average_reward_list, open('{}/report/test_report.p'.format(self.exp_dir), 'wb'))

        return self.reward_list, self.average_reward_list


###############################################################################
# Test method
###############################################################################

    def run_test(self, test_parameters):
        """
        The agent uses its current training state to test the training progress. The stored data can later be analysed with plot functions.
        """
        self._initialize_test(test_parameters)
        for ep in range(self.test_episodes):

            self._initialize_test_episode(ep)
            for step in range(self.test_timesteps):
                self._initialize_test_timestep()
                self.Q = self.estimator.predict(self.state)
                self.action_idx, self.action = greedy(self.Q, self.action_space)
                self.next_state, self.reward, done = self._act(self.action, vanilla=True)
                self._analyze_test_timestep()
                if done:
                    break
            self._analyze_test_episode(ep)
        return self.create_test_report(self.reward_list_test)

###############################################################################
# Full learning test
###############################################################################

    def run_n_learning_sessions(self, N_runs, train_parameters):
        # Create directories...
        actual_dir = train_parameters['ACTUAL_DIR']
        fm.create_plots_dir(actual_dir)
        fm.create_report_dir(actual_dir)
        # ... and report object
        multireport = []

        # Perform N runs
        for run in range(N_runs):
            self._reset_agent()  # reset agent models
            # train_parameters['ACTUAL_DIR'] = '{}/run{}'.format(actual_dir, run)
            print_timestamp('Starting run {}'.format(run))
            training_report, test_report = self.learn(train_parameters, run)
            # self.plotter.plot_training(actual_dir, training_report, run)
            # self.plotter.plot_test(actual_dir, test_report, run, self.test_each)
            multireport.append([training_report, test_report, self.test_each])
        # Create plot...

        # ... and save the report file
        pickle.dump(multireport, open('{}/report/multiReport.p'.format(actual_dir), 'wb'))
        self.plotter.plot_test_multireport(multireport, actual_dir, 'multireport_test')
        self.plotter.plot_training_multireport(multireport, actual_dir, 'multireport_training')
        return multireport

    def _reset_agent(self):
        self.estimator = self._build_model(self.architecture)
        self.target = self._build_model(self.architecture)

###############################################################################
# Parameter sweeping method
###############################################################################

    def parameter_sweep(self, parameter, sweep_vector, train_parameters, N_runs):
        """
        Sweeping one parameter according to the sweep_vector
        """
        # Create directories...
        actual_dir = train_parameters['ACTUAL_DIR']

        fm.create_report_dir(actual_dir)
        fm.create_plots_dir(actual_dir)
        # ... and report object
        sweepingReport = {}


        description = train_parameters['DESCRIPTION']
        descriptionfile = open('{}/description.txt'.format(actual_dir),"w")
        descriptionfile.write(description)
        descriptionfile.close()

        # Sweep through the parameter vector
        for sweep_parameter in sweep_vector:
            # Adjust the actual directory
            train_parameters['ACTUAL_DIR'] = '{}/{}_{}'.format(actual_dir, parameter, sweep_parameter)
            # Adjust the sweeped parameter
            train_parameters[parameter] = sweep_parameter

            print_timestamp('Parameter Sweep: Starting sweep on parameter {} with value {}'.format(parameter, sweep_parameter))
            multiReport = self.run_n_learning_sessions(N_runs, train_parameters)
            # Add multiReport to the sweepingReport
            sweepingReport.update({sweep_parameter: multiReport})
        # Create plot...

        # ... and save the report file
        pickle.dump(sweepingReport, open('{}/report/sweepReport.p'.format(actual_dir), 'wb'))
        return sweepingReport


###############################################################################
# Code dumpster
###############################################################################
