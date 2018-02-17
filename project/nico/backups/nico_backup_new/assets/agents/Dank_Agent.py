import numpy as np
from assets.agents.DQN_Agent import _DQN_Agent
from assets.helperFunctions.timestamps import print_timestamp
from assets.helperFunctions.discretize import discretize
from assets.policies.policies import epsilon_greedy, greedy, greedy_batch
from assets.plotter.PolarHeatmapPlotter import PolarHeatmapPlotter

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
                heat = PolarHeatmapPlotter(2, self.target, self.experiment_dir)
                heat.plot(ep, average_reward, run)
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
        report = self._prepare_report()
        return report

###############################################################################
# Full learning test
###############################################################################

    def run_n_learning_sessions(self, N_sessions, train_parameters):
        report = []
        for run in range(N_sessions):
            self._reset_agent()
            print_timestamp('Starting Run {}'.format(run))
            training_report, test_report = self.learn(train_parameters, run)
            report.append([training_report, test_report, self.test_each])
        return report

    def _reset_agent(self):
        self.estimator = self._build_model(self.architecture)
        self.target = self._build_model(self.architecture)

###############################################################################
# Parameter sweeping method
###############################################################################

# # Sweeping one parameter
# for sweep_parameter in batch_size_sweep:
#     training_parameters["BATCH_SIZE"] = sweep_parameter
#     print_timestamp('Parameter Sweep: Starting training with batchsize
#                     {}'.format(sweep_parameter))
#     dankAgent.train(training_parameters, training_weight_file)  # Train agent
#     report = dankAgent.perform(model)  # perform with weights
#     dankAgent.present() # Plot the results



###############################################################################
# Code dumpster
###############################################################################
