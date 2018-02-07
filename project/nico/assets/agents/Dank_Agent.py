import numpy as np
from assets.agents.DQN_Agent import _DQN_Agent
from assets.helperFunctions.timestamps import print_timestamp
from assets.helperFunctions.discretize import discretize
from assets.policies.policies import epsilon_greedy, greedy, greedy_batch


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
        self._init_env_parameters(env)
        self._unzip_hyperparameters(hyperparameters)
        self.action_space = discretize(self.action_low, self.action_high, self.D_action)
        architecture = self._create_architecture(model)
        self.estimator = self._build_model(architecture)
        self.target = self._build_model(architecture)


###############################################################################
# Update method
###############################################################################

    def update_on_batch(self, batch_memory):
        """

        """
        # Unzip the batch
        batch_state, batch_action, batch_reward, batch_state_next = self.memory.unzip_batch(batch_memory)

        q_target = self.estimator.model.predict(batch_state)
        q_next1 = self.estimator.model.predict(batch_state_next)
        q_next2 = self.target.model.predict(batch_state_next)

        # Find the index of the best estimated action
        _, q_next_max_idx = greedy_batch(q_next1)

        batch_index11 = np.arange(self.batch_size, dtype=np.int32)
        q_next_max = q_next2[batch_index11, q_next_max_idx]
        # print("q_next2: {}".format(q_next2))
        # print("q_next_max: {}".format(q_next_max))
        # print(q_next_max)
        q_target[batch_index11, batch_action] = batch_reward + self.gamma * q_next_max

        self.history = self.estimator.model.fit(batch_state, q_target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.learn_counter += 1

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

    def train(self, training_parameters, weights_file=None):
        """The agent uses his training method on the given environment"""
        self._initialize_training(training_parameters, weights_file)
        for ep in range(self.training_episodes):
            self._initialize_episode(ep)
            for t in range(self.training_timesteps):
                self._initialize_timestep()
                self.Q = self.estimator.predict(self.state)
                self.action = epsilon_greedy(self.Q, self.epsilon, self.D_action)
                # self.action = self._select_action(self.state, self.policy)
                self.next_state, self.reward, done = self._act(self.action)  # Perform action
                self._analyze_timestep()

                if done:
                    break
            self._analyze_episode(ep)
            print_timestamp("Episode {}/{}\t| Reward: {}\t| epsilon: {:.2f}\t".format(ep, self.training_episodes, self.episode_reward, self.epsilon,))
            self._decrease_epsilon()
            if self.evaluate:
                report = self.training_evaluation(self.eval_parameters)
        return report


###############################################################################
# Evaluation method
###############################################################################

    def training_evaluation(self, evaluation_parameters):
        """
        The agent uses its current training state to evaluate the training progress. The stored data can later be analysed with plot functions.
        """
        self._initialize_evaluation(evaluation_parameters)
        for ep in range(self.evaluation_episodes):
            self.env.seed(ep)  # For comparability
            self._initialize_episode(ep)
            for step in range(self.evaluation_timesteps):
                self._initialize_evaluation_timestep()
                self.Q = self.estimator.predict(self.state)
                self.action, self.action_idx = greedy(self.Q)
                self.next_state, self.reward, done = self._act(self.action)
                self._analyze_evaluation_timestep()
                if(ep > 10):
                    # print_timestamp("Updating on batch")
                    self.update_on_batch(self.memory.get_batch(self.batch_size))
                if done:
                    break
            self._analyze_episode(ep)
            if(ep > 10):
                # print_timestamp("Updating target model weights")
                self.update_target_model()
        # print_timestamp("Updating target model")
        report = self._prepare_report()
        return report

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

    def _prepare_report(self):
        eval_reward_average = np.mean(self.reward_list)
        print(eval_reward_average)
        return [self.reward_list, eval_reward_average]

###############################################################################
# Code dumpster
###############################################################################
