import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from assets.agents.DQN_Agent import DQN_Agent
from assets.helperFunctions.timestamps import print_timestamp
from assets.helperFunctions.discretize import discretize
# from assets.policies.reinforce import reinforce
from assets.policies.improvement.epsilon_greedy import *

class Reinforce_Agent(DQN_Agent):
    """
    A model-based RL agent which uses 2 networks:
    estimator model: Tries to estimate the Q-values in a given state
    target model: The actual model, which will select an action
    env: The pendulum environment with binary rewards
    hyperparameters: A set of hyperparameters
    model: Specifications for the neural networks
    """

    def __init__(self, env, hyperparameters, model):
        """
        - Initializing parameters according to the environment.
        - Initializing the agents hyperparameter set.
        - Discretizing the action_space for the usage of neural networks.
        - Creating the architecture file which is needed to build the neural
          network.
        - Initializing the estimator and target model.
        """
        self._init_env_parameters(env)
        self._unzip_hyperparameters(hyperparameters)
        self.action_space = discretize(self.action_low, self.action_high, self.D_action)
        architecture = self._create_architecture(model)
        self.estimator = self._build_model(architecture)
        self.target = self._build_model(architecture)

        EpisodeStats = namedtuple("Stats",["epis_lengths", "epis_rewards"])

    def reinforce(self, policy):
        for t, ep in enumerate(self.episode_history):
            generate_return(ep[t:])
            policy.update(sess,s,a,total_return)

    def generate_return(self, episode_history_from_t):
        """
        Calculates the return
        """
        state_history = episode_history_from_t[0]
        action_history = episode_history_from_t[1]
        reward_history = episode_history_from_t[2]

        total_return = sum(self.gamma**i * reward_history for i,t in enumerate(episode[t:]))
        return state_history

    def update_policy(self, ep):
        """
        """
        if(ep > 15): # MAGIC NUMBER
            sample_index = np.random.choice(self.memory_size, size=self.batch_size) # Selects [batch size] many indizes
            self.batch = self.memory[self.sample_index, :]
            if self.learn_counter % 200 == 0:
                self.update_target_model()
                self.learn_counter = 0

            batch_state = batch_memory[:, :self.state_size]
            batch_action = batch_memory[:, self.state_size].astype(int)
            batch_reward = batch_memory[:, self.state_size+1]
            batch_state_next = batch_memory[:, -self.state_size-1:-1]
            batch_done = batch_memory[:, -1]

            Q_target = self.model.predict(batch_state)
            q_next1 = self.model.predict(batch_state_next)
            q_next2 = self.target_model.predict(batch_state_next)
            batch_action_withMaxQ = np.argmax(q_next1, axis=1)
            batch_index11 = np.arange(self.batch_size, dtype=np.int32)
            q_next_Max = q_next2[batch_index11, batch_action_withMaxQ]
            q_target[batch_index11, batch_action] = batch_reward + self.gamma * q_next_Max
            self.history = self.model.fit(batch_state, q_target,                batch_size=32, epochs=1, verbose=0)
            self.learn_counter +=1
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon -= 0.00001

###############################################################################
# Training methods
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
                self.update_policy(ep)
                if done:
                    break
            self._analyze_episode(ep)
            if self.evaluate:
                report = _evaluate(evaluation_parameters)
            # reinforce(self, )

    def _evaluate(self, evaluation_parameters):
        """
        The agent uses its current training state to evaluate the training progress. The stored data can later be analysed with plot functions.
        """
        self._initialize_evaluation(evaluation_parameters)
        for ep in range(self.evaluation_episodes):
            self.env.seed(ep)  # For comparability
            self._initialize_episode(ep)
            for step in range(self.evaluation_timesteps):
                self._initialize_timestep()
                self.Q = self.estimator.predict(self.state)
                self.action, self.action_idx = greedy(self.Q)
                self.next_state, self.reward, done = self._act(self.action)  #
                self._analyze_timestep()
                if done:
                    break
        report = self._prepare_report()
        return report

    def _prepare_report(self):
        return None

###############################################################################
# Code dumpster
###############################################################################

    # # def select_action(self, state, policy):
    # def select_action(self, state):
    #     """
    #     The agents selects the next action according to its policy evaluation method (in general epsilon-greedy).
    #     """
    #     if False:
    #         pass
    #         policy = make_epsilon_greedy_policy(self.estimator, self.epsilon,
    #                         self.action_space)
    #         action = np.random.choice(np.arange(len(policy(state))),p=policy(state))
    #     else:
    #         action = np.random.random(1)
    #         return action


    # def perform(self, evaluation_parameters, weights_file=None):
    #     """
    #     After training is done. The agent can now show how well it will perform with its training set. A lot of statistics and plot are supported.
    #     """
    #     self._initialize_evaluation(evaluation_parameters, weight_file)
    #     for ep in range(self.evaluation_episodes):
    #         self._initialize_episode(ep)
    #         for t in range(self.evaluation_timesteps):
    #             self._initialize_timestep
    #             self.Q = self.estimator.predict(state)
    #             self.action, self.action_idx = greedy(self.Q)
    #             # self.policy = greedy_policy(self.Q)  # Less efficient than Liors code but no ifs
    #             # self.action = self.select_action(self.state)
    #             self.action = self.best_Q
    #             self.next_state, self.reward, done = self._act(self.env, self.action)  # Perform action
    #             self._analyze_timestep()
    #             if done:
    #                 break
    #         self._analyze_episode(ep)
    #     return None
