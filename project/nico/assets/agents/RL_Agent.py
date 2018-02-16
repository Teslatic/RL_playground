# from os import path
import numpy as np
# import pickle
# if "../" not in sys.path:
#     sys.path.append("../")
from assets.helperFunctions.timestamps import print_timestamp
from assets.memory.memory import TransitionBuffer
from assets.policies.policies import make_epsilon_greedy_policy
from assets.helperFunctions.discretize import discretize
from assets.reports.Report import TrainingReport
from assets.plotter.DankPlotters import Plotter
# from assets.helperFunctions.FileManager import
import assets.helperFunctions.FileManager as fm


class RL_Agent():
    """Basic RL agent class"""

###############################################################################
# Initializing methods
###############################################################################

    def __init__(self, env, hyperparameters):
        """
        Sets the agents basic parameters according to the environment.
        Unzips the given hyperparameters for the agent.
        Discretizes the action space (imported method).
        Checked --> Works correctly.
        """
        self._init_env_parameters(env)
        self._unzip_hyperparameters(hyperparameters)
        self.action_space = discretize(self.action_low, self.action_high, self.D_action)

    def _init_env_parameters(self, env):
        """
        Initializing parameters according to the environment.
        Checked:
        print("D_state: {}, action_high: {}, action_low: {}".format(self.D_state, self.action_high, self.action_low))
        """
        self.env = env
        self.D_state = self.env.observation_space.shape[0]
        self.action_high = self.env.action_space.high[0]
        self.action_low = self.env.action_space.low[0]


    def _unzip_hyperparameters(self, hyperparameters):
        """Unzips the hyperparameter set and writes it into single variables"""
        self.D_action = hyperparameters['D_ACTION']
        self.gamma = hyperparameters['GAMMA']

###############################################################################
# General layout of an RL task
###############################################################################

    def learn(self, training_parameters, weights_file):
        """The agent uses his training method on the given environment"""
        self._initialize_learning(training_parameters, weights_file)
        for ep in range(self.training_episodes):
            self._initialize_episode(ep)
            for t in range(self.training_timesteps):
                self._initialize_timestep()
                self.Q = self.estimator.predict(self.state)
                self.policy = make_epsilon_greedy_policy(self.Q, self.epsilon, self.D_action)
                self.action = self._select_action(self.state, self.policy)
                self.next_state, self.reward, done, _ = self._perform_action(self.env, self.action)  # Perform action
                self._analyze_timestep()
                self._update_policy()
                if done:
                    break
            self._analyze_episode(ep)

###############################################################################
# Initializing and analysing episodes and timesteps
###############################################################################

    def _initialize_learning(self, training_parameters):
        """
        This method initialize all members needed for the training.
        """
        self._unzip_training_parameters(training_parameters)

        # Create folder structure
        # fm.create_experiment(training_parameters['EXPERIMENT_DIR'])
        self.exp_dir = fm.create_path_and_experiment(self.exp_dir, 'run')

        # Create the Plotter
        self.plotter = Plotter(self.exp_root_dir)

        # Initialize Flags
        self.save = False  # indicates if weight file has to be saved
        self.render = False  # indicates if progess has to be rendered
        self.store = False  # indicates if progress has to be stored
        self.test = False  # indicates if testrun has to be started
        self.update = False  # indicates if model has to be updated
        # Initialize memory
        self.memory_depth = 8  # magic: 2xD_state = 6 + action, reward
        self.memory = TransitionBuffer(self.memory_size, self.memory_depth)
        self.memory.reset_memory()
        # Initialize report objects
        self.report = TrainingReport(self.training_episodes)
        self.reward_list = []  # Has to be a report
        self.average_reward_list = []

    def _unzip_training_parameters(self, training_parameters):
        self.exp_root_dir = training_parameters['EXPERIMENT_ROOT_DIR']
        self.exp_dir = training_parameters['ACTUAL_DIR']
        self.training_episodes = training_parameters['TRAINING_EPISODES']
        self.training_timesteps = training_parameters['TRAINING_TIMESTEPS']
        self.batch_size = training_parameters['BATCH_SIZE']
        self.memory_size = training_parameters['MEMORY_SIZE']
        self.auto_saver = training_parameters['AUTO_SAVER']

        self.show_progress = training_parameters['SHOW_PROGRESS']
        self.store_progress = training_parameters['STORE_PROGRESS']
        self.test_each = training_parameters['TEST_EACH']
        self.weights_file = training_parameters['TRAINING_FILE']

        # Epsilon: Maybe new class for epsilon handling
        self.epsilon_parameters = training_parameters['EPSILON_PARAM']
        self.epsilon = self.epsilon_parameters['EPSILON']
        self.epsilon_min = self.epsilon_parameters['EPSILON_MIN']
        self.epsilon_init = self.epsilon_parameters['EPSILON_INIT']
        self.epsilon_decay_rate = self.epsilon_parameters['EPS_DECAY_RATE']
        self.const_decay = self.epsilon_parameters['CONST_DECAY']

        # test parameters
        self.test_parameters = training_parameters['TEST_PARAMETERS']

    def _initialize_test(self, test_parameters):

        self.test_episodes = test_parameters['TEST_EPISODES']
        self.test_timesteps = test_parameters['TEST_TIMESTEPS']
        self.show_test_progress = test_parameters['RENDER_TEST']  # Rework
        print_timestamp("Starting intermediate test with {} episodes ({} timesteps each).".format(self.test_episodes, self.test_timesteps))
        self.reward_list_test = []  # Has to be a report

    def _initialize_test_episode(self, ep):
        self._initialize_episode(ep)
        self.render_test = self.set_flag_every(self.show_test_progress, ep)
        self.env.seed(ep)  # For comparability
        self.state = self.env.reset()

    def _initialize_episode(self, ep):
        """
        Every show_progress_each steps the render flag is set to true.
        Every save_weights_each steps the save flag is set to true.
        """
        self.state = self.env.reset()
        self.episode_reward = 0
        self.episode_history = []  # Clearing the history
        # FLAGS
        if ep != 0:  # No flags for the first episode
            self._set_episode_flags(ep)

    def _set_episode_flags(self, ep):
        """
        This private method sets and resets the flags:
          render: indicates if progess has to be rendered
          store: indicates if progress has to be stored
          save: indicates if weight file has to be saved
          test: indicates if testrun has to be started
        """
        self.render = self.set_flag_every(self.show_progress, ep)
        self.store = self.set_flag_every(self.store_progress, ep)
        self.save = self.set_flag_every(self.auto_saver, ep)
        self.test = self.set_flag_every(self.test_each, ep)
        self.update = self.set_flag_from(10, ep)  # magic

    def set_flag_every(self, var_input, ep):
        if var_input is None:
            return False
        else:
            return True if (ep % var_input == 0) else False

    def set_flag_from(self, min_limit, ep):
        return False if (ep < min_limit) else True

    def _analyze_episode(self, ep):
        # self.episode_stats.reward[ep] = self.episode_reward
        # self.episode_stats.length[ep] = ep
        # This will be handled witha report object
        self.reward_list.append(self.episode_reward)

    def _analyze_test_episode(self, ep):
        # self.episode_stats.reward[ep] = self.episode_reward
        # self.episode_stats.length[ep] = ep
        # This will be handled witha report object
        # self.report.add2testreport(ep, self.test_timesteps, self.episode_reward)
        self.reward_list_test.append(self.episode_reward)

    def _decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_rate

    def _initialize_timestep(self):
        """This method initialize the actual timestep."""
        if self.render:
            self.env.render()

    def _initialize_test_timestep(self):
        """This method initialize the actual timestep."""
        if self.render_test:
            self.env.render()

    def _analyze_timestep(self):
        """
        """
        self.episode_reward += self.reward
        trans = self.memory.create_transition(self.state, self.action, self.reward, self.next_state)
        self.memory.store(trans)
        #  self.episode_history.append((self.next_state, self.action, self.reward))
        self.state = self.next_state

    def _analyze_test_timestep(self):
        """
        """
        self.episode_reward += self.reward
        self.state = self.next_state

    def _select_action(self, state, policy):
        """
        The agent performs the given action on the environment and returns:
        observation: state information
        reward: reward by the environment
        done: flag to check if environment has terminated
        info: additional information
        """
        return policy(state)  # Returns the action according to the policy

    def _act(self, action, vanilla):
        """
        The agent performs the given action on the environment and returns:
        observation: state information
        reward: reward by the environment
        done: flag to check if environment has terminated
        info: additional information
        """
        next_state, reward, done, _ = self.env.step(action, vanilla)
        return next_state, reward, done

    def create_test_report(self, reward_list):
        average_test_reward = np.mean(reward_list)
        testReport = [reward_list, average_test_reward]
        return testReport

###############################################################################
# Code dumpster
###############################################################################
