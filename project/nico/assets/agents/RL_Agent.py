# from os import path
import numpy as np
from collections import namedtuple

# if "../" not in sys.path:
#     sys.path.append("../")
from assets.helperFunctions.timestamps import print_timestamp
from assets.memory.memory import TransitionBuffer
from assets.policies.policies import make_epsilon_greedy_policy


class RL_Agent():
    """Basic RL agent class"""

    def __init__(self, env, hyperparameters):
        """
        Sets the agents basic parameters according to the environment.
        Unzips the given hyperparameters for the agent.
        """
        self._init_env_parameters(env)
        self._unzip_hyperparameters(hyperparameters)

    def _init_env_parameters(self, env):
        """
        Initializing parameters according to the environment.
        """
        self.env = env
        self.observation_space = self.env.observation_space
        self.D_state = self.env.observation_space.shape[0]
        self.action_space_cont = self.env.action_space
        self.action_high = self.action_space_cont.high[0]
        self.action_low = self.action_space_cont.low[0]
        self.highscore_reward = 0  # best ever achieved reward
        self.best_parameters = None

    def _unzip_hyperparameters(self, hyperparameters):
        """Unzips the hyperparameter set and writes it into single variables"""
        self.learning_rate = hyperparameters['LEARNING_RATE']
        self.render = hyperparameters['LEARNING_RATE']
        self.gamma = hyperparameters['GAMMA']
        self.epsilon = hyperparameters['EPSILON']
        self.epsilon_init = hyperparameters['EPSILON_INIT']
        self.epsilon_decay_rate = hyperparameters['EPS_DECAY_RATE']
        self.const_decay = hyperparameters['CONST_DECAY']
        self.step_length = hyperparameters['STEP_LENGTH']
        self.D_action = hyperparameters['D_ACTION']

    def _initialize_training(self, training_parameters, weights_file):
        """
        This method initialize all members needed for the training.
        """
        self.training_episodes = training_parameters['TRAINING_EPISODES']
        self.batch_size = training_parameters['BATCH_SIZE']
        self.training_timesteps = training_parameters['TRAINING_TIMESTEPS']
        self.memory_size = training_parameters['MEMORY_SIZE']
        self.auto_saver = training_parameters['AUTO_SAVER']
        self.show_progress = training_parameters['SHOW_PROGRESS']
        self.store_progress = training_parameters['STORE_PROGRESS']
        self.evaluate_each = training_parameters['EVALUATE_EACH']

        self.save = False  # indicates if weight file has to be saved
        self.render = False  # indictaes if progess has to be rendered
        self.store = False  # indicates if progress has to be stored
        self.evaluate = False
        # evaluation parameters
        self.eval_parameters = training_parameters['EVAL_PARAMETERS']
        self._initialize_evaluation(self.eval_parameters)
        self.weights_file = weights_file
        self.learn_counter = 0
        # Initialize memory
        self.memory_depth = 8  # magic: 2xD_state = 6 + action, reward
        self.memory = TransitionBuffer(self.memory_size, self.memory_depth)

        # Initialize report parameters
        zero_vector = np.zeros(self.training_episodes)
        self.EpisodeStats = namedtuple("Stats", ["length", "reward"])
        self.episode_stats = self.EpisodeStats(length=zero_vector, reward=zero_vector)

    def _initialize_evaluation(self, eval_parameters):
        print_timestamp("Starting evaluation. ")
        self.evaluation_episodes = eval_parameters['EVALUATION_EPISODES']
        self.evaluation_timesteps = eval_parameters['EVALUATION_TIMESTEPS']
        self.render_evaluation = eval_parameters['RENDER_EVALUATE']
        self.reward_list = []

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
            if self.show_progress is None:
                self.render = False
            else:
                self.render = True if (ep % self.show_progress == 0) else False

            if self.store_progress is None:
                self.store = False
            else:
                self.store = True if (ep % self.store_progress == 0) else False

            if self.auto_saver is None:
                self.save = False
            else:
                self.save = True if (ep % self.auto_saver == 0) else False

            if self.evaluate_each is None:
                self.evaluate = False
            else:
                self.evaluate = True if (ep % self.evaluate_each == 0) else False

    def _analyze_episode(self, ep):
        self.episode_stats.reward[ep] = self.episode_reward
        self.episode_stats.length[ep] = ep
        self.reward_list.append(self.episode_reward)

    def _decrease_epsilon(self):
        if self.epsilon > self.epsilon_init:
            self.epsilon -= self.epsilon_decay_rate

    def _initialize_timestep(self):
        """This method initialize the actual timestep."""
        if self.render:
            self.env.render()

    def _initialize_evaluation_timestep(self):
        """This method initialize the actual timestep."""
        if self.render_evaluation:
            self.env.render()

    def _analyze_timestep(self):
        """
        """
        self.episode_reward += self.reward
        trans = self.memory.create_transition(self.state, self.action, self.reward, self.next_state)
        self.memory.store(trans)
        self.episode_history.append((self.next_state, self.action, self.reward))
        self.state = self.next_state

    def _analyze_evaluation_timestep(self):
        """
        """
        self.episode_reward += self.reward
        # trans = self.memory.create_transition(self.state, self.action, self.reward, self.next_state)
        # self.memory.store(trans)
        # self.episode_history.append((self.next_state, self.action, self.reward))
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

    def _act(self, action):
        """
        The agent performs the given action on the environment and returns:
        observation: state information
        reward: reward by the environment
        done: flag to check if environment has terminated
        info: additional information
        """
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def train(self, training_parameters, weights_file):
        """The agent uses his training method on the given environment"""
        self._initialize_training(training_parameters, weights_file)
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
# Code dumpster
###############################################################################
        #
        # avg_reward = np.mean(self.episode_stats.reward)
        # print("average reward",avg_reward)
        # self.episode_stats.reward.append(self.episode_reward)
        # self.episode_stats.length.append(self.episode_length)
        # if self.store:
        #     self.avg_reward_list.append(self.average_return)

# print_timestamp('Start training with {} episodes'.format(self.training_episodes))
