import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.layers import Flatten,Dense, Dropout

from assets.agents.agents import RL_agent
from assets.agents.agents import MBRL_agent
import assets.policies
from assets.helperFunctions.timestamps import print_timestamp
from assets.policies.reinforce import reinforce


class Reinforce_Agent(MBRL_agent):
    """
    A model-based RL agent which uses :
    env: The pendulum environment with binary rewards
    policy: This is epsilon-greedy
    model: DQN
    hyperparameters:
    """

    def __init__(self, env, hyperparameters, model, weights_file=None):
        """
        Initializing the agent with its hyperparameter set. If a model is necessary (e.g. for prediction) it is also build.
        """
        # Unzip env members
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.action_interval = [-2, 2] # Magic number for now
        self.action_bin_size = 0.1 # Magic number for now
        self.n_actions = (np.abs(self.action_interval[0]) +                    np.abs(self.action_interval[-1])) / self.action_bin_size

        # Unzip hyperparameters
        self.learning_rate = hyperparameters['LEARNING_RATE']
        # self.training_timesteps = hyperparameters['TIMESTEPS']
        self.render = hyperparameters['LEARNING_RATE']
        self.batch_size = hyperparameters['BATCH_SIZE']
        self.gamma = hyperparameters['GAMMA']
        self.epsilon = hyperparameters['EPSILON']
        self.epsilon_init = hyperparameters['EPSILON_INIT']
        self.epsilon_decay_rate = hyperparameters['EPS_DECAY_RATE']
        self.const_decay = hyperparameters['CONST_DECAY']
        self.step_length = hyperparameters['STEP_LENGTH']

        # self.best_reward = 0
        self.total_reward = 0
        self.best_parameters = None
        self.episode_counter = 0

        # Initialize plotting parameters
        self.plt_reward = []
        self.plt_episode_counter = []

        if model["MODEL_TYPE"] == 'DQN':
            self.estimator = self._build_DQN_model(model["WEIGTH_FILE"])

    def _build_DQN_model(self, weights_file):
        """
        Builds a DQN model. Architecture should be defined by input parameters
        """
        self.model = Sequential()
        self.model.add(Dense(32,input_shape = (3,),activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(int(self.n_actions)))
        self.model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        self.model.summary() # Prints out a short summary of the architecture

        self.weights_file = weights_file
        if self.weights_file == None:
            pass
        else:
            self._load(self.weights_file)

    def _discretize_actions(self):
        """
        Private method which is used to discretize the action space.
        """
        self.action_bin_size = (np.abs(self.action_interval[0]) + np.abs(self.action_interval[-1])) / self.step_length
        self.discrete_actions = np.linspace(self.action_interval[0],self.action_interval[-1],self.ticks)
        return self.discrete_actions

    def select_action(self, state):
        """
        The agents selects the next action according to its policy evaluation method (in general epsilon-greedy).
        """
        if False:
            pass
            policy = make_epsilon_greedy_policy(self.estimator, self.epsilon,
                            self.action_space)
            action = np.random.choice(np.arange(len(policy(state))),p=policy(state))
        else:
            action = np.random.random(1)
            return action
            # print('ACTION: {}'.format(action))

    def perform_action(self, env, action):
        """
        The agent performs the given action on the environment
        """
        # observation, reward, done, info =
        return env.step(action)

###############################################################################
# Training methods
###############################################################################

    def train(self, training_parameters, model):
        """The agent uses his training method on the given environment"""
        self._initialize_training(training_parameters, model)
        for ep in range(self.training_episodes):

            self._initialize_episode(ep)
            # print('Training episode {}'.format(self.episode_counter))
            for t in range(self.training_timesteps):
                self._initialize_timestep()
                action = self.select_action(self.state)[0]  # Select action
                obs, reward, done, _ = self.perform_action(self.env, action)  # Perform action
                self.total_reward += reward
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
                # Prepare episode history for plotting
                self.plt_reward.append(self.total_reward)
                self.plt_episode_counter.append(ep)

    def _initialize_training(self, hyperparameters, weights_file):
        """
        This method unzips the hyperparameter set
        """
        self.episode_counter = 0  # Keeps track of current episode
        self.training_episodes = hyperparameters['TRAINING_EPISODES']
        self.timesteps = hyperparameters['TIMESTEPS']
        self.memory_size = hyperparameters['MEMORY_SIZE']
        self.auto_saver = hyperparameters['AUTO_SAVER']
        self.show_progress = hyperparameters['SHOW_PROGRESS']
        self.weights_file = weights_file
        self.save_weights_each = 50
        self.show_progress_each = 100
        print('TRAINING_EPISODES: {}'.format(self.training_episodes))

    def _initialize_episode(self, ep):
        """
        Every show_progress_each steps the render flag is set to true.
        Every save_weights_each steps the save flag is set to true.
        """
        print(show_progress_each)
        if ep % self.show_progress_each == 0:
            self.render = True
        else:
            self.render = False

        if ep % self.save_weights_each == 0:
            self._save(self.weights_file)
        self.total_reward = 0
        self.episode_counter += 1
        self.state = self.env.reset()

    def _initialize_timestep(self):

        if self.render:
            self.env.render()

    def _save(self, file_name):
        print_timestamp()
        self.model.save_weights(file_name)
        print("agent saved weights in file '{}' ".format(file_name))

    def _load(self, file_name):
        self.model.load_weights(file_name)
        print_timestamp()
        print("agent loaded weights from file '{}' ".format(file_name))
        self.update_target_model()

    def perform(self, load_existent_weights=False, weights_file=None):

        return None


###############################################################################
# Plotting funtions
###############################################################################

    def present(self, smoothing_window=10, noshow=False):
        # Plot the episode length over time
        fig_eps_length = plt.figure(figsize=(10,5))
        plt.plot(episode_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Episode Length over Time")
        fig1.savefig('episode_lengths.png')
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
