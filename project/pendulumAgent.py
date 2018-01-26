import random
from random import randrange
from time import time
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Flatten,Dense, Dropout
import datetime

class DankAgent():
    def __init__(self,action_interval):
        self.action_interval = action_interval
        self.step_length = 0.1
        self.disc_actions = self._discretize_actions()
        self.gamma = 0.95
        self.epsilon = 1.0
        self.init_epsilon = 0.9
        self.eps_decay_rate = 0.00015
        self.learning_rate = 0.001
        self.decay_const = 0.1
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _discretize_actions(self):
        self.ticks = (np.abs(self.action_interval[0]) + np.abs(self.action_interval[-1])) / self.step_length
        self.discrete_actions = np.linspace(self.action_interval[0],self.action_interval[-1],self.ticks)
        print("I discretized your action-space for you:")
        print("|---------------------------------------------------------------|")
        print("|Action Interval {}\t| Ticks: {:.0f}\t| Delta {:.4f}\t|".format(self.action_interval,self.ticks,np.diff(self.discrete_actions)[-1]))
        print("|---------------------------------------------------------------|")
        return self.discrete_actions

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32,input_shape = (3,),activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(int(self.ticks)))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, en_explore = True):
        if random.random() <= self.epsilon and en_explore:
            # action = random.choice(self.disc_actions)==self.disc_actions
            action = np.where(self.disc_actions==random.choice(self.disc_actions))[0]
        else:
            state = state.reshape((1,3))
            action_values = self.model.predict(state)

            # print("action values",action_values)
            # print("greedy pick from action values",np.argmax(action_values[0]))
            action = np.array((np.argmax(action_values),))
            # print("discrete actions", self.discrete_actions)

        return action

    # def train(self,state, action, next_state, reward, done):
    def train(self, batch):
        state, action, reward, next_state, done = batch
        state = state.reshape((1,3))
        next_state = next_state.reshape((1,3))

        target = self.model.predict(state)
        if done:
            target[0][action] = reward
        else:
            a = self.model.predict(next_state)[0]
            t = self.target_model.predict(next_state)[0]

            target[0][action] = reward + self.gamma * t[np.argmax(a)]

        self.history = self.model.fit(state, target, epochs=1, verbose=0)


    def load(self, file_name):
        self.model.load_weights(file_name)
        print_timestamp()
        print("agent loaded weights from file '{}' ".format(file_name))
        self.update_target_model()

    def save(self, file_name):
        print_timestamp()
        self.model.save_weights(file_name)
        print("agent saved weights in file '{}' ".format(file_name))


def print_timestamp(string = ""):
    now = datetime.datetime.now()
    print(string + now.strftime("%Y-%m-%d %H:%M"))
