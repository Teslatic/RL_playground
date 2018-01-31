import random
from random import randrange
from time import time
import numpy as np
import keras
from collections import deque
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, Nadam
from keras import backend as K
from keras.layers import Flatten,Dense, Dropout
import datetime

class DankAgent():
    def __init__(self,action_interval, input_shape, batch_size):
        self.input_shape = input_shape
        self.action_interval = action_interval
        self.batch_size = batch_size
        self.step_length = 0.2#def: 0.1
        self.disc_actions = self._discretize_actions()
        self.gamma = 0.9
        self.epsilon = 1.0
        self.init_epsilon = 0.95
        self.eps_decay_rate = 0.01# 0.0006  0.00008 #def for 30000 ep: 0.00015
        self.learning_rate = 0.005 #def: 0.001
        self.decay_const = 0.05
        self.model = self._build_model()
        self.target_model = self._build_target_model()
        self.cnt = 1
        self.train_marker = 20
        self.update_target_marker = 1


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
        model.add(Dense(32,input_shape = (self.input_shape,),activation = 'relu',))
        # model.add(Dropout(0.5))
        model.add(Dense(64, activation = 'relu',))
        # model.add(Dropout(0.5))
        # model.add(Dense(64, activation = 'relu',))
        # model.add(Dropout(0.5))
        model.add(Dense(int(self.ticks), activation = 'linear' ,))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        return model

    def _build_target_model(self):
        model = Sequential()
        model.add(Dense(32,input_shape = (self.input_shape,),activation = 'relu',))
        # model.add(Dropout(0.5))
        model.add(Dense(64, activation = 'relu',))
        # model.add(Dropout(0.5))
        # model.add(Dense(64, activation = 'relu',))
        # model.add(Dropout(0.5))
        model.add(Dense(int(self.ticks), activation = 'linear' ,))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, en_explore = True):

        if en_explore == True:
            pick = np.random.choice(['random','greedy'], p = [self.epsilon,1-self.epsilon])

            # if random.random() <= self.epsilon and en_explore:
            if pick == 'random':
                # print(pick)
                # action = random.choice(self.disc_actions)==self.disc_actions
                action = np.where(self.disc_actions==random.choice(self.disc_actions))[0]
            else:
                # print(pick)
                state = state.reshape((1,self.input_shape))
                action_values = self.model.predict(state)

                # print("action values",action_values)
                # print("greedy pick from action values",np.argmax(action_values[0]))
                action = np.array((np.argmax(action_values),))
                # print("discrete actions", self.discrete_actions)
        else:
            state = state.reshape((1,self.input_shape))
            action_values = self.model.predict(state)
            action = np.array((np.argmax(action_values),))
            print(action)
        return action

    # def train(self,state, action, reward, next_state, done):
    def train(self, batch, memory, en_explore = False):

        state_batch = np.array([x[0] for x in batch])
        action_batch = np.array([x[1] for x in batch])
        reward_batch = np.array([x[2] for x in batch])
        next_state_batch = np.array([x[3] for x in batch])
        done_batch = np.array([x[4] for x in batch])


        q_target = self.model.predict(state_batch, batch_size = self.batch_size)


        a = self.model.predict(next_state_batch, batch_size = self.batch_size)
        t = self.target_model.predict(next_state_batch, batch_size = self.batch_size)
        # print("q_target shape: ",q_target.shape)
        # print("reward batch shape: ",reward_batch.shape)
        # print("action prediction shape: ", a.shape)
        # print("target model predictions: ", t.shape)
        for idx,action in enumerate(action_batch):
            # print(q_target[idx][action])
            q_target[idx][action] = reward_batch[idx] + self.gamma * t[idx][np.argmax(a[idx][action])]
        # print(q_target.shape)
        # print(action_batch)
        # print("q targets:", q_target)
        # print("a predictions",a)
        # print("t predictions",t)
        # for state, action, reward, next_state, done in batch:
        #     # state, action, reward, next_state, done = batch
        #     state = state.reshape((1,self.input_shape))
        #     # print(next_state)
        #     next_state = next_state.reshape((1,self.input_shape))
        #
        #
        #     target = self.model.predict(state)
        #
        #     a = self.model.predict(next_state)[0]
        #     print(a)
        #     t = self.target_model.predict(next_state)[0]
        #
        #     target[0][action] = reward + self.gamma * t[np.argmax(a)]
        if self.cnt % self.train_marker == 0 and en_explore == False:
            # self.model.fit(state, target,  epochs=1, verbose=0)
            self.model.train_on_batch(state_batch, q_target)
            # print("updated")



        self.cnt += 1



    def load(self, file_name):
        self.model.load_weights(file_name)
        print_timestamp()
        print("agent loaded weights from file '{}' ".format(file_name))
        self.update_target_model()

    def save(self, file_name):
        # print_timestamp()
        self.model.save_weights(file_name)
        # print("agent saved weights in file '{}' ".format(file_name))


def print_timestamp(string = ""):
    now = datetime.datetime.now()
    print(string + now.strftime("%Y-%m-%d %H:%M"))
