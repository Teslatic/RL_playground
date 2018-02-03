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
    def __init__(self,action_interval, input_shape, batch_size, network_setup):
        self.input_shape = input_shape
        self.action_interval = action_interval
        self.batch_size = batch_size
        # self.step_length = 0.16#def: 0.1
        self.step_length = 1.0 #0.5
        self.disc_actions = self._discretize_actions()
        self.gamma = 0.9
        self.epsilon = 1.0
        self.init_epsilon = 0.95
        self.eps_decay_rate = 0.045# 0.0006  0.00008 #def for 30000 ep: 0.00015
        self.learning_rate = 0.001 #def: 0.001
        self.decay_const = 0.05
        self.setup = network_setup
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.cnt = 1
        self.train_marker = 1
        self.update_target_marker = 200
        self.memory_counter = 0
        self.memory_size = 3000
        self.memory = np.empty([self.memory_size,9])

    def _discretize_actions(self):
        self.ticks = (np.abs(self.action_interval[0]) + np.abs(self.action_interval[-1])) / self.step_length
        self.discrete_actions = np.linspace(self.action_interval[0],self.action_interval[-1],self.ticks)
        print("I discretized your action-space for you:")
        print("|---------------------------------------------------------------|")
        print("|Action Interval {}\t| Ticks: {:.0f}\t| Delta {:.4f}\t|".format(self.action_interval,self.ticks,np.diff(self.discrete_actions)[-1]))
        print("|---------------------------------------------------------------|")
        return self.discrete_actions

    def _build_model(self):
        if self.setup == 'Vanilla':
            model = Sequential()
            model.add(Dense(16,input_shape = (self.input_shape,),activation = 'relu',))
            model.add(Dense(16, activation = 'relu',))
            model.add(Dense(32, activation = 'relu',))
            model.add(Dense(int(self.ticks), activation = 'linear' ,))
            model.compile(loss = 'mse', optimizer = Nadam(lr = self.learning_rate))

        if self.setup == 'Dropout':
            model = Sequential()
            model.add(Dense(32,input_shape = (self.input_shape,),activation = 'relu',))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation = 'relu',))
            model.add(Dropout(0.5))
            model.add(Dense(int(self.ticks), activation = 'linear' ,))
            model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))

        if self.setup == 'Deeper':
            model = Sequential()
            model.add(Dense(32,input_shape = (self.input_shape,),activation = 'relu',))
            model.add(Dense(64, activation = 'relu',))
            model.add(Dense(64, activation = 'relu',))
            model.add(Dense(64, activation = 'relu',))
            model.add(Dense(int(self.ticks), activation = 'linear' ,))
            model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))

        if self.setup == 'Wider':
            model = Sequential()
            model.add(Dense(512,input_shape = (self.input_shape,),activation = 'relu',))
            model.add(Dense(1024, activation = 'relu',))
            model.add(Dense(int(self.ticks), activation = 'linear' ,))
            model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))

        if self.setup == 'DeepWideDrop':
            model = Sequential()
            model.add(Dense(32,input_shape = (self.input_shape,),activation = 'relu',))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation = 'relu',))
            model.add(Dropout(0.5))
            model.add(Dense(512, activation = 'relu',))
            model.add(Dropout(0.5))
            model.add(Dense(256, activation = 'relu',))
            model.add(Dropout(0.5))
            model.add(Dense(int(self.ticks), activation = 'linear' ,))
            model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))

        return model


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, en_explore = True):

        if en_explore == True:
            pick = np.random.choice(['random','greedy'], p = [self.epsilon,1-self.epsilon])

            if pick == 'random':

                action = np.where(self.disc_actions==random.choice(self.disc_actions))[0]

            else:

                #state = state.reshape((1,self.input_shape))
                action_values = self.model.predict(state)
                action = np.array((np.argmax(action_values),))
        else:
            #state = state.reshape((1,self.input_shape))
            action_values = self.model.predict(state)
            action = np.array((np.argmax(action_values),))
        return action



    def memory_store(self, state_now, action, reward, state_next, done):

        action = np.reshape(action, [1, 1])
        reward = np.reshape(reward, [1, 1])
        done = np.reshape(done, [1, 1])
        transition = np.hstack((state_now, action, reward, state_next, done))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def train(self, batch):

        state_batch = np.array([x[0] for x in batch])
        state_batch = state_batch.reshape((self.batch_size,3))
        action_batch = np.array([x[1] for x in batch])
        action_batch = action_batch.reshape((self.batch_size,1))
        reward_batch = np.array([x[2] for x in batch])
        reward_batch = reward_batch.reshape((self.batch_size,1))
        next_state_batch = np.array([x[3] for x in batch])
        next_state_batch = next_state_batch.reshape((self.batch_size,3))
        done_batch = np.array([x[4] for x in batch])
        done_batch = done_batch.reshape((self.batch_size,1))

        #print(state_batch.shape)
        q_target = self.model.predict(state_batch)#, batch_size = self.batch_size)
        # print("q_target")
        # print(q_target)
        # print(q_target.shape)

        a = self.model.predict(next_state_batch)#, batch_size = self.batch_size)
        # print("a = q_next1")
        # print(a)
        # print(a.shape)

        t = self.target_model.predict(next_state_batch)#, batch_size = self.batch_size)
        # print("t = q_nextt")
        # print(t)
        # print(t.shape)

        batch_action_withMaxQ = np.argmax(a, axis=0)
        # print("axis=0",batch_action_withMaxQ)
        batch_action_withMaxQ = np.argmax(a, axis=1)
        # print("axis=1",batch_action_withMaxQ)





        for idx,action in enumerate(action_batch):
            # print("a[idx]")
            # print(a[idx])
            # print(np.argmax(a[idx],axis=0))
            # print(a[idx][action])

            # q_target[idx][action] = reward_batch[idx] + self.gamma * t[idx][np.argmax(a[idx][action])]
            q_target[idx][action] = reward_batch[idx] + self.gamma * t[idx][np.argmax(a[idx],axis=0)]
        #self.model.train_on_batch(state_batch, q_target)
        # print(batch_memory)

        # batch = np.array(batch)

        # print(batch[:,:3])
        # raise()

        # batch_state = batch_memory[:, :3]
        # batch_action = batch_memory[:, 3].astype(int)
        # batch_reward = batch_memory[:, 3+1]
        # batch_state_next = batch_memory[:, -3-1:-1]
        # batch_done = batch_memory[:, -1]
        #
        # q_target = self.model.predict(batch_state)
        # q_next1 = self.model.predict(batch_state_next)
        # q_next2 = self.target_model.predict(batch_state_next)
        # batch_action_withMaxQ = np.argmax(q_next1, axis=1)
        # batch_index11 = np.arange(self.batch_size, dtype=np.int32)
        # q_next_Max = q_next2[batch_index11, batch_action_withMaxQ]
        #
        # # q_target[batch_index11, batch_action] = batch_reward + (1-batch_done)*self.gamma * q_next_Max
        # q_target[batch_index11, batch_action] = batch_reward + self.gamma * q_next_Max

        self.model.fit(state_batch, q_target, batch_size = self.batch_size, epochs = 1, verbose = 0)
        # self.model.fit(batch_state, q_target, batch_size = self.batch_size, epochs = 1, verbose = 0)
        if self.cnt % self.update_target_marker == 0:
            self.update_target_model()
            self.cnt = 0
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

#### CODE DUMPSTER #####
'''
# print("q_target shape: ",q_target.shape)
        # print("reward batch shape: ",reward_batch.shape)
        # print("action prediction shape: ", a.shape)
        # print("target model predictions: ", t.shape)


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
        # self.model.fit(state, target,  epochs=1, verbose=0)





'''
