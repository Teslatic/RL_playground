import random
from random import randrange
from time import time
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam, Nadam
from keras import backend as K
from keras.layers import Flatten,Dense, Dropout
import datetime

from keras.layers import Input, Dense, Add, RepeatVector, Reshape
from keras.models import Model
from keras.optimizers import RMSprop




class DankAgent():
    def __init__(self,state_size, action_dim):
        self.state_size = state_size
        #self.action_interval = action_interval
        #self.step_length = 0.15
        #sself.disc_actions = self._discretize_actions()
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.init_epsilon = 1.5
        self.eps_decay_rate = 0.001
        self.learning_rate = 0.001
        self.decay_const = 0.1
        self.action_dim = action_dim
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.learn_counter = 0
        self.memory_size = 3000

        self.memory_counter = 0
        self.memory = np.empty([self.memory_size,9])
        self.batch_size = 32


    def _build_model(self):
        model = Sequential()
        model.add(Dense(16,input_shape = (self.state_size, ),activation = 'relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(16, activation = 'relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(32, activation = 'relu'))
        #model.add(Dense(64, activation = 'relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(self.action_dim, activation = 'linear'))
        model.compile(loss = 'mse', optimizer = Nadam(lr = self.learning_rate))
        '''activation_curve = 'relu'
        unit_num = 20
        action_dim = self.action_dim
        input_eval = Input(shape=(self.state_size,))
        l1 = Dense(unit_num, activation=activation_curve)(input_eval)
        val_layer = Dense(1)(l1)
        val_layer = RepeatVector(action_dim)(val_layer)
        val_layer = Reshape(target_shape=(action_dim,), input_shape=(action_dim, 1,))(val_layer)
        adv_layer = Dense(action_dim)(l1)
        merge_layer = Add()([val_layer, adv_layer])
        model = Model(inputs=input_eval, outputs=merge_layer)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), )
        '''
        return model
    
    def _build_target_model(self):
        activation_curve = 'relu'
        unit_num = 20
        action_dim = self.action_dim
        input_eval = Input(shape=(self.state_size,))
        l1 = Dense(unit_num, activation=activation_curve)(input_eval)
        val_layer = Dense(1)(l1)
        val_layer = RepeatVector(action_dim)(val_layer)
        val_layer = Reshape(target_shape=(action_dim,), input_shape=(action_dim, 1,))(val_layer)
        adv_layer = Dense(action_dim)(l1)
        merge_layer = Add()([val_layer, adv_layer])
        model = Model(inputs=input_eval, outputs=merge_layer)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, en_explore = True):
        if random.random() <= self.epsilon and en_explore:
            # action = random.choice(self.disc_actions)==self.disc_actions
            #action = np.where(self.disc_actions==random.choice(self.disc_actions))[0]
            action = np.random.randint(0, self.action_dim)
        else:
            state = state.reshape((1,3))
            action_values = self.model.predict(state)
            
            # print("action values",action_values)
            # print("greedy pick from action values",np.argmax(action_values[0]))
            action = np.array((np.argmax(action_values),))
            # print("discrete actions", self.discrete_actions)

        return action
    def memory_store(self, state_now, action, reward, state_next, done):
        
        action = np.reshape(action, [1, 1])
        reward = np.reshape(reward, [1, 1])
        done = np.reshape(done, [1, 1])
        transition = np.hstack((state_now, action, reward, state_next, done))
        
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # def train(self,state, action, next_state, , reward, done):
    def train(self, batch_memory):
        
        if self.learn_counter % 200 == 0:
            self.update_target_model()
            self.learn_counter = 0
        batch_state = batch_memory[:, :self.state_size]
        batch_action = batch_memory[:, self.state_size].astype(int)
        batch_reward = batch_memory[:, self.state_size+1]
        batch_state_next = batch_memory[:, -self.state_size-1:-1]
        batch_done = batch_memory[:, -1]

        q_target = self.model.predict(batch_state)
        q_next1 = self.model.predict(batch_state_next)
        q_next2 = self.target_model.predict(batch_state_next)
        batch_action_withMaxQ = np.argmax(q_next1, axis=1)
        batch_index11 = np.arange(self.batch_size, dtype=np.int32)
        q_next_Max = q_next2[batch_index11, batch_action_withMaxQ]
        # q_target[batch_index11, batch_action] = batch_reward + (1-batch_done)*self.gamma * q_next_Max
        q_target[batch_index11, batch_action] = batch_reward + self.gamma * q_next_Max
        
        '''inputs = np.zeros((32, 3 ))   #32, 80, 80, 4
        targets = np.zeros((32, 26))
        #state_t = minibatch[1,:]
        minibatch = minibatch.reshape((32,5))
        print(minibatch.shape)   
        for i in range(0, len(minibatch)):
            state_t = minibatch[i][0]
            state_t = state_t.reshape((1,3))
            action_t = minibatch[i][1]   #This is action index
            reward_t = minibatch[i][2]
            state_t1 = minibatch[i][3]
            state_t1 = state_t1.reshape((1,3))
            terminal = minibatch[i][4]
            # if terminated, only equals reward

            inputs[i:i + 1] = state_t    #I saved down s_t

            targets[i] = self.model.predict(state_t)  # Hitting each buttom probability

            a = self.model.predict(state_t1)[0]
            #print(np.argmax(a))
            t = self.target_model.predict(state_t1)[0]
            #print(t[np.argmax(a)])
            targets[i, action_t] = reward_t + self.gamma * t[np.argmax(a)]
        
        inputs = inputs.reshape((32,3))'''
        #print("shape", inputs.shape)
        self.history = self.model.fit(
            batch_state, q_target,
            batch_size=32,
            epochs=1,
            verbose=0)
        self.learn_counter +=1
        #self.history = self.model.train_on_batch(inputs, targets)
        '''
        for state, action, reward, next_state, done in batch:
            state = state.reshape((1,2))
            next_state = next_state.reshape((1,2))

            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]

                target[0][action] = reward + self.gamma * t[np.argmax(a)]

            self.history = self.model.train_on_batch(state, target)
            #self.history = self.model.train_on_batch(state_batch,target_batch)'''

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
