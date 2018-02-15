from assets.helperFunctions.tensor_conversions import *


class TransitionBuffer():
    """
    This class is used to store all transistions which have been seen so far in a vector format. A transistion consist of an action-state-pair and the resulting observation of the next state and the reward for the taken action.
    """
    def __init__(self, memory_size, memory_depth):
        """
        Initializing
        """
        self.memory_size = memory_size
        self.memory_depth = memory_depth
        self.memory = np.empty([self.memory_size, self.memory_depth])
        self.memory_counter = 0
        self.learn_counter = 0
        self.batch_memory = []

    def create_transition(self, state, action, reward, next_state):
        """
        Creates a transistion out of the raw data.
        """
        # action = convert_scalar2vector(action)
        action = convert_scalar2tensor(action)
        state = convert_vector2tensor(state)
        reward = convert_scalar2tensor(reward)
        next_state = convert_vector2tensor(next_state)
        return np.hstack((state, action, reward, next_state))

    def store(self, transition):
        """
        Will be stored as a ring-buffer. The oldest transitions will be overwritten by newer ones.
        """
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter += 1

    def get_batch(self, batchsize):
        """
        Returns a batch with size [batch size]
        """
        sample_index = np.random.choice(self.memory_size, size=batchsize)
        return self.memory[sample_index]


    def unzip_batch(self, batch):
        batch_state = batch[:, :3]
        batch_action = batch[:, 3].astype(int)
        batch_reward = batch[:, 4].astype(int)
        batch_state_next = batch[:, 5:8]
        return batch_state, batch_action, batch_reward, batch_state_next

    def unzip_batch_as_tensor(self, batch):
        batch_state, batch_action, batch_reward, batch_state_next = self.unzip_batch(batch)
        # print("batch_state: {}".format(batch_state))
        # batch_state_np = np.asarray(batch_state, np.float32)
        # print("batch_state_asserted: {}".format(batch_state_np))
        batch_state_tens = convert_to_tensor(batch_state, np.float32)
        batch_state_next_tens = convert_to_tensor(batch_state_next, np.float32)
        batch_action_tens = convert_to_tensor(batch_action, np.float32)
        # print("batch_state_tensor: {}".format(batch_state_tens))
        # print("batch_action: {}".format(batch_action))
        # print("batch_reward: {}".format(batch_reward))
        # print("batch_state_next: {}".format(batch_state_next))
        # batch_action_tens = convert_vector2tensor(batch_action)
        # batch_state_next_tens = convert_vector2tensor(batch_state_next)
        return batch_state_tens, batch_action_tens, batch_reward, batch_state_next_tens

class EpisodeHistory():
    """
    """
    def __init__(self):
        pass

    def store(self, length, reward):
        pass

# dumpster

        # self.batch =
        # batch_state = batch[:, :self.D_state]
        # batch_action = batch[:, self.D_state].astype(int)
        # batch_reward = batch[:, self.D_state+1]
        # batch_state_next = batch[:, -self.D_state-1:-1]
        # # batch_done = batch_memory[:, -1]
        # return batch, batch_state, batch_action, batch_reward, batch_state_next
