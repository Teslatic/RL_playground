# Setup

self.step_length = 0.05
self.gamma = 0.9
self.epsilon = 1.0
self.init_epsilon = 0.9
self.eps_decay_rate = 0.0004# 0.0006  0.00008 #def for 30000 ep: 0.00015
self.learning_rate = 0.001 #def: 0.001
self.decay_const = 0.1
self.train_marker = 5
self.batch_size = 64
MEMORY_SIZE = 3000
memory = []

//#def: 30000
TRAINING_EPISODES = 5000
AUTO_SAVER = 50
SHOW_PROGRESS = 25
//# def: 2000

TIMESTEPS = 500
BATCH_SIZE = 32 (const in main and batch size in agent differ here!)
