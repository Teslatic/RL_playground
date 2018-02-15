#!/usr/bin/env python

import sys
if "../" not in sys.path:
    sys.path.append("../")
# if "../" not in sys.path:
#     sys.path.append("../")

import numpy as np
from lib.envs.pendulum import PendulumEnv
# from assets.helperFunctions.conver import convert_vector2tensor

env = PendulumEnv()
env.reset()
print(env.observation_space.shape[0])
env.seed(2)
for t in range(200):
    state = env.render()

    next_state, reward, done , _ = env.step(1.5)
    print(next_state)
    # next_state_asserted = convert_vector2tensor(next_state)
    # next_state_asserted = next_state.reshape((1,3))
    # print(next_state_asserted)
