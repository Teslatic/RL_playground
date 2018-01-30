#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

epsilon = 1.0
decay_rate = 0.0004
episodes = 5000
const = 0.1
init_eps = epsilon-const
epsilon = []

for i in range(episodes):
    new_eps = init_eps*np.exp(-decay_rate*i)+const
    epsilon.append(new_eps)

plt.figure()
plt.plot(epsilon)
plt.show()
