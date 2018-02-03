#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

epsilon = 0.95
decay_rate = 0.055
episodes = 500
const = 0.05
init_eps = epsilon-const
epsilon = []

for i in range(episodes):
    new_eps = init_eps*np.exp(-decay_rate*i)+const
    epsilon.append(new_eps)

plt.figure()
plt.plot(epsilon)
plt.show()
