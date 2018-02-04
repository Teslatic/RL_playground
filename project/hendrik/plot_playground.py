#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

x = 30
y = 50
histories = {32: np.random.rand(x,y),64: np.random.rand(x,y),128: np.random.rand(x,y), 256: np.random.rand(x,y), 512: np.random.rand(x,y)}
features = [_ for _ in histories]
mean_summary = [[] for _ in range(len(features))]

merged_hist = []

for idx,val in enumerate(features):
    mean_summary[idx].append(np.mean(histories[val], axis=0))

for idx,val in enumerate(features):
   merged_hist.append(histories[val])



merged_mean = np.mean(np.mean(merged_hist,axis=0),axis=0)
mean_std = np.std(mean_summary,axis=0)


# print('histories',histories)
# print('merged hist',merged_hist)
# print('features',features)
print('mean summary',mean_summary)
# print('mean_std',mean_std)
# print('merged mean',merged_mean)
# print(merged_mean+mean_std)
plt.figure()
for idx,val in enumerate(mean_summary):
    plt.plot(val[0], label = '{}'.format(features[idx]))
plt.plot(merged_mean, label ='mean of means')
plt.plot(merged_mean+mean_std[0], label='+', linestyle = '-.')
plt.plot(merged_mean-mean_std[0], label='+', linestyle = '-.')

plt.legend()
plt.show()
