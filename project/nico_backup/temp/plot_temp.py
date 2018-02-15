# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
#
# fig = plt.figure()
# ax = Axes3D(fig)
#
# rad = np.linspace(0, 5, 100)
# azm = np.linspace(0, 2 * np.pi, 100)
# r, th = np.meshgrid(rad, azm)
# z = (r ** 2.0) / 4.0
#
# plt.subplot(projection="polar")
#
# plt.pcolormesh(th, r, z)
# #plt.pcolormesh(th, z, r)
#
# plt.plot(azm, r, color='k', ls='none')
# plt.grid()
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'ko-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')




plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()
