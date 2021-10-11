#!/usr/bin/env python
# File: fd3d_1_1.py
# Name: D.Saravanan
# Date: 11/10/2021

""" Simulation in free space """

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 8,
    'axes.labelsize': 10, 'axes.titlesize': 10, 'figure.titlesize': 10})
import matplotlib.pyplot as plt

ke = 200
ex = np.zeros(ke)
hy = np.zeros(ke)

# Pulse parameters
kc = int(ke/2)
t0 = 40
spread = 12
nsteps = 100

# FDTD loop
for time_step in range(1, nsteps + 1):

    # calculate the Ex field
    for k in range(1, ke):
        ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])

    # put a Gaussian pulse in the middle
    ex[kc] = np.exp(-0.5 * ((t0 - time_step)/spread)**2)
     
    # calculate the Hy field
    for k in range(ke - 1):
        hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(ex, 'k', lw=1)
ax1.text(100, 0.5, 'T = {}'.format(time_step), horizontalalignment='center')
ax1.set(xlim=(0, 200), ylim=(-1.2, 1.2), ylabel=r'E$_x$')
ax1.set(xticks=range(0,201,20), yticks=np.arange(-1,1.2,1))

ax2.plot(hy, 'k', lw=1)
ax2.set(xlim=(0, 200), ylim=(-1.2, 1.2), xlabel=r'FDTD cells', ylabel=r'H$_y$')
ax2.set(xticks=range(0,201,20), yticks=np.arange(-1,1.2,1))

plt.subplots_adjust(bottom=0.2, hspace=0.45)
plt.savefig('fd3d_1_1.pdf')
