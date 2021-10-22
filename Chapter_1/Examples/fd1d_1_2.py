#!/usr/bin/env python
# File: fd1d_1_2.py
# Name: D.Saravanan
# Date: 19/10/2021

""" Simulation in free space - absorbing boundary condition added """

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 8,
                            'axes.labelsize': 10, 'axes.titlesize': 10, 'figure.titlesize': 10})

ke = 200
ex = np.zeros(ke)
hy = np.zeros(ke)

# Pulse parameters
kc = int(ke/2)
t0 = 40
spread = 12
nsteps = 250

boundary_low = [0, 0]
boundary_high = [0, 0]

# desired points for plotting
points = [
    {'num_steps': 100, 'data': None, 'label': ' '},
    {'num_steps': 225, 'data': None, 'label': ' '},
    {'num_steps': 250, 'data': None, 'label': 'FDTD cells'}
]

# FDTD loop
for time_step in range(1, nsteps + 1):

    # calculate the Ex field
    for k in range(1, ke):
        ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])

    # put a Gaussian pulse in the middle
    ex[kc] = np.exp(-0.5 * ((t0 - time_step)/spread)**2)

    # absorbing boundary conditions
    ex[0] = boundary_low.pop(0)
    boundary_low.append(ex[1])
    ex[ke - 1] = boundary_high.pop(0)
    boundary_high.append(ex[ke - 2])

    # calculate the Hy field
    for k in range(ke - 1):
        hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])

    # save data at certain points for plotting
    for plot_data in points:
        if time_step == plot_data['num_steps']:
            plot_data['data'] = np.copy(ex)

fig = plt.figure(figsize=(8, 5.25))
fig.suptitle(r'FDTD simulation with absorbing boundary conditions')

def plotting(data, timestep, label):
    """ plot of E field at a single time step """
    ax.plot(data, color='k', linewidth=1)
    ax.set(xlim=(0, 199), ylim=(-0.2, 1.2),
           xlabel=r'{}'.format(label), ylabel=r'E$_x$')
    ax.set(xticks=np.arange(0, 199, 20), yticks=np.arange(0, 1.2, 1))
    ax.text(100, 0.5, 'T = {}'.format(timestep), horizontalalignment='center')

for subplot_num, plot_data in enumerate(points):
    ax = fig.add_subplot(3, 1, subplot_num + 1)
    plotting(plot_data['data'], plot_data['num_steps'], plot_data['label'])

plt.tight_layout()
plt.savefig('fd1d_1_2.pdf')
