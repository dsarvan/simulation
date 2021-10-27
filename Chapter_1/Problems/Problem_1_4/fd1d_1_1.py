#!/usr/bin/env python
# File: fd1d_1_1.py
# Name: D.Saravanan
# Date: 21/10/2021

""" Simulation of a pulse hitting a dielectric medium """

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 8,
    'axes.labelsize': 10, 'axes.titlesize': 10, 'figure.titlesize': 10})

ke = 200
ex = np.zeros(ke)
hy = np.zeros(ke)

# Pulse parameters
t0 = 40
spread = 12

boundary_low = [0, 0]
boundary_high = [0, 0]

# Create Dielectric Profile
epsilon = 4
cb = np.ones(ke)
cb = 0.5 * cb
cb[100:] = 0.5/epsilon

nsteps = 440

# desired points for plotting
points = [
    {'num_steps': 100, 'data': None, 'label': ' '},
    {'num_steps': 220, 'data': None, 'label': ' '},
    {'num_steps': 320, 'data': None, 'label': ' '},
    {'num_steps': 440, 'data': None, 'label': 'FDTD cells'}
]

# FDTD loop
for time_step in range(1, nsteps + 1):

    # calculate the Ex field
    for k in range(1, ke):
        ex[k] = ex[k] + cb[k] * (hy[k - 1] - hy[k])

    # put a Gaussian pulse at the low end
    ex[5] = ex[5] + np.exp(-0.5 * ((t0 - time_step)/spread)**2)

    # absorbing boundary conditons
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
fig.suptitle(r'FDTD simulation of a pulse hitting a dielectric medium')

def plotting(data, timestep, label):
    """ plot of E field at a single time step """
    ax.plot(data, color='k', linewidth=1)
    ax.plot((0.5/cb - 1)/3, 'k--', linewidth=0.75)
    ax.set(xlim=(0, 199), ylim=(-0.5, 1.2),
           xlabel=r'{}'.format(label), ylabel=r'E$_x$')
    ax.set(xticks=np.arange(0, 199, 20), yticks=np.arange(-0.5, 1.2, 0.5))
    ax.text(70, 0.5, 'T = {}'.format(timestep), horizontalalignment='center')
    ax.text(170, 0.5, 'Eps = {}'.format(epsilon), horizontalalignment='center')

for subplot_num, plot_data in enumerate(points):
    ax = fig.add_subplot(4, 1, subplot_num + 1)
    plotting(plot_data['data'], plot_data['num_steps'], plot_data['label'])

plt.tight_layout()
plt.savefig('fd1d_1_1.pdf')
