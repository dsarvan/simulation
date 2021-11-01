#!/usr/bin/env python
# File: fd1d_1_2.py
# Name: D.Saravanan
# Date: 22/10/2021

""" Simulation of a propagating sinusoidal wave of incident frequency from 700 MHz upward
at intervals of 300 MHz striking a medium with a relative dielectric constant of 4 """

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 8,
                            'axes.labelsize': 10, 'axes.titlesize': 10, 'figure.titlesize': 10})

ke = 200
ex = np.zeros(ke)
hy = np.zeros(ke)

dx = 0.01          # Cell size
dt = dx/6e8        # Time step size
freq = 700e6       # Frequency 700 MHz

boundary_low = [0, 0]
boundary_high = [0, 0]

# Create Dielectric Profile
epsilon = 4
cb = np.ones(ke)
cb = 0.5 * cb
cb[100:] = 0.5/epsilon

nsteps = 425

# desired points for plotting
points = [
    {'num_steps': 150, 'data': None, 'label': ' '},
    {'num_steps': 425, 'data': None, 'label': 'FDTD cells'}
]

# FDTD loop
for time_step in range(1, nsteps + 1):

    # calculate the Ex field
    for k in range(1, ke):
        ex[k] = ex[k] + cb[k] * (hy[k - 1] - hy[k])

    # put a sinusoidal at the low end
    ex[5] = ex[5] + np.sin(2 * np.pi * freq * dt * time_step)

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

    freq = freq + 300e6

fig = plt.figure(figsize=(8, 3.5))
fig.suptitle(r'FDTD simulation of a sinusoidal hitting a dielectric medium')


def plotting(data, timestep, label):
    """ plot of E field at a single time step """
    ax.plot(data, color='k', linewidth=1)
    ax.plot((0.5/cb - 1)/3, 'k--', linewidth=0.75)
    ax.set(xlim=(0, 199), ylim=(-1.2, 1.2),
           xlabel=r'{}'.format(label), ylabel=r'E$_x$')
    ax.set(xticks=np.arange(0, 199, 20), yticks=np.arange(-1, 1.2, 1))
    ax.text(50, 0.5, 'T = {}'.format(timestep), horizontalalignment='center')
    ax.text(170, 0.5, 'Eps = {}'.format(epsilon), horizontalalignment='center')


for subplot_num, plot_data in enumerate(points):
    ax = fig.add_subplot(2, 1, subplot_num + 1)
    plotting(plot_data['data'], plot_data['num_steps'], plot_data['label'])

plt.subplots_adjust(bottom=0.2, hspace=0.45)
plt.savefig('fd1d_1_2.png')
