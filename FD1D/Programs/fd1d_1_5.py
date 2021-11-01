#!/usr/bin/env python
# File: fd1d_1_5.py
# Name: D.Saravanan
# Date: 29/10/2021

""" Simulation of a sinusoidal wave hitting a lossy dielectric """

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 8,
                            'axes.labelsize': 10, 'axes.titlesize': 10, 'figure.titlesize': 10})

ke = 200
ex = np.zeros(ke)
hy = np.zeros(ke)

dx = 0.01           # cell size
dt = dx/6e8         # time step size
freq = 700e6        # frequency 700 MHz

boundary_low = [0, 0]
boundary_high = [0, 0]

# Create Dielectric Profile
epsz = 8.854e-12    # vacuum permittivity (F/m)
epsilon = 4         # relative permittivity
sigma = 0.04        # conductivity (S/m)

ca = np.ones(ke)
cb = 0.5 * np.ones(ke)

eaf = dt * sigma/(2*epsz*epsilon)
ca[100:] = (1 - eaf)/(1 + eaf)
cb[100:] = 0.5/(epsilon * (1 + eaf))

nsteps = 500

# FDTD loop
for time_step in range(1, nsteps + 1):

    # calculate the Ex field
    for k in range(1, ke):
        ex[k] = ca[k] * ex[k] + cb[k] * (hy[k - 1] - hy[k])

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

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle(r'FDTD simulation of a sinusoidal wave hitting a lossy dielectric')
ax1.plot(ex, 'k', lw=1)
ax1.plot((0.5/cb - 1)/3, 'k--', lw=0.75)
ax1.text(50, 0.5, 'T = {}'.format(time_step), horizontalalignment='center')
ax1.text(170, 0.5, 'Eps = {}'.format(epsilon), horizontalalignment='center')
ax1.text(170, -0.5, 'Cond = {}'.format(sigma), horizontalalignment='center')
ax1.set(xlim=(0, 199), ylim=(-1.2, 1.2), ylabel=r'E$_x$')
ax1.set(xticks=range(0, 199, 20), yticks=np.arange(-1, 1.2, 1))
ax2.plot(hy, 'k', lw=1)
ax2.plot((0.5/cb - 1)/3, 'k--', lw=0.75)
ax2.text(50, 0.5, 'T = {}'.format(time_step), horizontalalignment='center')
ax2.text(170, 0.5, 'Eps = {}'.format(epsilon), horizontalalignment='center')
ax2.text(170, -0.5, 'Cond = {}'.format(sigma), horizontalalignment='center')
ax2.set(xlim=(0, 200), ylim=(-1.2, 1.2), xlabel=r'FDTD cells', ylabel=r'H$_y$')
ax2.set(xticks=range(0, 199, 20), yticks=np.arange(-1, 1.2, 1))
plt.subplots_adjust(bottom=0.2, hspace=0.45)
plt.savefig('fd1d_1_5.png')
