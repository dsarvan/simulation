#!/usr/bin/env python
# File: gaussian.py
# Name: D.Saravanan
# Date: 20/10/2021

""" Simulation of Gaussian pulse """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ke = 200
ex = np.zeros(ke)
hy = np.zeros(ke)

# Pulse parameters
kc = int(ke / 2)
t0 = 40
spread = 12
nsteps = 100

# time step for tha animation (s), maximum time to animate (s)
dt, tmax = 1, nsteps
# data to plot
t, M = [], []

# draw an empty plot, but preset the plot x- and y- limits
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set(xlim=(0, ke), ylim=(-1.2, 1.2))
ax.set(xlabel=r'FDTD cells', ylabel=r'Ex')

def init():
    return line,

def animate(i, t, M):

    # FDTD loop
    for time_step in range(1, nsteps+1):

        for k in range(1, ke):
            ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])
            t.append(k)

        ex[kc] = np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)

        M.append(ex)
        print(M)

        for k in range(ke - 1):
            hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])
    
    t.extend(range(1, 101))
    line.set_data(t, M)
    return line,

interval, nframes = 1000*dt, int(tmax/dt)
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nframes, 
            fargs=(t, M), repeat=False, interval=interval, blit=True)
anim.save('gaussian_anim.mp4')
