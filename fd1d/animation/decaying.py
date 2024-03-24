#!/usr/bin/env python
# File: decaying.py
# Name: D.Saravanan
# Date: 20/10/2021

""" An animation of a decaying sine curve """

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 8,
    'axes.labelsize': 10, 'axes.titlesize': 10, 'figure.titlesize': 10})
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# time step for the animation (s), max time to animate for (s)
dt, tmax = 0.01, 5
# signal frequency (s-1), decay constant (s-1)
f, alpha = 2.5, 1
# these lists will hold the data to plot
t, M = [], []

# draw an empty plot, but preset the plot x- and y- limits
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlim(0, tmax)
ax.set_ylim(-1, 1)
ax.set_xlabel('t/s')
ax.set_ylabel('M (arb. units)')

def init():
    return line,

def animate(i, t, M):
    """ draw the frame i of the animation """

    # append this time point and its data and set the plotted line data
    _t = i*dt
    t.append(_t)
    M.append(np.sin(2*np.pi*f*_t) * np.exp(-alpha * _t))
    line.set_data(t, M)
    return line,

# interval between frames in ms, total number of frames to use
interval, nframes = 1000 * dt, int(tmax/dt)
# animate once (set repeat=False so the animation doesn't loop)
animation = ani.FuncAnimation(fig, animate, frames=nframes, init_func=init, fargs=(t, M),
    repeat=False, interval=interval, blit=True)

plt.show()
