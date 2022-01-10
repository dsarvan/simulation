#!/usr/bin/env python
# File: gaussian.py
# Name: D.Saravanan
# Date: 20/10/2021

""" Simulation of Gaussian pulse """

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

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

# draw an empty plot, but preset the plot x- and y- limits
fig, ax = plt.subplots()
(line,) = ax.plot(ex)
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
ax.set(xlim=(0, ke), ylim=(-1.2, 1.2))
ax.set(xlabel=r"FDTD cells", ylabel=r"Ex")


def init():
    line.set_ydata(ex)
    time_text.set_text("")
    return line, time_text


def animate(i):

    # FDTD loop
    for time_step in range(1, nsteps + 1):

        for k in range(1, ke):
            ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])

        ex[kc] = np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)

        for k in range(ke - 1):
            hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])

        line.set_ydata(ex)
        time_text.set_text("T = {}".format(time_step))

    return line, time_text


interval, nframes = 1000 * dt, int(tmax / dt)
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=nframes, interval=interval, blit=True
)
anim.save("gaussian.mp4")
