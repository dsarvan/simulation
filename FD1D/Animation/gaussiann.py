#!/usr/bin/env python
# File: gaussiann.py
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
nsteps = 300

# define the meta data for the movie
fwriter = animation.writers["ffmpeg"]
data = dict(title = 'Gaussian pulse', artist = 'Saran', comment = 'Simulation of Gaussian pulse')
writer = fwriter(fps = 15, metadata = data)

# draw an empty plot, but preset the plot x- and y- limits
fig, ax = plt.subplots()
line, = ax.plot(ex)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set(xlim=(0, ke), ylim=(-1.2, 1.2))
ax.set(xlabel=r'FDTD cells', ylabel=r'Ex')

# FDTD loop
with writer.saving(fig, 'gaussiann.mp4', 100):
    for time_step in range(1, nsteps+1):

        for k in range(1, ke):
            ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])

        ex[kc] = np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)
    
        for k in range(ke - 1):
            hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])

        line.set_ydata(ex)
        time_text.set_text('T = {}'.format(time_step))
        writer.grab_frame()
