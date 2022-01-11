#!/usr/bin/env python
# File: fd0d_1_1.py
# Name: D.Saravanan
# Date: 11/10/2021

""" Simulation in free space """
# FDTD simulation of a pulse in free space.
# The pulse originated in the center and travels outward.

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 10,
    }
)

ke = 201
ex = np.zeros(ke)
hy = np.zeros(ke)

# Pulse parameters
kc = int(ke / 2)
t0 = 40
spread = 12
nsteps = 300

# define the meta data for the movie
fwriter = animation.writers["ffmpeg"]
data = dict(title="FDTD simulation of a pulse in free space")
writer = fwriter(fps=15, metadata=data)

# draw an empty plot, but preset the plot x- and y- limits
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle(r"FDTD simulation of a pulse in free space")
(line1,) = ax1.plot(ex, "k", lw=1)
(line2,) = ax2.plot(hy, "k", lw=1)
time_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes)
ax1.set(xlim=(0, ke), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
ax1.set(xticks=range(0, 220, 20), yticks=np.arange(-1, 1.2, 1))
ax2.set(xlim=(0, ke), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"$H_y$")
ax2.set(xticks=range(0, 220, 20), yticks=np.arange(-1, 1.2, 1))

# FDTD loop
with writer.saving(fig, "fd1d_1_1.mp4", 100):
    for time_step in range(1, nsteps + 1):

        # calculate the Ex field
        for k in range(1, ke):
            ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])

        # put a Gaussian pulse in the middle
        ex[kc] = np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)

        # calculate the Hy field
        for k in range(ke - 1):
            hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])

        line1.set_ydata(ex)
        time_text.set_text("T = {}".format(time_step))
        line2.set_ydata(hy)
        writer.grab_frame()
