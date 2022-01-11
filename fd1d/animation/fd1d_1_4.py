#!/usr/bin/env python
# File: fd1d_1_4.py
# Name: D.Saravanan
# Date: 22/10/2021

""" Simulation of a propagating sinusoidal wave of 700 MHz
striking a medium with a relative dielectric constant of 4 """

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

dx = 0.01  # Cell size
dt = dx / 6e8  # Time step size
freq = 700e6  # Frequency 700 MHz

boundary_low = [0, 0]
boundary_high = [0, 0]

# Create Dielectric Profile
epsilon = 4
cb = np.ones(ke)
cb = 0.5 * cb
cb[100:] = 0.5 / epsilon

nsteps = 1500

# define the meta data for the movie
fwriter = animation.writers["ffmpeg"]
data = dict(title="Simulation of a propagating sinusoidal wave of 700 MHz")
writer = fwriter(fps=15, metadata=data)

# draw an empty plot, but preset the plot x- and y- limits
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle(r"FDTD simulation of a sinusoidal hitting a dielectric medium")
(line1,) = ax1.plot(ex, "k", lw=1)
(line2,) = ax2.plot(hy, "k", lw=1)
time_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes)
epsn_text1 = ax1.text(0.80, 0.80, "", transform=ax1.transAxes)
epsn_text2 = ax2.text(0.80, 0.80, "", transform=ax2.transAxes)
ax1.plot((0.5 / cb - 1) / 3, "k--", linewidth=0.75)
ax1.set(xlim=(0, 200), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
ax1.set(xticks=range(0, 220, 20), yticks=np.arange(-1.2, 1.4, 0.4))
ax2.plot((0.5 / cb - 1) / 3, "k--", linewidth=0.75)
ax2.set(xlim=(0, 200), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"$H_y$")
ax2.set(xticks=range(0, 220, 20), yticks=np.arange(-1.2, 1.4, 0.4))
plt.tight_layout()

# FDTD loop
with writer.saving(fig, "fd1d_1_4.mp4", 100):
    for time_step in range(1, nsteps + 1):

        # calculate the Ex field
        for k in range(1, ke):
            ex[k] = ex[k] + cb[k] * (hy[k - 1] - hy[k])

        # put a sinusoidal at the low end
        ex[1] = ex[1] + np.sin(2 * np.pi * freq * dt * time_step)

        # absorbing boundary conditions
        ex[0] = boundary_low.pop(0)
        boundary_low.append(ex[1])
        ex[ke - 1] = boundary_high.pop(0)
        boundary_high.append(ex[ke - 2])

        # calculate the Hy field
        for k in range(ke - 1):
            hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])

        line1.set_ydata(ex)
        time_text.set_text("T = {}".format(time_step))
        epsn_text1.set_text("Eps = {}".format(epsilon))
        line2.set_ydata(hy)
        epsn_text2.set_text("Eps = {}".format(epsilon))
        writer.grab_frame()
