#!/usr/bin/env python
# File: fd1d_2_1.py
# Name: D.Saravanan
# Date: 25/11/2021

""" Simulation of a propagating sinusoidal wave striking a lossy dielectric material """

import matplotlib
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
ix = np.zeros(ke)
Dx = np.zeros(ke)
hy = np.zeros(ke)

dx = 0.01  # cell size
dt = dx / 6e8  # time step size
freq = 700e6  # frequency 700 MHz

boundary_low = [0, 0]
boundary_high = [0, 0]

# dielectric profile
epsz = 8.854e-12  # vacuum permittivity (F/m)
epsr = 4  # relative permittivity
sigma = 0.04  # conductivity (S/m)


gax = np.ones(ke)
gbx = np.zeros(ke)
gax[100:] = 1 / (epsr + (sigma * dt / epsz))
gbx[100:] = sigma * dt / epsz

nsteps = 500

# FDTD loop
for time_step in range(1, nsteps + 1):

    # calculate the Dx flux
    for k in range(1, ke):
        Dx[k] = Dx[k] + 0.5 * (hy[k - 1] - hy[k])

    # put a sinusoidal at the low end
    Dx[5] = Dx[5] + np.sin(2 * np.pi * freq * dt * time_step)

    # calculate the Ex field from Dx
    for k in range(1, ke):
        ex[k] = gax[k] * (Dx[k] - ix[k])
        ix[k] = ix[k] + gbx[k] * ex[k]

    # absorbing boundary conditions
    ex[0] = boundary_low.pop(0)
    boundary_low.append(ex[1])
    ex[ke - 1] = boundary_high.pop(0)
    boundary_high.append(ex[ke - 2])

    # calculate the Hy field
    for k in range(ke - 1):
        hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle(r"FDTD simulation of a sinusoidal wave striking a lossy dielectric")
ax1.plot(ex, "k", lw=1)
ax1.plot(gbx / gbx[100], "k--", lw=0.75)
ax1.text(50, 0.5, "T = {}".format(time_step), horizontalalignment="center")
ax1.text(170, 0.5, "Eps = {}".format(epsr), horizontalalignment="center")
ax1.text(170, -0.5, "Cond = {}".format(sigma), horizontalalignment="center")
ax1.set(xlim=(0, 200), ylim=(-1.2, 1.2), ylabel=r"E$_x$")
ax1.set(xticks=np.arange(0, 220, 20), yticks=np.arange(-1, 1.2, 1))
ax2.plot(hy, "k", lw=1)
ax2.plot(gbx / gbx[100], "k--", lw=0.75)
ax2.text(50, 0.5, "T = {}".format(time_step), horizontalalignment="center")
ax2.text(170, 0.5, "Eps = {}".format(epsr), horizontalalignment="center")
ax2.text(170, -0.5, "Cond = {}".format(sigma), horizontalalignment="center")
ax2.set(xlim=(0, 200), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"H$_y$")
ax2.set(xticks=np.arange(0, 220, 20), yticks=np.arange(-1, 1.2, 1))
plt.subplots_adjust(bottom=0.2, hspace=0.45)
plt.savefig("fd1d_2_1.png")
