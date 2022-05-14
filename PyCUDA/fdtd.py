#!/usr/bin/env python
# File: fdtd.py
# Name: D.Saravanan
# Date: 13/05/2022

""" Script for finite difference time domain method """

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 10,
    }
)

ke = np.int32(201)
ex = gpuarray.zeros(ke, dtype=np.float32)
hy = gpuarray.zeros(ke, dtype=np.float32)

# Pulse parameters
kc = np.int32(ke/2)
t0 = np.int32(40)
spread = np.int32(12)
nsteps = np.int32(100)

gaussian_pulse = ElementwiseKernel(
    "int t0, int time_step, int spread, int kc, float *ex",
    "ex[kc] = exp(-0.5 * pow(((t0 - time_step) / spread), 2))",
    "gaussian_pulse"
)

# FDTD loop
for time_step in range(1, nsteps + 1):

    # calculate the Ex field
    for k in range(1, ke):
        ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])

    # put a Gaussian pulse in the middle
    #ex[kc] = cumath.exp(-0.5 * ((t0 - time_step) / spread) ** 2)
    gaussian_pulse(t0, time_step, spread, kc, ex)

    # calculate the Hy field
    for k in range(ke - 1):
        hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])


fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle(r"FDTD simulation of a pulse in free space after 100 time steps")
ax1.plot(ex.get(), "k", lw=1)
ax1.text(100, 0.5, "T = {}".format(time_step), horizontalalignment="center")
ax1.set(xlim=(0, 200), ylim=(-1.2, 1.2), ylabel=r"E$_x$")
ax1.set(xticks=range(0, 220, 20), yticks=np.arange(-1, 1.2, 1))
ax2.plot(hy.get(), "k", lw=1)
ax2.set(xlim=(0, 200), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"H$_y$")
ax2.set(xticks=range(0, 220, 20), yticks=np.arange(-1, 1.2, 1))
plt.subplots_adjust(bottom=0.2, hspace=0.45)
plt.savefig("fdtd.png")
