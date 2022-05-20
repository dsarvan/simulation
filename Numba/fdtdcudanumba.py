#!/usr/bin/env python
# File: fdtdcudanumba.py
# Name: D.Saravanan
# Date: 16/05/2022

""" Script for 1-dimensional fdtd with gaussian pulse as source """

import numpy as np
import cupy as cp
from numba import cuda
import matplotlib.pyplot as plt

plt.style.use("classic")
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


def plotting(time_step, ex, hy):
    """plot function"""
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
    plt.savefig("fdtdcudanumba.png")


@cuda.jit
def electric(ke, ex, hy):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for k in range(start, ke, stride):
        ex[k] = ex[k] + 0.5 * (hy[k-1] - hy[k])


@cuda.jit
def magnetic(ke, ex, hy):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for k in range(start, ke, stride):
        hy[k] = hy[k] + 0.5 * (ex[k-1] - ex[k])


ke = 201
ex = cp.zeros(ke)
hy = cp.zeros(ke)

kc = cp.int32(ke/2)
t0 = 40
spread = 12
nsteps = 100

for time_step in range(1, nsteps + 1):

    electric[10, 32](ke, ex, hy)

    ex[kc] = cp.exp(-0.5 * ((t0 - time_step) / spread) ** 2)

    magnetic[10, 32](ke, ex, hy)


plotting(time_step, ex, hy)
