#!/usr/bin/env python
# File: fdtdjob.py
# Name: D.Saravanan
# Date: 16/05/2022

""" Script for 1-dimensional fdtd with gaussian pulse as source """

import numpy as np
import multiprocessing
from joblib import Parallel, delayed
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
    ax1.plot(ex, "k", lw=1)
    ax1.text(100, 0.5, "T = {}".format(time_step), horizontalalignment="center")
    ax1.set(xlim=(0, 200), ylim=(-1.2, 1.2), ylabel=r"E$_x$")
    ax1.set(xticks=range(0, 220, 20), yticks=np.arange(-1, 1.2, 1))
    ax2.plot(hy, "k", lw=1)
    ax2.set(xlim=(0, 200), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"H$_y$")
    ax2.set(xticks=range(0, 220, 20), yticks=np.arange(-1, 1.2, 1))
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fdtdjob.png")


def gaussian(t0, time_step, spread):
    return np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)


def electric(k, ex, hy):
    ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])


def magnetic(k, ex, hy):
    hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])


def main():

    ke = 20001
    ex = np.zeros(ke)
    hy = np.zeros(ke)

    # Pulse parameters
    kc = int(ke / 2)
    t0 = 40
    spread = 12
    nsteps = 100

    n_cpu = multiprocessing.cpu_count()

    for time_step in range(1, nsteps + 1):
    
        Parallel(n_jobs = n_cpu)(delayed(electric)(k, ex, hy) for k in range(1, ke))

        ex[kc] = gaussian(t0, time_step, spread)

        Parallel(n_jobs = n_cpu)(delayed(magnetic)(k, ex, hy) for k in range(ke - 1))

    plotting(time_step, ex, hy)


if __name__ == "__main__":
    main()
