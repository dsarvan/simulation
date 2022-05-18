#!/usr/bin/env python
# File: fdtd0.py
# Name: D.Saravanan
# Date: 16/05/2022

""" FDTD """

import numpy as np
from numba import jit, guvectorize, int64
import matplotlib.pyplot as plt

plt.style.use('classic')
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
    """ plot function """
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
    plt.savefig("fdtd0.png")



@guvectorize([(int64[:], int64[:])], '(n) -> (n)')
def electric(ex, hy):
    for k in range(1, 201):
        ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])


@guvectorize([(int64[:], int64[:])], '(n) -> (n)')
def magnetic(ex, hy):
    for k in range(200):
        hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])


def gaussian(t0, time_step, spread):
    return np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)

@profile
def main():

    ke = 201
    ex = np.zeros(ke).astype(np.int64)
    hy = np.zeros(ke).astype(np.int64)

    # Pulse parameters
    kc = int(ke/2)
    t0 = 40
    spread = 12
    nsteps = 100


    for time_step in range(1, nsteps + 1):
        
        ex = electric(ex, hy)
        ex[kc] = gaussian(t0, time_step, spread)
        hy = magnetic(ex, hy)

    plotting(time_step, ex, hy)



if __name__ == "__main__":
    main()
