#!/usr/bin/env python
# File: fd1d_1_1.py
# Name: D.Saravanan
# Date: 11/10/2021

""" Simulation of a pulse in free space """

import numpy
from matplotlib import pyplot
from numpy.typing import NDArray

pyplot.style.use("classic")
pyplot.style.use("pyplot.mplstyle")

type F64Array = NDArray[numpy.float64]


def visualize(ns: int, nx: int, ex: F64Array):
    fig, axs = pyplot.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a pulse in free space")
    axs.plot(range(nx), ex, color="k", linewidth=1.0)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=range(0, nx+1, int(numpy.ceil(nx/500)*25)))
    axs.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axs.text(0.02, 0.90, rf"$T$ = {ns}", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("fd1d_1_1.png", dpi=100)


def gaussian(ts: int, t0: int, sigma: float):
    return numpy.exp(-0.5*((ts - t0)/sigma)**2)


def main():

    nx: int = 512  # number of grid points
    ns: int = 300  # number of time steps

    ex = numpy.zeros(nx, numpy.float64)
    hy = numpy.zeros(nx, numpy.float64)

    for ts in map(int,range(1,ns+1)):
        # calculate the Ex field
        ex[1:nx] += 0.5 * (hy[0:nx-1] - hy[1:nx])
        # put a Gaussian pulse in the middle
        ex[nx//2] = gaussian(ts, 40, 12.0)
        # calculate the Hy field
        hy[0:nx-1] += 0.5 * (ex[0:nx-1] - ex[1:nx])

    visualize(ns, nx, ex)


if __name__ == "__main__":
    main()
