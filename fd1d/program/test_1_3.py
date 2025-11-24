#!/usr/bin/env python
# File: test_1_3.py
# Name: D.Saravanan
# Date: 21/10/2021

""" Simulation of a pulse hitting a dielectric medium """

import time
import numpy
from matplotlib import pyplot
from numpy.typing import NDArray

pyplot.style.use("classic")
pyplot.style.use("pyplot.mplstyle")

type F32Array = NDArray[numpy.float32]


def visualize(ns: int, nx: int, epsr: float, cb: F32Array, ex: F32Array):
    fig, axs = pyplot.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a pulse striking dielectric material")
    medium = numpy.where(0.5/cb-1)[0] if epsr > 1 else (0.5/cb-1)
    axs.plot(range(nx), ex, color="k", linewidth=1.0)
    axs.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=range(0, nx+1, int(numpy.ceil(nx/500)*50)))
    axs.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axs.text(0.02, 0.90, rf"$T$ = {ns}", transform=axs.transAxes)
    axs.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("test_1_3.png", dpi=100)


def gaussian(ts: int, t0: int, sigma: float):
    return numpy.exp(-0.5*((ts - t0)/sigma)**2)


def dielectric(nx: int, epsr: float):
    cb = 0.5 + numpy.zeros(nx, numpy.float32)
    cb[nx//2:] = 0.5/epsr
    return cb


def main():

    nx: int = 38000  # number of grid points
    ns: int = 40000  # number of time steps

    ex = numpy.zeros(nx, numpy.float32)
    hy = numpy.zeros(nx, numpy.float32)

    bc = numpy.zeros(4, numpy.float32)

    epsr: float = 4.0  # relative permittivity
    cb: F32Array = dielectric(nx, epsr)

    stime = time.perf_counter()

    for ts in map(int,range(1,ns+1)):
        # calculate the Ex field
        ex[1:nx] += cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
        # put a Gaussian pulse at the low end
        ex[1] += gaussian(ts, 40, 12.0)
        # absorbing boundary conditions
        ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
        ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
        # calculate the Hy field
        hy[0:nx-1] += 0.5 * (ex[0:nx-1] - ex[1:nx])

    ntime = time.perf_counter()
    print(f"Total compute time on CPU: {ntime - stime:.3f} s")

    print(ex[0:50])
    visualize(ns, nx, epsr, cb, ex)


if __name__ == "__main__":
    main()
