#!/usr/bin/env python
# File: fd1d_1_4.py
# Name: D.Saravanan
# Date: 22/10/2021

""" Simulation of a propagating sinusoidal striking a dielectric medium """

import numpy
from numpy import pi, sin
from matplotlib import pyplot
from numpy.typing import NDArray

pyplot.style.use("classic")
pyplot.style.use("pyplot.mplstyle")

type F64Array = NDArray[numpy.float64]


def visualize(ns: int, nx: int, epsr: float, cb: F64Array, ex: F64Array):
    fig, axs = pyplot.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal striking dielectric material")
    medium = numpy.where(0.5/cb-1)[0] if epsr > 1 else (0.5/cb-1)
    axs.plot(range(nx), ex, color="k", linewidth=1.0)
    axs.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=range(0, nx+1, int(numpy.ceil(nx/500)*25)))
    axs.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axs.text(0.02, 0.90, rf"$T$ = {ns}", transform=axs.transAxes)
    axs.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("fd1d_1_4.png", dpi=100)


def sinusoidal(ts: int, ds: float, freq: float):
    dt: float = ds/6e8  # time step (s)
    return sin(2*pi*freq*dt*ts)


def dielectric(nx: int, epsr: float):
    cb = 0.5 + numpy.zeros(nx, numpy.float64)
    cb[nx//2:] = 0.5/epsr
    return cb


def main():

    nx: int = 512  # number of grid points
    ns: int = 740  # number of time steps

    ex = numpy.zeros(nx, numpy.float64)
    hy = numpy.zeros(nx, numpy.float64)

    bc = numpy.zeros(4, numpy.float64)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    epsr: float = 4.0  # relative permittivity
    cb: F64Array = dielectric(nx, epsr)

    for ts in map(int,range(1,ns+1)):
        # calculate the Ex field
        ex[1:nx] += cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
        # put a sinusoidal wave at the low end
        ex[1] += sinusoidal(ts, 0.01, 700e6)
        # absorbing boundary conditions
        ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
        ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
        # calculate the Hy field
        hy[0:nx-1] += 0.5 * (ex[0:nx-1] - ex[1:nx])

    visualize(ns, nx, epsr, cb, ex)


if __name__ == "__main__":
    main()
