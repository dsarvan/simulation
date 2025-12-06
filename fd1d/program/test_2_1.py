#!/usr/bin/env python
# File: test_2_1.py
# Name: D.Saravanan
# Date: 25/11/2021

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import time
import numpy
from numpy import pi, sin
from matplotlib import pyplot
from typing import NamedTuple
from numpy.typing import NDArray

pyplot.style.use("classic")
pyplot.style.use("pyplot.mplstyle")

type F32Array = NDArray[numpy.float32]


def visualize(ns: int, nx: int, epsr: float, sigma: float, nax: F32Array, ex: F32Array):
    fig, axs = pyplot.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal striking lossy dielectric material")
    medium = numpy.where(1.0/nax-1)[0] if epsr > 1 else (1.0/nax-1)
    axs.plot(range(nx), ex, color="k", linewidth=1.0)
    axs.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=range(0, nx+1, int(numpy.ceil(nx/500)*50)))
    axs.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axs.text(0.02, 0.90, rf"$T$ = {ns}", transform=axs.transAxes)
    axs.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=axs.transAxes)
    axs.text(0.85, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("test_2_1.png", dpi=100)


Medium = NamedTuple('Medium',[
    ('nax',F32Array), ('nbx',F32Array),
])


def sinusoidal(ts: int, ds: float, freq: float):
    dt: float = ds/6e8  # time step (s)
    return sin(2*pi*freq*dt*ts)


def dxfield(ts: int, nx: int, hy: F32Array, dx: F32Array):
    # calculate the electric flux density Dx
    dx[1:nx] += 0.5 * (hy[0:nx-1] - hy[1:nx])
    # put a sinusoidal wave at the low end
    dx[1] += sinusoidal(ts, 0.01, 700e6)


def exfield(nx: int, md: Medium, ix: F32Array, dx: F32Array, ex: F32Array):
    # calculate the Ex field from Dx
    ex[1:nx] = md.nax[1:nx] * (dx[1:nx] - ix[1:nx])
    ix[1:nx] += md.nbx[1:nx] * ex[1:nx]


def hyfield(nx: int, bc: F32Array, ex: F32Array, hy: F32Array):
    # absorbing boundary conditions
    ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
    ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
    # calculate the Hy field
    hy[0:nx-1] += 0.5 * (ex[0:nx-1] - ex[1:nx])


def dielectric(nx: int, dt: float, epsr: float, sigma: float):
    md = Medium(
        nax = numpy.full(nx, 1.0, numpy.float32),
        nbx = numpy.full(nx, 0.0, numpy.float32),
    )
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    md.nax[nx//2:] = 1/(epsr + sigma*dt/eps0)
    md.nbx[nx//2:] = sigma*dt/eps0
    return md


def main():

    nx: int = 38000  # number of grid points
    ns: int = 40000  # number of time steps

    dx = numpy.zeros(nx, numpy.float32)
    ex = numpy.zeros(nx, numpy.float32)
    ix = numpy.zeros(nx, numpy.float32)
    hy = numpy.zeros(nx, numpy.float32)

    bc = numpy.zeros(4, numpy.float32)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    epsr: float = 4.0  # relative permittivity
    sigma: float = 0.04  # conductivity (S/m)
    md: Medium = dielectric(nx, dt, epsr, sigma)

    stime = time.perf_counter()

    for ts in map(int,range(1,ns+1)):
        dxfield(ts, nx, hy, dx)
        exfield(nx, md, ix, dx, ex)
        hyfield(nx, bc, ex, hy)

    ntime = time.perf_counter()
    print(f"Total compute time on CPU: {ntime - stime:.3f} s")

    print(ex[0:50])
    visualize(ns, nx, epsr, sigma, md.nax, ex)


if __name__ == "__main__":
    main()
