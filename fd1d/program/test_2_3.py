#!/usr/bin/env python
# File: test_2_3.py
# Name: D.Saravanan
# Date: 10/01/2022

""" Simulation of a pulse striking a frequency-dependent dielectric material and
implements the discrete Fourier transform with a Gaussian pulse as its source """

import time
import numpy
from matplotlib import pyplot
from typing import NamedTuple
from numpy.typing import NDArray
from numpy import pi, exp, sin, cos, hypot, arctan2

pyplot.style.use("classic")
pyplot.style.use("pyplot.mplstyle")

type F32Array = NDArray[numpy.float32]


def visualize(ns: int, nx: int, epsr: float, sigma: float, nax: F32Array, ex: F32Array):
    fig, axs = pyplot.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a pulse striking Debye dielectric material")
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
    fig.savefig("test_2_3.png", dpi=100)


def amplitude(ns: int, nx: int, epsr: float, sigma: float, nax: F32Array, amp: F32Array):
    fig, axs = pyplot.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"The discrete Fourier transform with pulse as its source")
    medium = numpy.where(1.0/nax-1)[0] if epsr > 1 else (1.0/nax-1)
    axs.plot(range(nx), amp, color="k", linewidth=1.0)
    axs.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-0.2, 2.2))
    axs.set(xticks=range(0, nx+1, int(numpy.ceil(nx/500)*50)))
    axs.set(xlabel=r"$z\;(cm)$", ylabel=r"$Amplitude$")
    axs.text(0.02, 0.90, rf"$T$ = {ns}", transform=axs.transAxes)
    axs.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=axs.transAxes)
    axs.text(0.85, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("test_amp_2_3.png", dpi=100)


FTrans = NamedTuple('FTrans',[
    ('rpt',F32Array), ('ipt',F32Array),
    ('rin',F32Array), ('iin',F32Array),
])


Medium = NamedTuple('Medium',[
    ('nax',F32Array), ('nbx',F32Array),
    ('ncx',F32Array), ('ndx',F32Array),
])


def gaussian(ts: int, t0: int, sigma: float):
    return exp(-0.5*((ts - t0)/sigma)**2)


def fourier(ts: int, nf: int, nx: int, dt: float, freq: F32Array, ex: F32Array, ft: FTrans):
    # calculate the Fourier transform of input source
    ft.rin[0:nf] += cos(2*pi*freq[0:nf]*dt*ts) * ex[10]
    ft.iin[0:nf] -= sin(2*pi*freq[0:nf]*dt*ts) * ex[10]
    # calculate the Fourier transform of Ex field
    ft.rpt[0:nf,0:nx] += cos(2*pi*freq[0:nf]*dt*ts) * ex[0:nx]
    ft.ipt[0:nf,0:nx] -= sin(2*pi*freq[0:nf]*dt*ts) * ex[0:nx]


def dxfield(ts: int, nx: int, hy: F32Array, dx: F32Array):
    # calculate the electric flux density Dx
    dx[1:nx] += 0.5 * (hy[0:nx-1] - hy[1:nx])
    # put a Gaussian pulse at the low end
    dx[1] += gaussian(ts, 50, 10.0)


def exfield(nx: int, md: Medium, ix: F32Array, sx: F32Array, dx: F32Array, ex: F32Array):
    # calculate the Ex field from Dx
    ex[1:nx] = md.nax[1:nx] * (dx[1:nx] - ix[1:nx] - md.ncx[1:nx] * sx[1:nx])
    ix[1:nx] += md.nbx[1:nx] * ex[1:nx]
    sx[1:nx] = md.ncx[1:nx] * sx[1:nx] + md.ndx[1:nx] * ex[1:nx]


def hyfield(nx: int, bc: F32Array, ex: F32Array, hy: F32Array):
    # absorbing boundary conditions
    ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
    ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
    # calculate the Hy field
    hy[0:nx-1] += 0.5 * (ex[0:nx-1] - ex[1:nx])


def dielectric(nx: int, dt: float, chi: float, tau: float, epsr: float, sigma: float):
    md = Medium(
        nax = numpy.full(nx, 1.0, numpy.float32),
        nbx = numpy.full(nx, 0.0, numpy.float32),
        ncx = numpy.full(nx, 0.0, numpy.float32),
        ndx = numpy.full(nx, 0.0, numpy.float32),
    )
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    md.nax[nx//2:] = 1/(epsr + sigma*dt/eps0 + chi*dt/tau)
    md.nbx[nx//2:] = sigma*dt/eps0
    md.ncx[nx//2:] = exp(-dt/tau)
    md.ndx[nx//2:] = chi*dt/tau
    return md


def main():

    nx: int = 38000  # number of grid points
    ns: int = 40000  # number of time steps

    dx = numpy.zeros(nx, numpy.float32)
    ex = numpy.zeros(nx, numpy.float32)
    ix = numpy.zeros(nx, numpy.float32)
    sx = numpy.zeros(nx, numpy.float32)
    hy = numpy.zeros(nx, numpy.float32)

    bc = numpy.zeros(4, numpy.float32)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    chi: float = 2.0  # relaxation susceptibility
    tau: float = 0.001e-6  # relaxation time (s)
    epsr: float = 2.0  # relative permittivity
    sigma: float = 0.01  # conductivity (S/m)
    md: Medium = dielectric(nx, dt, chi, tau, epsr, sigma)

    # frequency 50 MHz, 200 MHz, 500 MHz
    freq = numpy.array(([50e6], [200e6], [500e6]), numpy.float32)
    nf: int = len(freq)  # number of frequencies

    ft = FTrans(
        rpt = numpy.zeros((nf,nx), numpy.float32),
        ipt = numpy.zeros((nf,nx), numpy.float32),
        rin = numpy.zeros((nf, 1), numpy.float32),
        iin = numpy.zeros((nf, 1), numpy.float32),
    )

    amplt = numpy.zeros((nf,nx), numpy.float32)
    phase = numpy.zeros((nf,nx), numpy.float32)

    stime = time.perf_counter()

    for ts in map(int,range(1,ns+1)):
        dxfield(ts, nx, hy, dx)
        exfield(nx, md, ix, sx, dx, ex)
        fourier(ts, nf, nx, dt, freq, ex, ft)
        hyfield(nx, bc, ex, hy)

    # calculate the amplitude and phase at each frequency
    amplt = 1/hypot(ft.rin,ft.iin) * hypot(ft.rpt,ft.ipt)
    phase = arctan2(ft.ipt,ft.rpt) - arctan2(ft.iin,ft.rin)

    ntime = time.perf_counter()
    print(f"Total compute time on CPU: {ntime - stime:.3f} s")

    print(ex[0:50])
    visualize(ns, nx, epsr, sigma, md.nax, ex)
    amplitude(ns, nx, epsr, sigma, md.nax, amplt[2])


if __name__ == "__main__":
    main()
