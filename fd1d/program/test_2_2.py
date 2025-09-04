#!/usr/bin/env python
# File: test_2_2.py
# Name: D.Saravanan
# Date: 07/12/2021

""" Simulation of a pulse striking a dielectric medium and implements
the discrete Fourier transform with a Gaussian pulse as its source """

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp, hypot, sin, cos, arctan2
from collections import namedtuple
import time

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, epsr: float, sigma: float, nax: np.ndarray, ex: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a pulse striking dielectric material")
    medium = (1-nax)/(1-nax[-1])*1e3 if epsr > 1 else (1-nax)
    medium[medium==0] = -1e3
    ax.plot(ex, color="black", linewidth=1)
    ax.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=range(0, nx+1, round(nx//10,-1)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=ax.transAxes)
    ax.text(0.85, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()


def amplitude(ns: int, nx: int, epsr: float, sigma: float, nax: np.ndarray, amp: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"The discrete Fourier transform with pulse as its source")
    medium = (1-nax)/(1-nax[-1])*1e3 if epsr > 1 else (1-nax)
    medium[medium==0] = -1e3
    ax.plot(amp, color="black", linewidth=1)
    ax.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-0.2, 2.2))
    ax.set(xticks=range(0, nx+1, round(nx//10,-1)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$Amp\;(V)$")
    ax.text(0.02, 0.90, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=ax.transAxes)
    ax.text(0.85, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()


medium = namedtuple('medium', (
    'nax', 'nbx',
))


ftrans = namedtuple('ftrans', (
    'r_pt', 'i_pt',
    'r_in', 'i_in',
))


def gaussian(t: int, t0: int, sigma: float) -> float:
    return np.exp(-0.5*((t - t0)/sigma)**2)


def fourier(t: int, nf: int, nx: int, dt: float, freq: np.ndarray, ex: np.ndarray, ft: ftrans) -> None:
    # calculate the Fourier transform of Ex field
    ft.r_pt[0:nf,0:nx] += cos(2*pi*freq[0:nf]*dt*t) * ex[0:nx]
    ft.i_pt[0:nf,0:nx] -= sin(2*pi*freq[0:nf]*dt*t) * ex[0:nx]
    if t < nx//2:
        # calculate the Fourier transform of input source
        ft.r_in[0:nf] += cos(2*pi*freq[0:nf]*dt*t) * ex[10]
        ft.i_in[0:nf] -= sin(2*pi*freq[0:nf]*dt*t) * ex[10]


def dxfield(t: int, nx: int, dx: np.ndarray, hy: np.ndarray) -> None:
    # calculate the electric flux density Dx
    dx[1:nx] += 0.5 * (hy[0:nx-1] - hy[1:nx])
    # put a Gaussian pulse at the low end
    dx[1] += gaussian(t, 50, 10.0)


def exfield(nx: int, md: medium, dx: np.ndarray, ix: np.ndarray, ex: np.ndarray) -> None:
    # calculate the Ex field from Dx
    ex[1:nx] = md.nax[1:nx] * (dx[1:nx] - ix[1:nx])
    ix[1:nx] += md.nbx[1:nx] * ex[1:nx]


def hyfield(nx: int, ex: np.ndarray, hy: np.ndarray, bc: np.ndarray) -> None:
    # absorbing boundary conditions
    ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
    ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
    # calculate the Hy field
    hy[0:nx-1] += 0.5 * (ex[0:nx-1] - ex[1:nx])


def dielectric(nx: int, dt: float, epsr: float, sigma: float) -> medium:
    md = medium(
        nax = np.full(nx, 1.0, dtype=np.float32),
        nbx = np.full(nx, 0.0, dtype=np.float32),
    )
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    md.nax[nx//2:] = 1/(epsr + sigma*dt/eps0)
    md.nbx[nx//2:] = sigma*dt/eps0
    return md


def main():

    nx: int = 38000  # number of grid points
    ns: int = 40000  # number of time steps

    dx = np.zeros(nx, dtype=np.float32)
    ex = np.zeros(nx, dtype=np.float32)
    ix = np.zeros(nx, dtype=np.float32)
    hy = np.zeros(nx, dtype=np.float32)

    bc = np.zeros(4, dtype=np.float32)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    epsr: float = 4.0  # relative permittivity
    sigma: float = 0.0  # conductivity (S/m)
    md: medium = dielectric(nx, dt, epsr, sigma)

    # frequency 100 MHz, 200 MHz, 500 MHz
    freq = np.array(([100e6], [200e6], [500e6]), dtype=np.float32)
    nf: int = len(freq)  # number of frequencies

    ft = ftrans(
        r_pt = np.zeros((nf, nx), dtype=np.float32),
        i_pt = np.zeros((nf, nx), dtype=np.float32),
        r_in = np.zeros((nf, 1), dtype=np.float32),
        i_in = np.zeros((nf, 1), dtype=np.float32),
    )

    amplt = np.zeros((nf, nx), dtype=np.float32)
    phase = np.zeros((nf, nx), dtype=np.float32)

    stime = time.perf_counter()

    for t in np.arange(1, ns+1).astype(np.int32):
        dxfield(t, nx, dx, hy)
        exfield(nx, md, dx, ix, ex)
        fourier(t, nf, nx, dt, freq, ex, ft)
        hyfield(nx, ex, hy, bc)

    # calculate the amplitude and phase at each frequency
    amplt = 1/hypot(ft.r_in,ft.i_in) * hypot(ft.r_pt,ft.i_pt)
    phase = arctan2(ft.i_pt,ft.r_pt) - arctan2(ft.i_in,ft.r_in)

    ntime = time.perf_counter()
    print(f"Total compute time on CPU: {ntime - stime:.3f} s")

    print(ex[0:50])
    visualize(ns, nx, epsr, sigma, md.nax, ex)
    amplitude(ns, nx, epsr, sigma, md.nax, amplt[2])


if __name__ == "__main__":
    main()
