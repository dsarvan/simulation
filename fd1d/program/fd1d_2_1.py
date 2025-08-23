#!/usr/bin/env python
# File: fd1d_2_1.py
# Name: D.Saravanan
# Date: 25/11/2021

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, epsr: float, sigma: float, nax: np.ndarray, ex: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal striking lossy dielectric material")
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
    plt.savefig("fd1d_2_1.png", dpi=100)


medium = namedtuple('medium', (
    'nax', 'nbx',
))


def sinusoidal(t: int, ds: float, freq: float) -> float:
    dt: float = ds/6e8  # time step (s)
    return np.sin(2*np.pi*freq*dt*t)


def dxfield(t: int, nx: int, dx: np.ndarray, hy: np.ndarray) -> None:
    # calculate the electric flux density Dx
    dx[1:nx] += 0.5 * (hy[0:nx-1] - hy[1:nx])
    # put a sinusoidal wave at the low end
    dx[1] += sinusoidal(t, 0.01, 700e6)


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
        nax = np.full(nx, 1.0, dtype=np.float64),
        nbx = np.full(nx, 0.0, dtype=np.float64),
    )
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    md.nax[nx//2:] = 1/(epsr + sigma*dt/eps0)
    md.nbx[nx//2:] = sigma*dt/eps0
    return md


def main():

    nx: int = 512  # number of grid points
    ns: int = 740  # number of time steps

    dx = np.zeros(nx, dtype=np.float64)
    ex = np.zeros(nx, dtype=np.float64)
    ix = np.zeros(nx, dtype=np.float64)
    hy = np.zeros(nx, dtype=np.float64)

    bc = np.zeros(4, dtype=np.float64)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    epsr: float = 4.0  # relative permittivity
    sigma: float = 0.04  # conductivity (S/m)
    md: medium = dielectric(nx, dt, epsr, sigma)

    for t in np.arange(1, ns+1).astype(np.int32):
        dxfield(t, nx, dx, hy)
        exfield(nx, md, dx, ix, ex)
        hyfield(nx, ex, hy, bc)

    visualize(ns, nx, epsr, sigma, md.nax, ex)


if __name__ == "__main__":
    main()
