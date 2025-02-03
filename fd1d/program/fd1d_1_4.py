#!/usr/bin/env python
# File: fd1d_1_4.py
# Name: D.Saravanan
# Date: 22/10/2021

""" Simulation of a propagating sinusoidal striking a dielectric medium """

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, epsr: float, cb: np.ndarray, ex: np.ndarray, hy: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal striking dielectric material")
    medium = (0.5/cb - 1)/(epsr - 1) if epsr > 1 else (0.5/cb - 1)
    medium[medium==0] = -1e3
    ax1.plot(ex, "k", lw=1)
    ax1.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    ax1.text(nx/4, 0.5, f"T = {ns}", horizontalalignment="center")
    ax1.text(3*nx/4, 0.5, f"epsr = {epsr}", horizontalalignment="center")
    ax1.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
    ax1.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
    ax2.plot(hy, "k", lw=1)
    ax2.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    ax2.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"$H_y$")
    ax2.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd1d_1_4.png")


def sinusoidal(t: int, ddx: float, freq: float) -> float:
    dt: float = ddx/6e8  # time step
    return np.sin(2 * np.pi * freq * dt * t)


def dielectric(nx: int, epsr: float) -> np.ndarray:
    cb = 0.5 * np.ones(nx, dtype=np.float64)
    cb[nx//2:] = 0.5/epsr
    return cb


def main():

    nx: int = 1024
    ns: int = 1500

    ex = np.zeros(nx, dtype=np.float64)
    hy = np.zeros(nx, dtype=np.float64)

    bc = np.zeros(4, dtype=np.float64)

    ddx: float = 0.01  # cell size (m)
    dt: float = ddx/6e8  # time step
    epsr: float = 4  # relative permittivity
    cb: np.ndarray = dielectric(nx, epsr)

    for t in np.arange(1, ns+1).astype(np.int32):
        # calculate the Ex field
        ex[1:nx] = ex[1:nx] + cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
        # put a sinusoidal wave at the low end
        ex[1] = ex[1] + sinusoidal(t, 0.01, 700e6)
        # absorbing boundary conditions
        ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
        ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
        # calculate the Hy field
        hy[0:nx-1] = hy[0:nx-1] + 0.5 * (ex[0:nx-1] - ex[1:nx])

    visualize(ns, nx, epsr, cb, ex, hy)


if __name__ == "__main__":
    main()
