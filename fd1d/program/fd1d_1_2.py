#!/usr/bin/env python
# File: fd1d_1_2.py
# Name: D.Saravanan
# Date: 19/10/2021

""" Simulation of a pulse with absorbing boundary conditions """

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, ex: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a pulse with absorbing boundary conditions")
    ax.plot(range(nx), ex, color="k", linewidth=1.0)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=range(0, nx+1, int(np.ceil(nx/500)*25)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, rf"$T$ = {ns}", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd1d_1_2.png", dpi=100)


def gaussian(t: int, t0: int, sigma: float) -> float:
    return np.exp(-0.5*((t - t0)/sigma)**2)


def main():

    nx: int = 512  # number of grid points
    ns: int = 570  # number of time steps

    ex = np.zeros(nx, dtype=np.float64)
    hy = np.zeros(nx, dtype=np.float64)

    bc = np.zeros(4, dtype=np.float64)

    for t in np.arange(1, ns+1).astype(np.int32):
        # calculate the Ex field
        ex[1:nx] += 0.5 * (hy[0:nx-1] - hy[1:nx])
        # put a Gaussian pulse in the middle
        ex[nx//2] = gaussian(t, 40, 12.0)
        # absorbing boundary conditions
        ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
        ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
        # calculate the Hy field
        hy[0:nx-1] += 0.5 * (ex[0:nx-1] - ex[1:nx])

    visualize(ns, nx, ex)


if __name__ == "__main__":
    main()
