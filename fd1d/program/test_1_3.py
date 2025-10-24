#!/usr/bin/env python
# File: test_1_3.py
# Name: D.Saravanan
# Date: 21/10/2021

""" Simulation of a pulse hitting a dielectric medium """

import matplotlib.pyplot as plt
import numpy as np
import time

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, epsr: float, cb: np.ndarray, ex: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a pulse striking dielectric material")
    medium = np.where(0.5/cb-1)[0] if epsr > 1 else (0.5/cb-1)
    ax.plot(range(nx), ex, color="k", linewidth=1.0)
    ax.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=range(0, nx+1, int(np.ceil(nx/500)*50)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("test_1_3.png", dpi=100)


def gaussian(t: int, t0: int, sigma: float) -> float:
    return np.exp(-0.5*((t - t0)/sigma)**2)


def dielectric(nx: int, epsr: float) -> np.ndarray:
    cb = 0.5 + np.zeros(nx, dtype=np.float32)
    cb[nx//2:] = 0.5/epsr
    return cb


def main():

    nx: int = 38000  # number of grid points
    ns: int = 40000  # number of time steps

    ex = np.zeros(nx, dtype=np.float32)
    hy = np.zeros(nx, dtype=np.float32)

    bc = np.zeros(4, dtype=np.float32)

    epsr: float = 4.0  # relative permittivity
    cb: np.ndarray = dielectric(nx, epsr)

    stime = time.perf_counter()

    for t in np.arange(1, ns+1).astype(np.int32):
        # calculate the Ex field
        ex[1:nx] += cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
        # put a Gaussian pulse at the low end
        ex[1] += gaussian(t, 40, 12.0)
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
