#!/usr/bin/env python
# File: test_1_5.py
# Name: D.Saravanan
# Date: 29/10/2021

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import matplotlib.pyplot as plt
import numpy as np
import time

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, epsr: float, sigma: float, cb: np.ndarray, ex: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal striking lossy dielectric material")
    medium = (0.5/cb-1)/(epsr-1)*1e3 if epsr > 1 else (0.5/cb-1)
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


def sinusoidal(t: int, ds: float, freq: float) -> float:
    dt: float = ds/6e8  # time step (s)
    return np.sin(2*np.pi*freq*dt*t)


def dielectric(nx: int, dt: float, epsr: float, sigma: float) -> tuple:
    ca = 1.0 + np.zeros(nx, dtype=np.float32)
    cb = 0.5 + np.zeros(nx, dtype=np.float32)
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    epsf: float = dt*sigma/(2*eps0*epsr)
    ca[nx//2:] = (1 - epsf)/(1 + epsf)
    cb[nx//2:] = 0.5/(epsr*(1 + epsf))
    return ca, cb


def main():

    nx: int = 38000  # number of grid points
    ns: int = 40000  # number of time steps

    ex = np.zeros(nx, dtype=np.float32)
    hy = np.zeros(nx, dtype=np.float32)

    bc = np.zeros(4, dtype=np.float32)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    epsr: float = 4.0  # relative permittivity
    sigma: float = 0.04  # conductivity (S/m)
    ca, cb = dielectric(nx, dt, epsr, sigma)

    stime = time.perf_counter()

    for t in np.arange(1, ns+1).astype(np.int32):
        # calculate the Ex field
        ex[1:nx] = ca[1:nx] * ex[1:nx] + cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
        # put a sinusoidal wave at the low end
        ex[1] += sinusoidal(t, 0.01, 700e6)
        # absorbing boundary conditions
        ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
        ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
        # calculate the Hy field
        hy[0:nx-1] += 0.5 * (ex[0:nx-1] - ex[1:nx])

    ntime = time.perf_counter()
    print(f"Total compute time on CPU: {ntime - stime:.3f} s")

    print(ex[0:50])
    visualize(ns, nx, epsr, sigma, cb, ex)


if __name__ == "__main__":
    main()
