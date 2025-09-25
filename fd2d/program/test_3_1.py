#!/usr/bin/env python
# File: test_3_1.py
# Name: D.Saravanan
# Date: 17/01/2022

""" Simulation of a pulse in free space in the transverse magnetic (TM) mode """

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
import numpy as np
import time

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def surfaceplot(ns: int, nx: int, ny: int, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    fig.suptitle(r"FDTD simulation of a pulse in free space")
    yv, xv = np.meshgrid(range(ny), range(nx))
    ax.plot_surface(xv, yv, ez, rstride=1, cstride=1, cmap="gray", lw=0.25)
    ax.text2D(0.1, 0.7, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(False); ax.view_init(elev=20.0, azim=45)
    plt.show()


def contourplot(ns: int, nx: int, ny: int, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a pulse in free space")
    yv, xv = np.meshgrid(range(ny), range(nx))
    ax.contourf(xv, yv, ez, cmap="gray", alpha=0.75)
    ax.contour(xv, yv, ez, colors="k", linewidths=0.25)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()


def gaussian(t: int, t0: int, sigma: float) -> float:
    return np.exp(-0.5*((t - t0)/sigma)**2)


def dfield(t: int, nx: int, ny: int, dz: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> None:
    """ calculate the electric flux density Dz """
    dz[1:nx,1:ny] += 0.5 * (hy[1:nx,1:ny] - hy[0:nx-1,1:ny] - hx[1:nx,1:ny] + hx[1:nx,0:ny-1])
    # put a Gaussian pulse in the middle
    dz[nx//2,ny//2] = gaussian(t, 20, 6.0)


def efield(nx: int, ny: int, naz: np.ndarray, dz: np.ndarray, ez: np.ndarray) -> None:
    """ calculate the Ez field from Dz """
    ez[0:nx,0:ny] = naz[0:nx,0:ny] * dz[0:nx,0:ny]


def hfield(nx: int, ny: int, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> None:
    """ calculate the Hx and Hy field """
    hx[0:nx-1,0:ny-1] += 0.5 * (ez[0:nx-1,0:ny-1] - ez[0:nx-1,1:ny])
    hy[0:nx-1,0:ny-1] -= 0.5 * (ez[0:nx-1,0:ny-1] - ez[1:nx,0:ny-1])


def main():

    nx: int = 1024  # number of grid points
    ny: int = 1024  # number of grid points

    ns: int = 5000  # number of time steps

    dz = np.zeros((nx, ny), dtype=np.float32)
    ez = np.zeros((nx, ny), dtype=np.float32)
    hx = np.zeros((nx, ny), dtype=np.float32)
    hy = np.zeros((nx, ny), dtype=np.float32)

    naz = np.ones((nx, ny), dtype=np.float32)

    stime = time.perf_counter()

    for t in np.arange(1, ns+1).astype(np.int32):
        dfield(t, nx, ny, dz, hx, hy)
        efield(nx, ny, naz, dz, ez)
        hfield(nx, ny, ez, hx, hy)

    ntime = time.perf_counter()
    print(f"Total compute time on CPU: {ntime - stime:.3f} s")

    print(ez[2][0:50])
    surfaceplot(ns, nx, ny, ez)
    contourplot(ns, nx, ny, ez)


if __name__ == "__main__":
    main()
