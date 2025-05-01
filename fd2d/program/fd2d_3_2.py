#!/usr/bin/env python
# File: fd2d_3_2.py
# Name: D.Saravanan
# Date: 18/01/2022

""" Simulation of a propagating sinusoidal in free space in the transverse
magnetic (TM) mode with the two-dimensional perfectly matched layer (PML) """

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from collections import namedtuple
import numpy as np

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def surfaceplot(ns: int, nx: int, ny: int, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.suptitle(r"FDTD simulation of a sinusoidal in free space with PML")
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    ax.plot_surface(yv, xv, ez, rstride=1, cstride=1, cmap="gray", lw=0.25)
    ax.text2D(0.1, 0.7, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(False); ax.view_init(elev=20.0, azim=45)
    plt.savefig("fd2d_surface_3_2.png", dpi=100)


def contourplot(ns: int, nx: int, ny: int, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal in free space with PML")
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    ax.contourf(yv, xv, ez, cmap="gray", alpha=0.75)
    ax.contour(yv, xv, ez, colors="k", linewidths=0.25)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd2d_contour_3_2.png", dpi=100)


pmlayer = namedtuple('pmlayer', (
    'fx1',
    'fx2',
    'fx3',
    'fy1',
    'fy2',
    'fy3',
    'gx2',
    'gx3',
    'gy2',
    'gy3',
))


def sinusoidal(t: int, ds: float, freq: float) -> float:
    dt: float = ds/6e8  # time step (s)
    return np.sin(2 * np.pi * freq * dt * t)


def pmlparam(npml: int, nx: int, ny: int, pml: pmlayer) -> None:
    """ calculate the two-dimensional perfectly matched layer (PML) parameters """
    for n in range(npml):
        xm = 0.33 * ((npml-n)/npml)**3
        xn = 0.33 * ((npml-n-0.5)/npml)**3
        pml.fx1[n] = pml.fx1[nx-2-n] = pml.fy1[n] = pml.fy1[ny-2-n] = xn
        pml.fx2[n] = pml.fx2[nx-2-n] = pml.fy2[n] = pml.fy2[ny-2-n] = 1/(1+xn)
        pml.gx2[n] = pml.gx2[nx-1-n] = pml.gy2[n] = pml.gy2[ny-1-n] = 1/(1+xm)
        pml.fx3[n] = pml.fx3[nx-2-n] = pml.fy3[n] = pml.fy3[ny-2-n] = (1-xn)/(1+xn)
        pml.gx3[n] = pml.gx3[nx-1-n] = pml.gy3[n] = pml.gy3[ny-1-n] = (1-xm)/(1+xm)


def dfield(t: int, nx: int, ny: int, pml: pmlayer, dz: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> None:
    """ calculate the electric flux density Dz """
    dz[1:nx,1:ny] = pml.gx3[1:nx,None] * pml.gy3[1:ny] * dz[1:nx,1:ny] + pml.gx2[1:nx,None] * pml.gy2[1:ny] * 0.5 * (hy[1:nx,1:ny] - hy[0:nx-1,1:ny] - hx[1:nx,1:ny] + hx[1:nx,0:ny-1])
    # put a sinusoidal source at a point that is offset five cells
    # from the center of the problem space in each direction
    dz[nx//2-5,ny//2-5] = sinusoidal(t, 0.01, 1500e6)


def efield(nx: int, ny: int, naz: np.ndarray, dz: np.ndarray, ez: np.ndarray) -> None:
    """ calculate the Ez field from Dz """
    ez[0:nx,0:ny] = naz[0:nx,0:ny] * dz[0:nx,0:ny]


def hfield(nx: int, ny: int, pml: pmlayer, ez: np.ndarray, ihx: np.ndarray, ihy: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> None:
    """ calculate the Hx and Hy field """
    curl_em = ez[0:nx-1,0:ny-1] - ez[0:nx-1,1:ny]
    curl_en = ez[0:nx-1,0:ny-1] - ez[1:nx,0:ny-1]
    ihx[0:nx-1,0:ny-1] += curl_em
    ihy[0:nx-1,0:ny-1] += curl_en
    hx[0:nx-1,0:ny-1] = pml.fy3[0:ny-1] * hx[0:nx-1,0:ny-1] + pml.fy2[0:ny-1] * (0.5 * curl_em + pml.fx1[0:nx-1,None] * ihx[0:nx-1,0:ny-1])
    hy[0:nx-1,0:ny-1] = pml.fx3[0:nx-1,None] * hy[0:nx-1,0:ny-1] - pml.fx2[0:nx-1,None] * (0.5 * curl_en + pml.fy1[0:ny-1] * ihy[0:nx-1,0:ny-1])


def main():

    nx: int = 60  # number of grid points
    ny: int = 60  # number of grid points

    ns: int = 100  # number of time steps

    dz = np.zeros((nx, ny), dtype=np.float64)
    ez = np.zeros((nx, ny), dtype=np.float64)
    hx = np.zeros((nx, ny), dtype=np.float64)
    hy = np.zeros((nx, ny), dtype=np.float64)

    ihx = np.zeros((nx, ny), dtype=np.float64)
    ihy = np.zeros((nx, ny), dtype=np.float64)

    naz = np.ones((nx, ny), dtype=np.float64)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)

    pml = pmlayer(
        fx1 = np.zeros(nx, dtype=np.float64),
        fx2 = np.ones(nx, dtype=np.float64),
        fx3 = np.ones(nx, dtype=np.float64),
        fy1 = np.zeros(ny, dtype=np.float64),
        fy2 = np.ones(ny, dtype=np.float64),
        fy3 = np.ones(ny, dtype=np.float64),
        gx2 = np.ones(nx, dtype=np.float64),
        gx3 = np.ones(nx, dtype=np.float64),
        gy2 = np.ones(ny, dtype=np.float64),
        gy3 = np.ones(ny, dtype=np.float64),
    )

    npml: int = 8  # pml thickness
    pmlparam(npml, nx, ny, pml)

    for t in np.arange(1, ns+1).astype(np.int32):
        dfield(t, nx, ny, pml, dz, hx, hy)
        efield(nx, ny, naz, dz, ez)
        hfield(nx, ny, pml, ez, ihx, ihy, hx, hy)

    surfaceplot(ns, nx, ny, ez)
    contourplot(ns, nx, ny, ez)


if __name__ == "__main__":
    main()
