#!/usr/bin/env python
# File: test_3_3.py
# Name: D.Saravanan
# Date: 19/01/2022

""" Simulation of a plane wave pulse propagating in free space in the transverse
magnetic (TM) mode with the two-dimensional perfectly matched layer (PML) """

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from collections import namedtuple
import numpy as np
import time

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def surfaceplot(ns: int, nx: int, ny: int, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    fig.suptitle(r"FDTD simulation of a plane wave in free space with PML")
    yv, xv = np.meshgrid(range(ny), range(nx))
    ax.plot_surface(xv, yv, ez, rstride=1, cstride=1, cmap="gray", lw=0.25)
    ax.text2D(0.1, 0.7, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(False); ax.view_init(elev=20.0, azim=45)
    plt.show()


def contourplot(ns: int, nx: int, ny: int, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a plane wave in free space with PML")
    yv, xv = np.meshgrid(range(ny), range(nx))
    ax.contourf(xv, yv, ez, cmap="gray", alpha=0.75)
    ax.contour(xv, yv, ez, colors="k", linewidths=0.25)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()


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


def gaussian(t: int, t0: int, sigma: float) -> float:
    return np.exp(-0.5*((t - t0)/sigma)**2)


def ezinct(ny: int, ezi: np.ndarray, hxi: np.ndarray, bc: np.ndarray) -> None:
    """ calculate the incident Ez """
    ezi[1:ny] += 0.5 * (hxi[0:ny-1] - hxi[1:ny])
    # absorbing boundary conditions
    ezi[0], bc[0], bc[1] = bc[0], bc[1], ezi[1]
    ezi[ny-1], bc[3], bc[2] = bc[3], bc[2], ezi[ny-2]


def dfield(t: int, nx: int, ny: int, pml: pmlayer, ezi: np.ndarray, dz: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> None:
    """ calculate the electric flux density Dz """
    dz[1:nx,1:ny] = pml.gx3[1:nx,None] * pml.gy3[1:ny] * dz[1:nx,1:ny] + pml.gx2[1:nx,None] * pml.gy2[1:ny] * 0.5 * (hy[1:nx,1:ny] - hy[0:nx-1,1:ny] - hx[1:nx,1:ny] + hx[1:nx,0:ny-1])
    # put a Gaussian pulse at the low end
    ezi[3] = gaussian(t, 20, 8.0)


def inctdz(nx: int, ny: int, npml: int, hxi: np.ndarray, dz: np.ndarray) -> None:
    """ incident Dz values """
    dz[npml-1:nx-npml+1,npml-1] += 0.5 * hxi[npml-2]
    dz[npml-1:nx-npml+1,ny-npml] -= 0.5 * hxi[ny-npml]


def efield(nx: int, ny: int, naz: np.ndarray, dz: np.ndarray, ez: np.ndarray) -> None:
    """ calculate the Ez field from Dz """
    ez[0:nx,0:ny] = naz[0:nx,0:ny] * dz[0:nx,0:ny]


def hxinct(ny: int, ezi: np.ndarray, hxi: np.ndarray) -> None:
    """ calculate the incident Hx """
    hxi[0:ny-1] += 0.5 * (ezi[0:ny-1] - ezi[1:ny])


def hfield(nx: int, ny: int, pml: pmlayer, ez: np.ndarray, ihx: np.ndarray, ihy: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> None:
    """ calculate the Hx and Hy field """
    curl_em = ez[0:nx-1,0:ny-1] - ez[0:nx-1,1:ny]
    curl_en = ez[0:nx-1,0:ny-1] - ez[1:nx,0:ny-1]
    ihx[0:nx-1,0:ny-1] += curl_em
    ihy[0:nx-1,0:ny-1] += curl_en
    hx[0:nx-1,0:ny-1] = pml.fy3[0:ny-1] * hx[0:nx-1,0:ny-1] + pml.fy2[0:ny-1] * (0.5 * curl_em + pml.fx1[0:nx-1,None] * ihx[0:nx-1,0:ny-1])
    hy[0:nx-1,0:ny-1] = pml.fx3[0:nx-1,None] * hy[0:nx-1,0:ny-1] - pml.fx2[0:nx-1,None] * (0.5 * curl_en + pml.fy1[0:ny-1] * ihy[0:nx-1,0:ny-1])


def incthx(nx: int, ny: int, npml: int, ezi: np.ndarray, hx: np.ndarray) -> None:
    """ incident Hx values """
    hx[npml-1:nx-npml+1,npml-2] += 0.5 * ezi[npml-1]
    hx[npml-1:nx-npml+1,ny-npml] -= 0.5 * ezi[ny-npml]


def incthy(nx: int, ny: int, npml: int, ezi: np.ndarray, hy: np.ndarray) -> None:
    """ incident Hy values """
    hy[npml-2,npml-1:ny-npml+1] -= 0.5 * ezi[npml-1:ny-npml+1]
    hy[nx-npml,npml-1:ny-npml+1] += 0.5 * ezi[npml-1:ny-npml+1]


def pmlparam(nx: int, ny: int, npml: int, pml: pmlayer) -> None:
    """ calculate the two-dimensional perfectly matched layer (PML) parameters """
    for n in range(npml):
        xm = 0.33*((npml-n)/npml)**3
        xn = 0.33*((npml-n-0.5)/npml)**3
        pml.fx1[n] = pml.fx1[nx-2-n] = pml.fy1[n] = pml.fy1[ny-2-n] = xn
        pml.fx2[n] = pml.fx2[nx-2-n] = pml.fy2[n] = pml.fy2[ny-2-n] = 1/(1+xn)
        pml.gx2[n] = pml.gx2[nx-1-n] = pml.gy2[n] = pml.gy2[ny-1-n] = 1/(1+xm)
        pml.fx3[n] = pml.fx3[nx-2-n] = pml.fy3[n] = pml.fy3[ny-2-n] = (1-xn)/(1+xn)
        pml.gx3[n] = pml.gx3[nx-1-n] = pml.gy3[n] = pml.gy3[ny-1-n] = (1-xm)/(1+xm)


def main():

    nx: int = 1024  # number of grid points
    ny: int = 1024  # number of grid points

    ns: int = 5000  # number of time steps

    ezi = np.zeros(ny, dtype=np.float32)
    hxi = np.zeros(ny, dtype=np.float32)

    dz = np.zeros((nx, ny), dtype=np.float32)
    ez = np.zeros((nx, ny), dtype=np.float32)
    hx = np.zeros((nx, ny), dtype=np.float32)
    hy = np.zeros((nx, ny), dtype=np.float32)

    ihx = np.zeros((nx, ny), dtype=np.float32)
    ihy = np.zeros((nx, ny), dtype=np.float32)

    naz = np.ones((nx, ny), dtype=np.float32)

    bc = np.zeros(4, dtype=np.float32)

    pml = pmlayer(
        fx1 = np.full(nx, 0.0, dtype=np.float32),
        fx2 = np.full(nx, 1.0, dtype=np.float32),
        fx3 = np.full(nx, 1.0, dtype=np.float32),
        fy1 = np.full(ny, 0.0, dtype=np.float32),
        fy2 = np.full(ny, 1.0, dtype=np.float32),
        fy3 = np.full(ny, 1.0, dtype=np.float32),
        gx2 = np.full(nx, 1.0, dtype=np.float32),
        gx3 = np.full(nx, 1.0, dtype=np.float32),
        gy2 = np.full(ny, 1.0, dtype=np.float32),
        gy3 = np.full(ny, 1.0, dtype=np.float32),
    )

    npml: int = 8  # pml thickness
    pmlparam(nx, ny, npml, pml)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)

    stime = time.perf_counter()

    for t in np.arange(1, ns+1).astype(np.int32):
        ezinct(ny, ezi, hxi, bc)
        dfield(t, nx, ny, pml, ezi, dz, hx, hy)
        inctdz(nx, ny, npml, hxi, dz)
        efield(nx, ny, naz, dz, ez)
        hxinct(ny, ezi, hxi)
        hfield(nx, ny, pml, ez, ihx, ihy, hx, hy)
        incthx(nx, ny, npml, ezi, hx)
        incthy(nx, ny, npml, ezi, hy)

    ntime = time.perf_counter()
    print(f"Total compute time on CPU: {ntime - stime:.3f} s")

    print(ez[2][0:50])
    surfaceplot(ns, nx, ny, ez)
    contourplot(ns, nx, ny, ez)


if __name__ == "__main__":
    main()
