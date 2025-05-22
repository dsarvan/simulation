#!/usr/bin/env python
# File: fd2d_3_3.py
# Name: D.Saravanan
# Date: 19/01/2022

""" Simulation of a plane wave pulse propagating in free space in the transverse
magnetic (TM) mode with the two-dimensional perfectly matched layer (PML) """

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from collections import namedtuple
import numba as nb
import numpy as np

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


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


@nb.jit(nopython=True, fastmath=True)
def gaussian(t: int, t0: int, sigma: float) -> float:
    return np.exp(-0.5 * ((t - t0)/sigma)**2)


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


@nb.jit(nopython=True, parallel=True, fastmath=True)
def ezinct(ny: int, ezi: np.ndarray, hxi: np.ndarray, bc: np.ndarray) -> None:
    """ calculate the incident Ez """
    for j in nb.prange(1, ny):
        ezi[j] += 0.5 * (hxi[j-1] - hxi[j])
    # absorbing boundary conditions
    ezi[0], bc[0], bc[1] = bc[0], bc[1], ezi[1]
    ezi[ny-1], bc[3], bc[2] = bc[3], bc[2], ezi[ny-2]


@nb.jit(nopython=True, parallel=True, fastmath=True)
def dfield(t: int, nx: int, ny: int, pml: pmlayer, ezi: np.ndarray, dz: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> None:
    """ calculate the electric flux density Dz """
    for i in nb.prange(1, nx):
        for j in nb.prange(1, ny):
            dz[i,j] = pml.gx3[i] * pml.gy3[j] * dz[i,j] + pml.gx2[i] * pml.gy2[j] * 0.5 * (hy[i,j] - hy[i-1,j] - hx[i,j] + hx[i,j-1])
    # put a Gaussian pulse at the low end
    ezi[3] = gaussian(t, 20, 8)


@nb.jit(nopython=True, parallel=True, fastmath=True)
def inctdz(nx: int, ny: int, npml: int, hxi: np.ndarray, dz: np.ndarray) -> None:
    """ incident Dz values """
    for i in nb.prange(npml-1, nx-npml+1):
        dz[i,npml-1] += 0.5 * hxi[npml-2]
        dz[i,ny-npml] -= 0.5 * hxi[ny-npml]


@nb.jit(nopython=True, parallel=True, fastmath=True)
def efield(nx: int, ny: int, naz: np.ndarray, dz: np.ndarray, ez: np.ndarray) -> None:
    """ calculate the Ez field from Dz """
    for i in nb.prange(0, nx):
        for j in nb.prange(0, ny):
            ez[i,j] = naz[i,j] * dz[i,j]


@nb.jit(nopython=True, parallel=True, fastmath=True)
def hxinct(ny: int, ezi: np.ndarray, hxi: np.ndarray) -> None:
    """ calculate the incident Hx """
    for j in nb.prange(0, ny-1):
        hxi[j] += 0.5 * (ezi[j] - ezi[j+1])


@nb.jit(nopython=True, parallel=True, fastmath=True)
def hfield(nx: int, ny: int, pml: pmlayer, ez: np.ndarray, ihx: np.ndarray, ihy: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> None:
    """ calculate the Hx and Hy field """
    for i in nb.prange(0, nx-1):
        for j in nb.prange(0, ny-1):
            ihx[i,j] += ez[i,j] - ez[i,j+1]
            ihy[i,j] += ez[i,j] - ez[i+1,j]
            hx[i,j] = pml.fy3[j] * hx[i,j] + pml.fy2[j] * (0.5 * ez[i,j] - 0.5 * ez[i,j+1] + pml.fx1[i] * ihx[i,j])
            hy[i,j] = pml.fx3[i] * hy[i,j] - pml.fx2[i] * (0.5 * ez[i,j] - 0.5 * ez[i+1,j] + pml.fy1[j] * ihy[i,j])


@nb.jit(nopython=True, parallel=True, fastmath=True)
def incthx(nx: int, ny: int, npml: int, ezi: np.ndarray, hx: np.ndarray) -> None:
    """ incident Hx values """
    for i in nb.prange(npml-1, nx-npml+1):
        hx[i,npml-2] += 0.5 * ezi[npml-1]
        hx[i,ny-npml] -= 0.5 * ezi[ny-npml]


@nb.jit(nopython=True, parallel=True, fastmath=True)
def incthy(nx: int, ny: int, npml: int, ezi: np.ndarray, hy: np.ndarray) -> None:
    """ incident Hy values """
    for j in nb.prange(npml-1, ny-npml+1):
        hy[npml-2,j] -= 0.5 * ezi[j]
        hy[nx-npml,j] += 0.5 * ezi[j]


def main():

    nx: int = 60  # number of grid points
    ny: int = 60  # number of grid points

    ns: int = 200  # number of time steps

    ezi = np.zeros(ny, dtype=np.float64)
    hxi = np.zeros(ny, dtype=np.float64)

    dz = np.zeros((nx, ny), dtype=np.float64)
    ez = np.zeros((nx, ny), dtype=np.float64)
    hx = np.zeros((nx, ny), dtype=np.float64)
    hy = np.zeros((nx, ny), dtype=np.float64)

    ihx = np.zeros((nx, ny), dtype=np.float64)
    ihy = np.zeros((nx, ny), dtype=np.float64)

    bc = np.zeros(4, dtype=np.float64)

    naz = np.ones((nx, ny), dtype=np.float64)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)

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

    ezdata = [ez]
    for t in np.arange(1, ns+1).astype(np.int32):
        ezinct(ny, ezi, hxi, bc)
        dfield(t, nx, ny, pml, ezi, dz, hx, hy)
        inctdz(nx, ny, npml, hxi, dz)
        efield(nx, ny, naz, dz, ez)
        hxinct(ny, ezi, hxi)
        hfield(nx, ny, pml, ez, ihx, ihy, hx, hy)
        incthx(nx, ny, npml, ezi, hx)
        incthy(nx, ny, npml, ezi, hy)
        ezdata.append(ez.copy())

    ez = np.array(ezdata)

    # define the meta data for the movie
    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Simulation of a plane wave in the transverse magnetic (TM) mode"}
    writer = fwriter(fps=15, codec='h264', bitrate=2000, metadata=data)

    # draw an empty plot, but preset the plot x-, y- and z- limits
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.suptitle(r"FDTD simulation of a plane wave in free space with PML")
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    ax.plot_surface(yv, xv, ez[0], rstride=1, cstride=1, cmap="gray", lw=0.25)
    ax.text2D(0.1, 0.7, "", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(False)
    ax.view_init(elev=20.0, azim=45)
    plt.subplots_adjust(bottom=0.1, hspace=0.45)

    with writer.saving(fig, "fd2d_surface_3_3.mp4", 300):
        for t in np.arange(1, ns+1).astype(np.int32):
            ax.clear()
            ax.plot_surface(yv, xv, ez[t], rstride=1, cstride=1, cmap="gray", lw=0.25)
            ax.text2D(0.1, 0.7, rf"$T$ = {t}", transform=ax.transAxes)
            ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
            ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
            writer.grab_frame()

    plt.close(fig)

    # draw an empty plot, but preset the plot x-, y- and z- limits
    fig, ax = plt.subplots(gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a plane wave in free space with PML")
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    ax.contourf(yv, xv, ez[0], cmap="gray", alpha=0.75)
    ax.contour(yv, xv, ez[0], colors="k", linewidths=0.25)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.1, hspace=0.45)

    with writer.saving(fig, "fd2d_contour_3_3.mp4", 300):
        for t in np.arange(1, ns+1).astype(np.int32):
            ax.clear()
            ax.contourf(yv, xv, ez[t], cmap="gray", alpha=0.75)
            ax.contour(yv, xv, ez[t], colors="k", linewidths=0.25)
            ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
            ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
            writer.grab_frame()

    plt.close(fig)


if __name__ == "__main__":
    main()
