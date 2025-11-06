#!/usr/bin/env python
# File: fd2d_3_2.py
# Name: D.Saravanan
# Date: 18/01/2022

""" Simulation of a propagating sinusoidal in free space in the transverse
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
    'fx1', 'fx2', 'fx3',
    'fy1', 'fy2', 'fy3',
    'gx2', 'gx3',
    'gy2', 'gy3',
))


@nb.jit(nopython=True, fastmath=True)
def sinusoidal(t: int, ds: float, freq: float) -> float:
    dt: float = ds/6e8  # time step (s)
    return np.sin(2*np.pi*freq*dt*t)


@nb.jit(nopython=True, parallel=True, fastmath=True)
def dfield(t: int, nx: int, ny: int, pml: pmlayer, dz: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> None:
    """ calculate the electric flux density Dz """
    for i in nb.prange(1, nx):
        for j in nb.prange(1, ny):
            dz[i,j] = pml.gx3[i] * pml.gy3[j] * dz[i,j] + pml.gx2[i] * pml.gy2[j] * 0.5 * (hy[i,j] - hy[i-1,j] - hx[i,j] + hx[i,j-1])
    # put a sinusoidal source at a point that is offset five cells
    # from the center of the problem space in each direction
    dz[nx//2-5,ny//2-5] = sinusoidal(t, 0.01, 1500e6)


@nb.jit(nopython=True, parallel=True, fastmath=True)
def efield(nx: int, ny: int, naz: np.ndarray, dz: np.ndarray, ez: np.ndarray) -> None:
    """ calculate the Ez field from Dz """
    for i in nb.prange(0, nx):
        for j in nb.prange(0, ny):
            ez[i,j] = naz[i,j] * dz[i,j]


@nb.jit(nopython=True, parallel=True, fastmath=True)
def hfield(nx: int, ny: int, pml: pmlayer, ez: np.ndarray, ihx: np.ndarray, ihy: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> None:
    """ calculate the Hx and Hy field """
    for i in nb.prange(0, nx-1):
        for j in nb.prange(0, ny-1):
            ihx[i,j] += ez[i,j] - ez[i,j+1]
            ihy[i,j] += ez[i,j] - ez[i+1,j]
            hx[i,j] = pml.fy3[j] * hx[i,j] + pml.fy2[j] * (0.5 * ez[i,j] - 0.5 * ez[i,j+1] + pml.fx1[i] * ihx[i,j])
            hy[i,j] = pml.fx3[i] * hy[i,j] - pml.fx2[i] * (0.5 * ez[i,j] - 0.5 * ez[i+1,j] + pml.fy1[j] * ihy[i,j])


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

    nx: int = 100  # number of grid points
    ny: int = 100  # number of grid points

    ns: int = 250  # number of time steps

    dz = np.zeros((nx, ny), dtype=np.float64)
    ez = np.zeros((nx, ny), dtype=np.float64)
    hx = np.zeros((nx, ny), dtype=np.float64)
    hy = np.zeros((nx, ny), dtype=np.float64)

    ihx = np.zeros((nx, ny), dtype=np.float64)
    ihy = np.zeros((nx, ny), dtype=np.float64)

    naz = np.ones((nx, ny), dtype=np.float64)

    pml = pmlayer(
        fx1 = np.full(nx, 0.0, dtype=np.float64),
        fx2 = np.full(nx, 1.0, dtype=np.float64),
        fx3 = np.full(nx, 1.0, dtype=np.float64),
        fy1 = np.full(ny, 0.0, dtype=np.float64),
        fy2 = np.full(ny, 1.0, dtype=np.float64),
        fy3 = np.full(ny, 1.0, dtype=np.float64),
        gx2 = np.full(nx, 1.0, dtype=np.float64),
        gx3 = np.full(nx, 1.0, dtype=np.float64),
        gy2 = np.full(ny, 1.0, dtype=np.float64),
        gy3 = np.full(ny, 1.0, dtype=np.float64),
    )

    npml: int = 8  # pml thickness
    pmlparam(nx, ny, npml, pml)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)

    ezdata = [ez]
    for t in np.arange(1, ns+1).astype(np.int32):
        dfield(t, nx, ny, pml, dz, hx, hy)
        efield(nx, ny, naz, dz, ez)
        hfield(nx, ny, pml, ez, ihx, ihy, hx, hy)
        ezdata.append(ez.copy())

    ez = np.array(ezdata)

    # define the meta data for the movie
    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Simulation in the transverse magnetic (TM) mode with PML"}
    writer = fwriter(fps=15, codec='h264', bitrate=2000, metadata=data)

    # draw an empty plot, but preset the plot x-, y- and z- limits
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    fig.suptitle(r"FDTD simulation of a sinusoidal in free space with PML")
    yv, xv = np.meshgrid(range(ny), range(nx)); levels = [0.50,1.50]
    pmlmsk = ((xv < npml)|(xv >= nx-npml)|(yv < npml)|(yv >= ny-npml))
    ax.plot_surface(xv, yv, ez[0], rstride=1, cstride=1, cmap="gray", lw=10/nx)
    ax.contourf(xv, yv, pmlmsk, levels, offset=0, colors="k", alpha=0.40)
    ax.text2D(0.1, 0.7, "", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(False); ax.view_init(elev=20.0, azim=45)
    plt.subplots_adjust(bottom=0.1, hspace=0.45)

    with writer.saving(fig, "fd2d_surface_3_2.mp4", 300):
        for t in np.arange(1, ns+1).astype(np.int32):
            ax.clear()
            ax.plot_surface(xv, yv, ez[t], rstride=1, cstride=1, cmap="gray", lw=10/nx)
            ax.contourf(xv, yv, pmlmsk, levels, offset=0, colors="k", alpha=0.40)
            ax.text2D(0.1, 0.7, rf"$T$ = {t}", transform=ax.transAxes)
            ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
            ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
            writer.grab_frame()

    plt.close(fig)

    # draw an empty plot, but preset the plot x-, y- and z- limits
    fig, ax = plt.subplots(gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal in free space with PML")
    yv, xv = np.meshgrid(range(ny), range(nx)); ezmax = np.abs(ez).max()
    levels = np.linspace(-ezmax, ezmax, int(2/0.04))
    pmlmsk = ((xv < npml)|(xv >= nx-npml)|(yv < npml)|(yv >= ny-npml))
    ax.contour(xv, yv, ez[0], levels, cmap="gray", alpha=1.0, linewidths=1.5)
    ax.contourf(xv, yv, pmlmsk, levels=[0.50,1.50], colors="k", alpha=0.40)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)

    with writer.saving(fig, "fd2d_contour_3_2.mp4", 300):
        for t in np.arange(1, ns+1).astype(np.int32):
            ax.clear()
            ax.contour(xv, yv, ez[t], levels, cmap="gray", alpha=1.0, linewidths=1.5)
            ax.contourf(xv, yv, pmlmsk, levels=[0.50,1.50], colors="k", alpha=0.40)
            ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
            ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
            writer.grab_frame()

    plt.close(fig)


if __name__ == "__main__":
    main()
