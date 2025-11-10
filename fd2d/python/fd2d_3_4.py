#!/usr/bin/env python
# File: fd2d_3_4.py
# Name: D.Saravanan
# Date: 20/01/2022

""" Simulation of a plane wave pulse striking a dielectric medium in the transverse
magnetic (TM) mode with PML and implements the discrete Fourier transform analysis """

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from numpy import pi, exp, hypot, sqrt, sin, cos, arctan2
from collections import namedtuple
import numba as nb
import numpy as np

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def surfaceplot(ns: int, nx: int, ny: int, npml: int, epsr: float, naz: np.ndarray, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    fig.suptitle(r"FDTD simulation of plane wave striking dielectric material")
    yv, xv = np.meshgrid(range(ny), range(nx)); ezmax = np.abs(ez).max()
    medium = np.stack([1.0/naz-1.0]*1, axis=2); levels = [0.50,1.50]
    pmlmsk = ((xv < npml)|(xv >= nx-npml)|(yv < npml)|(yv >= ny-npml))
    ax.plot_surface(xv, yv, ez, rstride=1, cstride=1, cmap="gray", alpha=0.5, lw=10/nx)
    ax.voxels(medium, color="y", edgecolor="k", shade=True, alpha=0.5, linewidths=1/nx)
    ax.contourf(xv, yv, pmlmsk, levels, offset=0, colors="k", alpha=0.40)
    ax.text2D(0.1, 0.7, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(False); ax.view_init(elev=20.0, azim=45)
    plt.savefig("fd2d_surface_3_4.png", dpi=100)


def contourplot(ns: int, nx: int, ny: int, npml: int, epsr: float, naz: np.ndarray, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of plane wave striking dielectric material")
    yv, xv = np.meshgrid(range(ny), range(nx)); ezmax = np.abs(ez).max()
    medium = 1.0/naz-1.0; levels = np.linspace(-ezmax, ezmax, int(2/0.04))
    pmlmsk = ((xv < npml)|(xv >= nx-npml)|(yv < npml)|(yv >= ny-npml))
    ax.contour(xv, yv, ez, levels, cmap="gray", alpha=1.0, linewidths=1.5)
    ax.contourf(xv, yv, medium, [0.001,medium.max()], colors="y", alpha=0.7)
    ax.contourf(xv, yv, pmlmsk, levels=[0.50,1.50], colors="k", alpha=0.40)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd2d_contour_3_4.png", dpi=100)


def amplitudeplot(ns: int, ny: int, rgrid: int, epsr: float, sigma: float, amp: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"The discrete Fourier transform with plane wave as its source")
    ax.plot(range(-ny//2, ny//2), amp, color="k", linewidth=1.0)
    ax.set(xlim=(-rgrid-1, rgrid+1), ylim=(0.0, 1.0))
    ax.set(xticks=[-rgrid, -rgrid//2, 0, rgrid//2, rgrid])
    ax.set(xlabel=r"$y\;(cm)$", ylabel=r"$Amplitude$")
    ax.text(0.03, 0.90, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.text(0.80, 0.90, rf"$\epsilon_r$ = {epsr}", transform=ax.transAxes)
    ax.text(0.75, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd2d_amplitude_3_4.png", dpi=100)


medium = namedtuple('medium', (
    'naz', 'nbz',
))


ftrans = namedtuple('ftrans', (
    'r_pt', 'i_pt',
    'r_in', 'i_in',
))


pmlayer = namedtuple('pmlayer', (
    'fx1', 'fx2', 'fx3',
    'fy1', 'fy2', 'fy3',
    'gx2', 'gx3',
    'gy2', 'gy3',
))


@nb.jit(nopython=True, fastmath=True)
def gaussian(t: int, t0: int, sigma: float) -> float:
    return np.exp(-0.5*((t - t0)/sigma)**2)


@nb.jit(nopython=True, fastmath=True)
def fourier(t: int, nf: int, nx: int, ny: int, dt: float, freq: np.ndarray, ezi: np.ndarray, ez: np.ndarray, ft: ftrans) -> None:
    for n in range(0, nf):
        # calculate the Fourier transform of input source
        ft.r_in[n] += cos(2*pi*freq[n]*dt*t) * ezi[6]
        ft.i_in[n] -= sin(2*pi*freq[n]*dt*t) * ezi[6]
        for i in range(0, nx):
            for j in range(0, ny):
                # calculate the Fourier transform of Ez field
                ft.r_pt[n,i,j] += cos(2*pi*freq[n]*dt*t) * ez[i,j]
                ft.i_pt[n,i,j] -= sin(2*pi*freq[n]*dt*t) * ez[i,j]


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
    ezi[3] = gaussian(t, 20, 8.0)


@nb.jit(nopython=True, parallel=True, fastmath=True)
def inctdz(nx: int, ny: int, npml: int, hxi: np.ndarray, dz: np.ndarray) -> None:
    """ incident Dz values """
    for i in nb.prange(npml-1, nx-npml+1):
        dz[i,npml-1] += 0.5 * hxi[npml-2]
        dz[i,ny-npml] -= 0.5 * hxi[ny-npml]


@nb.jit(nopython=True, parallel=True, fastmath=True)
def efield(nx: int, ny: int, md: medium, dz: np.ndarray, iz: np.ndarray, ez: np.ndarray) -> None:
    """ calculate the Ez field from Dz """
    for i in nb.prange(0, nx):
        for j in nb.prange(0, ny):
            ez[i,j] = md.naz[i,j] * (dz[i,j] - iz[i,j])
            iz[i,j] += md.nbz[i,j] * ez[i,j]


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


@nb.jit(nopython=True, fastmath=True)
def dielectric(nx: int, ny: int, npml: int, rgrid: int, dt: float, epsr: float, sigma: float) -> medium:
    md = medium(
        naz = np.full((nx, ny), 1.0, dtype=np.float64),
        nbz = np.full((nx, ny), 0.0, dtype=np.float64),
    )
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    for i in range(npml, nx-npml):
        for j in range(npml, ny-npml):
            epsn: float = 1.0
            cond: float = 0.0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    x: float = nx/2-1-i+m/3
                    y: float = ny/2-1-j+n/3
                    d: float = sqrt(x**2 + y**2)
                    if d <= rgrid:
                        epsn += (epsr - 1)/9
                        cond += sigma/9
            md.naz[i,j] = 1/(epsn + cond*dt/eps0)
            md.nbz[i,j] = cond*dt/eps0
    return md


def pmlparam(nx: int, ny: int, npml: int, pml: pmlayer) -> None:
    """ calculate the two-dimensional perfectly matched layer (PML) parameters """
    for n in range(npml):
        xm: float = 0.33*((npml-n)/npml)**3
        xn: float = 0.33*((npml-n-0.5)/npml)**3
        pml.fx1[n] = pml.fx1[nx-2-n] = pml.fy1[n] = pml.fy1[ny-2-n] = xn
        pml.fx2[n] = pml.fx2[nx-2-n] = pml.fy2[n] = pml.fy2[ny-2-n] = 1/(1+xn)
        pml.gx2[n] = pml.gx2[nx-1-n] = pml.gy2[n] = pml.gy2[ny-1-n] = 1/(1+xm)
        pml.fx3[n] = pml.fx3[nx-2-n] = pml.fy3[n] = pml.fy3[ny-2-n] = (1-xn)/(1+xn)
        pml.gx3[n] = pml.gx3[nx-1-n] = pml.gy3[n] = pml.gy3[ny-1-n] = (1-xm)/(1+xm)


def main():

    nx: int = 100  # number of grid points
    ny: int = 100  # number of grid points

    ns: int = 120  # number of time steps

    ezi = np.zeros(ny, dtype=np.float64)
    hxi = np.zeros(ny, dtype=np.float64)

    dz = np.zeros((nx, ny), dtype=np.float64)
    ez = np.zeros((nx, ny), dtype=np.float64)
    iz = np.zeros((nx, ny), dtype=np.float64)
    hx = np.zeros((nx, ny), dtype=np.float64)
    hy = np.zeros((nx, ny), dtype=np.float64)

    ihx = np.zeros((nx, ny), dtype=np.float64)
    ihy = np.zeros((nx, ny), dtype=np.float64)

    bc = np.zeros(4, dtype=np.float64)

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
    epsr: float = 30.0  # relative permittivity
    sigma: float = 0.30  # conductivity (S/m)
    radius: float = 0.15  # cylinder radius (m)
    rgrid: int = int(radius/ds-1)  # radius in FDTD grid cell units
    md: medium = dielectric(nx, ny, npml, rgrid, dt, epsr, sigma)

    # frequency 50 MHz, 300 MHz, 700 MHz
    freq = np.array((50e6, 300e6, 700e6), dtype=np.float64)
    nf: int = len(freq)  # number of frequencies

    ft = ftrans(
        r_pt = np.zeros((nf, nx, ny), dtype=np.float64),
        i_pt = np.zeros((nf, nx, ny), dtype=np.float64),
        r_in = np.zeros(nf, dtype=np.float64),
        i_in = np.zeros(nf, dtype=np.float64),
    )

    amplt = np.zeros((nf, ny), dtype=np.float64)
    phase = np.zeros((nf, ny), dtype=np.float64)

    for t in np.arange(1, ns+1).astype(np.int32):
        ezinct(ny, ezi, hxi, bc)
        dfield(t, nx, ny, pml, ezi, dz, hx, hy)
        inctdz(nx, ny, npml, hxi, dz)
        efield(nx, ny, md, dz, iz, ez)
        fourier(t, nf, nx, ny, dt, freq, ezi, ez, ft)
        hxinct(ny, ezi, hxi)
        hfield(nx, ny, pml, ez, ihx, ihy, hx, hy)
        incthx(nx, ny, npml, ezi, hx)
        incthy(nx, ny, npml, ezi, hy)

    # calculate the amplitude and phase at each frequency
    for n in range(0, nf):
        for j in range(npml-1, ny-npml+1):
            m = (n,j); k = (n,nx//2-1,j)
            amplt[m] = 1/hypot(ft.r_in[n],ft.i_in[n]) * hypot(ft.r_pt[k],ft.i_pt[k])
            phase[m] = arctan2(ft.i_pt[k],ft.r_pt[k]) - arctan2(ft.i_in[n],ft.r_in[n])

    surfaceplot(ns, nx, ny, npml, epsr, md.naz, ez)
    contourplot(ns, nx, ny, npml, epsr, md.naz, ez)
    amplitudeplot(ns, ny, rgrid, epsr, sigma, amplt[2])


if __name__ == "__main__":
    main()
