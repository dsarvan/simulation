#!/usr/bin/env python
# File: fd1d_2_2.py
# Name: D.Saravanan
# Date: 07/12/2021

""" Simulation of a pulse striking a dielectric medium and implements
the discrete Fourier transform with a Gaussian pulse as its source """

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp, hypot, sin, cos, arctan2
from collections import namedtuple

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


medium = namedtuple('medium', (
    'nax', 'nbx',
))


ftrans = namedtuple('ftrans', (
    'r_pt', 'i_pt',
    'r_in', 'i_in',
))


def gaussian(t: int, t0: int, sigma: float) -> float:
    return np.exp(-0.5 * ((t - t0)/sigma)**2)


def fourier(t: int, nf: int, nx: int, dt: float, freq: np.ndarray, ex: np.ndarray, ft: ftrans) -> None:
    # calculate the Fourier transform of Ex field
    ft.r_pt[0:nf,0:nx] += cos(2*pi*freq[0:nf]*dt*t) * ex[0:nx]
    ft.i_pt[0:nf,0:nx] -= sin(2*pi*freq[0:nf]*dt*t) * ex[0:nx]
    if t < nx//2:
        # calculate the Fourier transform of input source
        ft.r_in[0:nf] += cos(2*pi*freq[0:nf]*dt*t) * ex[10]
        ft.i_in[0:nf] -= sin(2*pi*freq[0:nf]*dt*t) * ex[10]


def dxfield(t: int, nx: int, dx: np.ndarray, hy: np.ndarray) -> None:
    # calculate the electric flux density Dx
    dx[1:nx] = dx[1:nx] + 0.5 * (hy[0:nx-1] - hy[1:nx])
    # put a Gaussian pulse at the low end
    dx[1] = dx[1] + gaussian(t, 50, 10)


def exfield(nx: int, md: medium, dx: np.ndarray, ix: np.ndarray, ex: np.ndarray) -> None:
    # calculate the Ex field from Dx
    ex[1:nx] = md.nax[1:nx] * (dx[1:nx] - ix[1:nx])
    ix[1:nx] = ix[1:nx] + md.nbx[1:nx] * ex[1:nx]


def hyfield(nx: int, ex: np.ndarray, hy: np.ndarray, bc: np.ndarray) -> None:
    # absorbing boundary conditions
    ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
    ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
    # calculate the Hy field
    hy[0:nx-1] = hy[0:nx-1] + 0.5 * (ex[0:nx-1] - ex[1:nx])


def dielectric(nx: int, dt: float, epsr: float, sigma: float) -> medium:
    md = medium(
        nax = np.ones(nx, dtype=np.float64),
        nbx = np.zeros(nx, dtype=np.float64),
    )
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    md.nax[nx//2:] = 1/(epsr + (sigma * dt/eps0))
    md.nbx[nx//2:] = sigma * dt/eps0
    return md


def main():

    nx: int = 512  # number of grid points
    ns: int = 740  # number of time steps

    dx = np.zeros(nx, dtype=np.float64)
    ex = np.zeros(nx, dtype=np.float64)
    ix = np.zeros(nx, dtype=np.float64)
    hy = np.zeros(nx, dtype=np.float64)

    bc = np.zeros(4, dtype=np.float64)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    epsr: float = 4  # relative permittivity
    sigma: float = 0  # conductivity (S/m)
    md: medium = dielectric(nx, dt, epsr, sigma)

    # frequency 100 MHz, 200 MHz, 500 MHz
    freq = np.array(([100e6], [200e6], [500e6]), dtype=np.float64)
    nf: int = len(freq)  # number of frequencies

    ft = ftrans(
        r_pt = np.zeros((nf, nx), dtype=np.float64),
        i_pt = np.zeros((nf, nx), dtype=np.float64),
        r_in = np.zeros((nf, 1), dtype=np.float64),
        i_in = np.zeros((nf, 1), dtype=np.float64),
    )

    amplt = np.zeros((nf, nx), dtype=np.float64)
    phase = np.zeros((nf, nx), dtype=np.float64)

    # define the meta data for the movie
    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Calculating the frequency domain output"}
    writer = fwriter(fps=15, codec='h264', bitrate=2000, metadata=data)

    # draw an empty plot, but preset the plot x- and y- limits
    fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.4})
    fig.suptitle(r"The discrete Fourier transform with pulse as its source")
    medium = (1 - md.nax)/(1 - md.nax[-1])*1e3 if epsr > 1 else (1 - md.nax)
    medium[medium==0] = -1e3
    axline1, = ax1.plot(ex, color="black", linewidth=1)
    ax1.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    ax1.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax1.set(xticks=range(0, nx+1, round(nx//10,-1)))
    ax1.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axtime1 = ax1.text(0.02, 0.90, "", transform=ax1.transAxes)
    axepsr1 = ax1.text(0.90, 0.90, "", transform=ax1.transAxes)
    axsigm1 = ax1.text(0.85, 0.80, "", transform=ax1.transAxes)
    axline2, = ax2.plot(amplt[2], color="black", linewidth=1)
    ax2.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    ax2.set(xlim=(0, nx-1), ylim=(-0.2, 2.2))
    ax2.set(xticks=range(0, nx+1, round(nx//10,-1)))
    ax2.set(xlabel=r"$z\;(cm)$", ylabel=r"$Amp\;(V)$")
    axtime2 = ax2.text(0.02, 0.90, "", transform=ax2.transAxes)
    axepsr2 = ax2.text(0.90, 0.90, "", transform=ax2.transAxes)
    axsigm2 = ax2.text(0.85, 0.80, "", transform=ax2.transAxes)
    plt.subplots_adjust(bottom=0.1, hspace=0.45)

    with writer.saving(fig, "fd1d_2_2.mp4", 300):
        for t in np.arange(1, ns+1).astype(np.int32):
            dxfield(t, nx, dx, hy)
            exfield(nx, md, dx, ix, ex)
            fourier(t, nf, nx, dt, freq, ex, ft)
            hyfield(nx, ex, hy, bc)

            # calculate the amplitude and phase at each frequency
            amplt = 1/hypot(ft.r_in,ft.i_in) * hypot(ft.r_pt,ft.i_pt)
            phase = arctan2(ft.i_pt,ft.r_pt) - arctan2(ft.i_in,ft.r_in)

            axline1.set_ydata(ex)
            axtime1.set_text(rf"$T$ = {t}")
            axepsr1.set_text(rf"$\epsilon_r$ = {epsr}")
            axsigm1.set_text(rf"$\sigma$ = {sigma} $S/m$")

            axline2.set_ydata(amplt[2])
            axtime2.set_text(rf"$T$ = {t}")
            axepsr2.set_text(rf"$\epsilon_r$ = {epsr}")
            axsigm2.set_text(rf"$\sigma$ = {sigma} $S/m$")

            writer.grab_frame()


if __name__ == "__main__":
    main()
