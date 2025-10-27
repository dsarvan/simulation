#!/usr/bin/env python
# File: fd1d_1_5.py
# Name: D.Saravanan
# Date: 29/10/2021

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def sinusoidal(t: int, ds: float, freq: float) -> float:
    dt: float = ds/6e8  # time step (s)
    return np.sin(2*np.pi*freq*dt*t)


def dielectric(nx: int, dt: float, epsr: float, sigma: float) -> tuple:
    ca = 1.0 + np.zeros(nx, dtype=np.float64)
    cb = 0.5 + np.zeros(nx, dtype=np.float64)
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    epsf: float = dt*sigma/(2*eps0*epsr)
    ca[nx//2:] = (1 - epsf)/(1 + epsf)
    cb[nx//2:] = 0.5/(epsr*(1 + epsf))
    return ca, cb


def main():

    nx: int = 512  # number of grid points
    ns: int = 740  # number of time steps

    ex = np.zeros(nx, dtype=np.float64)
    hy = np.zeros(nx, dtype=np.float64)

    bc = np.zeros(4, dtype=np.float64)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    epsr: float = 4.0  # relative permittivity
    sigma: float = 0.04  # conductivity (S/m)
    ca, cb = dielectric(nx, dt, epsr, sigma)

    # define the meta data for the movie
    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Simulation of a sinusoidal hitting lossy dielectric medium"}
    writer = fwriter(fps=15, codec='h264', bitrate=2000, metadata=data)

    # draw an empty plot, but preset the plot x- and y- limits
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal striking lossy dielectric material")
    medium = np.where(0.5/cb-1)[0] if epsr > 1 else (0.5/cb-1)
    axline, = ax.plot(range(nx), ex, color="k", linewidth=1.0)
    ax.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=range(0, nx+1, int(np.ceil(nx/500)*25)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axtime = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    axepsr = ax.text(0.90, 0.90, "", transform=ax.transAxes)
    axsigm = ax.text(0.85, 0.80, "", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)

    with writer.saving(fig, "fd1d_1_5.mp4", 300):
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

            axline.set_ydata(ex)
            axtime.set_text(rf"$T$ = {t}")
            axepsr.set_text(rf"$\epsilon_r$ = {epsr}")
            axsigm.set_text(rf"$\sigma$ = {sigma} $S/m$")
            writer.grab_frame()


if __name__ == "__main__":
    main()
