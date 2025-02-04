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


def sinusoidal(t: int, ddx: float, freq: float) -> float:
    dt: float = ddx/6e8  # time step
    return np.sin(2 * np.pi * freq * dt * t)


def dielectric(nx: int, dt: float, epsr: float, sigma: float) -> tuple:
    ca = 1.0 * np.ones(nx, dtype=np.float64)
    cb = 0.5 * np.ones(nx, dtype=np.float64)
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    epsf: float = dt * sigma/(2 * eps0 * epsr)
    ca[nx//2:] = (1 - epsf)/(1 + epsf)
    cb[nx//2:] = 0.5/(epsr * (1 + epsf))
    return ca, cb


def main():

    nx: int = 512
    ns: int = 3000

    ex = np.zeros(nx, dtype=np.float64)
    hy = np.zeros(nx, dtype=np.float64)

    bc = np.zeros(4, dtype=np.float64)

    ddx: float = 0.01  # cell size (m)
    dt: float = ddx/6e8  # time step
    epsr: float = 4  # relative permittivity
    sigma: float = 0.04  # conductivity (S/m)
    ca, cb = dielectric(nx, dt, epsr, sigma)

    # define the meta data for the movie
    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Simulation of a sinusoidal hitting lossy dielectric medium"}
    writer = fwriter(fps=15, metadata=data)

    # draw an empty plot, but preset the plot x- and y- limits
    fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal striking lossy dielectric material")
    medium = (0.5/cb - 1)/(epsr - 1) if epsr > 1 else (0.5/cb - 1)
    medium[medium==0] = -1e3
    line1, = ax1.plot(ex, "k", lw=1)
    ax1.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    time_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes)
    epsr_txt1 = ax1.text(0.80, 0.80, "", transform=ax1.transAxes)
    cond_txt1 = ax1.text(0.80, 0.70, "", transform=ax1.transAxes)
    ax1.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
    ax1.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
    line2, = ax2.plot(hy, "k", lw=1)
    ax2.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    epsr_txt2 = ax2.text(0.80, 0.80, "", transform=ax2.transAxes)
    cond_txt2 = ax2.text(0.80, 0.70, "", transform=ax2.transAxes)
    ax2.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"$H_y$")
    ax2.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))

    with writer.saving(fig, "fd1d_1_5.mp4", 300):
        for t in np.arange(1, ns+1).astype(np.int32):
            # calculate the Ex field
            ex[1:nx] = ca[1:nx] * ex[1:nx] + cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
            # put a sinusoidal wave at the low end
            ex[1] = ex[1] + sinusoidal(t, 0.01, 700e6)
            # absorbing boundary conditions
            ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
            ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
            # calculate the Hy field
            hy[0:nx-1] = hy[0:nx-1] + 0.5 * (ex[0:nx-1] - ex[1:nx])

            line1.set_ydata(ex)
            time_text.set_text(rf"T = {t}")
            epsr_txt1.set_text(rf"epsr = {epsr}")
            cond_txt1.set_text(rf"$\sigma$ = {sigma}")

            line2.set_ydata(hy)
            epsr_txt2.set_text(rf"epsr = {epsr}")
            cond_txt2.set_text(rf"$\sigma$ = {sigma}")

            writer.grab_frame()


if __name__ == "__main__":
    main()
