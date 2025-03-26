#!/usr/bin/env python
# File: fd1d_1_3.py
# Name: D.Saravanan
# Date: 21/10/2021

""" Simulation of a pulse hitting a dielectric medium """

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def gaussian(t: int, t0: int, sigma: float) -> float:
    return np.exp(-0.5 * ((t - t0)/sigma)**2)


def dielectric(nx: int, epsr: float) -> np.ndarray:
    cb = 0.5 * np.ones(nx, dtype=np.float64)
    cb[nx//2:] = 0.5/epsr
    return cb


def main():

    nx: int = 512  # number of grid points
    ns: int = 740  # number of time steps

    ex = np.zeros(nx, dtype=np.float64)
    hy = np.zeros(nx, dtype=np.float64)

    bc = np.zeros(4, dtype=np.float64)

    epsr: float = 4  # relative permittivity
    cb: np.ndarray = dielectric(nx, epsr)

    # define the meta data for the movie
    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Simulation of a pulse hitting dielectric medium"}
    writer = fwriter(fps=15, codec='h264', bitrate=2000, metadata=data)

    # draw an empty plot, but preset the plot x- and y- limits
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a pulse striking dielectric material")
    medium = (0.5/cb - 1)/(epsr - 1)*1e3 if epsr > 1 else (0.5/cb - 1)
    medium[medium==0] = -1e3
    axline, = ax.plot(ex, color="black", linewidth=1)
    ax.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=range(0, nx+1, round(nx//10,-1)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axtime = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    axepsr = ax.text(0.90, 0.90, "", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)

    with writer.saving(fig, "fd1d_1_3.mp4", 300):
        for t in np.arange(1, ns+1).astype(np.int32):
            # calculate the Ex field
            ex[1:nx] = ex[1:nx] + cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
            # put a Gaussian pulse at the low end
            ex[1] = ex[1] + gaussian(t, 40, 12)
            # absorbing boundary conditions
            ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
            ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
            # calculate the Hy field
            hy[0:nx-1] = hy[0:nx-1] + 0.5 * (ex[0:nx-1] - ex[1:nx])

            axline.set_ydata(ex)
            axtime.set_text(rf"$T$ = {t}")
            axepsr.set_text(rf"$\epsilon_r$ = {epsr}")
            writer.grab_frame()


if __name__ == "__main__":
    main()
