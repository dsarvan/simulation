#!/usr/bin/env python
# File: fd0d_1_1.py
# Name: D.Saravanan
# Date: 11/10/2021

""" Simulation of a pulse in free space """

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def gaussian(t: int, t0: int, sigma: float) -> float:
    return np.exp(-0.5 * ((t - t0)/sigma)**2)


def field(t: int, nx: int, ex: np.ndarray, hy: np.ndarray):
    # calculate the Hy field
    hy[0:nx-1] = hy[0:nx-1] + 0.5 * (ex[0:nx-1] - ex[1:nx])
    # calculate the Ex field
    ex[1:nx] = ex[1:nx] + 0.5 * (hy[0:nx-1] - hy[1:nx])
    # put a Gaussian pulse in the middle
    ex[nx//2] = gaussian(t, 40, 12)


def main():

    nx: int = 201
    ns: int = 300

    ex = np.zeros(nx, dtype=np.float64)
    hy = np.zeros(nx, dtype=np.float64)

    # define the meta data for the movie
    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Simulation in free space"}
    writer = fwriter(fps=15, metadata=data)

    # draw an empty plot, but preset the plot x- and y- limits
    fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a pulse in free space")
    line1, = ax1.plot(ex, "k", lw=1)
    time_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes)
    ax1.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
    ax1.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
    line2, = ax2.plot(hy, "k", lw=1)
    ax2.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"$H_y$")
    ax2.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))

    with writer.saving(fig, "fd1d_1_1.mp4", 300):
        for t in range(1, ns+1):
            field(t, nx, ex, hy)

            line1.set_ydata(ex)
            time_text.set_text(f"T = {t}")
            line2.set_ydata(hy)
            writer.grab_frame()


if __name__ == "__main__":
    main()
