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


def main():

    nx: int = 512  # number of grid points
    ns: int = 650  # number of time steps

    ex = np.zeros(nx, dtype=np.float64)
    hy = np.zeros(nx, dtype=np.float64)

    # define the meta data for the movie
    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Simulation in free space"}
    writer = fwriter(fps=15, metadata=data)

    # draw an empty plot, but preset the plot x- and y- limits
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a pulse in free space")
    axline, = ax.plot(ex, color="black", linewidth=1)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=range(0, nx+1, round(nx//10,-1)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axtime = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)

    with writer.saving(fig, "fd1d_1_1.mp4", 300):
        for t in np.arange(1, ns+1).astype(np.int32):
            # calculate the Ex field
            ex[1:nx] = ex[1:nx] + 0.5 * (hy[0:nx-1] - hy[1:nx])
            # put a Gaussian pulse in the middle
            ex[nx//2] = gaussian(t, 40, 12)
            # calculate the Hy field
            hy[0:nx-1] = hy[0:nx-1] + 0.5 * (ex[0:nx-1] - ex[1:nx])

            axline.set_ydata(ex)
            axtime.set_text(rf"$T$ = {t}")
            writer.grab_frame()


if __name__ == "__main__":
    main()
