#!/usr/bin/env python
# File: fd1d_1_1.py
# Name: D.Saravanan
# Date: 11/10/2021

""" Simulation in free space """
# FDTD simulation of a pulse in free space.
# The pulse originated in the center and travels outward.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

plt.style.use("classic")
plt.style.use("../../pyplot.mplstyle")

def animate(Ex: np.ndarray, Hy: np.ndarray) -> None:
    """Data animation with Matplotlib"""

    fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a pulse in free space")
    (line1,) = ax1.plot(Ex[0], "k", lw=1)
    (line2,) = ax2.plot(Hy[0], "k", lw=1)
    time_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes)
    ax1.set(xlim=(0, 200), ylim=(-1.2, 1.2), ylabel=r"E$_x$")
    ax1.set(xticks=range(0, 220, 20), yticks=np.arange(-1, 1.2, 1))
    ax2.set(xlim=(0, 200), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"H$_y$")
    ax2.set(xticks=range(0, 220, 20), yticks=np.arange(-1, 1.2, 1))

    data = {"title": "Simulation in free space"}
    writer = animation.FFMpegWriter(fps=15, metadata=data, bitrate=1800)

    with writer.saving(fig, "fd1d_1_1.mp4", dpi=300):
        for t in range(1, 1000):
            time_text.set_text(f"T = {t}")
            line1.set_ydata(Ex[t])
            line2.set_ydata(Hy[t])
            writer.grab_frame()


def gaussian(t: int, t0: int = 40, sigma: float = 12) -> float:
    """
    Gaussian pulse source

    :param int t: an integer counter that serves as the temporal index
    :param int t0: time step at which gaussian function is maximum, default 40
    :param float sigma: width of the gaussian pulse, default 12

    :return: gaussian pulse
    :rtype: float

    """

    return np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def simulate(ke: int, ex: np.ndarray, hy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference time-domain method

    :param int ke: number of electric and magnetic field nodes
    :param np.ndarray ex: electric field oriented in the x direction
    :param np.ndarray hy: magnetic field oriented in the y direction

    :return: ex, hy: electric and magnetic field
    :rtype: tuple[np.ndarray, np.ndarray]

    """

    kc: int = ke//2
    nsteps: int = 1000

    Ex = np.empty((0, ex.shape[0]))
    Hy = np.empty((0, hy.shape[0]))

    # FDTD simulation loop
    for t in range(1, nsteps + 1):

        # calculate the Ex field
        ex[1:ke] = ex[1:ke] + 0.5 * (hy[0:ke-1] - hy[1:ke])

        # put a Gaussian pulse in the middle
        ex[kc] = gaussian(t, 40, 12)

        # calculate the Hy field
        hy[0:ke-1] = hy[0:ke-1] + 0.5 * (ex[0:ke-1] - ex[1:ke])

        Ex = np.vstack((Ex, ex))
        Hy = np.vstack((Hy, hy))

    return Ex, Hy

def main():
    """Main function"""

    ke = 201
    ex = np.zeros(ke, dtype=np.float64)
    hy = np.zeros(ke, dtype=np.float64)

    Ex, Hy = simulate(ke, ex, hy)
    animate(Ex, Hy)


if __name__ == "__main__":
    main()
