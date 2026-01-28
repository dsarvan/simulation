#!/usr/bin/env python
# File: fdtd13.py
# Name: D.Saravanan
# Date: 21/10/2021

""" Simulation of a pulse hitting a dielectric medium """

import numpy
from matplotlib import pyplot
from matplotlib import animation
from numpy.typing import NDArray

pyplot.style.use("classic")
pyplot.style.use("pyplot.mplstyle")

type F64Array = NDArray[numpy.float64]


def gaussian(ts: int, t0: int, sigma: float):
    return numpy.exp(-0.5*((ts - t0)/sigma)**2)


def dielectric(nx: int, epsr: float):
    cb = numpy.full(nx, 0.5, numpy.float64)
    cb[nx//2:] = 0.5/epsr
    return cb


def main():

    nx: int = 512  # number of grid points
    ns: int = 740  # number of time steps

    ex = numpy.zeros(nx, numpy.float64)
    hy = numpy.zeros(nx, numpy.float64)

    bc = numpy.zeros(4, numpy.float64)

    epsr: float = 4.0  # relative permittivity
    cb: F64Array = dielectric(nx, epsr)

    # define the meta data for the movie
    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Simulation of a pulse hitting dielectric medium"}
    args = ['-vcodec','libx264','-crf','17','-pix_fmt','yuv420p','-tune','animation']
    writer = fwriter(fps=30, codec='h264', metadata=data, extra_args=args)

    # draw an empty plot, but preset the plot x- and y- limits
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=True)
    fig.suptitle(r"FDTD simulation of a pulse striking dielectric material")
    medium = numpy.where(0.5/cb-1)[0] if epsr > 1 else (0.5/cb-1)
    axline, = axs.plot(range(nx), ex, color="k", linewidth=1.0)
    axs.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=range(0, nx+1, int(numpy.ceil(nx/500)*25)))
    axs.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axtime = axs.text(0.02, 0.90, "", transform=axs.transAxes)
    axepsr = axs.text(0.90, 0.90, "", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)

    with writer.saving(fig, "fdtd13.mp4", 300):
        for ts in map(int,range(1,ns+1)):
            # calculate the Ex field
            ex[1:nx] += cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
            # put a Gaussian pulse at the low end
            ex[1] += gaussian(ts, 40, 12.0)
            # absorbing boundary conditions
            ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
            ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
            # calculate the Hy field
            hy[0:nx-1] += 0.5 * (ex[0:nx-1] - ex[1:nx])

            axline.set_ydata(ex)
            axtime.set_text(rf"$T$ = {ts}")
            axepsr.set_text(rf"$\epsilon_r$ = {epsr}")
            writer.grab_frame()

    pyplot.close(fig)


if __name__ == "__main__":
    main()
