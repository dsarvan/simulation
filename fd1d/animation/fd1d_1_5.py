#!/usr/bin/env python
# File: fdtd15.py
# Name: D.Saravanan
# Date: 29/10/2021

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import numpy
from numpy import pi, sin
from matplotlib import pyplot
from matplotlib import animation
from numpy.typing import NDArray

pyplot.style.use("classic")
pyplot.style.use("pyplot.mplstyle")

type F64Array = NDArray[numpy.float64]


def sinusoidal(ts: int, ds: float, freq: float):
    dt: float = ds/6e8  # time step (s)
    return sin(2*pi*freq*dt*ts)


def dielectric(nx: int, dt: float, epsr: float, sigma: float):
    ca = numpy.full(nx, 1.0, numpy.float64)
    cb = numpy.full(nx, 0.5, numpy.float64)
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    epsf: float = dt*sigma/(2*eps0*epsr)
    ca[nx//2:] = (1 - epsf)/(1 + epsf)
    cb[nx//2:] = 0.5/(epsr*(1 + epsf))
    return ca, cb


def main():

    nx: int = 512  # number of grid points
    ns: int = 740  # number of time steps

    ex = numpy.zeros(nx, numpy.float64)
    hy = numpy.zeros(nx, numpy.float64)

    bc = numpy.zeros(4, numpy.float64)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    epsr: float = 4.0  # relative permittivity
    sigma: float = 0.04  # conductivity (S/m)
    ca, cb = dielectric(nx, dt, epsr, sigma)

    # define the meta data for the movie
    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Simulation of a sinusoidal hitting lossy dielectric medium"}
    args = ['-vcodec','libx264','-crf','17','-pix_fmt','yuv420p','-tune','animation']
    writer = fwriter(fps=30, codec='h264', metadata=data, extra_args=args)

    # draw an empty plot, but preset the plot x- and y- limits
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=True)
    fig.suptitle(r"FDTD simulation of a sinusoidal striking lossy dielectric material")
    medium = numpy.where(0.5/cb-1)[0] if epsr > 1 else (0.5/cb-1)
    axline, = axs.plot(range(nx), ex, color="k", linewidth=1.0)
    axs.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=range(0, nx+1, int(numpy.ceil(nx/500)*25)))
    axs.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axtime = axs.text(0.02, 0.90, "", transform=axs.transAxes)
    axepsr = axs.text(0.90, 0.90, "", transform=axs.transAxes)
    axsigm = axs.text(0.85, 0.80, "", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)

    with writer.saving(fig, "fdtd15.mp4", 300):
        for ts in map(int,range(1,ns+1)):
            # calculate the Ex field
            ex[1:nx] = ca[1:nx] * ex[1:nx] + cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
            # put a sinusoidal wave at the low end
            ex[1] += sinusoidal(ts, 0.01, 700e6)
            # absorbing boundary conditions
            ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
            ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]
            # calculate the Hy field
            hy[0:nx-1] += 0.5 * (ex[0:nx-1] - ex[1:nx])

            axline.set_ydata(ex)
            axtime.set_text(rf"$T$ = {ts}")
            axepsr.set_text(rf"$\epsilon_r$ = {epsr}")
            axsigm.set_text(rf"$\sigma$ = {sigma} $S/m$")
            writer.grab_frame()

    pyplot.close(fig)


if __name__ == "__main__":
    main()
