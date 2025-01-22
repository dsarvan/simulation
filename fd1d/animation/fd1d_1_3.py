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


def field(t: int, nx: int, cb: np.ndarray, ex: np.ndarray, hy: np.ndarray, bc: np.ndarray):
	# calculate the Hy field
	hy[0:nx-1] = hy[0:nx-1] + 0.5 * (ex[0:nx-1] - ex[1:nx])
	# calculate the Ex field
	ex[1:nx] = ex[1:nx] + cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
	# put a Gaussian pulse at the low end
	ex[1] = ex[1] + gaussian(t, 40, 12)
	# absorbing boundary conditions
	ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
	ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]


def dielectric(nx: int, epsr: float = 1) -> np.ndarray:
	cb = 0.5 * np.ones(nx, dtype=np.float64)
	cb[nx//2:] = 0.5/epsr
	return cb


def main():

	nx: int = 201
	ns: int = 1800

	ex = np.zeros(nx, dtype=np.float64)
	hy = np.zeros(nx, dtype=np.float64)

	bc = np.zeros(4, dtype=np.float64)

	epsr: float = 4 # relative permittivity
	cb: np.ndarray = dielectric(nx, epsr)

	# define the meta data for the movie
	fwriter = animation.writers["ffmpeg"]
	data = {"title": "Simulation of a pulse hitting dielectric medium"}
	writer = fwriter(fps=15, metadata=data)

	# draw an empty plot, but preset the plot x- and y- limits
	fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.2})
	fig.suptitle(r"FDTD simulation of a pulse striking dielectric material")
	medium = (0.5/cb - 1)/(epsr - 1) if epsr > 1 else (0.5/cb - 1)
	medium[medium==0] = -1.5
	line1, = ax1.plot(ex, "k", lw=1)
	ax1.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
	time_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes)
	epsr_txt1 = ax1.text(0.80, 0.80, "", transform=ax1.transAxes)
	ax1.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
	ax1.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
	line2, = ax2.plot(hy, "k", lw=1)
	ax2.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
	epsr_txt2 = ax2.text(0.80, 0.80, "", transform=ax2.transAxes)
	ax2.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"$H_y$")
	ax2.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))

	with writer.saving(fig, "fd1d_1_3.mp4", 300):
		for t in range(1, ns+1):
			field(t, nx, cb, ex, hy, bc)

			line1.set_ydata(ex)
			time_text.set_text(f"T = {t}")
			epsr_txt1.set_text(f"epsr = {epsr}")
			line2.set_ydata(hy)
			epsr_txt2.set_text(f"epsr = {epsr}")
			writer.grab_frame()


if __name__ == "__main__":
	main()
