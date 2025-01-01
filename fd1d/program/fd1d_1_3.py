#!/usr/bin/env python
# File: fd1d_1_3.py
# Name: D.Saravanan
# Date: 21/10/2021

""" Simulation of a pulse hitting a dielectric medium """

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, epsr: int, cb: np.ndarray, ex: np.ndarray, hy: np.ndarray) -> None:
	fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.2})
	fig.suptitle(r"FDTD simulation of a pulse striking dielectric material")
	medium = (0.5/cb - 1)/3
	medium[medium==0] = -1.5
	ax1.plot(ex, "k", lw=1)
	ax1.plot(medium, 'k--', lw=0.75)
	ax1.text(nx/4, 0.5, f"T = {ns}", horizontalalignment="center")
	ax1.text(3*nx/4, 0.5, f"epsr = {epsr}", horizontalalignment="center")
	ax1.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
	ax1.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
	ax2.plot(hy, "k", lw=1)
	ax2.plot(medium, 'k--', lw=0.75)
	ax2.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"$H_y$")
	ax2.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
	plt.subplots_adjust(bottom=0.2, hspace=0.45)
	plt.savefig("fd1d_1_3.png")


def gaussian(t: int, t0: int, sigma: float) -> float:
	return np.exp(-0.5 * ((t - t0)/sigma)**2)


def field(t: int, nx: int, cb: np.ndarray, ex: np.ndarray, hy: np.ndarray, lb: np.ndarray, hb: np.ndarray):
	# calculate the Hy field
	hy[0:nx-1] = hy[0:nx-1] + 0.5 * (ex[0:nx-1] - ex[1:nx])
	# calculate the Ex field
	ex[1:nx] = ex[1:nx] + cb[1:nx] * (hy[0:nx-1] - hy[1:nx])
	# put a Gaussian pulse at the low end
	ex[1] = ex[1] + gaussian(t, 40, 12)
	# absorbing boundary conditions
	ex[0], lb[0], lb[1] = lb[0], lb[1], ex[1]
	ex[nx-1], hb[0], hb[1] = hb[0], hb[1], ex[nx-2]


def dielectric(nx: int, epsr: float = 1) -> np.ndarray:
	cb = 0.5 * np.ones(nx, dtype=np.float64)
	cb[nx//2:] = 0.5/epsr
	return cb


def main():

	nx: int = 201
	ns: int = 320

	ex = np.zeros(nx, dtype=np.float64)
	hy = np.zeros(nx, dtype=np.float64)

	lb = np.zeros(2, dtype=np.float64)
	hb = np.zeros(2, dtype=np.float64)

	epsr: float = 4 # relative permittivity
	cb: np.ndarray = dielectric(nx, epsr)

	for t in range(1, ns+1):
		field(t, nx, cb, ex, hy, lb, hb)

	visualize(ns, nx, epsr, cb, ex, hy)


if __name__ == "__main__":
	main()
