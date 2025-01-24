#!/usr/bin/env python
# File: fd1d_2_1.py
# Name: D.Saravanan
# Date: 25/11/2021

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, epsr: float, sigma: float, gbx: np.ndarray, ex: np.ndarray, hy: np.ndarray) -> None:
	fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.2})
	fig.suptitle(r"FDTD simulation of a sinusoidal wave striking lossy dielectric")
	medium = gbx/gbx[nx//2]
	medium[medium==0] = -1.5
	ax1.plot(ex, "k", lw=1)
	ax1.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
	ax1.text(nx/4, 0.5, f"T = {ns}", horizontalalignment="center")
	ax1.text(3*nx/4, 0.5, f"epsr = {epsr}", horizontalalignment="center")
	ax1.text(3*nx/4, -0.5, rf"$\sigma$ = {sigma}", horizontalalignment="center")
	ax1.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
	ax1.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
	ax2.plot(hy, "k", lw=1)
	ax2.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
	ax2.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"$H_y$")
	ax2.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
	plt.subplots_adjust(bottom=0.2, hspace=0.45)
	plt.savefig("fd1d_2_1.png")


def sinusoidal(t: int, ddx: float = 0.01, freq: float = 700e6) -> float:
	dt: float = ddx/6e8  # time step
	return np.sin(2 * np.pi * freq * dt * t)


def field(t: int, nx: int, gax: np.ndarray, gbx: np.ndarray, dx: np.ndarray, ex: np.ndarray, ix: np.ndarray, hy: np.ndarray, bc: np.ndarray):
	# calculate the Hy field
	hy[0:nx-1] = hy[0:nx-1] + 0.5 * (ex[0:nx-1] - ex[1:nx])
	# calculate the electric flux density Dx
	dx[1:nx] = dx[1:nx] + 0.5 * (hy[0:nx-1] - hy[1:nx])
	# put a sinusoidal wave at the low end
	dx[1] = dx[1] + sinusoidal(t, 0.01, 700e6)
	# calculate the Ex field from Dx
	ex[1:nx] = gax[1:nx] * (dx[1:nx] - ix[1:nx])
	ix[1:nx] = ix[1:nx] + gbx[1:nx] * ex[1:nx]
	# absorbing boundary conditions
	ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
	ex[nx-1], bc[3], bc[2] = bc[3], bc[2], ex[nx-2]


def dielectric(nx: int, epsr: float = 1, sigma: float = 0.04, ddx: float = 0.01):
	gax = np.ones(nx, dtype=np.float64)
	gbx = np.zeros(nx, dtype=np.float64)
	dt: float = ddx/6e8  # time step
	eps0: float = 8.854e-12  # vacuum permittivity (F/m)
	gax[nx//2:] = 1/(epsr + (sigma * dt/eps0))
	gbx[nx//2:] = sigma * dt/eps0
	return gax, gbx


def main():

	nx: int = 1024
	ns: int = 1500

	dx = np.zeros(nx, dtype=np.float64)
	ex = np.zeros(nx, dtype=np.float64)
	ix = np.zeros(nx, dtype=np.float64)
	hy = np.zeros(nx, dtype=np.float64)

	bc = np.zeros(4, dtype=np.float64)

	ddx: float = 0.01  # cell size (m)
	epsr: float = 4  # relative permittivity
	sigma: float = 0.04  # conductivity (S/m)
	gax, gbx = dielectric(nx, epsr, sigma, ddx)

	for t in range(1, ns+1):
		field(t, nx, gax, gbx, dx, ex, ix, hy, bc)

	visualize(ns, nx, epsr, sigma, gbx, ex, hy)


if __name__ == "__main__":
	main()
