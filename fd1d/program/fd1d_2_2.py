#!/usr/bin/env python
# File: fd1d_2_2.py
# Name: D.Saravanan
# Date: 07/12/2021

""" Simulation of a pulse striking a dielectric medium and implements
the discrete Fourier transform with a Gaussian pulse as its source """

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp, sqrt, sin, cos, atan2

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, epsr: float, sigma: float, gax: np.ndarray, ex: np.ndarray, amp: np.ndarray) -> None:
	fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.2})
	fig.suptitle(r"Implements the discrete Fourier transform with pulse as its source")
	medium = (1 - gax)/(1 - gax)[-1]
	medium[medium==0] = -1.5
	ax1.plot(ex, "k", lw=1)
	ax1.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
	ax1.text(nx/4, 0.5, f"T = {ns}", horizontalalignment="center")
	ax1.text(3*nx/4, 0.5, rf"$\epsilon_r$ = {epsr}", horizontalalignment="center")
	ax1.text(3*nx/4, -0.5, rf"$\sigma$ = {sigma}", horizontalalignment="center")
	ax1.set(xlim=(0, nx-1), ylim=(-1.2, 2.2), ylabel=r"$E_x$")
	ax1.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 2.2, 1))
	ax2.plot(amp, "k", lw=1)
	ax2.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
	ax2.set(xlim=(0, nx-1), ylim=(-1.2, 2.2), xlabel=r"FDTD cells", ylabel=r"$Amp$")
	ax2.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 2.2, 1))
	plt.subplots_adjust(bottom=0.2, hspace=0.45)
	plt.savefig("fd1d_2_2.png")


def gaussian(t: int, t0: int, sigma: float) -> float:
	return np.exp(-0.5 * ((t - t0)/sigma)**2)


def fieldfourier(t, dt, nx, nf, freq, ex, r_pt, i_pt):
	r_pt[0:nf, 0:nx] = r_pt[0:nf, 0:nx] + cos(2*pi*freq[0:nf]*dt*t) * ex[0:nx]
	i_pt[0:nf, 0:nx] = i_pt[0:nf, 0:nx] - sin(2*pi*freq[0:nf]*dt*t) * ex[0:nx]


def pulsefourier(t, dt, nx, nf, freq, ex, r_in, i_in):
	r_in[0:nf] = r_in[0:nf] + cos(2*pi*freq[0:nf]*dt*t) * ex[10]
	i_in[0:nf] = i_in[0:nf] - sin(2*pi*freq[0:nf]*dt*t) * ex[10]


def field(t, dt, nx, nf, freq, gax, gbx, r_pt, i_pt, r_in, i_in, dx, ex, ix, hy, bc):
	# calculate the Hy field
	hy[0:nx-1] = hy[0:nx-1] + 0.5 * (ex[0:nx-1] - ex[1:nx])
	# calculate the electric flux density Dx
	dx[1:nx] = dx[1:nx] + 0.5 * (hy[0:nx-1] - hy[1:nx])
	# put a Gaussian pulse at the low end
	dx[1] = dx[1] + gaussian(t, 50, 10)
	# calculate the Ex field from Dx
	ex[1:nx] = gax[1:nx] * (dx[1:nx] - ix[1:nx])
	ix[1:nx] = ix[1:nx] + gbx[1:nx] * ex[1:nx]
	fieldfourier(t, dt, nx, nf, freq, ex, r_pt, i_pt)
	if t < nx//2: pulsefourier(t, dt, nx, nf, freq, ex, r_in, i_in)
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

	nx: int = 201
	ns: int = 400

	dx = np.zeros(nx, dtype=np.float64)
	ex = np.zeros(nx, dtype=np.float64)
	ix = np.zeros(nx, dtype=np.float64)
	hy = np.zeros(nx, dtype=np.float64)

	bc = np.zeros(4, dtype=np.float64)

	ddx: float = 0.01  # cell size (m)
	dt: float = ddx/6e8  # time step
	epsr: float = 4  # relative permittivity
	sigma: float = 0  # conductivity (S/m)
	gax, gbx = dielectric(nx, epsr, sigma, ddx)

	# frequency 100 MHz, 200 MHz, 500 MHz
	freq = np.array((100e6, 200e6, 500e6)).reshape(-1,1)
	nf = len(freq)	# number of frequencies

	r_pt = np.zeros((nf, nx), dtype=np.float64)
	i_pt = np.zeros((nf, nx), dtype=np.float64)

	r_in = np.zeros(nf, dtype=np.float64).reshape(-1,1)
	i_in = np.zeros(nf, dtype=np.float64).reshape(-1,1)

	amplt = np.zeros((nf, nx), dtype=np.float64)
	phase = np.zeros((nf, nx), dtype=np.float64)

	for t in range(1, ns+1):
		field(t, dt, nx, nf, freq, gax, gbx, r_pt, i_pt, r_in, i_in, dx, ex, ix, hy, bc)

	amplt = (1/sqrt(r_in**2 + i_in**2)) * sqrt(r_pt**2 + i_pt**2)
	phase = atan2(i_pt, r_pt) - atan2(i_in, r_in)

	visualize(ns, nx, epsr, sigma, gax, ex, amplt[2])

if __name__ == "__main__":
	main()
