#!/usr/bin/env python
# File: fd1d_2_2.py
# Name: D.Saravanan
# Date: 07/12/2021

""" Simulation of a pulse striking a dielectric medium and implements
the discrete Fourier transform with a Gaussian pulse as its source """

import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, arctan2

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

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
	plt.show()


kernel = """
__device__ float
gaussian(int t, int t0, float sigma) {
	return exp(-0.5 * ((t - t0)/sigma) * ((t - t0)/sigma));
}


__global__ void
field(int t, int nx, float *gax, float *gbx, float *dx, float *ex, float *ix, float *hy) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	/* calculate the Hy field */
	for (int i = index; i < nx - 1; i += stride)
		hy[i] = hy[i] + 0.5 * (ex[i] - ex[i+1]);

	__syncthreads();

	/* calculate the electric flux density Dx */
	for (int i = index + 1; i < nx; i += stride)
		dx[i] = dx[i] + 0.5 * (hy[i-1] - hy[i]);

	__syncthreads();

	/* put a Gaussian pulse at the low end */
	dx[1] = dx[1] + gaussian(t, 50, 10);

	/* calculate the Ex field from Dx */
	for (int i = index + 1; i < nx; i += stride)
		ex[i] = gax[i] * (dx[i] - ix[i]);

	__syncthreads();

	for (int i = index + 1; i < nx; i += stride)
		ix[i] = ix[i] + gbx[i] * ex[i];

	__syncthreads();
}


__global__ void
fieldfourier(int t, int nf, int nx, float dt, float *freq, float *ex, float *r_pt, float *i_pt) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < nx; i += stride) {
		for (int n = threadIdx.y; n < nf; n += blockDim.y) {
			r_pt[n*nx + i] = r_pt[n*nx + i] + cos(2*M_PI*freq[n]*dt*t) * ex[i];
			i_pt[n*nx + i] = i_pt[n*nx + i] - sin(2*M_PI*freq[n]*dt*t) * ex[i];
		}
	}

	__syncthreads();
}


__global__ void
pulsefourier(int t, int nf, float dt, float *freq, float *ex, float *r_in, float *i_in) {
	for (int n = threadIdx.y; n < nf; n += blockDim.y) {
		r_in[n] = r_in[n] + cos(2*M_PI*freq[n]*dt*t) * ex[10];
		i_in[n] = i_in[n] - sin(2*M_PI*freq[n]*dt*t) * ex[10];
	}

	__syncthreads();
}
"""


def dielectric(nx: int, epsr: float = 1, sigma: float = 0.04, ddx: float = 0.01):
	gax = np.ones(nx, dtype=np.float32)
	gbx = np.zeros(nx, dtype=np.float32)
	dt: float = ddx/6e8  # time step
	eps0: float = 8.854e-12  # vacuum permittivity (F/m)
	gax[nx//2:] = 1/(epsr + (sigma * dt/eps0))
	gbx[nx//2:] = sigma * dt/eps0
	return gax, gbx


def main():

	nx = np.int32(10240)
	ns = np.int32(15000)

	dx = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))
	ex = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))
	ix = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))
	hy = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))

	bc = gpuarray.to_gpu(np.zeros(4, dtype=np.float32))

	ddx = np.float32(0.01)	# cell size (m)
	dt = np.float32(ddx/6e8)  # time step
	epsr: float = 4  # relative permittivity
	sigma: float = 0  # conductivity (S/m)
	gax, gbx = dielectric(nx, epsr, sigma, ddx)

	gax = gpuarray.to_gpu(gax)
	gbx = gpuarray.to_gpu(gbx)

	# frequency 100 MHz, 200 MHz, 500 MHz
	freq = gpuarray.to_gpu(np.array((100e6, 200e6, 500e6), dtype=np.float32).reshape(-1,1))
	nf = len(freq)	# number of frequencies

	r_pt = gpuarray.to_gpu(np.zeros((nf, nx), dtype=np.float32))
	i_pt = gpuarray.to_gpu(np.zeros((nf, nx), dtype=np.float32))

	r_in = gpuarray.to_gpu(np.zeros(nf, dtype=np.float32).reshape(-1,1))
	i_in = gpuarray.to_gpu(np.zeros(nf, dtype=np.float32).reshape(-1,1))

	amplt = np.zeros((nf, nx), dtype=np.float32)
	phase = np.zeros((nf, nx), dtype=np.float32)

	blockDimx = 256
	blockDimy = nf
	gridDimx = int((nx + blockDimx - 1)/blockDimx)

	mod = SourceModule(kernel)
	field = mod.get_function("field")
	fieldfourier = mod.get_function("fieldfourier")
	pulsefourier = mod.get_function("pulsefourier")

	gridDim = (gridDimx,1)
	blockDim = (blockDimx,blockDimy,1)

	for t in range(1, ns+1):
		field(np.int32(t), nx, gax, gbx, dx, ex, ix, hy, grid=gridDim, block=blockDim)
		fieldfourier(np.int32(t), np.int32(nf), nx, dt, freq, ex, r_pt, i_pt, grid=gridDim, block=blockDim)
		if t < nx//2: pulsefourier(np.int32(t), np.int32(nf), dt, freq, ex, r_in, i_in, grid=gridDim, block=blockDim)
		# absorbing boundary conditions
		ex[0], bc[0], bc[1] = bc[0], bc[1], ex[1]
		ex[-1], bc[3], bc[2] = bc[3], bc[2], ex[-2]

	drv.Context.synchronize()

	amplt = (1/sqrt(r_in.get()**2 + i_in.get()**2)) * sqrt(r_pt.get()**2 + i_pt.get()**2)
	phase = arctan2(i_pt.get(), r_pt.get()) - arctan2(i_in.get(), r_in.get())

	visualize(ns, nx, epsr, sigma, gax.get(), ex.get(), amplt[2])

if __name__ == "__main__":
	main()
