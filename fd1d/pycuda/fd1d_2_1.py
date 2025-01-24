#!/usr/bin/env python
# File: fd1d_2_1.py
# Name: D.Saravanan
# Date: 25/11/2021

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import matplotlib.pyplot as plt
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

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
	plt.show()


kernel = """
__device__ float sinusoidal(int t, float ddx, float freq) {
	float dt = ddx/6e8; /* time step */
	return sin(2 * M_PI * freq * dt * t);
}


__global__ void field(int t, int nx, float *gax, float *gbx, float *dx, float *ex, float *ix, float *hy, float *bc) {
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

	/* put a sinusoidal wave at the low end */
	dx[1] = dx[1] + sinusoidal(t, 0.01, 700e6);

	/* calculate the Ex field from Dx */
	for (int i = index + 1; i < nx; i += stride)
		ex[i] = gax[i] * (dx[i] - ix[i]);

	__syncthreads();

	for (int i = index + 1; i < nx; i += stride)
		ix[i] = ix[i] + gbx[i] * ex[i];

	__syncthreads();

	/* absorbing boundary conditions */
	ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
	ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
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

	nx = np.int32(1024)
	ns = np.int32(1500)

	dx = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))
	ex = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))
	ix = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))
	hy = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))

	bc = gpuarray.to_gpu(np.zeros(4, dtype=np.float32))

	ddx: float = 0.01  # cell size (m)
	epsr: float = 4  # relative permittivity
	sigma: float = 0.04  # conductivity (S/m)
	gax, gbx = dielectric(nx, epsr, sigma, ddx)

	gax = gpuarray.to_gpu(gax)
	gbx = gpuarray.to_gpu(gbx)

	blockDimx = 256
	gridDimx = int((nx + blockDimx - 1)/blockDimx)

	mod = SourceModule(kernel)
	field = mod.get_function("field")

	gridDim = (gridDimx,1)
	blockDim = (blockDimx,1,1)

	for t in range(1, ns+1):
		field(np.int32(t), nx, gax, gbx, dx, ex, ix, hy, bc, grid=gridDim, block=blockDim)

	drv.Context.synchronize()

	visualize(ns, nx, epsr, sigma, gbx.get(), ex.get(), hy.get())


if __name__ == "__main__":
	main()
