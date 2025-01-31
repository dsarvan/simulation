#!/usr/bin/env python
# File: fd1d_1_1.py
# Name: D.Saravanan
# Date: 11/10/2021

""" Simulation of a pulse in free space """

import matplotlib.pyplot as plt
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, ex: np.ndarray, hy: np.ndarray) -> None:
	fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.2})
	fig.suptitle(r"FDTD simulation of a pulse in free space")
	ax1.plot(ex, "k", lw=1)
	ax1.text(nx/4, 0.5, f"T = {ns}", horizontalalignment="center")
	ax1.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
	ax1.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
	ax2.plot(hy, "k", lw=1)
	ax2.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"$H_y$")
	ax2.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
	plt.subplots_adjust(bottom=0.2, hspace=0.45)
	plt.show()


kernel = """
#define idx (blockIdx.x * blockDim.x + threadIdx.x)
#define stx (blockDim.x * gridDim.x)


__device__ float gaussian(int t, int t0, float sigma) {
	return exp(-0.5 * ((t - t0)/sigma) * ((t - t0)/sigma));
}


__global__ void exfield(int t, int nx, float *ex, float *hy) {
	/* calculate the Ex field */
	for (int i = idx + 1; i < nx; i += stx)
		ex[i] = ex[i] + 0.5 * (hy[i-1] - hy[i]);
	__syncthreads();
	/* put a Gaussian pulse in the middle */
	if (idx == nx/2) ex[nx/2] = gaussian(t, 40, 12);
}


__global__ void hyfield(int nx, float *ex, float *hy) {
	/* calculate the Hy field */
	for (int i = idx; i < nx - 1; i += stx)
		hy[i] = hy[i] + 0.5 * (ex[i] - ex[i+1]);
	__syncthreads();
}
"""


def main():

	nx = np.int32(201)
	ns = np.int32(100)

	ex = gpuarray.zeros(nx, dtype=np.float32)
	hy = gpuarray.zeros(nx, dtype=np.float32)

	numSM: int = drv.Device(0).multiprocessor_count

	blockDimx: int = 256
	gridDimx: int = 32*numSM

	gridDim = (gridDimx,1)
	blockDim = (blockDimx,1,1)

	mod = SourceModule(kernel)
	exfield = mod.get_function("exfield")
	hyfield = mod.get_function("hyfield")

	for t in np.arange(1, ns+1).astype(np.int32):
		exfield(t, nx, ex, hy, grid=gridDim, block=blockDim)
		hyfield(nx, ex, hy, grid=gridDim, block=blockDim)

	drv.Context.synchronize()

	visualize(ns, nx, ex.get(), hy.get())


if __name__ == "__main__":
	main()
