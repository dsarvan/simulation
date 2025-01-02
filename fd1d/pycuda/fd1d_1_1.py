#!/usr/bin/env python
# File: fd1d_1_1.py
# Name: D.Saravanan
# Date: 11/10/2021

""" Simulation in free space """
# FDTD simulation of a pulse in free space after 100 steps.
# The pulse originated in the center and travels outward.

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
	ax1.text(nx/2, 0.5, f"T = {ns}", horizontalalignment="center")
	ax1.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
	ax1.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
	ax2.plot(hy, "k", lw=1)
	ax2.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), xlabel=r"FDTD cells", ylabel=r"$H_y$")
	ax2.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
	plt.subplots_adjust(bottom=0.2, hspace=0.45)
	plt.show()


kernel = """
__device__ float gaussian(int t, int t0, float sigma) {
	return exp(-0.5 * ((t - t0)/sigma) * ((t - t0)/sigma));
}


__global__ void field(int t, int nx, float *ex, float *hy) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int idx = index; idx < nx - 1; idx += stride)
		hy[idx] = hy[idx] + 0.5 * (ex[idx] - ex[idx + 1]);

	__syncthreads();

	for (int idx = index + 1; idx < nx; idx += stride)
		ex[idx] = ex[idx] + 0.5 * (hy[idx - 1] - hy[idx]);

	__syncthreads();

	ex[nx/2] = gaussian(t, 40, 12);
}
"""


def main():

	nx = np.int32(201)
	ns = np.int32(100)

	ex = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))
	hy = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))

	blockDimx = 256
	gridDimx = int((nx + blockDimx - 1)/blockDimx)

	mod = SourceModule(kernel)
	field = mod.get_function("field")

	for t in range(1, ns+1):
		field(np.int32(t), nx, ex, hy, grid=(gridDimx,1), block=(blockDimx,1,1))

	drv.Context.synchronize()

	visualize(ns, nx, ex.get(), hy.get())


if __name__ == "__main__":
	main()
