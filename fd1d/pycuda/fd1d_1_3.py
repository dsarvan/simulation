#!/usr/bin/env python
# File: fd1d_1_3.py
# Name: D.Saravanan
# Date: 21/10/2021

""" Simulation of a pulse hitting a dielectric medium """

import matplotlib.pyplot as plt
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

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
	plt.show()


kernel = """
__device__ float gaussian(int t, int t0, float sigma) {
	return exp(-0.5 * ((t - t0)/sigma) * ((t - t0)/sigma));
}


__global__ void field(int t, int nx, float *cb, float *ex, float *hy, float *lb, float *hb) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int idx = index; idx < nx - 1; idx += stride)
		hy[idx] = hy[idx] + 0.5 * (ex[idx] - ex[idx + 1]);

	__syncthreads();

	for (int idx = index + 1; idx < nx; idx += stride)
		ex[idx] = ex[idx] + cb[idx] * (hy[idx - 1] - hy[idx]);

	__syncthreads();

	ex[1] = ex[1] + gaussian(t, 40, 12);

	ex[0] = lb[0], lb[0] = lb[1], lb[1] = ex[1];
	ex[nx-1] = hb[0], hb[0] = hb[1], hb[1] = ex[nx-2];
}
"""


def dielectric(nx: int, epsr: float = 1) -> np.ndarray:
	cb = 0.5 * np.ones(nx, dtype=np.float32)
	cb[nx//2:] = 0.5/epsr
	return cb


def main():

	nx = np.int32(201)
	ns = np.int32(320)

	ex = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))
	hy = gpuarray.to_gpu(np.zeros(nx, dtype=np.float32))

	lb = gpuarray.to_gpu(np.zeros(2, dtype=np.float32))
	hb = gpuarray.to_gpu(np.zeros(2, dtype=np.float32))

	epsr: float = 4 # relative permittivity
	cb: np.ndarray = gpuarray.to_gpu(dielectric(nx, epsr))

	blockDimx = 256
	gridDimx = int((nx + blockDimx - 1)/blockDimx)

	mod = SourceModule(kernel)
	field = mod.get_function("field")

	gridDim = (gridDimx,1)
	blockDim = (blockDimx,1,1)

	for t in range(1, ns+1):
		field(np.int32(t), nx, cb, ex, hy, lb, hb, grid=gridDim, block=blockDim)

	drv.Context.synchronize()

	visualize(ns, nx, epsr, cb.get(), ex.get(), hy.get())


if __name__ == "__main__":
	main()
