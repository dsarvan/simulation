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


def visualize(ns: int, nx: int, epsr: float, cb: np.ndarray, ex: np.ndarray, hy: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(2, sharex=False, gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a pulse striking dielectric material")
    medium = (0.5/cb - 1)/(epsr - 1) if epsr > 1 else (0.5/cb - 1)
    medium[medium==0] = -1e3
    ax1.plot(ex, "k", lw=1)
    ax1.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    ax1.text(nx/4, 0.5, f"T = {ns}", horizontalalignment="center")
    ax1.text(3*nx/4, 0.5, f"epsr = {epsr}", horizontalalignment="center")
    ax1.set(xlim=(0, nx-1), ylim=(-1.2, 1.2), ylabel=r"$E_x$")
    ax1.set(xticks=range(0, nx+1, round(nx//10,-1)), yticks=np.arange(-1, 1.2, 1))
    ax2.plot(hy, "k", lw=1)
    ax2.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
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


__global__ void exfield(int t, int nx, float *cb, float *ex, float *hy, float *bc) {
    /* calculate the Ex field */
    for (int i = idx + 1; i < nx; i += stx)
        ex[i] = ex[i] + cb[i] * (hy[i-1] - hy[i]);
    __syncthreads();
    /* put a Gaussian pulse at the low end */
    if (idx == 1) ex[1] = ex[1] + gaussian(t, 40, 12);
    /* absorbing boundary conditions */
    if (idx == 0) ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    if (idx == nx-1) ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
}


__global__ void hyfield(int nx, float *ex, float *hy) {
    /* calculate the Hy field */
    for (int i = idx; i < nx - 1; i += stx)
        hy[i] = hy[i] + 0.5 * (ex[i] - ex[i+1]);
    __syncthreads();
}
"""


def dielectric(nx: int, epsr: float) -> np.ndarray:
    cb = 0.5 * gpuarray.ones(nx, dtype=np.float32)
    cb[nx//2:] = 0.5/epsr
    return cb


def main():

    nx = np.int32(201)
    ns = np.int32(320)

    ex = gpuarray.zeros(nx, dtype=np.float32)
    hy = gpuarray.zeros(nx, dtype=np.float32)

    bc = gpuarray.zeros(4, dtype=np.float32)

    epsr: float = 4 # relative permittivity
    cb: np.ndarray = dielectric(nx, epsr)

    numSM: int = drv.Device(0).multiprocessor_count

    blockDimx: int = 256
    gridDimx: int = 32*numSM

    gridDim = (gridDimx,1)
    blockDim = (blockDimx,1,1)

    mod = SourceModule(kernel)
    exfield = mod.get_function("exfield")
    hyfield = mod.get_function("hyfield")

    for t in np.arange(1, ns+1).astype(np.int32):
        exfield(t, nx, cb, ex, hy, bc, grid=gridDim, block=blockDim)
        hyfield(nx, ex, hy, grid=gridDim, block=blockDim)

    drv.Context.synchronize()

    visualize(ns, nx, epsr, cb.get(), ex.get(), hy.get())


if __name__ == "__main__":
    main()
