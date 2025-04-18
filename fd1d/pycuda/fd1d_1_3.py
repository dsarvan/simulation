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


def visualize(ns: int, nx: int, epsr: float, cb: np.ndarray, ex: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a pulse striking dielectric material")
    medium = (0.5/cb - 1)/(epsr - 1)*1e3 if epsr > 1 else (0.5/cb - 1)
    medium[medium==0] = -1e3
    ax.plot(ex, color="black", linewidth=1)
    ax.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=range(0, nx+1, round(nx//10,-1)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()


kernel = """
#define idx (blockIdx.x * blockDim.x + threadIdx.x)
#define stx (blockDim.x * gridDim.x)


__device__
float gaussian(int t, int t0, float sigma) {
    return exp(-0.5 * ((t - t0)/sigma) * ((t - t0)/sigma));
}


__global__
void exfield(int t, int nx, float *cb, float *ex, float *hy) {
    /* calculate the Ex field */
    for (int i = idx + 1; i < nx; i += stx)
        ex[i] = ex[i] + cb[i] * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse at the low end */
    if (idx == 1) ex[1] = ex[1] + gaussian(t, 40, 12);
}


__global__
void hyfield(int nx, float *ex, float *hy, float *bc) {
    /* absorbing boundary conditions */
    if (idx == 0) ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    if (idx == nx-1) ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = idx; i < nx - 1; i += stx)
        hy[i] = hy[i] + 0.5 * (ex[i] - ex[i+1]);
}
"""


def dielectric(nx: int, epsr: float) -> np.ndarray:
    cb = 0.5 * gpuarray.ones(nx, dtype=np.float32)
    cb[nx//2:] = 0.5/epsr
    return cb


def main():

    nx = np.int32(512)  # number of grid points
    ns = np.int32(740)  # number of time steps

    ex = gpuarray.zeros(nx, dtype=np.float32)
    hy = gpuarray.zeros(nx, dtype=np.float32)

    bc = gpuarray.zeros(4, dtype=np.float32)

    epsr: float = 4  # relative permittivity
    cb: np.ndarray = dielectric(nx, epsr)

    numSM: int = drv.Device(0).multiprocessor_count

    blockDimx: int = 256
    gridDimx: int = 32*numSM

    gridDim = (gridDimx,1,1)
    blockDim = (blockDimx,1,1)

    mod = SourceModule(kernel)
    exfield = mod.get_function("exfield")
    hyfield = mod.get_function("hyfield")

    for t in np.arange(1, ns+1).astype(np.int32):
        exfield(t, nx, cb, ex, hy, grid=gridDim, block=blockDim)
        hyfield(nx, ex, hy, bc, grid=gridDim, block=blockDim)

    drv.Context.synchronize()

    visualize(ns, nx, epsr, cb.get(), ex.get())


if __name__ == "__main__":
    main()
