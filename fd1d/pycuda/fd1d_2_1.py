#!/usr/bin/env python
# File: fd1d_2_1.py
# Name: D.Saravanan
# Date: 25/11/2021

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, epsr: float, sigma: float, nax: np.ndarray, ex: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace": 0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal striking lossy dielectric material")
    medium = (1 - nax)/(1 - nax[-1])*1e3 if epsr > 1 else (1 - nax)
    medium[medium==0] = -1e3
    ax.plot(ex, color="black", linewidth=1)
    ax.fill_between(range(nx), medium, medium[0], color='y', alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=range(0, nx+1, round(nx//10,-1)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=ax.transAxes)
    ax.text(0.85, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()


medium = namedtuple('medium', (
    'nax', 'nbx',
))


kernel = """
#define idx (blockIdx.x * blockDim.x + threadIdx.x)
#define stx (blockDim.x * gridDim.x)


typedef struct {
    float *nax, *nbx;
} medium;


__device__
float sinusoidal(int t, float ds, float freq) {
    float dt = ds/6e8;  /* time step (s) */
    return sin(2 * M_PI * freq * dt * t);
}


__global__
void dxfield(int t, int nx, float *dx, float *hy) {
    /* calculate the electric flux density Dx */
    for (int i = idx + 1; i < nx; i += stx)
        dx[i] = dx[i] + 0.5 * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    if (idx == 1) dx[1] = dx[1] + sinusoidal(t, 0.01, 700e6);
}


__global__
void exfield(int nx, medium *md, float *dx, float *ix, float *ex) {
    /* calculate the Ex field from Dx */
    for (int i = idx + 1; i < nx; i += stx) {
        ex[i] = md->nax[i] * (dx[i] - ix[i]);
        ix[i] = ix[i] + md->nbx[i] * ex[i];
    }
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


def dielectric(nx: int, dt: float, epsr: float, sigma: float) -> medium:
    md = medium(
        nax = gpuarray.ones(nx, dtype=np.float32),
        nbx = gpuarray.zeros(nx, dtype=np.float32),
    )
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    md.nax[nx//2:] = 1/(epsr + (sigma * dt/eps0))
    md.nbx[nx//2:] = sigma * dt/eps0
    return md


def main():

    nx = np.int32(512)  # number of grid points
    ns = np.int32(740)  # number of time steps

    dx = gpuarray.zeros(nx, dtype=np.float32)
    ex = gpuarray.zeros(nx, dtype=np.float32)
    ix = gpuarray.zeros(nx, dtype=np.float32)
    hy = gpuarray.zeros(nx, dtype=np.float32)

    bc = gpuarray.zeros(4, dtype=np.float32)

    ds = np.float32(0.01)  # spatial step (m)
    dt = np.float32(ds/6e8)  # time step (s)
    epsr: float = 4  # relative permittivity
    sigma: float = 0.04  # conductivity (S/m)
    md: medium = dielectric(nx, dt, epsr, sigma)

    mdptr = gpuarray.to_gpu(np.array([
        md.nax.ptr, md.nbx.ptr,
    ], dtype=np.uint64)).gpudata

    numSM: int = drv.Device(0).multiprocessor_count

    blockDimx: int = 256
    gridDimx: int = 32*numSM

    gridDim = (gridDimx,1,1)
    blockDim = (blockDimx,1,1)

    mod = SourceModule(kernel)
    dxfield = mod.get_function("dxfield")
    exfield = mod.get_function("exfield")
    hyfield = mod.get_function("hyfield")

    for t in np.arange(1, ns+1).astype(np.int32):
        dxfield(t, nx, dx, hy, grid=gridDim, block=blockDim)
        exfield(nx, mdptr, dx, ix, ex, grid=gridDim, block=blockDim)
        hyfield(nx, ex, hy, bc, grid=gridDim, block=blockDim)

    drv.Context.synchronize()

    visualize(ns, nx, epsr, sigma, md.nax.get(), ex.get())


if __name__ == "__main__":
    main()
