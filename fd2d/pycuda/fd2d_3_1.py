#!/usr/bin/env python
# File: fd2d_3_1.py
# Name: D.Saravanan
# Date: 17/01/2022

""" Simulation of a pulse in free space in the transverse magnetic (TM) mode """

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def surfaceplot(ns: int, nx: int, ny: int, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    fig.suptitle(r"FDTD simulation of a pulse in free space")
    yv, xv = np.meshgrid(range(ny), range(nx))
    ax.plot_surface(xv, yv, ez, rstride=1, cstride=1, cmap="gray", lw=10/nx)
    ax.text2D(0.1, 0.7, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(False); ax.view_init(elev=20.0, azim=45)
    plt.savefig("fd2d_surface_3_1.png", dpi=100)


def contourplot(ns: int, nx: int, ny: int, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a pulse in free space")
    yv, xv = np.meshgrid(range(ny), range(nx)); ezmax = np.abs(ez).max()
    levels = np.linspace(-ezmax, ezmax, int(2/0.04))
    ax.contour(xv, yv, ez, levels, cmap="gray", linewidths=1.5)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd2d_contour_3_1.png", dpi=100)


kernel = """
#define idx blockIdx.x*blockDim.x+threadIdx.x
#define idy blockIdx.y*blockDim.y+threadIdx.y
#define stx blockDim.x*gridDim.x
#define sty blockDim.y*gridDim.y


__device__
float gaussian(int t, int t0, float sigma) {
    return expf(-0.5f*(t - t0)/sigma*(t - t0)/sigma);
}


__global__
void dfield(int t, int nx, int ny, float *dz, float *hx, float *hy) {
    /* calculate the electric flux density Dz */
    for (int i = idy+1; i < nx; i += sty) {
        for (int j = idx+1; j < ny; j += stx) {
            int n = i*ny+j;
            dz[n] += 0.5f * (hy[n] - hy[n-ny] - hx[n] + hx[n-1]);
        }
    }
    __syncthreads();
    /* put a Gaussian pulse in the middle */
    if (idy == nx/2 && idx == ny/2) dz[nx/2*ny+ny/2] = gaussian(t, 20, 6.0f);
}


__global__
void efield(int nx, int ny, float *naz, float *dz, float *ez) {
    /* calculate the Ez field from Dz */
    for (int i = idy; i < nx; i += sty) {
        for (int j = idx; j < ny; j += stx) {
            int n = i*ny+j;
            ez[n] = naz[n] * dz[n];
        }
    }
}


__global__
void hfield(int nx, int ny, float *ez, float *hx, float *hy) {
    /* calculate the Hx and Hy field */
    for (int i = idy; i < nx-1; i += sty) {
        for (int j = idx; j < ny-1; j += stx) {
            int n = i*ny+j;
            hx[n] += 0.5f * (ez[n] - ez[n+1]);
            hy[n] -= 0.5f * (ez[n] - ez[n+ny]);
        }
    }
}
"""


def main():

    nx = np.int32(100)  # number of grid points
    ny = np.int32(100)  # number of grid points

    ns = np.int32(90)  # number of time steps

    dz = gpuarray.zeros(nx*ny, dtype=np.float32)
    ez = gpuarray.zeros(nx*ny, dtype=np.float32)
    hx = gpuarray.zeros(nx*ny, dtype=np.float32)
    hy = gpuarray.zeros(nx*ny, dtype=np.float32)

    naz = gpuarray.ones(nx*ny, dtype=np.float32)

    blockDimx: int = 16
    blockDimy: int = 16
    gridDimx: int = int((ny+blockDimx-1)/blockDimx)
    gridDimy: int = int((nx+blockDimy-1)/blockDimy)

    gridDim = (gridDimx,gridDimy,1)
    blockDim = (blockDimx,blockDimy,1)

    mod = SourceModule(kernel)
    dfield = mod.get_function("dfield")
    efield = mod.get_function("efield")
    hfield = mod.get_function("hfield")

    for t in np.arange(1, ns+1).astype(np.int32):
        dfield(t, nx, ny, dz, hx, hy, grid=gridDim, block=blockDim)
        efield(nx, ny, naz, dz, ez, grid=gridDim, block=blockDim)
        hfield(nx, ny, ez, hx, hy, grid=gridDim, block=blockDim)

    drv.Context.synchronize()

    surfaceplot(ns, nx, ny, ez.get().reshape(nx,ny))
    contourplot(ns, nx, ny, ez.get().reshape(nx,ny))


if __name__ == "__main__":
    main()
