#!/usr/bin/env python
# File: test_3_2.py
# Name: D.Saravanan
# Date: 18/01/2022

""" Simulation of a propagating sinusoidal in free space in the transverse
magnetic (TM) mode with the two-dimensional perfectly matched layer (PML) """

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from collections import namedtuple
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def surfaceplot(ns: int, nx: int, ny: int, npml: int, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    fig.suptitle(r"FDTD simulation of a sinusoidal in free space with PML")
    yv, xv = np.meshgrid(range(ny), range(nx)); levels = [0.50,1.50]
    pmlmsk = ((xv < npml)|(xv >= nx-npml)|(yv < npml)|(yv >= ny-npml))
    ax.plot_surface(xv, yv, ez, rstride=1, cstride=1, cmap="gray", lw=10/nx)
    ax.contourf(xv, yv, pmlmsk, levels, offset=0, colors="k", alpha=0.40)
    ax.text2D(0.1, 0.7, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(False); ax.view_init(elev=20.0, azim=45)
    plt.savefig("test_surface_3_2.png", dpi=100)


def contourplot(ns: int, nx: int, ny: int, npml: int, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a sinusoidal in free space with PML")
    yv, xv = np.meshgrid(range(ny), range(nx)); ezmax = np.abs(ez).max()
    levels = np.linspace(-ezmax, ezmax, int(2/0.04))
    pmlmsk = ((xv < npml)|(xv >= nx-npml)|(yv < npml)|(yv >= ny-npml))
    ax.contour(xv, yv, ez, levels, cmap="gray", alpha=1.0, linewidths=1.5)
    ax.contourf(xv, yv, pmlmsk, levels=[0.50,1.50], colors="k", alpha=0.40)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("test_contour_3_2.png", dpi=100)


pmlayer = namedtuple('pmlayer', (
    'fx1', 'fx2', 'fx3',
    'fy1', 'fy2', 'fy3',
    'gx2', 'gx3',
    'gy2', 'gy3',
))


kernel = """
#define idx blockIdx.x*blockDim.x+threadIdx.x
#define idy blockIdx.y*blockDim.y+threadIdx.y
#define stx blockDim.x*gridDim.x
#define sty blockDim.y*gridDim.y


typedef struct {
    float *fx1, *fx2, *fx3;
    float *fy1, *fy2, *fy3;
    float *gx2, *gx3;
    float *gy2, *gy3;
} pmlayer;


__device__
float sinusoidal(int t, float ds, float freq) {
    float dt = ds/6e8f;  /* time step (s) */
    return sinf(2*M_PI*freq*dt*t);
}


__global__
void dfield(int t, int nx, int ny, pmlayer *pml, float *dz, float *hx, float *hy) {
    /* calculate the electric flux density Dz */
    for (int i = idy+1; i < nx; i += sty) {
        for (int j = idx+1; j < ny; j += stx) {
            int n = i*ny+j;
            dz[n] = pml->gx3[i] * pml->gy3[j] * dz[n] + pml->gx2[i] * pml->gy2[j] * 0.5f * (hy[n] - hy[n-ny] - hx[n] + hx[n-1]);
        }
    }
    __syncthreads();
    /* put a sinusoidal source at a point that is offset five cells
     * from the center of the problem space in each direction */
    if (idy == nx/2-5 && idx == ny/2-5)
        dz[(nx/2-5)*ny+(ny/2-5)] = sinusoidal(t, 0.01f, 1500e6f);
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
void hfield(int nx, int ny, pmlayer *pml, float *ez, float *ihx, float *ihy, float *hx, float *hy) {
    /* calculate the Hx and Hy field */
    for (int i = idy; i < nx-1; i += sty) {
        for (int j = idx; j < ny-1; j += stx) {
            int n = i*ny+j;
            ihx[n] += ez[n] - ez[n+1];
            ihy[n] += ez[n] - ez[n+ny];
            hx[n] = pml->fy3[j] * hx[n] + pml->fy2[j] * (0.5f * ez[n] - 0.5f * ez[n+1] + pml->fx1[i] * ihx[n]);
            hy[n] = pml->fx3[i] * hy[n] - pml->fx2[i] * (0.5f * ez[n] - 0.5f * ez[n+ny] + pml->fy1[j] * ihy[n]);
        }
    }
}
"""


def pmlparam(nx: int, ny: int, npml: int, pml: pmlayer) -> None:
    """ calculate the two-dimensional perfectly matched layer (PML) parameters """
    for n in range(npml):
        xm = 0.33*((npml-n)/npml)**3
        xn = 0.33*((npml-n-0.5)/npml)**3
        pml.fx1[n] = pml.fx1[nx-2-n] = pml.fy1[n] = pml.fy1[ny-2-n] = xn
        pml.fx2[n] = pml.fx2[nx-2-n] = pml.fy2[n] = pml.fy2[ny-2-n] = 1/(1+xn)
        pml.gx2[n] = pml.gx2[nx-1-n] = pml.gy2[n] = pml.gy2[ny-1-n] = 1/(1+xm)
        pml.fx3[n] = pml.fx3[nx-2-n] = pml.fy3[n] = pml.fy3[ny-2-n] = (1-xn)/(1+xn)
        pml.gx3[n] = pml.gx3[nx-1-n] = pml.gy3[n] = pml.gy3[ny-1-n] = (1-xm)/(1+xm)


def main():

    nx = np.int32(1024)  # number of grid points
    ny = np.int32(1024)  # number of grid points

    ns = np.int32(5000)  # number of time steps

    dz = gpuarray.zeros(nx*ny, dtype=np.float32)
    ez = gpuarray.zeros(nx*ny, dtype=np.float32)
    hx = gpuarray.zeros(nx*ny, dtype=np.float32)
    hy = gpuarray.zeros(nx*ny, dtype=np.float32)

    ihx = gpuarray.zeros(nx*ny, dtype=np.float32)
    ihy = gpuarray.zeros(nx*ny, dtype=np.float32)

    naz = gpuarray.ones(nx*ny, dtype=np.float32)

    pml = pmlayer(
        fx1 = gpuarray.empty(nx, dtype=np.float32).fill(0.0),
        fx2 = gpuarray.empty(nx, dtype=np.float32).fill(1.0),
        fx3 = gpuarray.empty(nx, dtype=np.float32).fill(1.0),
        fy1 = gpuarray.empty(ny, dtype=np.float32).fill(0.0),
        fy2 = gpuarray.empty(ny, dtype=np.float32).fill(1.0),
        fy3 = gpuarray.empty(ny, dtype=np.float32).fill(1.0),
        gx2 = gpuarray.empty(nx, dtype=np.float32).fill(1.0),
        gx3 = gpuarray.empty(nx, dtype=np.float32).fill(1.0),
        gy2 = gpuarray.empty(ny, dtype=np.float32).fill(1.0),
        gy3 = gpuarray.empty(ny, dtype=np.float32).fill(1.0),
    )

    pmlptr = gpuarray.to_gpu(np.array([
        pml.fx1.ptr,
        pml.fx2.ptr,
        pml.fx3.ptr,
        pml.fy1.ptr,
        pml.fy2.ptr,
        pml.fy3.ptr,
        pml.gx2.ptr,
        pml.gx3.ptr,
        pml.gy2.ptr,
        pml.gy3.ptr,
    ], dtype=np.uint64)).gpudata

    npml: int = np.int32(80)  # pml thickness
    pmlparam(nx, ny, npml, pml)

    ds: float = np.float32(0.01)  # spatial step (m)
    dt: float = np.float32(ds/6e8)  # time step (s)

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

    stime = drv.Event()
    ntime = drv.Event()

    stime.record()

    for t in np.arange(1, ns+1).astype(np.int32):
        dfield(t, nx, ny, pmlptr, dz, hx, hy, grid=gridDim, block=blockDim)
        efield(nx, ny, naz, dz, ez, grid=gridDim, block=blockDim)
        hfield(nx, ny, pmlptr, ez, ihx, ihy, hx, hy, grid=gridDim, block=blockDim)

    drv.Context.synchronize()

    ntime.record()
    ntime.synchronize()

    time = stime.time_till(ntime)*1e-3
    print(f"Total compute time on GPU: {time:.3f} s")

    print(ez[2*ny:2*ny+50])
    surfaceplot(ns, nx, ny, npml, ez.get().reshape(nx,ny))
    contourplot(ns, nx, ny, npml, ez.get().reshape(nx,ny))


if __name__ == "__main__":
    main()
