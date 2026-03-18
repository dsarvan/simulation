#!/usr/bin/env python
# File: fdtd21.py
# Name: D.Saravanan
# Date: 25/11/2021

""" Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m) """

import numpy
from pycuda import driver
from pycuda import autoinit
from pycuda import gpuarray
from matplotlib import pyplot
from typing import NamedTuple
from pycuda.compiler import SourceModule

pyplot.style.use("classic")
pyplot.style.use("pyplot.mplstyle")

type F32Array = gpuarray.GPUArray


def visualize(ns: int, nx: int, epsr: float, sigma: float, nax: F32Array, ex: F32Array):
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=True)
    fig.suptitle(r"FDTD simulation of a sinusoidal striking lossy dielectric material")
    medium = numpy.where(1.0/nax-1)[0] if epsr > 1 else (1.0/nax-1)
    axs.plot(range(nx), ex, color="k", linewidth=1.0)
    axs.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=range(0, nx+1, int(numpy.ceil(nx/500)*25)))
    axs.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axs.text(0.02, 0.90, rf"$T$ = {ns}", transform=axs.transAxes)
    axs.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=axs.transAxes)
    axs.text(0.85, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("fdtd21.png", dpi=100)


Medium = NamedTuple('Medium',[
    ('nax',F32Array), ('nbx',F32Array),
])


kernel = """
#define idx blockIdx.x*blockDim.x+threadIdx.x
#define stx blockDim.x*gridDim.x


typedef struct {
    float *nax, *nbx;
} Medium;


__device__
float sinusoidal(int ts, float ds, float freq) {
    float dt = ds/6e8f;  /* time step (s) */
    return sinf(2*M_PI*freq*dt*ts);
}


__global__
void dxfield(int ts, int nx, float *hy, float *dx) {
    /* calculate the electric flux density Dx */
    for (int i = idx+1; i < nx; i += stx)
        dx[i] += 0.5f * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    if (idx == 1) dx[1] += sinusoidal(ts, 0.01f, 700e6f);
}


__global__
void exfield(int nx, Medium md, float *ix, float *dx, float *ex) {
    /* calculate the Ex field from Dx */
    for (int i = idx+1; i < nx; i += stx) {
        ex[i] = md.nax[i] * (dx[i] - ix[i]);
        ix[i] += md.nbx[i] * ex[i];
    }
}


__global__
void hyfield(int nx, float *bc, float *ex, float *hy) {
    /* absorbing boundary conditions */
    if (idx == 0) ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    if (idx == nx-1) ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = idx; i < nx-1; i += stx)
        hy[i] += 0.5f * (ex[i] - ex[i+1]);
}
"""


def dielectric(nx: int, dt: float, epsr: float, sigma: float):
    md = Medium(
        nax = gpuarray.empty(nx, numpy.float32).fill(1.0),
        nbx = gpuarray.empty(nx, numpy.float32).fill(0.0),
    )
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    md.nax[nx//2:] = 1/(epsr + sigma*dt/eps0)
    md.nbx[nx//2:] = sigma*dt/eps0
    return md


def main():

    nx = numpy.int32(512)  # number of grid points
    ns = numpy.int32(740)  # number of time steps

    dx = gpuarray.zeros(nx, numpy.float32)
    ex = gpuarray.zeros(nx, numpy.float32)
    ix = gpuarray.zeros(nx, numpy.float32)
    hy = gpuarray.zeros(nx, numpy.float32)

    bc = gpuarray.zeros(4, numpy.float32)

    ds = numpy.float32(0.01)  # spatial step (m)
    dt = numpy.float32(ds/6e8)  # time step (s)
    epsr: float = 4.0  # relative permittivity
    sigma: float = 0.04  # conductivity (S/m)
    md: Medium = dielectric(nx, dt, epsr, sigma)

    numSM: int = driver.Device(0).multiprocessor_count

    blkDimx: int = 256
    grdDimx: int = 32*numSM

    blkDim = (blkDimx,1,1)
    grdDim = (grdDimx,1,1)

    module = SourceModule(kernel, options=['-O3','-lineinfo','--use_fast_math'])
    dxfield = module.get_function("dxfield")
    exfield = module.get_function("exfield")
    hyfield = module.get_function("hyfield")

    for ts in map(numpy.int32,range(1,ns+1)):
        dxfield(ts, nx, hy, dx, block=blkDim, grid=grdDim)
        exfield(nx,*md, ix, dx, ex, block=blkDim, grid=grdDim)
        hyfield(nx, bc, ex, hy, block=blkDim, grid=grdDim)

    driver.Context.synchronize()

    visualize(ns, nx, epsr, sigma, md.nax.get(), ex.get())


if __name__ == "__main__":
    main()
