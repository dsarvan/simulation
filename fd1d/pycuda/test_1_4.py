#!/usr/bin/env python
# File: test14.py
# Name: D.Saravanan
# Date: 22/10/2021

""" Simulation of a propagating sinusoidal striking a dielectric medium """

import numpy
from pycuda import driver
from pycuda import autoinit
from pycuda import gpuarray
from matplotlib import pyplot
from pycuda.compiler import SourceModule

pyplot.style.use("classic")
pyplot.style.use("pyplot.mplstyle")

type F32Array = gpuarray.GPUArray


def visualize(ns: int, nx: int, epsr: float, cb: F32Array, ex: F32Array):
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=True)
    fig.suptitle(r"FDTD simulation of a sinusoidal striking dielectric material")
    medium = numpy.where(0.5/cb-1)[0] if epsr > 1 else (0.5/cb-1)
    axs.plot(range(nx), ex, color="k", linewidth=1.0)
    axs.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    axs.set(xticks=range(0, nx+1, int(numpy.ceil(nx/500)*50)))
    axs.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    axs.text(0.02, 0.90, rf"$T$ = {ns}", transform=axs.transAxes)
    axs.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("test14.png", dpi=100)


kernel = """
#define idx blockIdx.x*blockDim.x+threadIdx.x
#define stx blockDim.x*gridDim.x


__device__
float sinusoidal(int ts, float ds, float freq) {
    float dt = ds/6e8f;  /* time step (s) */
    return sinf(2*M_PI*freq*dt*ts);
}


__global__
void exfield(int ts, int nx, float *cb, float *hy, float *ex) {
    /* calculate the Ex field */
    for (int i = idx+1; i < nx; i += stx)
        ex[i] += cb[i] * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    if (idx == 1) ex[1] += sinusoidal(ts, 0.01f, 700e6f);
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


def dielectric(nx: int, epsr: float):
    cb = gpuarray.empty(nx, numpy.float32).fill(0.5)
    cb[nx//2:] = 0.5/epsr
    return cb


def main():

    nx = numpy.int32(38000)  # number of grid points
    ns = numpy.int32(40000)  # number of time steps

    ex = gpuarray.zeros(nx, numpy.float32)
    hy = gpuarray.zeros(nx, numpy.float32)

    bc = gpuarray.zeros(4, numpy.float32)

    ds: float = 0.01  # spatial step (m)
    dt: float = ds/6e8  # time step (s)
    epsr: float = 4.0  # relative permittivity
    cb: F32Array = dielectric(nx, epsr)

    numSM: int = driver.Device(0).multiprocessor_count

    blkDimx: int = 256
    grdDimx: int = 32*numSM

    blkDim = (blkDimx,1,1)
    grdDim = (grdDimx,1,1)

    module = SourceModule(kernel)
    exfield = module.get_function("exfield")
    hyfield = module.get_function("hyfield")

    stime = driver.Event()
    ntime = driver.Event()

    stime.record()

    for ts in map(numpy.int32,range(1,ns+1)):
        exfield(ts, nx, cb, hy, ex, block=blkDim, grid=grdDim)
        hyfield(nx, bc, ex, hy, block=blkDim, grid=grdDim)

    driver.Context.synchronize()

    ntime.record()
    ntime.synchronize()

    time = stime.time_till(ntime)*1e-3
    print(f"Total compute time on GPU: {time:.3f} s")

    print(ex[0:50])
    visualize(ns, nx, epsr, cb.get(), ex.get())


if __name__ == "__main__":
    main()
