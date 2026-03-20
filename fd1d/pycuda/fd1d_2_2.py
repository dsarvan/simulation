#!/usr/bin/env python
# File: fdtd22.py
# Name: D.Saravanan
# Date: 07/12/2021

""" Simulation of a pulse striking a dielectric medium and implements
the discrete Fourier transform with a Gaussian pulse as its source """

import numpy
from pycuda import driver
from pycuda import autoinit
from pycuda import gpuarray
from matplotlib import pyplot
from typing import NamedTuple
from numpy import hypot, arctan2
from pycuda.compiler import SourceModule

pyplot.style.use("classic")
pyplot.style.use("pyplot.mplstyle")

type F32Array = gpuarray.GPUArray


def visualize(ns: int, nx: int, epsr: float, sigma: float, nax: F32Array, ex: F32Array):
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=True)
    fig.suptitle(r"FDTD simulation of a pulse striking dielectric material")
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
    fig.savefig("fdtd22.png", dpi=100)


def amplitude(ns: int, nx: int, epsr: float, sigma: float, nax: F32Array, amp: F32Array):
    fig, axs = pyplot.subplots(figsize=(8,3), frameon=True)
    fig.suptitle(r"The discrete Fourier transform with pulse as its source")
    medium = numpy.where(1.0/nax-1)[0] if epsr > 1 else (1.0/nax-1)
    axs.plot(range(nx), amp, color="k", linewidth=1.0)
    axs.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    axs.set(xlim=(0, nx-1), ylim=(-0.2, 2.2))
    axs.set(xticks=range(0, nx+1, int(numpy.ceil(nx/500)*25)))
    axs.set(xlabel=r"$z\;(cm)$", ylabel=r"$Amplitude$")
    axs.text(0.02, 0.90, rf"$T$ = {ns}", transform=axs.transAxes)
    axs.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=axs.transAxes)
    axs.text(0.85, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=axs.transAxes)
    fig.subplots_adjust(bottom=0.2, hspace=0.45)
    fig.savefig("fdtdamplt22.png", dpi=100)


Medium = NamedTuple('Medium',[
    ('nax',F32Array), ('nbx',F32Array),
])


FTrans = NamedTuple('FTrans',[
    ('rpt',F32Array), ('ipt',F32Array),
    ('rin',F32Array), ('iin',F32Array),
])


kernel = """
#define idx blockIdx.x*blockDim.x+threadIdx.x
#define stx blockDim.x*gridDim.x


typedef struct {
    float *nax, *nbx;
} Medium;


typedef struct {
    float *rpt, *ipt;
    float *rin, *iin;
} FTrans;


__device__
float gaussian(int ts, int t0, float sigma) {
    return expf(-0.5f*(ts - t0)/sigma*(ts - t0)/sigma);
}


__global__
void fourier(int ts, int nf, int nx, float dt, float *freq, float *ex, FTrans ft) {
    for (int n = 0; n < nf; n++) {
        /* calculate the Fourier transform of input source */
        if (idx == 0) ft.rin[n] += cosf(2*M_PI*freq[n]*dt*ts) * ex[10];
        if (idx == 0) ft.iin[n] -= sinf(2*M_PI*freq[n]*dt*ts) * ex[10];
        for (int i = idx; i < nx; i += stx) {
            /* calculate the Fourier transform of Ex field */
            ft.rpt[n*nx+i] += cosf(2*M_PI*freq[n]*dt*ts) * ex[i];
            ft.ipt[n*nx+i] -= sinf(2*M_PI*freq[n]*dt*ts) * ex[i];
        }
    }
}


__global__
void dxfield(int ts, int nx, float *hy, float *dx) {
    /* calculate the electric flux density Dx */
    for (int i = idx+1; i < nx; i += stx)
        dx[i] += 0.5f * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse at the low end */
    if (idx == 1) dx[1] += gaussian(ts, 50, 10.0f);
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
    sigma: float = 0.0  # conductivity (S/m)
    md: Medium = dielectric(nx, dt, epsr, sigma)

    # frequency 100 MHz, 200 MHz, 500 MHz
    freq = gpuarray.to_gpu(numpy.array((100e6,200e6,500e6), numpy.float32))
    nf = numpy.int32(len(freq))  # number of frequencies

    ft = FTrans(
        rpt = gpuarray.zeros((nf,nx), numpy.float32),
        ipt = gpuarray.zeros((nf,nx), numpy.float32),
        rin = gpuarray.zeros((nf, 1), numpy.float32),
        iin = gpuarray.zeros((nf, 1), numpy.float32),
    )

    amplt = numpy.zeros((nf,nx), numpy.float32)
    phase = numpy.zeros((nf,nx), numpy.float32)

    blkDimx: int = 256
    grdDimx: int = int((nx+blkDimx-1)/blkDimx)

    blkDim = (blkDimx,1,1)
    grdDim = (grdDimx,1,1)

    module = SourceModule(kernel, options=['-O3','-lineinfo','--use_fast_math'])
    fourier = module.get_function("fourier")
    dxfield = module.get_function("dxfield")
    exfield = module.get_function("exfield")
    hyfield = module.get_function("hyfield")

    for ts in map(numpy.int32,range(1,ns+1)):
        dxfield(ts, nx, hy, dx, block=blkDim, grid=grdDim)
        exfield(nx,*md, ix, dx, ex, block=blkDim, grid=grdDim)
        fourier(ts, nf, nx, dt, freq, ex,*ft, block=blkDim, grid=grdDim)
        hyfield(nx, bc, ex, hy, block=blkDim, grid=grdDim)

    driver.Context.synchronize()

    # calculate the amplitude and phase at each frequency
    amplt = 1/hypot(ft.rin.get(),ft.iin.get()) * hypot(ft.rpt.get(),ft.ipt.get())
    phase = arctan2(ft.ipt.get(),ft.rpt.get()) - arctan2(ft.iin.get(),ft.rin.get())

    visualize(ns, nx, epsr, sigma, md.nax.get(), ex.get())
    amplitude(ns, nx, epsr, sigma, md.nax.get(), amplt[2])


if __name__ == "__main__":
    main()
