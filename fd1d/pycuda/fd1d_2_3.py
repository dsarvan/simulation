#!/usr/bin/env python
# File: fd1d_2_3.py
# Name: D.Saravanan
# Date: 10/01/2022

""" Simulation of a pulse striking a frequency-dependent dielectric material and
implements the discrete Fourier transform with a Gaussian pulse as its source """

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp, hypot, sin, cos, arctan2
from collections import namedtuple

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def visualize(ns: int, nx: int, epsr: float, sigma: float, nax: np.ndarray, ex: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of a pulse striking Debye dielectric material")
    medium = np.where(1.0/nax-1)[0] if epsr > 1 else (1.0/nax-1)
    ax.plot(range(nx), ex, color="k", linewidth=1.0)
    ax.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-1.2, 1.2))
    ax.set(xticks=range(0, nx+1, int(np.ceil(nx/500)*25)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$E_x\;(V/m)$")
    ax.text(0.02, 0.90, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=ax.transAxes)
    ax.text(0.85, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd1d_2_3.png", dpi=100)


def amplitude(ns: int, nx: int, epsr: float, sigma: float, nax: np.ndarray, amp: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8,3), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"The discrete Fourier transform with pulse as its source")
    medium = np.where(1.0/nax-1)[0] if epsr > 1 else (1.0/nax-1)
    ax.plot(range(nx), amp, color="k", linewidth=1.0)
    ax.axvspan(medium[0], medium[-1], color="y", alpha=0.3)
    ax.set(xlim=(0, nx-1), ylim=(-0.2, 2.2))
    ax.set(xticks=range(0, nx+1, int(np.ceil(nx/500)*25)))
    ax.set(xlabel=r"$z\;(cm)$", ylabel=r"$Amplitude$")
    ax.text(0.02, 0.90, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.text(0.90, 0.90, rf"$\epsilon_r$ = {epsr}", transform=ax.transAxes)
    ax.text(0.85, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("fd1d_amp_2_3.png", dpi=100)


medium = namedtuple('medium', (
    'nax', 'nbx',
    'ncx', 'ndx',
))


ftrans = namedtuple('ftrans', (
    'r_pt', 'i_pt',
    'r_in', 'i_in',
))


kernel = """
#define idx blockIdx.x*blockDim.x+threadIdx.x
#define stx blockDim.x*gridDim.x


typedef struct {
    float *nax, *nbx;
    float *ncx, *ndx;
} medium;


typedef struct {
    float *r_pt, *i_pt;
    float *r_in, *i_in;
} ftrans;


__device__
float gaussian(int t, int t0, float sigma) {
    return expf(-0.5f*(t - t0)/sigma*(t - t0)/sigma);
}


__global__
void fourier(int t, int nf, int nx, float dt, float *freq, float *ex, ftrans *ft) {
    for (int n = threadIdx.y; n < nf; n += blockDim.y) {
        /* calculate the Fourier transform of input source */
        if (idx == 0) ft->r_in[n] += cosf(2*M_PI*freq[n]*dt*t) * ex[10];
        if (idx == 0) ft->i_in[n] -= sinf(2*M_PI*freq[n]*dt*t) * ex[10];
        for (int i = idx; i < nx; i += stx) {
            /* calculate the Fourier transform of Ex field */
            int m = n*nx+i;
            ft->r_pt[m] += cosf(2*M_PI*freq[n]*dt*t) * ex[i];
            ft->i_pt[m] -= sinf(2*M_PI*freq[n]*dt*t) * ex[i];
        }
    }
}


__global__
void dxfield(int t, int nx, float *dx, float *hy) {
    /* calculate the electric flux density Dx */
    for (int i = idx+1; i < nx; i += stx)
        dx[i] += 0.5f * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse at the low end */
    if (idx == 1) dx[1] += gaussian(t, 50, 10.0f);
}


__global__
void exfield(int nx, medium *md, float *dx, float *ix, float *sx, float *ex) {
    /* calculate the Ex field from Dx */
    for (int i = idx+1; i < nx; i += stx) {
        ex[i] = md->nax[i] * (dx[i] - ix[i] - md->ncx[i] * sx[i]);
        ix[i] += md->nbx[i] * ex[i];
        sx[i] = md->ncx[i] * sx[i] + md->ndx[i] * ex[i];
    }
}


__global__
void hyfield(int nx, float *ex, float *hy, float *bc) {
    /* absorbing boundary conditions */
    if (idx == 0) ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    if (idx == nx-1) ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = idx; i < nx-1; i += stx)
        hy[i] += 0.5f * (ex[i] - ex[i+1]);
}
"""


def dielectric(nx: int, dt: float, chi: float, tau: float, epsr: float, sigma: float) -> medium:
    md = medium(
        nax = gpuarray.empty(nx, dtype=np.float32).fill(1.0),
        nbx = gpuarray.empty(nx, dtype=np.float32).fill(0.0),
        ncx = gpuarray.empty(nx, dtype=np.float32).fill(0.0),
        ndx = gpuarray.empty(nx, dtype=np.float32).fill(0.0),
    )
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    md.nax[nx//2:] = 1/(epsr + sigma*dt/eps0 + chi*dt/tau)
    md.nbx[nx//2:] = sigma*dt/eps0
    md.ncx[nx//2:] = exp(-dt/tau)
    md.ndx[nx//2:] = chi*dt/tau
    return md


def main():

    nx = np.int32(512)  # number of grid points
    ns = np.int32(740)  # number of time steps

    dx = gpuarray.zeros(nx, dtype=np.float32)
    ex = gpuarray.zeros(nx, dtype=np.float32)
    ix = gpuarray.zeros(nx, dtype=np.float32)
    sx = gpuarray.zeros(nx, dtype=np.float32)
    hy = gpuarray.zeros(nx, dtype=np.float32)

    bc = gpuarray.zeros(4, dtype=np.float32)

    ds = np.float32(0.01)  # spatial step (m)
    dt = np.float32(ds/6e8)  # time step (s)
    chi: float = 2.0  # relaxation susceptibility
    tau: float = 0.001e-6  # relaxation time (s)
    epsr: float = 2.0  # relative permittivity
    sigma: float = 0.01  # conductivity (S/m)
    md: medium = dielectric(nx, dt, chi, tau, epsr, sigma)

    mdptr = gpuarray.to_gpu(np.array([
        md.nax.ptr, md.nbx.ptr,
        md.ncx.ptr, md.ndx.ptr,
    ], dtype=np.uint64)).gpudata

    # frequency 50 MHz, 200 MHz, 500 MHz
    freq = gpuarray.to_gpu(np.array((50e6, 200e6, 500e6), dtype=np.float32))
    nf = np.int32(len(freq))  # number of frequencies

    ft = ftrans(
        r_pt = gpuarray.zeros((nf, nx), dtype=np.float32),
        i_pt = gpuarray.zeros((nf, nx), dtype=np.float32),
        r_in = gpuarray.zeros((nf, 1), dtype=np.float32),
        i_in = gpuarray.zeros((nf, 1), dtype=np.float32),
    )

    ftptr = gpuarray.to_gpu(np.array([
        ft.r_pt.ptr, ft.i_pt.ptr,
        ft.r_in.ptr, ft.i_in.ptr,
    ], dtype=np.uint64)).gpudata

    amplt = np.zeros((nf, nx), dtype=np.float32)
    phase = np.zeros((nf, nx), dtype=np.float32)

    blockDimx: int = 256
    gridDimx: int = int((nx+blockDimx-1)/blockDimx)

    gridDim = (gridDimx,1,1)
    blockDim = (blockDimx,1,1)

    mod = SourceModule(kernel)
    fourier = mod.get_function("fourier")
    dxfield = mod.get_function("dxfield")
    exfield = mod.get_function("exfield")
    hyfield = mod.get_function("hyfield")

    for t in np.arange(1, ns+1).astype(np.int32):
        dxfield(t, nx, dx, hy, grid=gridDim, block=blockDim)
        exfield(nx, mdptr, dx, ix, sx, ex, grid=gridDim, block=blockDim)
        fourier(t, nf, nx, dt, freq, ex, ftptr, grid=gridDim, block=(256,4,1))
        hyfield(nx, ex, hy, bc, grid=gridDim, block=blockDim)

    drv.Context.synchronize()

    # calculate the amplitude and phase at each frequency
    amplt = 1/hypot(ft.r_in.get(),ft.i_in.get()) * hypot(ft.r_pt.get(),ft.i_pt.get())
    phase = arctan2(ft.i_pt.get(),ft.r_pt.get()) - arctan2(ft.i_in.get(),ft.r_in.get())

    visualize(ns, nx, epsr, sigma, md.nax.get(), ex.get())
    amplitude(ns, nx, epsr, sigma, md.nax.get(), amplt[2])


if __name__ == "__main__":
    main()
