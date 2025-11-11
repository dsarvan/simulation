#!/usr/bin/env python
# File: test_3_4.py
# Name: D.Saravanan
# Date: 20/01/2022

""" Simulation of a plane wave pulse striking a dielectric medium in the transverse
magnetic (TM) mode with PML and implements the discrete Fourier transform analysis """

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from numpy import pi, exp, hypot, sqrt, sin, cos, arctan2
from collections import namedtuple
import numba as nb
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

plt.style.use("classic")
plt.style.use("../pyplot.mplstyle")


def surfaceplot(ns: int, nx: int, ny: int, npml: int, epsr: float, naz: np.ndarray, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    fig.suptitle(r"FDTD simulation of plane wave striking dielectric material")
    yv, xv = np.meshgrid(range(ny), range(nx)); ezmax = np.abs(ez).max()
    medium = np.stack([1.0/naz-1.0]*1, axis=2); levels = [0.50,1.50]
    pmlmsk = ((xv < npml)|(xv >= nx-npml)|(yv < npml)|(yv >= ny-npml))
    ax.plot_surface(xv, yv, ez, rstride=1, cstride=1, cmap="gray", alpha=0.5, lw=10/nx)
    ax.voxels(medium, color="y", edgecolor="k", shade=True, alpha=0.5, linewidths=1/nx)
    ax.contourf(xv, yv, pmlmsk, levels, offset=0, colors="k", alpha=0.40)
    ax.text2D(0.1, 0.7, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, 1))
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$", zlabel=r"$E_z\;(V/m)$")
    ax.zaxis.set_rotate_label(False); ax.view_init(elev=20.0, azim=45)
    plt.savefig("test_surface_3_4.png", dpi=100)


def contourplot(ns: int, nx: int, ny: int, npml: int, epsr: float, naz: np.ndarray, ez: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"FDTD simulation of plane wave striking dielectric material")
    yv, xv = np.meshgrid(range(ny), range(nx)); ezmax = np.abs(ez).max()
    medium = 1.0/naz-1.0; levels = np.linspace(-ezmax, ezmax, int(2/0.04))
    pmlmsk = ((xv < npml)|(xv >= nx-npml)|(yv < npml)|(yv >= ny-npml))
    ax.contour(xv, yv, ez, levels, cmap="gray", alpha=1.0, linewidths=1.5)
    ax.contourf(xv, yv, medium, [0.001,medium.max()], colors="y", alpha=0.7)
    ax.contourf(xv, yv, pmlmsk, levels=[0.50,1.50], colors="k", alpha=0.40)
    ax.set(xlim=(0, nx-1), ylim=(0, ny-1), aspect="equal")
    ax.set(xlabel=r"$x\;(cm)$", ylabel=r"$y\;(cm)$")
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("test_contour_3_4.png", dpi=100)


def amplitudeplot(ns: int, ny: int, rgrid: int, epsr: float, sigma: float, amp: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4,4), gridspec_kw={"hspace":0.2})
    fig.suptitle(r"The discrete Fourier transform with plane wave as its source")
    ax.plot(range(-ny//2, ny//2), amp, color="k", linewidth=1.0)
    ax.set(xlim=(-rgrid-1, rgrid+1), ylim=(0.0, 1.0))
    ax.set(xticks=[-rgrid, -rgrid//2, 0, rgrid//2, rgrid])
    ax.set(xlabel=r"$y\;(cm)$", ylabel=r"$Amplitude$")
    ax.text(0.03, 0.90, rf"$T$ = {ns}", transform=ax.transAxes)
    ax.text(0.80, 0.90, rf"$\epsilon_r$ = {epsr}", transform=ax.transAxes)
    ax.text(0.75, 0.80, rf"$\sigma$ = {sigma} $S/m$", transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.savefig("test_amplitude_3_4.png", dpi=100)


medium = namedtuple('medium', (
    'naz', 'nbz',
))


ftrans = namedtuple('ftrans', (
    'r_pt', 'i_pt',
    'r_in', 'i_in',
))


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
    float *naz, *nbz;
} medium;


typedef struct {
    float *r_pt, *i_pt;
    float *r_in, *i_in;
} ftrans;


typedef struct {
    float *fx1, *fx2, *fx3;
    float *fy1, *fy2, *fy3;
    float *gx2, *gx3;
    float *gy2, *gy3;
} pmlayer;


__device__
float gaussian(int t, int t0, float sigma) {
    return expf(-0.5f*(t - t0)/sigma*(t - t0)/sigma);
}


__global__
void fourier(int t, int nf, int nx, int ny, float dt, float *freq, float *ezi, float *ez, ftrans *ft) {
    for (int n = threadIdx.z; n < nf; n += blockDim.z) {
        /* calculate the Fourier transform of input source */
        if (idx == 0 && idy == 0) ft->r_in[n] += cosf(2*M_PI*freq[n]*dt*t) * ezi[6];
        if (idx == 0 && idy == 0) ft->i_in[n] -= sinf(2*M_PI*freq[n]*dt*t) * ezi[6];
        for (int i = idy; i < nx; i += sty) {
            for (int j = idx; j < ny; j += stx) {
                /* calculate the Fourier transform of Ex field */
                int m = n*nx*ny+i*ny+j;
                ft->r_pt[m] += cosf(2*M_PI*freq[n]*dt*t) * ez[i*ny+j];
                ft->i_pt[m] -= sinf(2*M_PI*freq[n]*dt*t) * ez[i*ny+j];
            }
        }
    }
}


__global__
void ezinct(int ny, float *ezi, float *hxi, float *bc) {
    /* calculate the incident Ez */
    for (int j = idx+1; j < ny; j += stx) {
        ezi[j] += 0.5f * (hxi[j-1] - hxi[j]);
    }
    /* absorbing boundary conditions */
    if (idx == 0) ezi[0] = bc[0], bc[0] = bc[1], bc[1] = ezi[1];
    if (idx == 0) ezi[ny-1] = bc[3], bc[3] = bc[2], bc[2] = ezi[ny-2];
}


__global__
void dfield(int t, int nx, int ny, pmlayer *pml, float *ezi, float *dz, float *hx, float *hy) {
    /* calculate the electric flux density Dz */
    for (int i = idy+1; i < nx; i += sty) {
        for (int j = idx+1; j < ny; j += stx) {
            int n = i*ny+j;
            dz[n] = pml->gx3[i] * pml->gy3[j] * dz[n] + pml->gx2[i] * pml->gy2[j] * 0.5f * (hy[n] - hy[n-ny] - hx[n] + hx[n-1]);
        }
    }
    /* put a Gaussian pulse at the low end */
    if (idy == 0 && idx == 3) ezi[3] = gaussian(t, 20, 8.0f);
}


__global__
void inctdz(int nx, int ny, int npml, float *hxi, float *dz) {
    /* incident Dz values */
    for (int i = idx+npml-1; i <= nx-npml; i += stx) {
        dz[i*ny+(npml-1)] += 0.5f * hxi[npml-2];
        dz[i*ny+(ny-npml)] -= 0.5f * hxi[ny-npml];
    }
}


__global__
void efield(int nx, int ny, medium *md, float *dz, float *iz, float *ez) {
    /* calculate the Ez field from Dz */
    for (int i = idy; i < nx; i += sty) {
        for (int j = idx; j < ny; j += stx) {
            int n = i*ny+j;
            ez[n] = md->naz[n] * (dz[n] - iz[n]);
            iz[n] += md->nbz[n] * ez[n];
        }
    }
}


__global__
void hxinct(int ny, float *ezi, float *hxi) {
    /* calculate the incident Hx */
    for (int j = idx; j < ny-1; j += stx) {
        hxi[j] += 0.5f * (ezi[j] - ezi[j+1]);
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


__global__
void incthx(int nx, int ny, int npml, float *ezi, float *hx) {
    /* incident Hx values */
    for (int i = idx+npml-1; i <= nx-npml; i += stx) {
        hx[i*ny+(npml-2)] += 0.5f * ezi[npml-1];
        hx[i*ny+(ny-npml)] -= 0.5f * ezi[ny-npml];
    }
}


__global__
void incthy(int nx, int ny, int npml, float *ezi, float *hy) {
    /* incident Hy values */
    for (int j = idx+npml-1; j <= ny-npml; j += stx) {
        hy[(npml-2)*ny+j] -= 0.5f * ezi[j];
        hy[(nx-npml)*ny+j] += 0.5f * ezi[j];
    }
}
"""


@nb.jit(nopython=True, fastmath=True)
def dielectric(nx: int, ny: int, npml: int, rgrid: int, dt: float, epsr: float, sigma: float) -> medium:
    md = medium(
        naz = np.full((nx, ny), 1.0, dtype=np.float32),
        nbz = np.full((nx, ny), 0.0, dtype=np.float32),
    )
    eps0: float = 8.854e-12  # vacuum permittivity (F/m)
    for i in range(npml, nx-npml):
        for j in range(npml, ny-npml):
            epsn: float = 1.0
            cond: float = 0.0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    x: float = nx/2-1-i+m/3
                    y: float = ny/2-1-j+n/3
                    d: float = sqrt(x**2 + y**2)
                    if d <= rgrid:
                        epsn += (epsr - 1)/9
                        cond += sigma/9
            md.naz[i,j] = 1/(epsn + cond*dt/eps0)
            md.nbz[i,j] = cond*dt/eps0
    return md


def pmlparam(nx: int, ny: int, npml: int, pml: pmlayer) -> None:
    """ calculate the two-dimensional perfectly matched layer (PML) parameters """
    for n in range(npml):
        xm: float = 0.33*((npml-n)/npml)**3
        xn: float = 0.33*((npml-n-0.5)/npml)**3
        pml.fx1[n] = pml.fx1[nx-2-n] = pml.fy1[n] = pml.fy1[ny-2-n] = xn
        pml.fx2[n] = pml.fx2[nx-2-n] = pml.fy2[n] = pml.fy2[ny-2-n] = 1/(1+xn)
        pml.gx2[n] = pml.gx2[nx-1-n] = pml.gy2[n] = pml.gy2[ny-1-n] = 1/(1+xm)
        pml.fx3[n] = pml.fx3[nx-2-n] = pml.fy3[n] = pml.fy3[ny-2-n] = (1-xn)/(1+xn)
        pml.gx3[n] = pml.gx3[nx-1-n] = pml.gy3[n] = pml.gy3[ny-1-n] = (1-xm)/(1+xm)


def main():

    nx = np.int32(1024)  # number of grid points
    ny = np.int32(1024)  # number of grid points

    ns = np.int32(5000)  # number of time steps

    ezi = gpuarray.zeros(ny, dtype=np.float32)
    hxi = gpuarray.zeros(ny, dtype=np.float32)

    dz = gpuarray.zeros(nx*ny, dtype=np.float32)
    ez = gpuarray.zeros(nx*ny, dtype=np.float32)
    iz = gpuarray.zeros(nx*ny, dtype=np.float32)
    hx = gpuarray.zeros(nx*ny, dtype=np.float32)
    hy = gpuarray.zeros(nx*ny, dtype=np.float32)

    ihx = gpuarray.zeros(nx*ny, dtype=np.float32)
    ihy = gpuarray.zeros(nx*ny, dtype=np.float32)

    bc = gpuarray.zeros(4, dtype=np.float32)

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
        pml.fx1.ptr, pml.fx2.ptr, pml.fx3.ptr,
        pml.fy1.ptr, pml.fy2.ptr, pml.fy3.ptr,
        pml.gx2.ptr, pml.gx3.ptr,
        pml.gy2.ptr, pml.gy3.ptr,
    ], dtype=np.uint64)).gpudata

    npml: int = np.int32(80)  # pml thickness
    pmlparam(nx, ny, npml, pml)

    ds: float = np.float32(0.01)  # spatial step (m)
    dt: float = np.float32(ds/6e8)  # time step (s)
    epsr: float = 30.0  # relative permittivity
    sigma: float = 0.30  # conductivity (S/m)
    radius: float = 1.50  # cylinder radius (m)
    rgrid: int = int(radius/ds-1)  # radius in FDTD grid cell units
    md: medium = dielectric(nx, ny, npml, rgrid, dt, epsr, sigma)
    md: medium = medium(gpuarray.to_gpu(md.naz), gpuarray.to_gpu(md.nbz))

    mdptr = gpuarray.to_gpu(np.array([
        md.naz.ptr, md.nbz.ptr,
    ], dtype=np.uint64)).gpudata

    # frequency 50 MHz, 300 MHz, 700 MHz
    freq = gpuarray.to_gpu(np.array((50e6, 300e6, 700e6), dtype=np.float32))
    nf = np.int32(len(freq))  # number of frequencies

    ft = ftrans(
        r_pt = gpuarray.zeros((nf, nx, ny), dtype=np.float32),
        i_pt = gpuarray.zeros((nf, nx, ny), dtype=np.float32),
        r_in = gpuarray.zeros(nf, dtype=np.float32),
        i_in = gpuarray.zeros(nf, dtype=np.float32),
    )

    ftptr = gpuarray.to_gpu(np.array([
        ft.r_pt.ptr, ft.i_pt.ptr,
        ft.r_in.ptr, ft.i_in.ptr,
    ], dtype=np.uint64)).gpudata

    amplt = np.zeros((nf, ny), dtype=np.float32)
    phase = np.zeros((nf, ny), dtype=np.float32)

    blockDimx: int = 16
    blockDimy: int = 16
    gridDimx: int = int((ny+blockDimx-1)/blockDimx)
    gridDimy: int = int((nx+blockDimy-1)/blockDimy)

    gridDim = (gridDimx,gridDimy,1)
    blockDim = (blockDimx,blockDimy,1)

    mod = SourceModule(kernel)
    fourier = mod.get_function("fourier")
    ezinct = mod.get_function("ezinct")
    dfield = mod.get_function("dfield")
    inctdz = mod.get_function("inctdz")
    efield = mod.get_function("efield")
    hxinct = mod.get_function("hxinct")
    hfield = mod.get_function("hfield")
    incthx = mod.get_function("incthx")
    incthy = mod.get_function("incthy")

    stime = drv.Event()
    ntime = drv.Event()

    stime.record()

    for t in np.arange(1, ns+1).astype(np.int32):
        ezinct(ny, ezi, hxi, bc, grid=(int((ny+255)/256),1,1), block=(256,1,1))
        dfield(t, nx, ny, pmlptr, ezi, dz, hx, hy, grid=gridDim, block=blockDim)
        inctdz(nx, ny, npml, hxi, dz, grid=(int((nx+255)/256),1,1), block=(256,1,1))
        efield(nx, ny, mdptr, dz, iz, ez, grid=gridDim, block=blockDim)
        fourier(t, nf, nx, ny, dt, freq, ezi, ez, ftptr, grid=gridDim, block=(16,16,4))
        hxinct(ny, ezi, hxi, grid=(int((ny+255)/256),1,1), block=(256,1,1))
        hfield(nx, ny, pmlptr, ez, ihx, ihy, hx, hy, grid=gridDim, block=blockDim)
        incthx(nx, ny, npml, ezi, hx, grid=(int((nx+255)/256),1,1), block=(256,1,1))
        incthy(nx, ny, npml, ezi, hy, grid=(int((ny+255)/256),1,1), block=(256,1,1))

    drv.Context.synchronize()

    ft_r_pt, ft_i_pt = ft.r_pt.get(), ft.i_pt.get()
    ft_r_in, ft_i_in = ft.r_in.get(), ft.i_in.get()

    # calculate the amplitude and phase at each frequency
    for n in range(0, nf):
        for j in range(npml-1, ny-npml+1):
            m = (n,j); k = (n,nx//2-1,j)
            amplt[m] = 1/hypot(ft_r_in[n],ft_i_in[n]) * hypot(ft_r_pt[k],ft_i_pt[k])
            phase[m] = arctan2(ft_i_pt[k],ft_r_pt[k]) - arctan2(ft_i_in[n],ft_r_in[n])

    ntime.record()
    ntime.synchronize()

    time = stime.time_till(ntime)*1e-3
    print(f"Total compute time on GPU: {time:.3f} s")

    print(ez[2*ny:2*ny+50])
    print(amplt[2][0:ny-50])
    surfaceplot(ns, nx, ny, npml, epsr, md.naz.get(), ez.get().reshape(nx,ny))
    contourplot(ns, nx, ny, npml, epsr, md.naz.get(), ez.get().reshape(nx,ny))
    amplitudeplot(ns, ny, rgrid, epsr, sigma, amplt[2])


if __name__ == "__main__":
    main()
