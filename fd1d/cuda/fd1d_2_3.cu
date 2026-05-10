/* File: fdtd23.cu
 * Name: D.Saravanan
 * Date: 10/01/2022
 * Simulation of a pulse striking a frequency-dependent dielectric material and
 * implements the discrete Fourier transform with a Gaussian pulse as its source
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx (blockIdx.x*blockDim.x+threadIdx.x)
#define stx (blockDim.x*gridDim.x)


typedef struct {
    float *rpt, *ipt;
    float *rin, *iin;
} FTrans;


typedef struct {
    float *nax, *nbx;
    float *ncx, *ndx;
} Medium;


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
void exfield(int nx, Medium md, float *ix, float *sx, float *dx, float *ex) {
    /* calculate the Ex field from Dx */
    for (int i = idx+1; i < nx; i += stx) {
        ex[i] = md.nax[i] * (dx[i] - ix[i] - md.ncx[i] * sx[i]);
        ix[i] += md.nbx[i] * ex[i];
        sx[i] = md.ncx[i] * sx[i] + md.ndx[i] * ex[i];
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


Medium dielectric(int nx, float dt, float chi, float tau, float epsr, float sigma) {
    Medium md;
    cudaMallocManaged(&md.nax, nx*sizeof(*md.nax));
    cudaMallocManaged(&md.nbx, nx*sizeof(*md.nbx));
    cudaMallocManaged(&md.ncx, nx*sizeof(*md.ncx));
    cudaMallocManaged(&md.ndx, nx*sizeof(*md.ndx));
    for (int i = 0; i < nx; md.nax[i] = 1.0f, i++);
    for (int i = 0; i < nx; md.nbx[i] = 0.0f, i++);
    for (int i = 0; i < nx; md.ncx[i] = 0.0f, i++);
    for (int i = 0; i < nx; md.ndx[i] = 0.0f, i++);
    float eps0 = 8.854e-12f;  /* vacuum permittivity (F/m) */
    for (int i = nx/2; i < nx; i++) {
        md.nax[i] = 1/(epsr + sigma*dt/eps0 + chi*dt/tau);
        md.nbx[i] = sigma*dt/eps0;
        md.ncx[i] = expf(-dt/tau);
        md.ndx[i] = chi*dt/tau;
    }
    return md;
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 740;  /* number of time steps */

    float *dx, *ex, *ix, *sx, *hy;
    cudaMallocManaged(&dx, nx*sizeof(*dx));
    cudaMallocManaged(&ex, nx*sizeof(*ex));
    cudaMallocManaged(&ix, nx*sizeof(*ix));
    cudaMallocManaged(&sx, nx*sizeof(*sx));
    cudaMallocManaged(&hy, nx*sizeof(*hy));

    for (int i = 0; i < nx; i++) {
        dx[i] = 0.0f;
        ex[i] = 0.0f;
        ix[i] = 0.0f;
        sx[i] = 0.0f;
        hy[i] = 0.0f;
    }

    float *bc;
    cudaMallocManaged(&bc, 4*sizeof(*bc));
    for (int i = 0; i < 4; bc[i] = 0.0f, i++);

    float ds = 0.01f;  /* spatial step (m) */
    float dt = ds/6e8f;  /* time step (s) */
    float chi = 2.0f;  /* relaxation susceptibility */
    float tau = 0.001e-6f;  /* relaxation time (s) */
    float epsr = 2.0f;  /* relative permittivity */
    float sigma = 0.01f;  /* conductivity (S/m) */
    Medium md = dielectric(nx, dt, chi, tau, epsr, sigma);

    int nf = 3;  /* number of frequencies */
    /* frequency 50 MHz, 200 MHz, 500 MHz */
    float *freq;
    cudaMallocManaged(&freq, nf*sizeof(*freq));
    freq[0] = 50e6f; freq[1] = 200e6f; freq[2] = 500e6f;

    FTrans ft;
    cudaMallocManaged(&ft.rpt, nf*nx*sizeof(*ft.rpt));
    cudaMallocManaged(&ft.ipt, nf*nx*sizeof(*ft.ipt));
    cudaMallocManaged(&ft.rin, nf*sizeof(*ft.rin));
    cudaMallocManaged(&ft.iin, nf*sizeof(*ft.iin));

    for (int i = 0; i < nf*nx; i++) {
        ft.rpt[i] = 0.0f;
        ft.ipt[i] = 0.0f;
    }

    for (int i = 0; i < nf; i++) {
        ft.rin[i] = 0.0f;
        ft.iin[i] = 0.0f;
    }

    float *amplt, *phase;
    cudaMallocManaged(&amplt, nf*nx*sizeof(*amplt));
    cudaMallocManaged(&phase, nf*nx*sizeof(*phase));

    for (int i = 0; i < nf*nx; i++) {
        amplt[i] = 0.0f;
        phase[i] = 0.0f;
    }

    dim3 grdDim, blkDim;
    blkDim.x = 256;
    grdDim.x = (nx+blkDim.x-1)/blkDim.x;

    for (int ts = 1; ts <= ns; ts++) {
        dxfield<<<grdDim, blkDim>>>(ts, nx, hy, dx);
        exfield<<<grdDim, blkDim>>>(nx, md, ix, sx, dx, ex);
        fourier<<<grdDim, blkDim>>>(ts, nf, nx, dt, freq, ex, ft);
        hyfield<<<grdDim, blkDim>>>(nx, bc, ex, hy);
    }

    cudaDeviceSynchronize();

    /* calculate the amplitude and phase at each frequency */
    for (int n = 0; n < nf; n++) {
        for (int i = 0; i < nx; i++) {
            int m = n*nx+i;
            amplt[m] = 1/hypotf(ft.rin[n],ft.iin[n]) * hypotf(ft.rpt[m],ft.ipt[m]);
            phase[m] = atan2f(ft.ipt[m],ft.rpt[m]) - atan2f(ft.iin[n],ft.rin[n]);
        }
    }

    cudaFree(ft.rpt);
    cudaFree(ft.ipt);
    cudaFree(ft.rin);
    cudaFree(ft.iin);
    cudaFree(md.nax);
    cudaFree(md.nbx);
    cudaFree(md.ncx);
    cudaFree(md.ndx);
    cudaFree(amplt);
    cudaFree(phase);
    cudaFree(freq);
    cudaFree(bc);
    cudaFree(dx);
    cudaFree(ex);
    cudaFree(ix);
    cudaFree(sx);
    cudaFree(hy);

    return 0;
}
