/* File: test_2_2.cu
 * Name: D.Saravanan
 * Date: 07/12/2021
 * Simulation of a pulse striking a dielectric medium and implements
 * the discrete Fourier transform with a Gaussian pulse as its source
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx blockIdx.x*blockDim.x+threadIdx.x
#define stx blockDim.x*gridDim.x


typedef struct {
    float *nax, *nbx;
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
void exfield(int nx, medium *md, float *dx, float *ix, float *ex) {
    /* calculate the Ex field from Dx */
    for (int i = idx+1; i < nx; i += stx) {
        ex[i] = md->nax[i] * (dx[i] - ix[i]);
        ix[i] += md->nbx[i] * ex[i];
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


medium dielectric(int nx, float dt, float epsr, float sigma) {
    medium md;
    cudaMallocManaged(&md.nax, nx*sizeof(*md.nax));
    cudaMallocManaged(&md.nbx, nx*sizeof(*md.nbx));
    for (int i = 0; i < nx; md.nax[i] = 1.0f, i++);
    for (int i = 0; i < nx; md.nbx[i] = 0.0f, i++);
    float eps0 = 8.854e-12f;  /* vacuum permittivity (F/m) */
    for (int i = nx/2; i < nx; i++) {
        md.nax[i] = 1/(epsr + sigma*dt/eps0);
        md.nbx[i] = sigma*dt/eps0;
    }
    return md;
}


int main() {

    int nx = 38000;  /* number of grid points */
    int ns = 40000;  /* number of time steps */

    float *dx, *ex, *ix, *hy;
    /* allocate unified memory accessible from host or device */
    cudaMallocManaged(&dx, nx*sizeof(*dx));
    cudaMallocManaged(&ex, nx*sizeof(*ex));
    cudaMallocManaged(&ix, nx*sizeof(*ix));
    cudaMallocManaged(&hy, nx*sizeof(*hy));

    /* initialize dx, ex, ix and hy arrays on the host */
    for (int i = 0; i < nx; i++) {
        dx[i] = 0.0f;
        ex[i] = 0.0f;
        ix[i] = 0.0f;
        hy[i] = 0.0f;
    }

    float *bc;
    cudaMallocManaged(&bc, 4*sizeof(*bc));
    for (int i = 0; i < 4; bc[i] = 0.0f, i++);

    float ds = 0.01f;  /* spatial step (m) */
    float dt = ds/6e8f;  /* time step (s) */
    float epsr = 4.0f;  /* relative permittivity */
    float sigma = 0.0f;  /* conductivity (S/m) */
    medium md = dielectric(nx, dt, epsr, sigma);

    int nf = 3;  /* number of frequencies */
    /* frequency 100 MHz, 200 MHz, 500 MHz */
    float *freq;
    cudaMallocManaged(&freq, nf*sizeof(*freq));
    freq[0] = 100e6f; freq[1] = 200e6f; freq[2] = 500e6f;

    ftrans ft;
    cudaMallocManaged(&ft.r_pt, nf*nx*sizeof(*ft.r_pt));
    cudaMallocManaged(&ft.i_pt, nf*nx*sizeof(*ft.i_pt));
    cudaMallocManaged(&ft.r_in, nf*sizeof(*ft.r_in));
    cudaMallocManaged(&ft.i_in, nf*sizeof(*ft.i_in));

    for (int i = 0; i < nf*nx; i++) {
        ft.r_pt[i] = 0.0f;
        ft.i_pt[i] = 0.0f;
    }

    for (int i = 0; i < nf; i++) {
        ft.r_in[i] = 0.0f;
        ft.i_in[i] = 0.0f;
    }

    float *amplt, *phase;
    cudaMallocManaged(&amplt, nf*nx*sizeof(*amplt));
    cudaMallocManaged(&phase, nf*nx*sizeof(*phase));

    for (int i = 0; i < nf*nx; i++) {
        amplt[i] = 0.0f;
        phase[i] = 0.0f;
    }

    dim3 gridDim, blockDim;
    blockDim.x = 256;
    gridDim.x = (nx+blockDim.x-1)/blockDim.x;

    cudaEvent_t stime, ntime;
    cudaEventCreate(&stime);
    cudaEventCreate(&ntime);

    cudaEventRecord(stime, 0);

    for (int t = 1; t <= ns; t++) {
        dxfield<<<gridDim, blockDim>>>(t, nx, dx, hy);
        exfield<<<gridDim, blockDim>>>(nx, &md, dx, ix, ex);
        fourier<<<gridDim, dim3(256,4)>>>(t, nf, nx, dt, freq, ex, &ft);
        hyfield<<<gridDim, blockDim>>>(nx, ex, hy, bc);
    }

    cudaDeviceSynchronize();

    /* calculate the amplitude and phase at each frequency */
    for (int n = 0; n < nf; n++) {
        for (int i = 0; i < nx; i++) {
            int m = n*nx+i;
            amplt[m] = 1/hypotf(ft.r_in[n],ft.i_in[n]) * hypotf(ft.r_pt[m],ft.i_pt[m]);
            phase[m] = atan2f(ft.i_pt[m],ft.r_pt[m]) - atan2f(ft.i_in[n],ft.r_in[n]);
        }
    }

    cudaEventRecord(ntime, 0);
    cudaEventSynchronize(ntime);

    float time;
    cudaEventElapsedTime(&time, stime, ntime);
    printf("Total compute time on GPU: %.3f s\n", time/1000.0f);

    cudaEventDestroy(stime);
    cudaEventDestroy(ntime);

    for (int i = 0; i < 50; i++)
        printf("%e\n", ex[i]);

    cudaFree(ft.r_pt);
    cudaFree(ft.i_pt);
    cudaFree(ft.r_in);
    cudaFree(ft.i_in);
    cudaFree(md.nax);
    cudaFree(md.nbx);
    cudaFree(amplt);
    cudaFree(phase);
    cudaFree(freq);
    cudaFree(bc);
    cudaFree(dx);
    cudaFree(ex);
    cudaFree(ix);
    cudaFree(hy);

    return 0;
}
