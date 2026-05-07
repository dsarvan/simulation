/* File: test21.cu
 * Name: D.Saravanan
 * Date: 25/11/2021
 * Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
 * dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx (blockIdx.x*blockDim.x+threadIdx.x)
#define stx (blockDim.x*gridDim.x)


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


Medium dielectric(int nx, float dt, float epsr, float sigma) {
    Medium md;
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
    cudaMallocManaged(&dx, nx*sizeof(*dx));
    cudaMallocManaged(&ex, nx*sizeof(*ex));
    cudaMallocManaged(&ix, nx*sizeof(*ix));
    cudaMallocManaged(&hy, nx*sizeof(*hy));

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
    float sigma = 0.04f;  /* conductivity (S/m) */
    Medium md = dielectric(nx, dt, epsr, sigma);

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);

    dim3 grdDim, blkDim;
    blkDim.x = 256;
    grdDim.x = 32*numSM;

    cudaEvent_t stime, ntime;
    cudaEventCreate(&stime);
    cudaEventCreate(&ntime);

    cudaEventRecord(stime, 0);

    for (int ts = 1; ts <= ns; ts++) {
        dxfield<<<grdDim, blkDim>>>(ts, nx, hy, dx);
        exfield<<<grdDim, blkDim>>>(nx, md, ix, dx, ex);
        hyfield<<<grdDim, blkDim>>>(nx, bc, ex, hy);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(ntime, 0);
    cudaEventSynchronize(ntime);

    float time;
    cudaEventElapsedTime(&time, stime, ntime);
    printf("Total compute time on GPU: %.3f s\n", time/1000.0f);

    cudaEventDestroy(stime);
    cudaEventDestroy(ntime);

    for (int i = 0; i < 50; i++)
        printf("%e\n", ex[i]);

    cudaFree(md.nax);
    cudaFree(md.nbx);
    cudaFree(bc);
    cudaFree(dx);
    cudaFree(ex);
    cudaFree(ix);
    cudaFree(hy);

    return 0;
}
