/* File: fd1d_2_1.cu
 * Name: D.Saravanan
 * Date: 25/11/2021
 * Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
 * dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx (blockIdx.x * blockDim.x + threadIdx.x)
#define stx (blockDim.x * gridDim.x)


typedef struct {
    float *gax;
    float *gbx;
} tuple;


__device__
float sinusoidal(int t, float ds, float freq) {
    float dt = ds/6e8;  /* time step (s) */
    return sin(2 * M_PI * freq * dt * t);
}


__global__
void dxfield(int t, int nx, float *dx, float *hy) {
    /* calculate the electric flux density Dx */
    for (int i = idx + 1; i < nx; i += stx)
        dx[i] = dx[i] + 0.5 * (hy[i-1] - hy[i]);
    __syncthreads();
    /* put a sinusoidal wave at the low end */
    if (idx == 1) dx[1] = dx[1] + sinusoidal(t, 0.01, 700e6);
}


__global__
void exfield(int nx, float *gax, float *gbx, float *dx, float *ix, float *ex) {
    /* calculate the Ex field from Dx */
    for (int i = idx + 1; i < nx; i += stx) {
        ex[i] = gax[i] * (dx[i] - ix[i]);
        ix[i] = ix[i] + gbx[i] * ex[i];
    }
    __syncthreads();
}


__global__
void hyfield(int nx, float *ex, float *hy, float *bc) {
    /* absorbing boundary conditions */
    if (idx == 0) ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    if (idx == nx-1) ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = idx; i < nx - 1; i += stx)
        hy[i] = hy[i] + 0.5 * (ex[i] - ex[i+1]);
    __syncthreads();
}


tuple dielectric(int nx, float dt, float epsr, float sigma) {
    tuple n;
    cudaMallocManaged(&n.gax, nx*sizeof(float));
    cudaMallocManaged(&n.gbx, nx*sizeof(float));
    for (int i = 0; i < nx; n.gax[i] = 1.0f, i++);
    float eps0 = 8.854e-12;  /* vacuum permittivity (F/m) */
    for (int i = nx/2; i < nx; i++) {
        n.gax[i] = 1/(epsr + (sigma * dt/eps0));
        n.gbx[i] = sigma * dt/eps0;
    }
    return n;
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 740;  /* number of time steps */

    float *dx, *ex, *ix, *hy;
    /* allocate unified memory accessible from host or device */
    cudaMallocManaged(&dx, nx*sizeof(float));
    cudaMallocManaged(&ex, nx*sizeof(float));
    cudaMallocManaged(&ix, nx*sizeof(float));
    cudaMallocManaged(&hy, nx*sizeof(float));

    /* initialize dx, ex, ix and hy arrays on the host */
    for (int i = 0; i < nx; i++) {
        dx[i] = 0.0f;
        ex[i] = 0.0f;
        ix[i] = 0.0f;
        hy[i] = 0.0f;
    }

    float *bc;
    cudaMallocManaged(&bc, 4*sizeof(float));
    bc[4] = {0.0f};

    float ds = 0.01;  /* spatial step (m) */
    float dt = ds/6e8;  /* time step (s) */
    float epsr = 4;  /* relative permittivity */
    float sigma = 0.04;  /* conductivity (S/m) */
    tuple n = dielectric(nx, dt, epsr, sigma);

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);

    dim3 gridDim, blockDim;
    blockDim.x = 256;
    gridDim.x = 32*numSM;

    for (int t = 1; t <= ns; t++) {
        dxfield<<<gridDim, blockDim>>>(t, nx, dx, hy);
        exfield<<<gridDim, blockDim>>>(nx, n.gax, n.gbx, dx, ix, ex);
        hyfield<<<gridDim, blockDim>>>(nx, ex, hy, bc);
    }

    cudaDeviceSynchronize();

    cudaFree(bc);
    cudaFree(n.gax);
    cudaFree(n.gbx);
    cudaFree(dx);
    cudaFree(ex);
    cudaFree(ix);
    cudaFree(hy);

    return 0;
}
