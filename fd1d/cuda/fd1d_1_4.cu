/* File: fd1d_1_4.cu
 * Name: D.Saravanan
 * Date: 22/10/2021
 * Simulation of a propagating sinusoidal striking a dielectric medium
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx (blockIdx.x * blockDim.x + threadIdx.x)
#define stx (blockDim.x * gridDim.x)


__device__
float sinusoidal(int t, float ds, float freq) {
    float dt = ds/6e8;  /* time step (s) */
    return sin(2 * M_PI * freq * dt * t);
}


__global__
void exfield(int t, int nx, float *cb, float *ex, float *hy) {
    /* calculate the Ex field */
    for (int i = idx + 1; i < nx; i += stx)
        ex[i] = ex[i] + cb[i] * (hy[i-1] - hy[i]);
    __syncthreads();
    /* put a sinusoidal wave at the low end */
    if (idx == 1) ex[1] = ex[1] + sinusoidal(t, 0.01, 700e6);
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


float *dielectric(int nx, float epsr) {
    float *cb;
    cudaMallocManaged(&cb, nx*sizeof(float));
    for (int i = 0; i < nx; cb[i] = 0.5f, i++);
    for (int i = nx/2; i < nx; cb[i] = 0.5/epsr, i++);
    return cb;
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 740;  /* number of time steps */

    float *ex, *hy;
    /* allocate unified memory accessible from host or device */
    cudaMallocManaged(&ex, nx*sizeof(float));
    cudaMallocManaged(&hy, nx*sizeof(float));

    /* initialize ex and hy arrays on the host */
    for (int i = 0; i < nx; i++) {
        ex[i] = 0.0f;
        hy[i] = 0.0f;
    }

    float *bc;
    cudaMallocManaged(&bc, 4*sizeof(float));
    bc[4] = {0.0f};

    float ds = 0.01;  /* spatial step (m) */
    float dt = ds/6e8;  /* time step (s) */
    float epsr = 4;  /* relative permittivity */
    float *cb = dielectric(nx, epsr);

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);

    dim3 gridDim, blockDim;
    blockDim.x = 256;
    gridDim.x = 32*numSM;

    for (int t = 1; t <= ns; t++) {
        exfield<<<gridDim, blockDim>>>(t, nx, cb, ex, hy);
        hyfield<<<gridDim, blockDim>>>(nx, ex, hy, bc);
    }

    cudaDeviceSynchronize();

    cudaFree(bc);
    cudaFree(cb);
    cudaFree(ex);
    cudaFree(hy);

    return 0;
}
