/* File: fd1d_1_2.cu
 * Name: D.Saravanan
 * Date: 19/10/2021
 * Simulation of a pulse with absorbing boundary conditions
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx blockIdx.x*blockDim.x+threadIdx.x
#define stx blockDim.x*gridDim.x


__device__
float gaussian(int t, int t0, float sigma) {
    return exp(-0.5*(t - t0)/sigma*(t - t0)/sigma);
}


__global__
void exfield(int t, int nx, float *ex, float *hy) {
    /* calculate the Ex field */
    for (int i = idx+1; i < nx; i += stx)
        ex[i] += 0.5 * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse in the middle */
    if (idx == nx/2) ex[nx/2] = gaussian(t, 40, 12.0f);
}


__global__
void hyfield(int nx, float *ex, float *hy, float *bc) {
    /* absorbing boundary conditions */
    if (idx == 0) ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    if (idx == nx-1) ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = idx; i < nx-1; i += stx)
        hy[i] += 0.5 * (ex[i] - ex[i+1]);
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 570;  /* number of time steps */

    float *ex, *hy;
    /* allocate unified memory accessible from host or device */
    cudaMallocManaged(&ex, nx*sizeof(*ex));
    cudaMallocManaged(&hy, nx*sizeof(*hy));

    /* initialize ex and hy arrays on the host */
    for (int i = 0; i < nx; i++) {
        ex[i] = 0.0f;
        hy[i] = 0.0f;
    }

    float *bc;
    cudaMallocManaged(&bc, 4*sizeof(*bc));
    for (int i = 0; i < 4; bc[i] = 0.0f, i++);

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);

    dim3 gridDim, blockDim;
    blockDim.x = 256;
    gridDim.x = 32*numSM;

    for (int t = 1; t <= ns; t++) {
        exfield<<<gridDim, blockDim>>>(t, nx, ex, hy);
        hyfield<<<gridDim, blockDim>>>(nx, ex, hy, bc);
    }

    cudaDeviceSynchronize();

    cudaFree(bc);
    cudaFree(ex);
    cudaFree(hy);

    return 0;
}
