/* File: fdtd11.cu
 * Name: D.Saravanan
 * Date: 11/10/2021
 * Simulation of a pulse in free space
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx (blockIdx.x*blockDim.x+threadIdx.x)
#define stx (blockDim.x*gridDim.x)


__device__
float gaussian(int ts, int t0, float sigma) {
    return expf(-0.5f*(ts - t0)/sigma*(ts - t0)/sigma);
}


__global__
void exfield(int ts, int nx, float *hy, float *ex) {
    /* calculate the Ex field */
    for (int i = idx+1; i < nx; i += stx)
        ex[i] += 0.5f * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse in the middle */
    if (idx == nx/2) ex[nx/2] = gaussian(ts, 40, 12.0f);
}


__global__
void hyfield(int nx, float *ex, float *hy) {
    /* calculate the Hy field */
    for (int i = idx; i < nx-1; i += stx)
        hy[i] += 0.5f * (ex[i] - ex[i+1]);
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 300;  /* number of time steps */

    float *ex, *hy;
    cudaMallocManaged(&ex, nx*sizeof(*ex));
    cudaMallocManaged(&hy, nx*sizeof(*hy));

    for (int i = 0; i < nx; i++) {
        ex[i] = 0.0f;
        hy[i] = 0.0f;
    }

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);

    dim3 grdDim, blkDim;
    blkDim.x = 256;
    grdDim.x = 32*numSM;

    for (int ts = 1; ts <= ns; ts++) {
        exfield<<<grdDim, blkDim>>>(ts, nx, hy, ex);
        hyfield<<<grdDim, blkDim>>>(nx, ex, hy);
    }

    cudaDeviceSynchronize();

    cudaFree(ex);
    cudaFree(hy);

    return 0;
}
