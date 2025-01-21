/* File: fd1d_1_1.cu
 * Name: D.Saravanan
 * Date: 11/10/2021
 * Simulation of a pulse in free space
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


__device__ float gaussian(int t, int t0, float sigma) {
    return exp(-0.5 * ((t - t0)/sigma) * ((t - t0)/sigma));
}


__global__ void field(int t, int nx, float *ex, float *hy) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    /* calculate the Hy field */
    for (int i = index; i < nx - 1; i += stride)
        hy[i] = hy[i] + 0.5 * (ex[i] - ex[i+1]);

    __syncthreads();

    /* calculate the Ex field */
    for (int i = index + 1; i < nx; i += stride)
        ex[i] = ex[i] + 0.5 * (hy[i-1] - hy[i]);

    __syncthreads();

    /* put a Gaussian pulse in the middle */
    ex[nx/2] = gaussian(t, 40, 12);
}


int main() {

    int nx = 201;
    int ns = 100;

    float *ex, *hy;

    /* allocate unified memory accessible from host or device */
    cudaMallocManaged(&ex, nx*sizeof(float));
    cudaMallocManaged(&hy, nx*sizeof(float));

    /* initialize ex and hy arrays on the host */
    for (size_t i = 0; i < nx; i++) {
        ex[i] = 0.0f;
        hy[i] = 0.0f;
    }

    dim3 gridDim, blockDim;

    blockDim.x = 256;
    gridDim.x = (nx + blockDim.x - 1)/blockDim.x;

    for (int t = 1; t <= ns; t++)
        field<<<gridDim, blockDim>>>(t, nx, ex, hy);

    cudaDeviceSynchronize();

    cudaFree(ex);
    cudaFree(hy);

    return 0;
}
