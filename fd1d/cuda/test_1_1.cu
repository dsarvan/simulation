/* File: test_1_1.cu
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
float gaussian(int t, int t0, float sigma) {
    return exp(-0.5*((t-t0)/sigma)*((t-t0)/sigma));
}


__global__
void exfield(int t, int nx, float *ex, float *hy) {
    /* calculate the Ex field */
    for (int i = idx+1; i < nx; i += stx)
        ex[i] += 0.5 * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse in the middle */
    if (idx == nx/2) ex[nx/2] = gaussian(t, 40, 12);
}


__global__
void hyfield(int nx, float *ex, float *hy) {
    /* calculate the Hy field */
    for (int i = idx; i < nx-1; i += stx)
        hy[i] += 0.5 * (ex[i] - ex[i+1]);
}


int main() {

    int nx = 38000;  /* number of grid points */
    int ns = 40000;  /* number of time steps */

    float *ex, *hy;
    /* allocate unified memory accessible from host or device */
    cudaMallocManaged(&ex, nx*sizeof(*ex));
    cudaMallocManaged(&hy, nx*sizeof(*hy));

    /* initialize ex and hy arrays on the host */
    for (int i = 0; i < nx; i++) {
        ex[i] = 0.0f;
        hy[i] = 0.0f;
    }

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);

    dim3 gridDim, blockDim;
    blockDim.x = 256;
    gridDim.x = 32*numSM;

    cudaEvent_t stime, ntime;
    cudaEventCreate(&stime);
    cudaEventCreate(&ntime);

    cudaEventRecord(stime, 0);

    for (int t = 1; t <= ns; t++) {
        exfield<<<gridDim, blockDim>>>(t, nx, ex, hy);
        hyfield<<<gridDim, blockDim>>>(nx, ex, hy);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(ntime, 0);
    cudaEventSynchronize(ntime);

    float time;
    cudaEventElapsedTime(&time, stime, ntime);
    printf("Total compute time on GPU: %f s\n", time/1000.0f);

    for (int i = 0; i < 50; i++)
        printf("%e\n", ex[i]);

    cudaEventDestroy(stime);
    cudaEventDestroy(ntime);

    cudaFree(ex);
    cudaFree(hy);

    return 0;
}
