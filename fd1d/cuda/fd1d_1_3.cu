/* File: fdtd13.cu
 * Name: D.Saravanan
 * Date: 19/10/2021
 * Simulation of a pulse hitting a dielectric medium
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
void exfield(int ts, int nx, float *cb, float *hy, float *ex) {
    /* calculate the Ex field */
    for (int i = idx+1; i < nx; i += stx)
        ex[i] += cb[i] * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse at the low end */
    if (idx == 1) ex[1] += gaussian(ts, 40, 12.0f);
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


float *dielectric(int nx, float epsr) {
    float *cb;
    cudaMallocManaged(&cb, nx*sizeof(*cb));
    for (int i = 0; i < nx; cb[i] = 0.5f, i++);
    for (int i = nx/2; i < nx; cb[i] = 0.5f/epsr, i++);
    return cb;
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 740;  /* number of time steps */

    float *ex, *hy;
    cudaMallocManaged(&ex, nx*sizeof(*ex));
    cudaMallocManaged(&hy, nx*sizeof(*hy));

    for (int i = 0; i < nx; i++) {
        ex[i] = 0.0f;
        hy[i] = 0.0f;
    }

    float *bc;
    cudaMallocManaged(&bc, 4*sizeof(*bc));
    for (int i = 0; i < 4; bc[i] = 0.0f, i++);

    float epsr = 4.0f;  /* relative permittivity */
    float *cb = dielectric(nx, epsr);

    int numSM;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);

    dim3 grdDim, blkDim;
    blkDim.x = 256;
    grdDim.x = 32*numSM;

    for (int ts = 1; ts <= ns; ts++) {
        exfield<<<grdDim, blkDim>>>(ts, nx, cb, hy, ex);
        hyfield<<<grdDim, blkDim>>>(nx, bc, ex, hy);
    }

    cudaDeviceSynchronize();

    cudaFree(cb);
    cudaFree(bc);
    cudaFree(ex);
    cudaFree(hy);

    return 0;
}
