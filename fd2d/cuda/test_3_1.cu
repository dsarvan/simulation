/* File: test_3_1.cu
 * Name: D.Saravanan
 * Date: 17/01/2022
 * Simulation of a pulse in free space in the transverse magnetic (TM) mode
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx blockIdx.x*blockDim.x+threadIdx.x
#define idy blockIdx.y*blockDim.y+threadIdx.y
#define stx blockDim.x*gridDim.x
#define sty blockDim.y*gridDim.y


__device__
float gaussian(int t, int t0, float sigma) {
    return exp(-0.5*(t - t0)/sigma*(t - t0)/sigma);
}


__global__
void dfield(int t, int nx, int ny, float *dz, float *hx, float *hy) {
    /* calculate the electric flux density Dz */
    for (int i = idy+1; i < nx; i += sty) {
        for (int j = idx+1; j < ny; j += stx) {
            int n = i*ny+j;
            dz[n] += 0.5 * (hy[n] - hy[n-ny] - hx[n] + hx[n-1]);
        }
    }
    __syncthreads();
    /* put a Gaussian pulse in the middle */
    if (idy == nx/2 && idx == ny/2) dz[nx/2*ny+ny/2] = gaussian(t, 20, 6.0f);
}


__global__
void efield(int nx, int ny, float *naz, float *dz, float *ez) {
    /* calculate the Ez field from Dz */
    for (int i = idy; i < nx; i += sty) {
        for (int j = idx; j < ny; j += stx) {
            int n = i*ny+j;
            ez[n] = naz[n] * dz[n];
        }
    }
}


__global__
void hfield(int nx, int ny, float *ez, float *hx, float *hy) {
    /* calculate the Hx and Hy field */
    for (int i = idy; i < nx-1; i += sty) {
        for (int j = idx; j < ny-1; j += stx) {
            int n = i*ny+j;
            hx[n] += 0.5 * (ez[n] - ez[n+1]);
            hy[n] -= 0.5 * (ez[n] - ez[n+ny]);
        }
    }
}


int main() {

    int nx = 1024;  /* number of grid points */
    int ny = 1024;  /* number of grid points */

    int ns = 5000;  /* number of time steps */

    float *dz, *ez, *hx, *hy;
    /* allocate unified memory accessible from host or device */
    cudaMallocManaged(&dz, nx*ny*sizeof(*dz));
    cudaMallocManaged(&ez, nx*ny*sizeof(*ez));
    cudaMallocManaged(&hx, nx*ny*sizeof(*hx));
    cudaMallocManaged(&hy, nx*ny*sizeof(*hy));

    /* initialize dz, ez, hx and hy arrays on the host */
    for (int i = 0; i < nx*ny; i++) {
        dz[i] = 0.0f;
        ez[i] = 0.0f;
        hx[i] = 0.0f;
        hy[i] = 0.0f;
    }

    float *naz;
    cudaMallocManaged(&naz, nx*ny*sizeof(*naz));
    for (int i = 0; i < nx*ny; naz[i] = 1.0f, i++);

    dim3 gridDim, blockDim;
    blockDim.x = 16;
    blockDim.y = 16;
    gridDim.x = (ny+blockDim.x-1)/blockDim.x;
    gridDim.y = (nx+blockDim.y-1)/blockDim.y;

    cudaEvent_t stime, ntime;
    cudaEventCreate(&stime);
    cudaEventCreate(&ntime);

    cudaEventRecord(stime, 0);

    for (int t = 1; t <= ns; t++) {
        dfield<<<gridDim, blockDim>>>(t, nx, ny, dz, hx, hy);
        efield<<<gridDim, blockDim>>>(nx, ny, naz, dz, ez);
        hfield<<<gridDim, blockDim>>>(nx, ny, ez, hx, hy);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(ntime, 0);
    cudaEventSynchronize(ntime);

    float time;
    cudaEventElapsedTime(&time, stime, ntime);
    printf("Total compute time on GPU: %.3f s\n", time/1000.0f);

    cudaEventDestroy(stime);
    cudaEventDestroy(ntime);

    for (int i = 2*ny; i < 2*ny+50; i++)
        printf("%e\n", ez[i]);

    cudaFree(naz);
    cudaFree(dz);
    cudaFree(ez);
    cudaFree(hx);
    cudaFree(hy);

    return 0;
}
