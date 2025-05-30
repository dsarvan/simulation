/* File: fd2d_3_1.cu
 * Name: D.Saravanan
 * Date: 17/01/2022
 * Simulation of a pulse in free space in the transverse magnetic (TM) mode
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx (blockIdx.x * blockDim.x + threadIdx.x)
#define idy (blockIdx.y * blockDim.y + threadIdx.y)
#define stx (blockDim.x * gridDim.x)
#define sty (blockDim.y * gridDim.y)


__device__
float gaussian(int t, int t0, float sigma) {
    return exp(-0.5 * ((t - t0)/sigma) * ((t - t0)/sigma));
}


__global__
void dfield(int t, int nx, int ny, float *dz, float *hx, float *hy) {
    /* calculate the electric flux density Dz */
    for (int j = idy + 1; j < ny; j += sty) {
        for (int i = idx + 1; i < nx; i += stx) {
            int n = j*nx+i;
            dz[n] += 0.5 * (hy[n] - hy[n-nx] - hx[n] + hx[n-1]);
        }
    }
    /* put a Gaussian pulse in the middle */
    if ((idy == ny/2) && (idx == nx/2)) dz[ny/2*nx+nx/2] = gaussian(t, 20, 6);
}


__global__
void efield(int nx, int ny, float *naz, float *dz, float *ez) {
    /* calculate the Ez field from Dz */
    for (int j = idy + 1; j < ny; j += sty) {
        for (int i = idx + 1; i < nx; i += stx) {
            int n = j*nx+i;
            ez[n] = naz[n] * dz[n];
        }
    }
}


__global__
void hfield(int nx, int ny, float *ez, float *hx, float *hy) {
    /* calculate the Hx and Hy field */
    for (int j = idy; j < ny - 1; j += sty) {
        for (int i = idx; i < nx - 1; i += stx) {
            int n = j*nx+i;
            hx[n] += 0.5 * (ez[n] - ez[n+1]);
            hy[n] += 0.5 * (ez[n+nx] - ez[n]);
        }
    }
}


int main() {

    int nx = 60;  /* number of grid points */
    int ny = 60;  /* number of grid points */

    int ns = 70;  /* number of time steps */

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
    gridDim.x = (nx + blockDim.x - 1)/blockDim.x;
    gridDim.y = (ny + blockDim.y - 1)/blockDim.y;

    for (int t = 1; t <= ns; t++) {
        dfield<<<gridDim, blockDim>>>(t, nx, ny, dz, hx, hy);
        efield<<<gridDim, blockDim>>>(nx, ny, naz, dz, ez);
        hfield<<<gridDim, blockDim>>>(nx, ny, ez, hx, hy);
    }

    cudaDeviceSynchronize();

    cudaFree(naz);
    cudaFree(dz);
    cudaFree(ez);
    cudaFree(hx);
    cudaFree(hy);

    return 0;
}
