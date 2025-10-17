/* File: fd2d_3_3.cu
 * Name: D.Saravanan
 * Date: 19/01/2022
 * Simulation of a plane wave pulse propagating in free space in the transverse
 * magnetic (TM) mode with the two-dimensional perfectly matched layer (PML)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx blockIdx.x*blockDim.x+threadIdx.x
#define idy blockIdx.y*blockDim.y+threadIdx.y
#define stx blockDim.x*gridDim.x
#define sty blockDim.y*gridDim.y


typedef struct {
    float *fx1;
    float *fx2;
    float *fx3;
    float *fy1;
    float *fy2;
    float *fy3;
    float *gx2;
    float *gx3;
    float *gy2;
    float *gy3;
} pmlayer;


__device__
float gaussian(int t, int t0, float sigma) {
    return exp(-0.5*(t - t0)/sigma*(t - t0)/sigma);
}


__global__
void ezinct(int ny, float *ezi, float *hxi, float *bc) {
    /* calculate the incident Ez */
    for (int j = idx+1; j < ny; j += stx) {
        ezi[j] += 0.5 * (hxi[j-1] - hxi[j]);
    }
    /* absorbing boundary conditions */
    if (idx == 0) ezi[0] = bc[0], bc[0] = bc[1], bc[1] = ezi[1];
    if (idx == ny-1) ezi[ny-1] = bc[3], bc[3] = bc[2], bc[2] = ezi[ny-2];
}


__global__
void dfield(int t, int nx, int ny, pmlayer *pml, float *ezi, float *dz, float *hx, float *hy) {
    /* calculate the electric flux density Dz */
    for (int i = idy+1; i < nx; i += sty) {
        for (int j = idx+1; j < ny; j += stx) {
            int n = i*ny+j;
            dz[n] = pml->gx3[i] * pml->gy3[j] * dz[n] + pml->gx2[i] * pml->gy2[j] * 0.5 * (hy[n] - hy[n-ny] - hx[n] + hx[n-1]);
        }
    }
    /* put a Gaussian pulse at the low end */
    if (idy == 0 && idx == 3) ezi[3] = gaussian(t, 20, 8.0f);
}


__global__
void inctdz(int nx, int ny, int npml, float *hxi, float *dz) {
    /* incident Dz values */
    for (int i = idx+npml-1; i <= nx-npml; i += stx) {
        dz[i*ny+(npml-1)] += 0.5 * hxi[npml-2];
        dz[i*ny+(ny-npml)] -= 0.5 * hxi[ny-npml];
    }
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
void hxinct(int ny, float *ezi, float *hxi) {
    /* calculate the incident Hx */
    for (int j = idx; j < ny-1; j += stx) {
        hxi[j] += 0.5 * (ezi[j] - ezi[j+1]);
    }
}


__global__
void hfield(int nx, int ny, pmlayer *pml, float *ez, float *ihx, float *ihy, float *hx, float *hy) {
    /* calculate the Hx and Hy field */
    for (int i = idy; i < nx-1; i += sty) {
        for (int j = idx; j < ny-1; j += stx) {
            int n = i*ny+j;
            ihx[n] += ez[n] - ez[n+1];
            ihy[n] += ez[n] - ez[n+ny];
            hx[n] = pml->fy3[j] * hx[n] + pml->fy2[j] * (0.5 * ez[n] - 0.5 * ez[n+1] + pml->fx1[i] * ihx[n]);
            hy[n] = pml->fx3[i] * hy[n] - pml->fx2[i] * (0.5 * ez[n] - 0.5 * ez[n+ny] + pml->fy1[j] * ihy[n]);
        }
    }
}


__global__
void incthx(int nx, int ny, int npml, float *ezi, float *hx) {
    /* incident Hx values */
    for (int i = idx+npml-1; i <= nx-npml; i += stx) {
        hx[i*ny+(npml-2)] += 0.5 * ezi[npml-1];
        hx[i*ny+(ny-npml)] -= 0.5 * ezi[ny-npml];
    }
}


__global__
void incthy(int nx, int ny, int npml, float *ezi, float *hy) {
    /* incident Hy values */
    for (int j = idx+npml-1; j <= ny-npml; j += stx) {
        hy[(npml-2)*ny+j] -= 0.5 * ezi[j];
        hy[(nx-npml)*ny+j] += 0.5 * ezi[j];
    }
}


void pmlparam(int nx, int ny, int npml, pmlayer *pml) {
    /* calculate the two-dimensional perfectly matched layer (PML) parameters */
    for (int n = 0; n < npml; n++) {
        float xm = 0.33*(npml-n)/npml*(npml-n)/npml*(npml-n)/npml;
        float xn = 0.33*(npml-n-0.5)/npml*(npml-n-0.5)/npml*(npml-n-0.5)/npml;
        pml->fx1[n] = pml->fx1[nx-2-n] = pml->fy1[n] = pml->fy1[ny-2-n] = xn;
        pml->fx2[n] = pml->fx2[nx-2-n] = pml->fy2[n] = pml->fy2[ny-2-n] = 1/(1+xn);
        pml->gx2[n] = pml->gx2[nx-1-n] = pml->gy2[n] = pml->gy2[ny-1-n] = 1/(1+xm);
        pml->fx3[n] = pml->fx3[nx-2-n] = pml->fy3[n] = pml->fy3[ny-2-n] = (1-xn)/(1+xn);
        pml->gx3[n] = pml->gx3[nx-1-n] = pml->gy3[n] = pml->gy3[ny-1-n] = (1-xm)/(1+xm);
    }
}


int main() {

    int nx = 60;  /* number of grid points */
    int ny = 60;  /* number of grid points */

    int ns = 115;  /* number of time steps */

    float *ezi, *hxi;
    /* allocate unified memory accessible from host or device */
    cudaMallocManaged(&ezi, ny*sizeof(*ezi));
    cudaMallocManaged(&hxi, ny*sizeof(*hxi));

    /* initialize ezi and hxi arrays on the host */
    for (int i = 0; i < ny; i++) {
        ezi[i] = 0.0f;
        hxi[i] = 0.0f;
    }

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

    float *ihx, *ihy;
    /* allocate unified memory accessible from host or device */
    cudaMallocManaged(&ihx, nx*ny*sizeof(*ihx));
    cudaMallocManaged(&ihy, nx*ny*sizeof(*ihy));

    /* initialize ihx and ihy arrays on the host */
    for (int i = 0; i < nx*ny; i++) {
        ihx[i] = 0.0f;
        ihy[i] = 0.0f;
    }

    float *naz;
    cudaMallocManaged(&naz, nx*ny*sizeof(*naz));
    for (int i = 0; i < nx*ny; naz[i] = 1.0f, i++);

    float *bc;
    cudaMallocManaged(&bc, 4*sizeof(*bc));
    for (int i = 0; i < 4; bc[i] = 0.0f, i++);

    pmlayer pml;
    cudaMallocManaged(&pml.fx1, nx*sizeof(*pml.fx1));
    cudaMallocManaged(&pml.fx2, nx*sizeof(*pml.fx2));
    cudaMallocManaged(&pml.fx3, nx*sizeof(*pml.fx3));
    cudaMallocManaged(&pml.fy1, ny*sizeof(*pml.fy1));
    cudaMallocManaged(&pml.fy2, ny*sizeof(*pml.fy2));
    cudaMallocManaged(&pml.fy3, ny*sizeof(*pml.fy3));
    cudaMallocManaged(&pml.gx2, nx*sizeof(*pml.gx2));
    cudaMallocManaged(&pml.gx3, nx*sizeof(*pml.gx3));
    cudaMallocManaged(&pml.gy2, ny*sizeof(*pml.gy2));
    cudaMallocManaged(&pml.gy3, ny*sizeof(*pml.gy3));

    for (int i = 0; i < nx; i++) {
        pml.fx1[i] = 0.0f;
        pml.fx2[i] = 1.0f;
        pml.fx3[i] = 1.0f;
        pml.gx2[i] = 1.0f;
        pml.gx3[i] = 1.0f;
    }

    for (int i = 0; i < ny; i++) {
        pml.fy1[i] = 0.0f;
        pml.fy2[i] = 1.0f;
        pml.fy3[i] = 1.0f;
        pml.gy2[i] = 1.0f;
        pml.gy3[i] = 1.0f;
    }

    int npml = 8;  /* pml thickness */
    pmlparam(nx, ny, npml, &pml);

    float ds = 0.01;  /* spatial step (m) */
    float dt = ds/6e8;  /* time step (s) */

    dim3 gridDim, blockDim;
    blockDim.x = 16;
    blockDim.y = 16;
    gridDim.x = (ny+blockDim.x-1)/blockDim.x;
    gridDim.y = (nx+blockDim.y-1)/blockDim.y;

    for (int t = 1; t <= ns; t++) {
        ezinct<<<(ny+255)/256, 256>>>(ny, ezi, hxi, bc);
        dfield<<<gridDim, blockDim>>>(t, nx, ny, &pml, ezi, dz, hx, hy);
        inctdz<<<(nx+255)/256, 256>>>(nx, ny, npml, hxi, dz);
        efield<<<gridDim, blockDim>>>(nx, ny, naz, dz, ez);
        hxinct<<<(ny+255)/256, 256>>>(ny, ezi, hxi);
        hfield<<<gridDim, blockDim>>>(nx, ny, &pml, ez, ihx, ihy, hx, hy);
        incthx<<<(nx+255)/256, 256>>>(nx, ny, npml, ezi, hx);
        incthy<<<(ny+255)/256, 256>>>(nx, ny, npml, ezi, hy);
    }

    cudaDeviceSynchronize();

    cudaFree(pml.fx1);
    cudaFree(pml.fx2);
    cudaFree(pml.fx3);
    cudaFree(pml.fy1);
    cudaFree(pml.fy2);
    cudaFree(pml.fy3);
    cudaFree(pml.gx2);
    cudaFree(pml.gx3);
    cudaFree(pml.gy2);
    cudaFree(pml.gy3);
    cudaFree(naz);
    cudaFree(ezi);
    cudaFree(hxi);
    cudaFree(ihx);
    cudaFree(ihy);
    cudaFree(bc);
    cudaFree(dz);
    cudaFree(ez);
    cudaFree(hx);
    cudaFree(hy);

    return 0;
}
