/* File: test_1_5.cu
 * Name: D.Saravanan
 * Date: 29/10/2021
 * Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
 * dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define idx blockIdx.x*blockDim.x+threadIdx.x
#define stx blockDim.x*gridDim.x


typedef struct {
    float *ca;
    float *cb;
} tuple;


__device__
float sinusoidal(int t, float ds, float freq) {
    float dt = ds/6e8f;  /* time step (s) */
    return sinf(2*M_PI*freq*dt*t);
}


__global__
void exfield(int t, int nx, float *ca, float *cb, float *ex, float *hy) {
    /* calculate the Ex field */
    for (int i = idx+1; i < nx; i += stx)
        ex[i] = ca[i] * ex[i] + cb[i] * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    if (idx == 1) ex[1] += sinusoidal(t, 0.01f, 700e6f);
}


__global__
void hyfield(int nx, float *ex, float *hy, float *bc) {
    /* absorbing boundary conditions */
    if (idx == 0) ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    if (idx == nx-1) ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = idx; i < nx-1; i += stx)
        hy[i] += 0.5f * (ex[i] - ex[i+1]);
}


tuple dielectric(int nx, float dt, float epsr, float sigma) {
    tuple n;
    cudaMallocManaged(&n.ca, nx*sizeof(*n.ca));
    cudaMallocManaged(&n.cb, nx*sizeof(*n.cb));
    for (int i = 0; i < nx; n.ca[i] = 1.0f, i++);
    for (int i = 0; i < nx; n.cb[i] = 0.5f, i++);
    float eps0 = 8.854e-12f;  /* vacuum permittivity (F/m) */
    float epsf = dt*sigma/(2*eps0*epsr);
    for (int i = nx/2; i < nx; n.ca[i] = (1 - epsf)/(1 + epsf), i++);
    for (int i = nx/2; i < nx; n.cb[i] = 0.5f/(epsr*(1 + epsf)), i++);
    return n;
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

    float *bc;
    cudaMallocManaged(&bc, 4*sizeof(*bc));
    for (int i = 0; i < 4; bc[i] = 0.0f, i++);

    float ds = 0.01f;  /* spatial step (m) */
    float dt = ds/6e8f;  /* time step (s) */
    float epsr = 4.0f;  /* relative permittivity */
    float sigma = 0.04f;  /* conductivity (S/m) */
    tuple n = dielectric(nx, dt, epsr, sigma);

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
        exfield<<<gridDim, blockDim>>>(t, nx, n.ca, n.cb, ex, hy);
        hyfield<<<gridDim, blockDim>>>(nx, ex, hy, bc);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(ntime, 0);
    cudaEventSynchronize(ntime);

    float time;
    cudaEventElapsedTime(&time, stime, ntime);
    printf("Total compute time on GPU: %.3f s\n", time/1000.0f);

    cudaEventDestroy(stime);
    cudaEventDestroy(ntime);

    for (int i = 0; i < 50; i++)
        printf("%e\n", ex[i]);

    cudaFree(n.ca);
    cudaFree(n.cb);
    cudaFree(bc);
    cudaFree(ex);
    cudaFree(hy);

    return 0;
}
