/* File: test_3_1.c
 * Name: D.Saravanan
 * Date: 17/01/2022
 * Simulation of a pulse in free space in the transverse magnetic (TM) mode
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


float gaussian(int t, int t0, float sigma) {
    return expf(-0.5f*(t - t0)/sigma*(t - t0)/sigma);
}


void dfield(int t, int nx, int ny, float *dz, float *hx, float *hy) {
    /* calculate the electric flux density Dz */
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            int n = i*ny+j;
            dz[n] += 0.5f * (hy[n] - hy[n-ny] - hx[n] + hx[n-1]);
        }
    }
    /* put a Gaussian pulse in the middle */
    dz[nx/2*ny+ny/2] = gaussian(t, 20, 6.0f);
}


void efield(int nx, int ny, float *naz, float *dz, float *ez) {
    /* calculate the Ez field from Dz */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int n = i*ny+j;
            ez[n] = naz[n] * dz[n];
        }
    }
}


void hfield(int nx, int ny, float *ez, float *hx, float *hy) {
    /* calculate the Hx and Hy field */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < ny-1; j++) {
            int n = i*ny+j;
            hx[n] += 0.5f * (ez[n] - ez[n+1]);
            hy[n] -= 0.5f * (ez[n] - ez[n+ny]);
        }
    }
}


int main() {

    int nx = 1024;  /* number of grid points */
    int ny = 1024;  /* number of grid points */

    int ns = 5000;  /* number of time steps */

    float *dz = (float*) calloc(nx*ny, sizeof(*dz));
    float *ez = (float*) calloc(nx*ny, sizeof(*ez));
    float *hx = (float*) calloc(nx*ny, sizeof(*hx));
    float *hy = (float*) calloc(nx*ny, sizeof(*hy));

    float *naz = (float*) calloc(nx*ny, sizeof(*naz));
    for (int i = 0; i < nx*ny; naz[i] = 1.0f, i++);

    float stime = omp_get_wtime();

    for (int t = 1; t <= ns; t++) {
        dfield(t, nx, ny, dz, hx, hy);
        efield(nx, ny, naz, dz, ez);
        hfield(nx, ny, ez, hx, hy);
    }

    float ntime = omp_get_wtime();
    printf("Total compute time on CPU: %.3f s\n", ntime - stime);

    for (int i = 2*ny; i < 2*ny+50; i++)
        printf("%e\n", ez[i]);

    free(naz);
    free(dz);
    free(ez);
    free(hx);
    free(hy);

    return 0;
}
