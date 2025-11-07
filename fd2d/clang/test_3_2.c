/* File: test_3_2.c
 * Name: D.Saravanan
 * Date: 18/01/2022
 * Simulation of a propagating sinusoidal in free space in the transverse
 * magnetic (TM) mode with the two-dimensional perfectly matched layer (PML)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


typedef struct {
    float *fx1, *fx2, *fx3;
    float *fy1, *fy2, *fy3;
    float *gx2, *gx3;
    float *gy2, *gy3;
} pmlayer;


float sinusoidal(int t, float ds, float freq) {
    float dt = ds/6e8f;  /* time step (s) */
    return sinf(2*M_PI*freq*dt*t);
}


void dfield(int t, int nx, int ny, pmlayer *pml, float *dz, float *hx, float *hy) {
    /* calculate the electric flux density Dz */
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            int n = i*ny+j;
            dz[n] = pml->gx3[i] * pml->gy3[j] * dz[n] + pml->gx2[i] * pml->gy2[j] * 0.5f * (hy[n] - hy[n-ny] - hx[n] + hx[n-1]);
        }
    }
    /* put a sinusoidal source at a point that is offset five cells
     * from the center of the problem space in each direction */
    dz[(nx/2-5)*ny+(ny/2-5)] = sinusoidal(t, 0.01f, 1500e6f);
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


void hfield(int nx, int ny, pmlayer *pml, float *ez, float *ihx, float *ihy, float *hx, float *hy) {
    /* calculate the Hx and Hy field */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < ny-1; j++) {
            int n = i*ny+j;
            ihx[n] += ez[n] - ez[n+1];
            ihy[n] += ez[n] - ez[n+ny];
            hx[n] = pml->fy3[j] * hx[n] + pml->fy2[j] * (0.5f * ez[n] - 0.5f * ez[n+1] + pml->fx1[i] * ihx[n]);
            hy[n] = pml->fx3[i] * hy[n] - pml->fx2[i] * (0.5f * ez[n] - 0.5f * ez[n+ny] + pml->fy1[j] * ihy[n]);
        }
    }
}


void pmlparam(int nx, int ny, int npml, pmlayer *pml) {
    /* calculate the two-dimensional perfectly matched layer (PML) parameters */
    for (int n = 0; n < npml; n++) {
        float xm = 0.33f*(npml-n)/npml*(npml-n)/npml*(npml-n)/npml;
        float xn = 0.33f*(npml-n-0.5f)/npml*(npml-n-0.5f)/npml*(npml-n-0.5f)/npml;
        pml->fx1[n] = pml->fx1[nx-2-n] = pml->fy1[n] = pml->fy1[ny-2-n] = xn;
        pml->fx2[n] = pml->fx2[nx-2-n] = pml->fy2[n] = pml->fy2[ny-2-n] = 1/(1+xn);
        pml->gx2[n] = pml->gx2[nx-1-n] = pml->gy2[n] = pml->gy2[ny-1-n] = 1/(1+xm);
        pml->fx3[n] = pml->fx3[nx-2-n] = pml->fy3[n] = pml->fy3[ny-2-n] = (1-xn)/(1+xn);
        pml->gx3[n] = pml->gx3[nx-1-n] = pml->gy3[n] = pml->gy3[ny-1-n] = (1-xm)/(1+xm);
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

    float *ihx = (float*) calloc(nx*ny, sizeof(*ihx));
    float *ihy = (float*) calloc(nx*ny, sizeof(*ihy));

    float *naz = (float*) calloc(nx*ny, sizeof(*naz));
    for (int i = 0; i < nx*ny; naz[i] = 1.0f, i++);

    pmlayer pml;
    pml.fx1 = (float*) calloc(nx, sizeof(*pml.fx1));
    pml.fx2 = (float*) calloc(nx, sizeof(*pml.fx2));
    pml.fx3 = (float*) calloc(nx, sizeof(*pml.fx3));
    pml.fy1 = (float*) calloc(ny, sizeof(*pml.fy1));
    pml.fy2 = (float*) calloc(ny, sizeof(*pml.fy2));
    pml.fy3 = (float*) calloc(ny, sizeof(*pml.fy3));
    pml.gx2 = (float*) calloc(nx, sizeof(*pml.gx2));
    pml.gx3 = (float*) calloc(nx, sizeof(*pml.gx3));
    pml.gy2 = (float*) calloc(ny, sizeof(*pml.gy2));
    pml.gy3 = (float*) calloc(ny, sizeof(*pml.gy3));

    for (int i = 0; i < nx; i++) {
        pml.fx2[i] = 1.0f;
        pml.fx3[i] = 1.0f;
        pml.gx2[i] = 1.0f;
        pml.gx3[i] = 1.0f;
    }

    for (int i = 0; i < ny; i++) {
        pml.fy2[i] = 1.0f;
        pml.fy3[i] = 1.0f;
        pml.gy2[i] = 1.0f;
        pml.gy3[i] = 1.0f;
    }

    int npml = 80;  /* pml thickness */
    pmlparam(nx, ny, npml, &pml);

    float ds = 0.01f;  /* spatial step (m) */
    float dt = ds/6e8f;  /* time step (s) */

    float stime = omp_get_wtime();

    for (int t = 1; t <= ns; t++) {
        dfield(t, nx, ny, &pml, dz, hx, hy);
        efield(nx, ny, naz, dz, ez);
        hfield(nx, ny, &pml, ez, ihx, ihy, hx, hy);
    }

    float ntime = omp_get_wtime();
    printf("Total compute time on CPU: %.3f s\n", ntime - stime);

    for (int i = 2*ny; i < 2*ny+50; i++)
        printf("%e\n", ez[i]);

    free(pml.fx1);
    free(pml.fx2);
    free(pml.fx3);
    free(pml.fy1);
    free(pml.fy2);
    free(pml.fy3);
    free(pml.gx2);
    free(pml.gx3);
    free(pml.gy2);
    free(pml.gy3);
    free(naz);
    free(ihx);
    free(ihy);
    free(dz);
    free(ez);
    free(hx);
    free(hy);

    return 0;
}
