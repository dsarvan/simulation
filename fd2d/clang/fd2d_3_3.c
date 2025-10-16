/* File: fd2d_3_3.c
 * Name: D.Saravanan
 * Date: 19/01/2022
 * Simulation of a plane wave pulse propagating in free space in the transverse
 * magnetic (TM) mode with the two-dimensional perfectly matched layer (PML)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


typedef struct {
    double *fx1;
    double *fx2;
    double *fx3;
    double *fy1;
    double *fy2;
    double *fy3;
    double *gx2;
    double *gx3;
    double *gy2;
    double *gy3;
} pmlayer;


double gaussian(int t, int t0, double sigma) {
    return exp(-0.5*(t - t0)/sigma*(t - t0)/sigma);
}


void ezinct(int ny, double *ezi, double *hxi, double *bc) {
    /* calculate the incident Ez */
    #pragma omp parallel for
    for (int j = 1; j < ny; j++) {
        ezi[j] += 0.5 * (hxi[j-1] - hxi[j]);
    }
    /* absorbing boundary conditions */
    ezi[0] = bc[0], bc[0] = bc[1], bc[1] = ezi[1];
    ezi[ny-1] = bc[3], bc[3] = bc[2], bc[2] = ezi[ny-2];
}


void dfield(int t, int nx, int ny, pmlayer *pml, double *ezi, double *dz, double *hx, double *hy) {
    /* calculate the electric flux density Dz */
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            int n = i*ny+j;
            dz[n] = pml->gx3[i] * pml->gy3[j] * dz[n] + pml->gx2[i] * pml->gy2[j] * 0.5 * (hy[n] - hy[n-ny] - hx[n] + hx[n-1]);
        }
    }
    /* put a Gaussian pulse at the low end */
    ezi[3] = gaussian(t, 20, 8.0);
}


void inctdz(int nx, int ny, int npml, double *hxi, double *dz) {
    /* incident Dz values */
    #pragma omp parallel for
    for (int i = npml-1; i <= nx-npml; i++) {
        dz[i*ny+(npml-1)] += 0.5 * hxi[npml-2];
        dz[i*ny+(ny-npml)] -= 0.5 * hxi[ny-npml];
    }
}


void efield(int nx, int ny, double *naz, double *dz, double *ez) {
    /* calculate the Ez field from Dz */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int n = i*ny+j;
            ez[n] = naz[n] * dz[n];
        }
    }
}


void hxinct(int ny, double *ezi, double *hxi) {
    /* calculate the incident Hx */
    #pragma omp parallel for
    for (int j = 0; j < ny-1; j++) {
        hxi[j] += 0.5 * (ezi[j] - ezi[j+1]);
    }
}


void hfield(int nx, int ny, pmlayer *pml, double *ez, double *ihx, double *ihy, double *hx, double *hy) {
    /* calculate the Hx and Hy field */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < ny-1; j++) {
            int n = i*ny+j;
            ihx[n] += ez[n] - ez[n+1];
            ihy[n] += ez[n] - ez[n+ny];
            hx[n] = pml->fy3[j] * hx[n] + pml->fy2[j] * (0.5 * ez[n] - 0.5 * ez[n+1] + pml->fx1[i] * ihx[n]);
            hy[n] = pml->fx3[i] * hy[n] - pml->fx2[i] * (0.5 * ez[n] - 0.5 * ez[n+ny] + pml->fy1[j] * ihy[n]);
        }
    }
}


void incthx(int nx, int ny, int npml, double *ezi, double *hx) {
    /* incident Hx values */
    #pragma omp parallel for
    for (int i = npml-1; i <= nx-npml; i++) {
        hx[i*ny+(npml-2)] += 0.5 * ezi[npml-1];
        hx[i*ny+(ny-npml)] -= 0.5 * ezi[ny-npml];
    }
}


void incthy(int nx, int ny, int npml, double *ezi, double *hy) {
    /* incident Hy values */
    #pragma omp parallel for
    for (int j = npml-1; j <= ny-npml; j++) {
        hy[(npml-2)*ny+j] -= 0.5 * ezi[j];
        hy[(nx-npml)*ny+j] += 0.5 * ezi[j];
    }
}


void pmlparam(int nx, int ny, int npml, pmlayer *pml) {
    /* calculate the two-dimensional perfectly matched layer (PML) parameters */
    for (int n = 0; n < npml; n++) {
        double xm = 0.33*(npml-n)/npml*(npml-n)/npml*(npml-n)/npml;
        double xn = 0.33*(npml-n-0.5)/npml*(npml-n-0.5)/npml*(npml-n-0.5)/npml;
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

    double *ezi = (double*) calloc(ny, sizeof(*ezi));
    double *hxi = (double*) calloc(ny, sizeof(*hxi));

    double *dz = (double*) calloc(nx*ny, sizeof(*dz));
    double *ez = (double*) calloc(nx*ny, sizeof(*ez));
    double *hx = (double*) calloc(nx*ny, sizeof(*hx));
    double *hy = (double*) calloc(nx*ny, sizeof(*hy));

    double *ihx = (double*) calloc(nx*ny, sizeof(*ihx));
    double *ihy = (double*) calloc(nx*ny, sizeof(*ihy));

    double *naz = (double*) calloc(nx*ny, sizeof(*naz));
    for (int i = 0; i < nx*ny; naz[i] = 1.0, i++);

    double bc[4] = {0.0};

    pmlayer pml;
    pml.fx1 = (double*) calloc(nx, sizeof(*pml.fx1));
    pml.fx2 = (double*) calloc(nx, sizeof(*pml.fx2));
    pml.fx3 = (double*) calloc(nx, sizeof(*pml.fx3));
    pml.fy1 = (double*) calloc(ny, sizeof(*pml.fy1));
    pml.fy2 = (double*) calloc(ny, sizeof(*pml.fy2));
    pml.fy3 = (double*) calloc(ny, sizeof(*pml.fy3));
    pml.gx2 = (double*) calloc(nx, sizeof(*pml.gx2));
    pml.gx3 = (double*) calloc(nx, sizeof(*pml.gx3));
    pml.gy2 = (double*) calloc(ny, sizeof(*pml.gy2));
    pml.gy3 = (double*) calloc(ny, sizeof(*pml.gy3));

    for (int i = 0; i < nx; i++) {
        pml.fx2[i] = 1.0;
        pml.fx3[i] = 1.0;
        pml.gx2[i] = 1.0;
        pml.gx3[i] = 1.0;
    }

    for (int i = 0; i < ny; i++) {
        pml.fy2[i] = 1.0;
        pml.fy3[i] = 1.0;
        pml.gy2[i] = 1.0;
        pml.gy3[i] = 1.0;
    }

    int npml = 8;  /* pml thickness */
    pmlparam(nx, ny, npml, &pml);

    double ds = 0.01;  /* spatial step (m) */
    double dt = ds/6e8;  /* time step (s) */

    for (int t = 1; t <= ns; t++) {
        ezinct(ny, ezi, hxi, bc);
        dfield(t, nx, ny, &pml, ezi, dz, hx, hy);
        inctdz(nx, ny, npml, hxi, dz);
        efield(nx, ny, naz, dz, ez);
        hxinct(ny, ezi, hxi);
        hfield(nx, ny, &pml, ez, ihx, ihy, hx, hy);
        incthx(nx, ny, npml, ezi, hx);
        incthy(nx, ny, npml, ezi, hy);
    }

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
    free(ezi);
    free(hxi);
    free(ihx);
    free(ihy);
    free(dz);
    free(ez);
    free(hx);
    free(hy);

    return 0;
}
