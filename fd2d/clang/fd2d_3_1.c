/* File: fd2d_3_1.c
 * Name: D.Saravanan
 * Date: 17/01/2022
 * Simulation of a pulse in free space in the transverse magnetic (TM) mode
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double gaussian(int t, int t0, double sigma) {
    return exp(-0.5 * ((t - t0)/sigma) * ((t - t0)/sigma));
}


void dfield(int t, int nx, int ny, double *dz, double *hx, double *hy) {
    /* calculate the electric flux density Dz */
    for (int j = 1; j < ny; j++) {
        for (int i = 1; i < nx; i++) {
            int n = j*nx+i;
            dz[n] += 0.5 * (hy[n] - hy[n-nx] - hx[n] + hx[n-1]);
        }
    }
    /* put a Gaussian pulse in the middle */
    dz[ny/2*nx+nx/2] = gaussian(t, 20, 6);
}


void efield(int nx, int ny, double *naz, double *dz, double *ez) {
    /* calculate the Ez field from Dz */
    for (int j = 1; j < ny; j++) {
        for (int i = 1; i < nx; i++) {
            int n = j*nx+i;
            ez[n] = naz[n] * dz[n];
        }
    }
}


void hfield(int nx, int ny, double *ez, double *hx, double *hy) {
    /* calculate the Hx and Hy field */
    for (int j = 0; j < ny - 1; j++) {
        for (int i = 0; i < nx - 1; i++) {
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

    double *dz = (double *) calloc(nx*ny, sizeof(*dz));
    double *ez = (double *) calloc(nx*ny, sizeof(*ez));
    double *hx = (double *) calloc(nx*ny, sizeof(*hx));
    double *hy = (double *) calloc(nx*ny, sizeof(*hy));

    double *naz = (double *) calloc(nx*ny, sizeof(*naz));
    for (int i = 0; i < nx*ny; naz[i] = 1.0f, i++);

    for (int t = 1; t <= ns; t++) {
        dfield(t, nx, ny, dz, hx, hy);
        efield(nx, ny, naz, dz, ez);
        hfield(nx, ny, ez, hx, hy);
    }

    free(naz);
    free(dz);
    free(ez);
    free(hx);
    free(hy);

    return 0;
}
