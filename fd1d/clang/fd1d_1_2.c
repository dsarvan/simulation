/* File: fdtd12.c
 * Name: D.Saravanan
 * Date: 19/10/2021
 * Simulation of a pulse with absorbing boundary conditions
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double gaussian(int ts, int t0, double sigma) {
    return exp(-0.5*(ts - t0)/sigma*(ts - t0)/sigma);
}


void exfield(int ts, int nx, double *hy, double *ex) {
    /* calculate the Ex field */
    for (int i = 1; i < nx; i++)
        ex[i] += 0.5 * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse in the middle */
    ex[nx/2] = gaussian(ts, 40, 12.0);
}


void hyfield(int nx, double *bc, double *ex, double *hy) {
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = 0; i < nx-1; i++)
        hy[i] += 0.5 * (ex[i] - ex[i+1]);
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 570;  /* number of time steps */

    double *ex = malloc(nx*sizeof(*ex));
    double *hy = malloc(nx*sizeof(*hy));

    for (int i = 0; i < nx; i++) {
        ex[i] = 0.0;
        hy[i] = 0.0;
    }

    double bc[4] = {0.0};

    for (int ts = 1; ts <= ns; ts++) {
        exfield(ts, nx, hy, ex);
        hyfield(nx, bc, ex, hy);
    }

    free(ex);
    free(hy);

    return 0;
}
