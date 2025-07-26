/* File: fd1d_1_3.c
 * Name: D.Saravanan
 * Date: 19/10/2021
 * Simulation of a pulse hitting a dielectric medium
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double gaussian(int t, int t0, double sigma) {
    return exp(-0.5*(t - t0)/sigma*(t - t0)/sigma);
}


void exfield(int t, int nx, double *cb, double *ex, double *hy) {
    /* calculate the Ex field */
    for (int i = 1; i < nx; i++)
        ex[i] += cb[i] * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse at the low end */
    ex[1] += gaussian(t, 40, 12.0);
}


void hyfield(int nx, double *ex, double *hy, double *bc) {
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = 0; i < nx-1; i++)
        hy[i] += 0.5 * (ex[i] - ex[i+1]);
}


double *dielectric(int nx, double epsr) {
    double *cb = (double*) calloc(nx, sizeof(*cb));
    for (int i = 0; i < nx; cb[i] = 0.5, i++);
    for (int i = nx/2; i < nx; cb[i] = 0.5/epsr, i++);
    return cb;
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 740;  /* number of time steps */

    double *ex = (double*) calloc(nx, sizeof(*ex));
    double *hy = (double*) calloc(nx, sizeof(*hy));

    double bc[4] = {0.0};

    double epsr = 4.0;  /* relative permittivity */
    double *cb = dielectric(nx, epsr);

    for (int t = 1; t <= ns; t++) {
        exfield(t, nx, cb, ex, hy);
        hyfield(nx, ex, hy, bc);
    }

    free(cb);
    free(ex);
    free(hy);

    return 0;
}
