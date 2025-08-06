/* File: fd1d_1_4.c
 * Name: D.Saravanan
 * Date: 22/10/2021
 * Simulation of a propagating sinusoidal striking a dielectric medium
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double sinusoidal(int t, double ds, double freq) {
    double dt = ds/6e8;  /* time step (s) */
    return sin(2*M_PI*freq*dt*t);
}


void exfield(int t, int nx, double *cb, double *ex, double *hy) {
    /* calculate the Ex field */
    for (int i = 1; i < nx; i++)
        ex[i] += cb[i] * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    ex[1] += sinusoidal(t, 0.01, 700e6);
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

    double ds = 0.01;  /* spatial step (m) */
    double dt = ds/6e8;  /* time step (s) */
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
