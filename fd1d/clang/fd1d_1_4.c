/* File: fd1d_1_4.c
 * Name: D.Saravanan
 * Date: 22/10/2021
 * Simulation of a propagating sinusoidal striking a dielectric medium
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double sinusoidal(int t, double ddx, double freq) {
    double dt = ddx/6e8;  /* time step */
    return sin(2 * M_PI * freq * dt * t);
}


void exfield(int t, int nx, double *cb, double *ex, double *hy, double *bc) {
    /* calculate the Ex field */
    for (int i = 1; i < nx; i++)
        ex[i] = ex[i] + cb[i] * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    ex[1] = ex[1] + sinusoidal(t, 0.01, 700e6);
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
}


void hyfield(int nx, double *ex, double *hy) {
    /* calculate the Hy field */
    for (int i = 0; i < nx - 1; i++)
        hy[i] = hy[i] + 0.5 * (ex[i] - ex[i+1]);
}


double *dielectric(int nx, double epsr) {
    double *cb = (double *) calloc(nx, sizeof(*cb));
    for (int i = 0; i < nx; cb[i] = 0.5f, i++);
    for (int i = nx/2; i < nx; cb[i] = 0.5/epsr, i++);
    return cb;
}


int main() {

    int nx = 1024;
    int ns = 1500;

    double *ex = (double *) calloc(nx, sizeof(*ex));
    double *hy = (double *) calloc(nx, sizeof(*hy));

    double bc[4] = {0.0f};

    double ddx = 0.01;  /* cell size (m) */
    double dt = ddx/6e8;  /* time step */
    double epsr = 4;  /* relative permittivity */
    double *cb = dielectric(nx, epsr);

    for (int t = 1; t <= ns; t++) {
        exfield(t, nx, cb, ex, hy, bc);
        hyfield(nx, ex, hy);
    }

    free(cb);
    free(ex);
    free(hy);

    return 0;
}
