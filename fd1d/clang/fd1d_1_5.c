/* File: fd1d_1_5.c
 * Name: D.Saravanan
 * Date: 29/10/2021
 * Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
 * dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    double *ca;
    double *cb;
} tuple;


double sinusoidal(int ts, double ds, double freq) {
    double dt = ds/6e8;  /* time step (s) */
    return sin(2*M_PI*freq*dt*ts);
}


void exfield(int ts, int nx, double *ca, double *cb, double *hy, double *ex) {
    /* calculate the Ex field */
    for (int i = 1; i < nx; i++)
        ex[i] = ca[i] * ex[i] + cb[i] * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    ex[1] += sinusoidal(ts, 0.01, 700e6);
}


void hyfield(int nx, double *bc, double *ex, double *hy) {
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = 0; i < nx-1; i++)
        hy[i] += 0.5 * (ex[i] - ex[i+1]);
}


tuple dielectric(int nx, double dt, double epsr, double sigma) {
    tuple n;
    n.ca = malloc(nx*sizeof(*n.ca));
    n.cb = malloc(nx*sizeof(*n.cb));
    for (int i = 0; i < nx; n.ca[i] = 1.0, i++);
    for (int i = 0; i < nx; n.cb[i] = 0.5, i++);
    double eps0 = 8.854e-12;  /* vacuum permittivity (F/m) */
    double epsf = dt*sigma/(2*eps0*epsr);
    for (int i = nx/2; i < nx; n.ca[i] = (1 - epsf)/(1 + epsf), i++);
    for (int i = nx/2; i < nx; n.cb[i] = 0.5/(epsr*(1 + epsf)), i++);
    return n;
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 740;  /* number of time steps */

    double *ex = malloc(nx*sizeof(*ex));
    double *hy = malloc(nx*sizeof(*hy));

    for (int i = 0; i < nx; i++) {
        ex[i] = 0.0;
        hy[i] = 0.0;
    }

    double bc[4] = {0.0};

    double ds = 0.01;  /* spatial step (m) */
    double dt = ds/6e8;  /* time step (s) */
    double epsr = 4.0;  /* relative permittivity */
    double sigma = 0.04;  /* conductivity (S/m) */
    tuple n = dielectric(nx, dt, epsr, sigma);

    for (int ts = 1; ts <= ns; ts++) {
        exfield(ts, nx, n.ca, n.cb, hy, ex);
        hyfield(nx, bc, ex, hy);
    }

    free(n.ca);
    free(n.cb);
    free(ex);
    free(hy);

    return 0;
}
