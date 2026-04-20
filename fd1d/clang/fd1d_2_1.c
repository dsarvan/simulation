/* File: fd1d_2_1.c
 * Name: D.Saravanan
 * Date: 25/11/2021
 * Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
 * dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    double *nax, *nbx;
} Medium;


double sinusoidal(int ts, double ds, double freq) {
    double dt = ds/6e8;  /* time step (s) */
    return sin(2*M_PI*freq*dt*ts);
}


void dxfield(int ts, int nx, double *hy, double *dx) {
    /* calculate the electric flux density Dx */
    for (int i = 1; i < nx; i++)
        dx[i] += 0.5 * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    dx[1] += sinusoidal(ts, 0.01, 700e6);
}


void exfield(int nx, Medium md, double *ix, double *dx, double *ex) {
    /* calculate the Ex field from Dx */
    for (int i = 1; i < nx; i++) {
        ex[i] = md.nax[i] * (dx[i] - ix[i]);
        ix[i] += md.nbx[i] * ex[i];
    }
}


void hyfield(int nx, double *bc, double *ex, double *hy) {
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = 0; i < nx-1; i++)
        hy[i] += 0.5 * (ex[i] - ex[i+1]);
}


Medium dielectric(int nx, double dt, double epsr, double sigma) {
    Medium md;
    md.nax = malloc(nx*sizeof(*md.nax));
    md.nbx = malloc(nx*sizeof(*md.nbx));
    for (int i = 0; i < nx; md.nax[i] = 1.0, i++);
    for (int i = 0; i < nx; md.nbx[i] = 0.0, i++);
    double eps0 = 8.854e-12;  /* vacuum permittivity (F/m) */
    for (int i = nx/2; i < nx; i++) {
        md.nax[i] = 1/(epsr + sigma*dt/eps0);
        md.nbx[i] = sigma*dt/eps0;
    }
    return md;
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 740;  /* number of time steps */

    double *dx = malloc(nx*sizeof(*dx));
    double *ex = malloc(nx*sizeof(*ex));
    double *ix = malloc(nx*sizeof(*ix));
    double *hy = malloc(nx*sizeof(*hy));

    for (int i = 0; i < nx; i++) {
        dx[i] = 0.0;
        ex[i] = 0.0;
        ix[i] = 0.0;
        hy[i] = 0.0;
    }

    double bc[4] = {0.0};

    double ds = 0.01;  /* spatial step (m) */
    double dt = ds/6e8;  /* time step (s) */
    double epsr = 4.0;  /* relative permittivity */
    double sigma = 0.04;  /* conductivity (S/m) */
    Medium md = dielectric(nx, dt, epsr, sigma);

    for (int ts = 1; ts <= ns; ts++) {
        dxfield(ts, nx, hy, dx);
        exfield(nx, md, ix, dx, ex);
        hyfield(nx, bc, ex, hy);
    }

    free(md.nax);
    free(md.nbx);
    free(dx);
    free(ex);
    free(ix);
    free(hy);

    return 0;
}
