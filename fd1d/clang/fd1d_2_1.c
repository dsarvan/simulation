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
} medium;


double sinusoidal(int t, double ds, double freq) {
    double dt = ds/6e8;  /* time step (s) */
    return sin(2 * M_PI * freq * dt * t);
}


void dxfield(int t, int nx, double *dx, double *hy) {
    /* calculate the electric flux density Dx */
    for (int i = 1; i < nx; i++)
        dx[i] = dx[i] + 0.5 * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    dx[1] = dx[1] + sinusoidal(t, 0.01, 700e6);
}


void exfield(int nx, medium *md, double *dx, double *ix, double *ex) {
    /* calculate the Ex field from Dx */
    for (int i = 1; i < nx; i++) {
        ex[i] = md->nax[i] * (dx[i] - ix[i]);
        ix[i] = ix[i] + md->nbx[i] * ex[i];
    }
}


void hyfield(int nx, double *ex, double *hy, double *bc) {
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = 0; i < nx - 1; i++)
        hy[i] = hy[i] + 0.5 * (ex[i] - ex[i+1]);
}


medium dielectric(int nx, double dt, double epsr, double sigma) {
    medium md;
    md.nax = (double *) calloc(nx, sizeof(*md.nax));
    md.nbx = (double *) calloc(nx, sizeof(*md.nbx));
    for (int i = 0; i < nx; md.nax[i] = 1.0f, i++);
    double eps0 = 8.854e-12;  /* vacuum permittivity (F/m) */
    for (int i = nx/2; i < nx; i++) {
        md.nax[i] = 1/(epsr + (sigma * dt/eps0));
        md.nbx[i] = sigma * dt/eps0;
    }
    return md;
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 740;  /* number of time steps */

    double *dx = (double *) calloc(nx, sizeof(*dx));
    double *ex = (double *) calloc(nx, sizeof(*ex));
    double *ix = (double *) calloc(nx, sizeof(*ix));
    double *hy = (double *) calloc(nx, sizeof(*hy));

    double bc[4] = {0.0f};

    double ds = 0.01;  /* spatial step (m) */
    double dt = ds/6e8;  /* time step (s) */
    double epsr = 4;  /* relative permittivity */
    double sigma = 0.04;  /* conductivity (S/m) */
    medium md = dielectric(nx, dt, epsr, sigma);

    for (int t = 1; t <= ns; t++) {
        dxfield(t, nx, dx, hy);
        exfield(nx, &md, dx, ix, ex);
        hyfield(nx, ex, hy, bc);
    }

    free(md.nax);
    free(md.nbx);
    free(dx);
    free(ex);
    free(ix);
    free(hy);

    return 0;
}
