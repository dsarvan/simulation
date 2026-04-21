/* File: test_2_1.c
 * Name: D.Saravanan
 * Date: 25/11/2021
 * Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
 * dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


typedef struct {
    float *nax, *nbx;
} Medium;


float sinusoidal(int ts, float ds, float freq) {
    float dt = ds/6e8f;  /* time step (s) */
    return sinf(2*M_PI*freq*dt*ts);
}


void dxfield(int ts, int nx, float *hy, float *dx) {
    /* calculate the electric flux density Dx */
    for (int i = 1; i < nx; i++)
        dx[i] += 0.5f * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    dx[1] += sinusoidal(ts, 0.01f, 700e6f);
}


void exfield(int nx, Medium md, float *ix, float *dx, float *ex) {
    /* calculate the Ex field from Dx */
    for (int i = 1; i < nx; i++) {
        ex[i] = md.nax[i] * (dx[i] - ix[i]);
        ix[i] += md.nbx[i] * ex[i];
    }
}


void hyfield(int nx, float *bc, float *ex, float *hy) {
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = 0; i < nx-1; i++)
        hy[i] += 0.5f * (ex[i] - ex[i+1]);
}


Medium dielectric(int nx, float dt, float epsr, float sigma) {
    Medium md;
    md.nax = malloc(nx*sizeof(*md.nax));
    md.nbx = malloc(nx*sizeof(*md.nbx));
    for (int i = 0; i < nx; md.nax[i] = 1.0f, i++);
    for (int i = 0; i < nx; md.nbx[i] = 0.0f, i++);
    float eps0 = 8.854e-12f;  /* vacuum permittivity (F/m) */
    for (int i = nx/2; i < nx; i++) {
        md.nax[i] = 1/(epsr + sigma*dt/eps0);
        md.nbx[i] = sigma*dt/eps0;
    }
    return md;
}


int main() {

    int nx = 38000;  /* number of grid points */
    int ns = 40000;  /* number of time steps */

    float *dx = malloc(nx*sizeof(*dx));
    float *ex = malloc(nx*sizeof(*ex));
    float *ix = malloc(nx*sizeof(*ix));
    float *hy = malloc(nx*sizeof(*hy));

    for (int i = 0; i < nx; i++) {
        dx[i] = 0.0f;
        ex[i] = 0.0f;
        ix[i] = 0.0f;
        hy[i] = 0.0f;
    }

    float bc[4] = {0.0f};

    float ds = 0.01f;  /* spatial step (m) */
    float dt = ds/6e8f;  /* time step (s) */
    float epsr = 4.0f;  /* relative permittivity */
    float sigma = 0.04f;  /* conductivity (S/m) */
    Medium md = dielectric(nx, dt, epsr, sigma);

    float stime = omp_get_wtime();

    for (int ts = 1; ts <= ns; ts++) {
        dxfield(ts, nx, hy, dx);
        exfield(nx, md, ix, dx, ex);
        hyfield(nx, bc, ex, hy);
    }

    float ntime = omp_get_wtime();
    printf("Total compute time on CPU: %.3f s\n", ntime - stime);

    for (int i = 0; i < 50; i++)
        printf("%e\n", ex[i]);

    free(md.nax);
    free(md.nbx);
    free(dx);
    free(ex);
    free(ix);
    free(hy);

    return 0;
}
