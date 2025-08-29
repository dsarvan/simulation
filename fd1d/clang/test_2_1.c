/* File: test_2_1.c
 * Name: D.Saravanan
 * Date: 25/11/2021
 * Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
 * dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


typedef struct {
    float *nax, *nbx;
} medium;


float sinusoidal(int t, float ds, float freq) {
    float dt = ds/6e8;  /* time step (s) */
    return sin(2*M_PI*freq*dt*t);
}


void dxfield(int t, int nx, float *dx, float *hy) {
    /* calculate the electric flux density Dx */
    for (int i = 1; i < nx; i++)
        dx[i] += 0.5 * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    dx[1] += sinusoidal(t, 0.01f, 700e6);
}


void exfield(int nx, medium *md, float *dx, float *ix, float *ex) {
    /* calculate the Ex field from Dx */
    for (int i = 1; i < nx; i++) {
        ex[i] = md->nax[i] * (dx[i] - ix[i]);
        ix[i] += md->nbx[i] * ex[i];
    }
}


void hyfield(int nx, float *ex, float *hy, float *bc) {
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = 0; i < nx-1; i++)
        hy[i] += 0.5 * (ex[i] - ex[i+1]);
}


medium dielectric(int nx, float dt, float epsr, float sigma) {
    medium md;
    md.nax = (float*) calloc(nx, sizeof(*md.nax));
    md.nbx = (float*) calloc(nx, sizeof(*md.nbx));
    for (int i = 0; i < nx; md.nax[i] = 1.0f, i++);
    float eps0 = 8.854e-12;  /* vacuum permittivity (F/m) */
    for (int i = nx/2; i < nx; i++) {
        md.nax[i] = 1/(epsr + sigma*dt/eps0);
        md.nbx[i] = sigma*dt/eps0;
    }
    return md;
}


int main() {

    int nx = 38000;  /* number of grid points */
    int ns = 40000;  /* number of time steps */

    float *dx = (float*) calloc(nx, sizeof(*dx));
    float *ex = (float*) calloc(nx, sizeof(*ex));
    float *ix = (float*) calloc(nx, sizeof(*ix));
    float *hy = (float*) calloc(nx, sizeof(*hy));

    float bc[4] = {0.0f};

    float ds = 0.01;  /* spatial step (m) */
    float dt = ds/6e8;  /* time step (s) */
    float epsr = 4.0;  /* relative permittivity */
    float sigma = 0.04;  /* conductivity (S/m) */
    medium md = dielectric(nx, dt, epsr, sigma);

    clock_t stime = clock();

    for (int t = 1; t <= ns; t++) {
        dxfield(t, nx, dx, hy);
        exfield(nx, &md, dx, ix, ex);
        hyfield(nx, ex, hy, bc);
    }

    clock_t ntime = clock();
    float time = (ntime - stime)*1000/CLOCKS_PER_SEC;
    printf("Total compute time on CPU: %.3f s\n", time/1000.0f);

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
