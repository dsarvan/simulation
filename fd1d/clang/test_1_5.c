/* File: test15.c
 * Name: D.Saravanan
 * Date: 29/10/2021
 * Simulation of a propagating sinusoidal wave of 700 MHz striking a lossy
 * dielectric with a dielectric constant of 4 and conductivity of 0.04 (S/m)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


typedef struct {
    float *ca;
    float *cb;
} tuple;


float sinusoidal(int ts, float ds, float freq) {
    float dt = ds/6e8f;  /* time step (s) */
    return sinf(2*M_PI*freq*dt*ts);
}


void exfield(int ts, int nx, float *ca, float *cb, float *hy, float *ex) {
    /* calculate the Ex field */
    for (int i = 1; i < nx; i++)
        ex[i] = ca[i] * ex[i] + cb[i] * (hy[i-1] - hy[i]);
    /* put a sinusoidal wave at the low end */
    ex[1] += sinusoidal(ts, 0.01f, 700e6f);
}


void hyfield(int nx, float *bc, float *ex, float *hy) {
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = 0; i < nx-1; i++)
        hy[i] += 0.5f * (ex[i] - ex[i+1]);
}


tuple dielectric(int nx, float dt, float epsr, float sigma) {
    tuple n;
    n.ca = malloc(nx*sizeof(*n.ca));
    n.cb = malloc(nx*sizeof(*n.cb));
    for (int i = 0; i < nx; n.ca[i] = 1.0f, i++);
    for (int i = 0; i < nx; n.cb[i] = 0.5f, i++);
    float eps0 = 8.854e-12f;  /* vacuum permittivity (F/m) */
    float epsf = dt*sigma/(2*eps0*epsr);
    for (int i = nx/2; i < nx; n.ca[i] = (1 - epsf)/(1 + epsf), i++);
    for (int i = nx/2; i < nx; n.cb[i] = 0.5f/(epsr*(1 + epsf)), i++);
    return n;
}


int main() {

    int nx = 38000;  /* number of grid points */
    int ns = 40000;  /* number of time steps */

    float *ex = malloc(nx*sizeof(*ex));
    float *hy = malloc(nx*sizeof(*hy));

    for (int i = 0; i < nx; i++) {
        ex[i] = 0.0f;
        hy[i] = 0.0f;
    }

    float bc[4] = {0.0f};

    float ds = 0.01f;  /* spatial step (m) */
    float dt = ds/6e8f;  /* time step (s) */
    float epsr = 4.0f;  /* relative permittivity */
    float sigma = 0.04f;  /* conductivity (S/m) */
    tuple n = dielectric(nx, dt, epsr, sigma);

    float stime = omp_get_wtime();

    for (int ts = 1; ts <= ns; ts++) {
        exfield(ts, nx, n.ca, n.cb, hy, ex);
        hyfield(nx, bc, ex, hy);
    }

    float ntime = omp_get_wtime();
    printf("Total compute time on CPU: %.3f s\n", ntime - stime);

    for (int i = 0; i < 50; i++)
        printf("%e\n", ex[i]);

    free(n.ca);
    free(n.cb);
    free(ex);
    free(hy);

    return 0;
}
