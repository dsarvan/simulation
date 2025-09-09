/* File: test_2_2.c
 * Name: D.Saravanan
 * Date: 07/12/2021
 * Simulation of a pulse striking a dielectric medium and implements
 * the discrete Fourier transform with a Gaussian pulse as its source
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


typedef struct {
    float *nax, *nbx;
} medium;


typedef struct {
    float *r_pt, *i_pt;
    float *r_in, *i_in;
} ftrans;


float gaussian(int t, int t0, float sigma) {
    return exp(-0.5*(t - t0)/sigma*(t - t0)/sigma);
}


void fourier(int t, int nf, int nx, float dt, float *freq, float *ex, ftrans *ft) {
    for (int n = 0; n < nf; n++) {
        for (int i = 0; i < nx; i++) {
            /* calculate the Fourier transform of Ex field */
            int m = n*nx+i;
            ft->r_pt[m] += cos(2*M_PI*freq[n]*dt*t) * ex[i];
            ft->i_pt[m] -= sin(2*M_PI*freq[n]*dt*t) * ex[i];
        }
        if (t < nx/2) {
            /* calculate the Fourier transform of input source */
            ft->r_in[n] += cos(2*M_PI*freq[n]*dt*t) * ex[10];
            ft->i_in[n] -= sin(2*M_PI*freq[n]*dt*t) * ex[10];
        }
    }
}


void dxfield(int t, int nx, float *dx, float *hy) {
    /* calculate the electric flux density Dx */
    for (int i = 1; i < nx; i++)
        dx[i] += 0.5 * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse at the low end */
    dx[1] += gaussian(t, 50, 10.0f);
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
    float sigma = 0.0;  /* conductivity (S/m) */
    medium md = dielectric(nx, dt, epsr, sigma);

    /* frequency 100 MHz, 200 MHz, 500 MHz */
    float freq[] = {100e6, 200e6, 500e6};
    int nf = (sizeof freq)/(sizeof freq[0]);  /* number of frequencies */

    ftrans ft;
    ft.r_pt = (float*) calloc(nf*nx, sizeof(*ft.r_pt));
    ft.i_pt = (float*) calloc(nf*nx, sizeof(*ft.i_pt));
    ft.r_in = (float*) calloc(nf, sizeof(*ft.r_in));
    ft.i_in = (float*) calloc(nf, sizeof(*ft.i_in));

    float *amplt = (float*) calloc(nf*nx, sizeof(*amplt));
    float *phase = (float*) calloc(nf*nx, sizeof(*phase));

    clock_t stime = clock();

    for (int t = 1; t <= ns; t++) {
        dxfield(t, nx, dx, hy);
        exfield(nx, &md, dx, ix, ex);
        fourier(t, nf, nx, dt, freq, ex, &ft);
        hyfield(nx, ex, hy, bc);
    }

    /* calculate the amplitude and phase at each frequency */
    for (int n = 0; n < nf; n++) {
        for (int i = 0; i < nx; i++) {
            int m = n*nx+i;
            amplt[m] = 1/hypotf(ft.r_in[n],ft.i_in[n]) * hypotf(ft.r_pt[m],ft.i_pt[m]);
            phase[m] = atan2f(ft.i_pt[m],ft.r_pt[m]) - atan2f(ft.i_in[n],ft.r_in[n]);
        }
    }

    clock_t ntime = clock();
    float time = (ntime - stime)*1000/CLOCKS_PER_SEC;
    printf("Total compute time on CPU: %.3f s\n", time/1000.0f);

    for (int i = 0; i < 50; i++)
        printf("%e\n", ex[i]);

    free(ft.r_pt);
    free(ft.i_pt);
    free(ft.r_in);
    free(ft.i_in);
    free(md.nax);
    free(md.nbx);
    free(amplt);
    free(phase);
    free(dx);
    free(ex);
    free(ix);
    free(hy);

    return 0;
}
