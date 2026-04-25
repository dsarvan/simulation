/* File: test22.c
 * Name: D.Saravanan
 * Date: 07/12/2021
 * Simulation of a pulse striking a dielectric medium and implements
 * the discrete Fourier transform with a Gaussian pulse as its source
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


typedef struct {
    float *nax, *nbx;
} Medium;


typedef struct {
    float *rpt, *ipt;
    float *rin, *iin;
} FTrans;


float gaussian(int ts, int t0, float sigma) {
    return expf(-0.5f*(ts - t0)/sigma*(ts - t0)/sigma);
}


void fourier(int ts, int nf, int nx, float dt, float *freq, float *ex, FTrans ft) {
    for (int n = 0; n < nf; n++) {
        /* calculate the Fourier transform of input source */
        ft.rin[n] += cosf(2*M_PI*freq[n]*dt*ts) * ex[10];
        ft.iin[n] -= sinf(2*M_PI*freq[n]*dt*ts) * ex[10];
        for (int i = 0; i < nx; i++) {
            /* calculate the Fourier transform of Ex field */
            ft.rpt[n*nx+i] += cosf(2*M_PI*freq[n]*dt*ts) * ex[i];
            ft.ipt[n*nx+i] -= sinf(2*M_PI*freq[n]*dt*ts) * ex[i];
        }
    }
}


void dxfield(int ts, int nx, float *hy, float *dx) {
    /* calculate the electric flux density Dx */
    for (int i = 1; i < nx; i++)
        dx[i] += 0.5f * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse at the low end */
    dx[1] += gaussian(ts, 50, 10.0f);
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
    float sigma = 0.0f;  /* conductivity (S/m) */
    Medium md = dielectric(nx, dt, epsr, sigma);

    /* frequency 100 MHz, 200 MHz, 500 MHz */
    float freq[] = {100e6f, 200e6f, 500e6f};
    int nf = (sizeof freq)/(sizeof freq[0]);  /* number of frequencies */

    FTrans ft;
    ft.rpt = malloc(nf*nx*sizeof(*ft.rpt));
    ft.ipt = malloc(nf*nx*sizeof(*ft.ipt));
    ft.rin = malloc(nf*sizeof(*ft.rin));
    ft.iin = malloc(nf*sizeof(*ft.iin));

    for (int i = 0; i < nf*nx; i++) {
        ft.rpt[i] = 0.0f;
        ft.ipt[i] = 0.0f;
    }

    for (int i = 0; i < nf; i++) {
        ft.rin[i] = 0.0f;
        ft.iin[i] = 0.0f;
    }

    float *amplt = malloc(nf*nx*sizeof(*amplt));
    float *phase = malloc(nf*nx*sizeof(*phase));

    for (int i = 0; i < nf*nx; i++) {
        amplt[i] = 0.0f;
        phase[i] = 0.0f;
    }

    float stime = omp_get_wtime();

    for (int ts = 1; ts <= ns; ts++) {
        dxfield(ts, nx, hy, dx);
        exfield(nx, md, ix, dx, ex);
        fourier(ts, nf, nx, dt, freq, ex, ft);
        hyfield(nx, bc, ex, hy);
    }

    /* calculate the amplitude and phase at each frequency */
    for (int n = 0; n < nf; n++) {
        for (int i = 0; i < nx; i++) {
            int m = n*nx+i;
            amplt[m] = 1/hypotf(ft.rin[n],ft.iin[n]) * hypotf(ft.rpt[m],ft.ipt[m]);
            phase[m] = atan2f(ft.ipt[m],ft.rpt[m]) - atan2f(ft.iin[n],ft.rin[n]);
        }
    }

    float ntime = omp_get_wtime();
    printf("Total compute time on CPU: %.3f s\n", ntime - stime);

    for (int i = 0; i < 50; i++)
        printf("%e\n", ex[i]);

    free(ft.rpt);
    free(ft.ipt);
    free(ft.rin);
    free(ft.iin);
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
