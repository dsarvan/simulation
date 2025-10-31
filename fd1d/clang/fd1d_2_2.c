/* File: fd1d_2_2.c
 * Name: D.Saravanan
 * Date: 07/12/2021
 * Simulation of a pulse striking a dielectric medium and implements
 * the discrete Fourier transform with a Gaussian pulse as its source
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    double *nax, *nbx;
} medium;


typedef struct {
    double *r_pt, *i_pt;
    double *r_in, *i_in;
} ftrans;


double gaussian(int t, int t0, double sigma) {
    return exp(-0.5*(t - t0)/sigma*(t - t0)/sigma);
}


void fourier(int t, int nf, int nx, double dt, double *freq, double *ex, ftrans *ft) {
    for (int n = 0; n < nf; n++) {
        /* calculate the Fourier transform of input source */
        ft->r_in[n] += cos(2*M_PI*freq[n]*dt*t) * ex[10];
        ft->i_in[n] -= sin(2*M_PI*freq[n]*dt*t) * ex[10];
        for (int i = 0; i < nx; i++) {
            /* calculate the Fourier transform of Ex field */
            int m = n*nx+i;
            ft->r_pt[m] += cos(2*M_PI*freq[n]*dt*t) * ex[i];
            ft->i_pt[m] -= sin(2*M_PI*freq[n]*dt*t) * ex[i];
        }
    }
}


void dxfield(int t, int nx, double *dx, double *hy) {
    /* calculate the electric flux density Dx */
    for (int i = 1; i < nx; i++)
        dx[i] += 0.5 * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse at the low end */
    dx[1] += gaussian(t, 50, 10.0);
}


void exfield(int nx, medium *md, double *dx, double *ix, double *ex) {
    /* calculate the Ex field from Dx */
    for (int i = 1; i < nx; i++) {
        ex[i] = md->nax[i] * (dx[i] - ix[i]);
        ix[i] += md->nbx[i] * ex[i];
    }
}


void hyfield(int nx, double *ex, double *hy, double *bc) {
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = 0; i < nx-1; i++)
        hy[i] += 0.5 * (ex[i] - ex[i+1]);
}


medium dielectric(int nx, double dt, double epsr, double sigma) {
    medium md;
    md.nax = (double*) calloc(nx, sizeof(*md.nax));
    md.nbx = (double*) calloc(nx, sizeof(*md.nbx));
    for (int i = 0; i < nx; md.nax[i] = 1.0, i++);
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

    double *dx = (double*) calloc(nx, sizeof(*dx));
    double *ex = (double*) calloc(nx, sizeof(*ex));
    double *ix = (double*) calloc(nx, sizeof(*ix));
    double *hy = (double*) calloc(nx, sizeof(*hy));

    double bc[4] = {0.0};

    double ds = 0.01;  /* spatial step (m) */
    double dt = ds/6e8;  /* time step (s) */
    double epsr = 4.0;  /* relative permittivity */
    double sigma = 0.0;  /* conductivity (S/m) */
    medium md = dielectric(nx, dt, epsr, sigma);

    /* frequency 100 MHz, 200 MHz, 500 MHz */
    double freq[] = {100e6, 200e6, 500e6};
    int nf = (sizeof freq)/(sizeof freq[0]);  /* number of frequencies */

    ftrans ft;
    ft.r_pt = (double*) calloc(nf*nx, sizeof(*ft.r_pt));
    ft.i_pt = (double*) calloc(nf*nx, sizeof(*ft.i_pt));
    ft.r_in = (double*) calloc(nf, sizeof(*ft.r_in));
    ft.i_in = (double*) calloc(nf, sizeof(*ft.i_in));

    double *amplt = (double*) calloc(nf*nx, sizeof(*amplt));
    double *phase = (double*) calloc(nf*nx, sizeof(*phase));

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
            amplt[m] = 1/hypot(ft.r_in[n],ft.i_in[n]) * hypot(ft.r_pt[m],ft.i_pt[m]);
            phase[m] = atan2(ft.i_pt[m],ft.r_pt[m]) - atan2(ft.i_in[n],ft.r_in[n]);
        }
    }

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
