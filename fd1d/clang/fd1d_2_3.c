/* File: fd1d_2_3.c
 * Name: D.Saravanan
 * Date: 10/01/2022
 * Simulation of a pulse striking a frequency-dependent dielectric material and
 * implements the discrete Fourier transform with a Gaussian pulse as its source
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    double *rpt, *ipt;
    double *rin, *iin;
} FTrans;


typedef struct {
    double *nax, *nbx;
    double *ncx, *ndx;
} Medium;


double gaussian(int ts, int t0, double sigma) {
    return exp(-0.5*(ts - t0)/sigma*(ts - t0)/sigma);
}


void fourier(int ts, int nf, int nx, double dt, double *freq, double *ex, FTrans ft) {
    for (int n = 0; n < nf; n++) {
        /* calculate the Fourier transform of input source */
        ft.rin[n] += cos(2*M_PI*freq[n]*dt*ts) * ex[10];
        ft.iin[n] -= sin(2*M_PI*freq[n]*dt*ts) * ex[10];
        for (int i = 0; i < nx; i++) {
            /* calculate the Fourier transform of Ex field */
            ft.rpt[n*nx+i] += cos(2*M_PI*freq[n]*dt*ts) * ex[i];
            ft.ipt[n*nx+i] -= sin(2*M_PI*freq[n]*dt*ts) * ex[i];
        }
    }
}


void dxfield(int ts, int nx, double *hy, double *dx) {
    /* calculate the electric flux density Dx */
    for (int i = 1; i < nx; i++)
        dx[i] += 0.5 * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse at the low end */
    dx[1] += gaussian(ts, 50, 10.0);
}


void exfield(int nx, Medium md, double *ix, double *sx, double *dx, double *ex) {
    /* calculate the Ex field from Dx */
    for (int i = 1; i < nx; i++) {
        ex[i] = md.nax[i] * (dx[i] - ix[i] - md.ncx[i] * sx[i]);
        ix[i] += md.nbx[i] * ex[i];
        sx[i] = md.ncx[i] * sx[i] + md.ndx[i] * ex[i];
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


Medium dielectric(int nx, double dt, double chi, double tau, double epsr, double sigma) {
    Medium md;
    md.nax = malloc(nx*sizeof(*md.nax));
    md.nbx = malloc(nx*sizeof(*md.nbx));
    md.ncx = malloc(nx*sizeof(*md.ncx));
    md.ndx = malloc(nx*sizeof(*md.ndx));
    for (int i = 0; i < nx; md.nax[i] = 1.0, i++);
    for (int i = 0; i < nx; md.nbx[i] = 0.0, i++);
    for (int i = 0; i < nx; md.ncx[i] = 0.0, i++);
    for (int i = 0; i < nx; md.ndx[i] = 0.0, i++);
    double eps0 = 8.854e-12;  /* vacuum permittivity (F/m) */
    for (int i = nx/2; i < nx; i++) {
        md.nax[i] = 1/(epsr + sigma*dt/eps0 + chi*dt/tau);
        md.nbx[i] = sigma*dt/eps0;
        md.ncx[i] = exp(-dt/tau);
        md.ndx[i] = chi*dt/tau;
    }
    return md;
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 740;  /* number of time steps */

    double *dx = malloc(nx*sizeof(*dx));
    double *ex = malloc(nx*sizeof(*ex));
    double *ix = malloc(nx*sizeof(*ix));
    double *sx = malloc(nx*sizeof(*sx));
    double *hy = malloc(nx*sizeof(*hy));

    for (int i = 0; i < nx; i++) {
        dx[i] = 0.0;
        ex[i] = 0.0;
        ix[i] = 0.0;
        sx[i] = 0.0;
        hy[i] = 0.0;
    }

    double bc[4] = {0.0};

    double ds = 0.01;  /* spatial step (m) */
    double dt = ds/6e8;  /* time step (s) */
    double chi = 2.0;  /* relaxation susceptibility */
    double tau = 0.001e-6;  /* relaxation time (s) */
    double epsr = 2.0;  /* relative permittivity */
    double sigma = 0.01;  /* conductivity (S/m) */
    Medium md = dielectric(nx, dt, chi, tau, epsr, sigma);

    /* frequency 50 MHz, 200 MHz, 500 MHz */
    double freq[] = {50e6, 200e6, 500e6};
    int nf = (sizeof freq)/(sizeof freq[0]);  /* number of frequencies */

    FTrans ft;
    ft.rpt = malloc(nf*nx*sizeof(*ft.rpt));
    ft.ipt = malloc(nf*nx*sizeof(*ft.ipt));
    ft.rin = malloc(nf*sizeof(*ft.rin));
    ft.iin = malloc(nf*sizeof(*ft.iin));

    for (int i = 0; i < nf*nx; i++) {
        ft.rpt[i] = 0.0;
        ft.ipt[i] = 0.0;
    }

    for (int i = 0; i < nf; i++) {
        ft.rin[i] = 0.0;
        ft.iin[i] = 0.0;
    }

    double *amplt = malloc(nf*nx*sizeof(*amplt));
    double *phase = malloc(nf*nx*sizeof(*phase));

    for (int i = 0; i < nf*nx; i++) {
        amplt[i] = 0.0;
        phase[i] = 0.0;
    }

    for (int ts = 1; ts <= ns; ts++) {
        dxfield(ts, nx, hy, dx);
        exfield(nx, md, ix, sx, dx, ex);
        fourier(ts, nf, nx, dt, freq, ex, ft);
        hyfield(nx, bc, ex, hy);
    }

    /* calculate the amplitude and phase at each frequency */
    for (int n = 0; n < nf; n++) {
        for (int i = 0; i < nx; i++) {
            int m = n*nx+i;
            amplt[m] = 1/hypot(ft.rin[n],ft.iin[n]) * hypot(ft.rpt[m],ft.ipt[m]);
            phase[m] = atan2(ft.ipt[m],ft.rpt[m]) - atan2(ft.iin[n],ft.rin[n]);
        }
    }

    free(ft.rpt);
    free(ft.ipt);
    free(ft.rin);
    free(ft.iin);
    free(md.nax);
    free(md.nbx);
    free(md.ncx);
    free(md.ndx);
    free(amplt);
    free(phase);
    free(dx);
    free(ex);
    free(ix);
    free(sx);
    free(hy);

    return 0;
}
