/* File: test_3_4.c
 * Name: D.Saravanan
 * Date: 20/01/2022
 * Simulation of a plane wave pulse striking a dielectric medium in the transverse
 * magnetic (TM) mode with PML and implements the discrete Fourier transform analysis
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


typedef struct {
    float *naz, *nbz;
} medium;


typedef struct {
    float *r_pt, *i_pt;
    float *r_in, *i_in;
} ftrans;


typedef struct {
    float *fx1, *fx2, *fx3;
    float *fy1, *fy2, *fy3;
    float *gx2, *gx3;
    float *gy2, *gy3;
} pmlayer;


float gaussian(int t, int t0, float sigma) {
    return expf(-0.5f*(t - t0)/sigma*(t - t0)/sigma);
}


void fourier(int t, int nf, int nx, int ny, float dt, float *freq, float *ezi, float *ez, ftrans *ft) {
    #pragma omp parallel for
    for (int n = 0; n < nf; n++) {
        /* calculate the Fourier transform of input source */
        ft->r_in[n] += cosf(2*M_PI*freq[n]*dt*t) * ezi[6];
        ft->i_in[n] -= sinf(2*M_PI*freq[n]*dt*t) * ezi[6];
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                /* calculate the Fourier transform of Ex field */
                int m = n*nx*ny+i*ny+j;
                ft->r_pt[m] += cosf(2*M_PI*freq[n]*dt*t) * ez[i*ny+j];
                ft->i_pt[m] -= sinf(2*M_PI*freq[n]*dt*t) * ez[i*ny+j];
            }
        }
    }
}


void ezinct(int ny, float *ezi, float *hxi, float *bc) {
    /* calculate the incident Ez */
    #pragma omp parallel for
    for (int j = 1; j < ny; j++) {
        ezi[j] += 0.5f * (hxi[j-1] - hxi[j]);
    }
    /* absorbing boundary conditions */
    ezi[0] = bc[0], bc[0] = bc[1], bc[1] = ezi[1];
    ezi[ny-1] = bc[3], bc[3] = bc[2], bc[2] = ezi[ny-2];
}


void dfield(int t, int nx, int ny, pmlayer *pml, float *ezi, float *dz, float *hx, float *hy) {
    /* calculate the electric flux density Dz */
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            int n = i*ny+j;
            dz[n] = pml->gx3[i] * pml->gy3[j] * dz[n] + pml->gx2[i] * pml->gy2[j] * 0.5f * (hy[n] - hy[n-ny] - hx[n] + hx[n-1]);
        }
    }
    /* put a Gaussian pulse at the low end */
    ezi[3] = gaussian(t, 20, 8.0f);
}


void inctdz(int nx, int ny, int npml, float *hxi, float *dz) {
    /* incident Dz values */
    #pragma omp parallel for
    for (int i = npml-1; i <= nx-npml; i++) {
        dz[i*ny+(npml-1)] += 0.5f * hxi[npml-2];
        dz[i*ny+(ny-npml)] -= 0.5f * hxi[ny-npml];
    }
}


void efield(int nx, int ny, medium *md, float *dz, float *iz, float *ez) {
    /* calculate the Ez field from Dz */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int n = i*ny+j;
            ez[n] = md->naz[n] * (dz[n] - iz[n]);
            iz[n] += md->nbz[n] * ez[n];
        }
    }
}


void hxinct(int ny, float *ezi, float *hxi) {
    /* calculate the incident Hx */
    #pragma omp parallel for
    for (int j = 0; j < ny-1; j++) {
        hxi[j] += 0.5f * (ezi[j] - ezi[j+1]);
    }
}


void hfield(int nx, int ny, pmlayer *pml, float *ez, float *ihx, float *ihy, float *hx, float *hy) {
    /* calculate the Hx and Hy field */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < ny-1; j++) {
            int n = i*ny+j;
            ihx[n] += ez[n] - ez[n+1];
            ihy[n] += ez[n] - ez[n+ny];
            hx[n] = pml->fy3[j] * hx[n] + pml->fy2[j] * (0.5f * ez[n] - 0.5f * ez[n+1] + pml->fx1[i] * ihx[n]);
            hy[n] = pml->fx3[i] * hy[n] - pml->fx2[i] * (0.5f * ez[n] - 0.5f * ez[n+ny] + pml->fy1[j] * ihy[n]);
        }
    }
}


void incthx(int nx, int ny, int npml, float *ezi, float *hx) {
    /* incident Hx values */
    #pragma omp parallel for
    for (int i = npml-1; i <= nx-npml; i++) {
        hx[i*ny+(npml-2)] += 0.5f * ezi[npml-1];
        hx[i*ny+(ny-npml)] -= 0.5f * ezi[ny-npml];
    }
}


void incthy(int nx, int ny, int npml, float *ezi, float *hy) {
    /* incident Hy values */
    #pragma omp parallel for
    for (int j = npml-1; j <= ny-npml; j++) {
        hy[(npml-2)*ny+j] -= 0.5f * ezi[j];
        hy[(nx-npml)*ny+j] += 0.5f * ezi[j];
    }
}


medium dielectric(int nx, int ny, int npml, int rgrid, float dt, float epsr, float sigma) {
    medium md;
    md.naz = (float*) calloc(nx*ny, sizeof(*md.naz));
    md.nbz = (float*) calloc(nx*ny, sizeof(*md.nbz));
    for (int i = 0; i < nx*ny; md.naz[i] = 1.0f, i++);
    float eps0 = 8.854e-12f;  /* vacuum permittivity (F/m) */
    #pragma omp parallel for collapse(2)
    for (int i = npml; i < nx-npml; i++) {
        for (int j = npml; j < ny-npml; j++) {
            float epsn = 1.0f;
            float cond = 0.0f;
            for (int m = -1; m <= 1; m++) {
                for (int n = -1; n <= 1; n++) {
                    float x = nx/2-1-i+m/3;
                    float y = ny/2-1-j+n/3;
                    float d = sqrtf(x*x + y*y);
                    if (d <= rgrid) {
                        epsn += (epsr - 1)/9;
                        cond += sigma/9;
                    }
                }
            }
            md.naz[i*ny+j] = 1/(epsn + cond*dt/eps0);
            md.nbz[i*ny+j] = cond*dt/eps0;
        }
    }
    return md;
}


void pmlparam(int nx, int ny, int npml, pmlayer *pml) {
    /* calculate the two-dimensional perfectly matched layer (PML) parameters */
    for (int n = 0; n < npml; n++) {
        float xm = 0.33f*(npml-n)/npml*(npml-n)/npml*(npml-n)/npml;
        float xn = 0.33f*(npml-n-0.5f)/npml*(npml-n-0.5f)/npml*(npml-n-0.5f)/npml;
        pml->fx1[n] = pml->fx1[nx-2-n] = pml->fy1[n] = pml->fy1[ny-2-n] = xn;
        pml->fx2[n] = pml->fx2[nx-2-n] = pml->fy2[n] = pml->fy2[ny-2-n] = 1/(1+xn);
        pml->gx2[n] = pml->gx2[nx-1-n] = pml->gy2[n] = pml->gy2[ny-1-n] = 1/(1+xm);
        pml->fx3[n] = pml->fx3[nx-2-n] = pml->fy3[n] = pml->fy3[ny-2-n] = (1-xn)/(1+xn);
        pml->gx3[n] = pml->gx3[nx-1-n] = pml->gy3[n] = pml->gy3[ny-1-n] = (1-xm)/(1+xm);
    }
}


int main() {

    int nx = 1024;  /* number of grid points */
    int ny = 1024;  /* number of grid points */

    int ns = 5000;  /* number of time steps */

    float *ezi = (float*) calloc(ny, sizeof(*ezi));
    float *hxi = (float*) calloc(ny, sizeof(*hxi));

    float *dz = (float*) calloc(nx*ny, sizeof(*dz));
    float *ez = (float*) calloc(nx*ny, sizeof(*ez));
    float *iz = (float*) calloc(nx*ny, sizeof(*iz));
    float *hx = (float*) calloc(nx*ny, sizeof(*hx));
    float *hy = (float*) calloc(nx*ny, sizeof(*hy));

    float *ihx = (float*) calloc(nx*ny, sizeof(*ihx));
    float *ihy = (float*) calloc(nx*ny, sizeof(*ihy));

    float bc[4] = {0.0f};

    pmlayer pml;
    pml.fx1 = (float*) calloc(nx, sizeof(*pml.fx1));
    pml.fx2 = (float*) calloc(nx, sizeof(*pml.fx2));
    pml.fx3 = (float*) calloc(nx, sizeof(*pml.fx3));
    pml.fy1 = (float*) calloc(ny, sizeof(*pml.fy1));
    pml.fy2 = (float*) calloc(ny, sizeof(*pml.fy2));
    pml.fy3 = (float*) calloc(ny, sizeof(*pml.fy3));
    pml.gx2 = (float*) calloc(nx, sizeof(*pml.gx2));
    pml.gx3 = (float*) calloc(nx, sizeof(*pml.gx3));
    pml.gy2 = (float*) calloc(ny, sizeof(*pml.gy2));
    pml.gy3 = (float*) calloc(ny, sizeof(*pml.gy3));

    for (int i = 0; i < nx; i++) {
        pml.fx2[i] = 1.0f;
        pml.fx3[i] = 1.0f;
        pml.gx2[i] = 1.0f;
        pml.gx3[i] = 1.0f;
    }

    for (int i = 0; i < ny; i++) {
        pml.fy2[i] = 1.0f;
        pml.fy3[i] = 1.0f;
        pml.gy2[i] = 1.0f;
        pml.gy3[i] = 1.0f;
    }

    int npml = 80;  /* pml thickness */
    pmlparam(nx, ny, npml, &pml);

    float ds = 0.01f;  /* spatial step (m) */
    float dt = ds/6e8f;  /* time step (s) */
    float epsr = 30.0f;  /* relative permittivity */
    float sigma = 0.30f;  /* conductivity (S/m) */
    float radius = 1.50f;  /* cylinder radius (m) */
    int rgrid = (int)(radius/ds-1);  /* radius in FDTD grid cell units */
    medium md = dielectric(nx, ny, npml, rgrid, dt, epsr, sigma);

    /* frequency 50 MHz, 300 MHz, 700 MHz */
    float freq[] = {50e6f, 300e6f, 700e6f};
    int nf = (sizeof freq)/(sizeof freq[0]);  /* number of frequencies */

    ftrans ft;
    ft.r_pt = (float*) calloc(nf*nx*ny, sizeof(*ft.r_pt));
    ft.i_pt = (float*) calloc(nf*nx*ny, sizeof(*ft.i_pt));
    ft.r_in = (float*) calloc(nf, sizeof(*ft.r_in));
    ft.i_in = (float*) calloc(nf, sizeof(*ft.i_in));

    float *amplt = (float*) calloc(nf*ny, sizeof(*amplt));
    float *phase = (float*) calloc(nf*ny, sizeof(*phase));

    float stime = omp_get_wtime();

    for (int t = 1; t <= ns; t++) {
        ezinct(ny, ezi, hxi, bc);
        dfield(t, nx, ny, &pml, ezi, dz, hx, hy);
        inctdz(nx, ny, npml, hxi, dz);
        efield(nx, ny, &md, dz, iz, ez);
        fourier(t, nf, nx, ny, dt, freq, ezi, ez, &ft);
        hxinct(ny, ezi, hxi);
        hfield(nx, ny, &pml, ez, ihx, ihy, hx, hy);
        incthx(nx, ny, npml, ezi, hx);
        incthy(nx, ny, npml, ezi, hy);
    }

    /* calculate the amplitude and phase at each frequency */
    for (int n = 0; n < nf; n++) {
        for (int j = npml-1; j <= ny-npml; j++) {
            int m = n*ny+j, k = n*nx*ny+(nx/2-1)*ny+j;
            amplt[m] = 1/hypotf(ft.r_in[n],ft.i_in[n]) * hypotf(ft.r_pt[k],ft.i_pt[k]);
            phase[m] = atan2f(ft.i_pt[k],ft.r_pt[k]) - atan2f(ft.i_in[n],ft.r_in[n]);
        }
    }

    float ntime = omp_get_wtime();
    printf("Total compute time on CPU: %.3f s\n", ntime - stime);

    for (int i = 2*ny; i < 2*ny+50; i++)
        printf("%e\n", ez[i]);
    for (int i = 2*ny; i < 2*ny+(ny-50); i++)
        printf("%e\n", amplt[i]);

    free(pml.fx1);
    free(pml.fx2);
    free(pml.fx3);
    free(pml.fy1);
    free(pml.fy2);
    free(pml.fy3);
    free(pml.gx2);
    free(pml.gx3);
    free(pml.gy2);
    free(pml.gy3);
    free(ft.r_pt);
    free(ft.i_pt);
    free(ft.r_in);
    free(ft.i_in);
    free(md.naz);
    free(md.nbz);
    free(amplt);
    free(phase);
    free(ezi);
    free(hxi);
    free(ihx);
    free(ihy);
    free(dz);
    free(ez);
    free(iz);
    free(hx);
    free(hy);

    return 0;
}
