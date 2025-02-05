/* File: fd1d_1_1.c
 * Name: D.Saravanan
 * Date: 11/10/2021
 * Simulation of a pulse in free space
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double gaussian(int t, int t0, double sigma) {
    return exp(-0.5 * ((t - t0)/sigma) * ((t - t0)/sigma));
}


void exfield(int t, int nx, double *ex, double *hy) {
    /* calculate the Ex field */
    for (int i = 1; i < nx; i++)
        ex[i] = ex[i] + 0.5 * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse in the middle */
    ex[nx/2] = gaussian(t, 40, 12);
}


void hyfield(int nx, double *ex, double *hy) {
    /* calculate the Hy field */
    for (int i = 0; i < nx - 1; i++)
        hy[i] = hy[i] + 0.5 * (ex[i] - ex[i+1]);
}


int main() {

    int nx = 512;  /* number of grid points */
    int ns = 300;  /* number of time steps */

    double *ex = (double *) calloc(nx, sizeof(*ex));
    double *hy = (double *) calloc(nx, sizeof(*hy));

    for (int t = 1; t <= ns; t++) {
        exfield(t, nx, ex, hy);
        hyfield(nx, ex, hy);
    }

    free(ex);
    free(hy);

    return 0;
}
