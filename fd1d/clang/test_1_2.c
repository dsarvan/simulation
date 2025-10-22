/* File: test_1_2.c
 * Name: D.Saravanan
 * Date: 19/10/2021
 * Simulation of a pulse with absorbing boundary conditions
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


float gaussian(int t, int t0, float sigma) {
    return expf(-0.5f*(t - t0)/sigma*(t - t0)/sigma);
}


void exfield(int t, int nx, float *ex, float *hy) {
    /* calculate the Ex field */
    for (int i = 1; i < nx; i++)
        ex[i] += 0.5f * (hy[i-1] - hy[i]);
    /* put a Gaussian pulse in the middle */
    ex[nx/2] = gaussian(t, 40, 12.0f);
}


void hyfield(int nx, float *ex, float *hy, float *bc) {
    /* absorbing boundary conditions */
    ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
    ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
    /* calculate the Hy field */
    for (int i = 0; i < nx-1; i++)
        hy[i] += 0.5f * (ex[i] - ex[i+1]);
}


int main() {

    int nx = 38000;  /* number of grid points */
    int ns = 40000;  /* number of time steps */

    float *ex = (float*) calloc(nx, sizeof(*ex));
    float *hy = (float*) calloc(nx, sizeof(*hy));

    float bc[4] = {0.0f};

    clock_t stime = clock();

    for (int t = 1; t <= ns; t++) {
        exfield(t, nx, ex, hy);
        hyfield(nx, ex, hy, bc);
    }

    clock_t ntime = clock();
    float time = (ntime - stime)*1000/CLOCKS_PER_SEC;
    printf("Total compute time on CPU: %.3f s\n", time/1000.0f);

    for (int i = 0; i < 50; i++)
        printf("%e\n", ex[i]);

    free(ex);
    free(hy);

    return 0;
}
