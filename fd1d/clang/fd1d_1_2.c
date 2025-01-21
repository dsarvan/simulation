/* File: fd1d_1_2.c
 * Name: D.Saravanan
 * Date: 19/10/2021
 * Simulation of a pulse with absorbing boundary conditions
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double gaussian(int t, int t0, double sigma) {
    return exp(-0.5 * ((t - t0)/sigma) * ((t - t0)/sigma));
}


void field(int t, int nx, double *ex, double *hy, double *bc) {
	/* calculate the Hy field */
    for (int i = 0; i < nx - 1; i++)
        hy[i] = hy[i] + 0.5 * (ex[i] - ex[i+1]);

	/* calculate the Ex field */
    for (int i = 1; i < nx; i++)
        ex[i] = ex[i] + 0.5 * (hy[i-1] - hy[i]);

	/* put a Gaussian pulse in the middle */
    ex[nx/2] = gaussian(t, 40, 12);

	/* absorbing boundary conditions */
	ex[0] = bc[0], bc[0] = bc[1], bc[1] = ex[1];
	ex[nx-1] = bc[3], bc[3] = bc[2], bc[2] = ex[nx-2];
}


int main() {

    int nx = 201;
    int ns = 260;

	double *ex = (double *) calloc(nx, sizeof(*ex));
	double *hy = (double *) calloc(nx, sizeof(*hy));

	/* initialize ex and hy arrays */
	for (size_t i = 0; i < nx; i++) {
		ex[i] = 0.0f;
		hy[i] = 0.0f;
	}

	double bc[4] = {0.0f};

    for (int t = 1; t <= ns; t++)
        field(t, nx, ex, hy, bc);

	for (size_t i = 0; i < nx; i++)
		printf("%e\n", ex[i]);


	free(ex);
	free(hy);

    return 0;
}
