#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv)
{
    Maxiter = 10;

    int grid_nx[] = {250, 500, 1000};
    int grid_ny[] = {100, 200, 400};
    int num_grids = 3;

    long particle_counts[] = {100L, 10000L, 1000000L, 100000000L};
    int num_counts = 4;

    for (int g = 0; g < num_grids; g++)
    {
        NX = grid_nx[g];
        NY = grid_ny[g];

        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        printf("\nNX=%d NY=%d\n", NX, NY);
        printf("----------\n");

        for (int c = 0; c < num_counts; c++)
        {
            long N = particle_counts[c];
            NUM_Points = (int)N;

            double *mesh_value = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
            Points *points = (Points *)malloc(NUM_Points * sizeof(Points));

            if (!mesh_value || !points)
            {
                printf("\nParticles = %ld (Allocation Failed)\n", N);
                free(mesh_value);
                free(points);
                continue;
            }

            initializepoints(points);

            printf("\nParticles = %ld\n\n", N);
            printf("Iter\tInterp(s)\tMover(s)\tTotal(s)\n");

            double total_interp = 0.0;
            double total_mover = 0.0;

            for (int iter = 0; iter < Maxiter; iter++)
            {
                clock_t start_interp = clock();
                interpolation(mesh_value, points);
                clock_t end_interp = clock();

                clock_t start_move = clock();
                mover_serial(points, dx, dy);
                clock_t end_move = clock();

                double interp_time = (double)(end_interp - start_interp) / CLOCKS_PER_SEC;
                double move_time = (double)(end_move - start_move) / CLOCKS_PER_SEC;
                double total = interp_time + move_time;

                total_interp += interp_time;
                total_mover += move_time;

                printf("%d\t%lf\t%lf\t%lf\n", iter + 1, interp_time, move_time, total);
            }

            double grand_total = total_interp + total_mover;
            printf("SUM\t%lf\t%lf\t%lf\n", total_interp, total_mover, grand_total);

            free(mesh_value);
            free(points);
        }
    }

    return 0;
}
