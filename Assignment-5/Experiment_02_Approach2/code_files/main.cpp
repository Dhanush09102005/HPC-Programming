#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

void run_experiment(Points *ref, Points *buf, double *mesh,
                    void (*mover_fn)(Points *, double, double),
                    double *sum_interp, double *sum_mover)
{
    memcpy(buf, ref, sizeof(Points) * NUM_Points);

    *sum_interp = 0.0;
    *sum_mover = 0.0;

    for (int iter = 0; iter < Maxiter; iter++)
    {
        double t0 = omp_get_wtime();
        interpolation(mesh, buf);
        double t1 = omp_get_wtime();

        mover_fn(buf, dx, dy);
        double t2 = omp_get_wtime();

        double interp_time = t1 - t0;
        double mover_time = t2 - t1;

        *sum_interp += interp_time;
        *sum_mover += mover_time;

        printf("%d\t%lf\t%lf\t%lf\n",
               iter + 1, interp_time, mover_time,
               interp_time + mover_time);
    }
}

int main(int argc, char **argv)
{
    Maxiter = 10;
    NUM_Points = 14000000;

    int grid_nx[] = {250, 500, 1000};
    int grid_ny[] = {100, 200, 400};
    int num_grids = 3;

    int threads[] = {1, 2, 4, 8, 16};
    int num_threads = 5;

    Points *points_ref = (Points *)malloc(NUM_Points * sizeof(Points));
    Points *points_buf = (Points *)malloc(NUM_Points * sizeof(Points));

    initializepoints(points_ref);

    printf("Particles = %d, Maxiter = %d\n", NUM_Points, Maxiter);

    for (int g = 0; g < num_grids; g++)
    {
        NX = grid_nx[g];
        NY = grid_ny[g];

        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        double *mesh = (double *)calloc(GRID_X * GRID_Y, sizeof(double));

        printf("\nNX=%d NY=%d\n", NX, NY);
        printf("----------\n");

        for (int t = 0; t < num_threads; t++)
        {
            int th = threads[t];
            omp_set_num_threads(th);

            printf("\n--- Threads = %d ---\n", th);
            printf("\n(With Insertion/Deletion)\n");
            printf("Iter\tInterp(s)\tMover(s)\tTotal(s)\n");

            double sum_i1, sum_m1;
            run_experiment(points_ref, points_buf, mesh,
                           (th == 1) ? mover_immediate_serial : mover_immediate_parallel,
                           &sum_i1, &sum_m1);

            double total1 = sum_i1 + sum_m1;
            printf("SUM\t%lf\t%lf\t%lf\n", sum_i1, sum_m1, total1);
            printf("\n(Without Insertion/Deletion)\n");
            printf("Iter\tInterp(s)\tMover(s)\tTotal(s)\n");

            double sum_i2, sum_m2;
            run_experiment(points_ref, points_buf, mesh, mover_simple_parallel, &sum_i2, &sum_m2);

            double total2 = sum_i2 + sum_m2;
            printf("SUM\t%lf\t%lf\t%lf\n", sum_i2, sum_m2, total2);
        }

        free(mesh);
    }

    free(points_ref);
    free(points_buf);
    return 0;
}