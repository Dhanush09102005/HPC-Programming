#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main() {

    // --- Experiment 03 parameters ---
    NX = 1000;
    NY = 400;
    NUM_Points = 14000000;  // 14 million particles
    Maxiter = 10;

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // Allocate memory for mesh and points
    double *mesh_value = (double*) calloc(GRID_X * GRID_Y, sizeof(double));
    if (!mesh_value) { printf("Mesh allocation failed\n"); return 1; }

    Points *points_serial = (Points*) calloc(NUM_Points, sizeof(Points));
    Points *points_parallel = (Points*) calloc(NUM_Points, sizeof(Points));
    if (!points_serial || !points_parallel) {
        printf("Points allocation failed\n");
        free(mesh_value);
        return 1;
    }

    // Initialize particles once and copy to both arrays
    initializepoints(points_serial);
    memcpy(points_parallel, points_serial, NUM_Points * sizeof(Points));

    // Set number of threads for parallel Mover
    omp_set_num_threads(4);

    printf("Experiment 03: Serial vs Parallel Mover\n");
    printf("Iteration\tInterp_Serial(s)\tMover_Serial(s)\tTotal_Serial(s)\n");

    double total_interp_serial = 0.0;
    double total_mover_serial = 0.0;

    // --- SERIAL MOVER ---
    for (int iter = 0; iter < Maxiter; iter++) {

        memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

        double t1 = omp_get_wtime();
        interpolation(mesh_value, points_serial);
        double t2 = omp_get_wtime();
        double interp_time = t2 - t1;
        total_interp_serial += interp_time;

        double t3 = omp_get_wtime();
        mover_serial(points_serial, dx, dy);
        double t4 = omp_get_wtime();
        double mover_time = t4 - t3;
        total_mover_serial += mover_time;

        double total_time = interp_time + mover_time;

        printf("%d\t\t%lf\t%lf\t%lf\n", iter + 1, interp_time, mover_time, total_time);
    }

    printf("\nCumulative Serial Times: Interp=%lf s, Mover=%lf s, Total=%lf s\n",
           total_interp_serial, total_mover_serial, total_interp_serial + total_mover_serial);

    // --- PARALLEL MOVER ---
    printf("\nRunning Parallel Mover...\n");
    printf("Iteration\tInterp_Parallel(s)\tMover_Parallel(s)\tTotal_Parallel(s)\n");

    double total_interp_parallel = 0.0;
    double total_mover_parallel = 0.0;

    for (int iter = 0; iter < Maxiter; iter++) {

        memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

        double t1 = omp_get_wtime();
        interpolation(mesh_value, points_parallel);
        double t2 = omp_get_wtime();
        double interp_time = t2 - t1;
        total_interp_parallel += interp_time;

        double t3 = omp_get_wtime();
        mover_parallel(points_parallel, dx, dy);
        double t4 = omp_get_wtime();
        double mover_time = t4 - t3;
        total_mover_parallel += mover_time;

        double total_time = interp_time + mover_time;

        printf("%d\t\t%lf\t%lf\t%lf\n", iter + 1, interp_time, mover_time, total_time);
    }

    printf("\nCumulative Parallel Times: Interp=%lf s, Mover=%lf s, Total=%lf s\n",
           total_interp_parallel, total_mover_parallel, total_interp_parallel + total_mover_parallel);

    // --- Compute Speedup ---
    double speedup_mover = total_mover_serial / total_mover_parallel;
    printf("\nSpeedup of Mover (Serial / Parallel): %lf\n", speedup_mover);

    // Free memory
    free(mesh_value);
    free(points_serial);
    free(points_parallel);

    return 0;
}
