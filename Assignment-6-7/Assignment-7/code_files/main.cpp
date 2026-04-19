#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

// Global simulation parameters
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        printf("Error opening input file\n");
        exit(1);
    }

    fread(&NX, sizeof(int), 1, file);
    fread(&NY, sizeof(int), 1, file);
    fread(&NUM_Points, sizeof(int), 1, file);
    fread(&Maxiter, sizeof(int), 1, file);

    GRID_X = NX + 1;
    GRID_Y = NY + 1;

    dx = 1.0 / NX;
    dy = 1.0 / NY;

    Points *original_points =
        (Points *)calloc(NUM_Points, sizeof(Points));

    read_points(file, original_points);
    fclose(file);

    FILE *csv = fopen("timing_results.csv", "w");
    if (!csv) {
        printf("Error creating timing_results.csv\n");
        exit(1);
    }

    fprintf(csv,
        "Threads,Interpolation,Normalization,Mover,Denormalization,Total,Voids\n");

    int thread_list[] = {1, 2, 4, 8, 16};
    int num_configs = 5;

    for (int cfg = 0; cfg < num_configs; cfg++) {

        int nthreads = thread_list[cfg];

        printf("\n========================================\n");
        printf("Running with %d Threads\n", nthreads);
        printf("========================================\n");

        double *mesh_value =
            (double *)calloc(GRID_X * GRID_Y, sizeof(double));

        Points *points =
            (Points *)calloc(NUM_Points, sizeof(Points));

        memcpy(points, original_points, NUM_Points * sizeof(Points));

        double total_int_time = 0.0;
        double total_norm_time = 0.0;
        double total_move_time = 0.0;
        double total_denorm_time = 0.0;

        for (int iter = 0; iter < Maxiter; iter++) {

            double t0 = omp_get_wtime();

            interpolation(mesh_value, points, nthreads);

            double t1 = omp_get_wtime();

            normalization(mesh_value, nthreads);

            double t2 = omp_get_wtime();

            mover(mesh_value, points, nthreads);

            double t3 = omp_get_wtime();

            denormalization(mesh_value, nthreads);

            double t4 = omp_get_wtime();

            total_int_time += (t1 - t0);
            total_norm_time += (t2 - t1);
            total_move_time += (t3 - t2);
            total_denorm_time += (t4 - t3);
        }

        double total_algorithm_time =
            total_int_time +
            total_norm_time +
            total_move_time +
            total_denorm_time;

        long long int voids = void_count(points);

        save_mesh(mesh_value);

        printf("Total Interpolation Time = %lf seconds\n", total_int_time);
        printf("Total Normalization Time = %lf seconds\n", total_norm_time);
        printf("Total Mover Time = %lf seconds\n", total_move_time);
        printf("Total Denormalization Time = %lf seconds\n", total_denorm_time);
        printf("Total Algorithm Time = %lf seconds\n", total_algorithm_time);
        printf("Total Number of Voids = %lld\n", voids);

        fprintf(csv,
            "%d,%lf,%lf,%lf,%lf,%lf,%lld\n",
            nthreads,
            total_int_time,
            total_norm_time,
            total_move_time,
            total_denorm_time,
            total_algorithm_time,
            voids);

        free(mesh_value);
        free(points);
    }

    fclose(csv);
    free(original_points);

    return 0;
}
