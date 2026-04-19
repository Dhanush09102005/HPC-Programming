#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "init.h"
#include "utils.h"

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

    // Read header
    fread(&NX, sizeof(int), 1, file);
    fread(&NY, sizeof(int), 1, file);
    fread(&NUM_Points, sizeof(int), 1, file);
    fread(&Maxiter, sizeof(int), 1, file);

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

    // Thread configurations
    int thread_list[] = {1, 2, 4, 8, 16};
    int num_configs = 5;

    // Create CSV file
    char csv_name[256];
    sprintf(csv_name, "%s_results.csv", argv[1]);
    FILE *csv = fopen(csv_name, "w");

    fprintf(csv, "threads,iteration,time\n");

    for (int t = 0; t < num_configs; t++) {

        int nthreads = thread_list[t];
        double total_time = 0.0;

        printf("Running with %d threads...\n", nthreads);

        for (int iter = 0; iter < Maxiter; iter++) {

            // Reset file pointer to start of points section
            fseek(file, sizeof(int)*4 + iter * NUM_Points * sizeof(Points), SEEK_SET);

            // Read points
            read_points(file, points);

            // Allocate fresh mesh (important!)
            double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));

            // Run interpolation
            double time_taken = interpolation(mesh_value, points, nthreads);

            total_time += time_taken;

            fprintf(csv, "%d,%d,%lf\n", nthreads, iter, time_taken);
            
            if (nthreads == 16 && iter == Maxiter - 1) {
		    save_mesh(mesh_value);
		}

            free(mesh_value);
        }

        printf("Avg time (%d threads): %lf sec\n", nthreads, total_time / Maxiter);
    }

    fclose(csv);
    fclose(file);
    free(points);

    printf("CSV saved.\n");

    return 0;
}
