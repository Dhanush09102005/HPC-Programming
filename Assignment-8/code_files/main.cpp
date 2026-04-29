#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            printf("Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    FILE *file = NULL;

    if (rank == 0) {
        file = fopen(argv[1], "rb");
        if (!file) {
            printf("Error opening input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fread(&NX, sizeof(int), 1, file);
        fread(&NY, sizeof(int), 1, file);
        fread(&NUM_Points, sizeof(int), 1, file);
        fread(&Maxiter, sizeof(int), 1, file);
    }

    MPI_Bcast(&NX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_Points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    GRID_X = NX + 1;
    GRID_Y = NY + 1;

    dx = 1.0 / NX;
    dy = 1.0 / NY;

    Points *original_points = (Points*)calloc(NUM_Points, sizeof(Points));

    if (rank == 0) {
        read_points(file, original_points);
        fclose(file);
    }

    MPI_Bcast(original_points,
              NUM_Points * sizeof(Points),
              MPI_BYTE, 0, MPI_COMM_WORLD);

    // ---------------- CSV ----------------
    FILE *csv = NULL;

    if (rank == 0) {
        csv = fopen("timing_results.csv", "a");

        if (!csv) {
            printf("CSV open failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0) {
            fprintf(csv,
                "MPI,OMP,Cores,Interp,Norm,Move,Denorm,Total,Voids\n");
        }
    }

    // ---------------- CONFIGS (2–64 only) ----------------
    int thread_list[] = {1, 2, 4, 8, 16};
    int num_cfg = 5;

    double *mesh_value = (double*)calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points*)calloc(NUM_Points, sizeof(Points));

    for (int c = 0; c < num_cfg; c++) {

        int nthreads = thread_list[c];

        // 🔥 CRITICAL FIX: force OpenMP correctly per MPI process
        omp_set_dynamic(0);
        omp_set_num_threads(nthreads);

        memcpy(points, original_points, NUM_Points * sizeof(Points));
        memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

        if (rank == 0) {
            printf("MPI=%d OMP=%d CORES=%d\n",
                   size, nthreads, size * nthreads);
        }

        double t_interp = 0, t_norm = 0, t_move = 0, t_denorm = 0;

        for (int iter = 0; iter < Maxiter; iter++) {

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            interpolation(mesh_value, points, nthreads);

            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();

            normalization(mesh_value, nthreads);

            MPI_Barrier(MPI_COMM_WORLD);
            double t2 = MPI_Wtime();

            mover(mesh_value, points, nthreads);

            MPI_Barrier(MPI_COMM_WORLD);
            double t3 = MPI_Wtime();

            denormalization(mesh_value, nthreads);

            MPI_Barrier(MPI_COMM_WORLD);
            double t4 = MPI_Wtime();

            t_interp += (t1 - t0);
            t_norm   += (t2 - t1);
            t_move   += (t3 - t2);
            t_denorm += (t4 - t3);
        }

        double local_total = t_interp + t_norm + t_move + t_denorm;
        double total = 0;

        MPI_Reduce(&local_total, &total, 1,
                   MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        long long voids = void_count(points);

        if (rank == 0) {
            fprintf(csv,
                "%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lld\n",
                size,
                nthreads,
                size * nthreads,
                t_interp,
                t_norm,
                t_move,
                t_denorm,
                total,
                voids
            );

            fflush(csv);
        }
    }

    if (rank == 0) {
        fclose(csv);
    }

    free(mesh_value);
    free(points);
    free(original_points);

    MPI_Finalize();
    return 0;
}
