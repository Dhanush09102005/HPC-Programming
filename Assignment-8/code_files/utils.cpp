#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "utils.h"

double min_val, max_val;

// ======================================================
// INTERPOLATION (Particle -> Grid)
// ======================================================
void interpolation(double *mesh_value, Points *points, int nthreads) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double inv_dx = (double)NX;
    double inv_dy = (double)NY;

    int total_size = GRID_X * GRID_Y;

    memset(mesh_value, 0, total_size * sizeof(double));

    // Domain decomposition (points)
    int points_per_proc = NUM_Points / size;
    int start = rank * points_per_proc;
    int end = (rank == size - 1) ? NUM_Points : start + points_per_proc;

    // Per-thread local meshes
    double **all_meshes = (double**)malloc(nthreads * sizeof(double*));

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        all_meshes[tid] = (double*)calloc(total_size, sizeof(double));

        #pragma omp for schedule(static)
        for (int p = start; p < end; p++) {

            if (points[p].is_void) continue;

            double x = points[p].x;
            double y = points[p].y;

            int i = (int)(x * inv_dx);
            int j = (int)(y * inv_dy);

            if (i >= NX) i = NX - 1;
            if (j >= NY) j = NY - 1;

            double lx = x - (i * dx);
            double ly = y - (j * dy);

            double wx_m = dx - lx;
            double wy_m = dy - ly;

            int base_idx = j * GRID_X + i;

            all_meshes[tid][base_idx]              += wx_m * wy_m;
            all_meshes[tid][base_idx + 1]          += lx * wy_m;
            all_meshes[tid][base_idx + GRID_X]     += wx_m * ly;
            all_meshes[tid][base_idx + GRID_X + 1] += lx * ly;
        }
    }

    // Thread reduction
    for (int t = 0; t < nthreads; t++) {
        for (int i = 0; i < total_size; i++) {
            mesh_value[i] += all_meshes[t][i];
        }
        free(all_meshes[t]);
    }
    free(all_meshes);

    // MPI reduction (combine all processes)
    double *global_mesh = (double*)malloc(total_size * sizeof(double));

    MPI_Allreduce(mesh_value, global_mesh, total_size,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    memcpy(mesh_value, global_mesh, total_size * sizeof(double));
    free(global_mesh);
}

// ======================================================
// NORMALIZATION (Grid -> [-1,1])
// ======================================================
double normalization(double *mesh_value, int nthreads) {

    int total_size = GRID_X * GRID_Y;

    double start_time = omp_get_wtime();

    double local_min = mesh_value[0];
    double local_max = mesh_value[0];

    #pragma omp parallel for num_threads(nthreads) reduction(min:local_min) reduction(max:local_max)
    for (int i = 0; i < total_size; i++) {
        if (mesh_value[i] < local_min) local_min = mesh_value[i];
        if (mesh_value[i] > local_max) local_max = mesh_value[i];
    }

    // MPI global min/max
    MPI_Allreduce(&local_min, &min_val, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &max_val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double range = max_val - min_val;

    if (range == 0.0) {
        #pragma omp parallel for num_threads(nthreads)
        for (int i = 0; i < total_size; i++) {
            mesh_value[i] = 0.0;
        }
        return omp_get_wtime() - start_time;
    }

    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < total_size; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - min_val) / range - 1.0;
    }

    return omp_get_wtime() - start_time;
}

// ======================================================
// REVERSE INTERPOLATION / MOVER (Grid -> Particle)
// ======================================================
double mover(double *mesh_value, Points *points, int nthreads) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double inv_dx = (double)NX;
    double inv_dy = (double)NY;

    double start_time = omp_get_wtime();

    int points_per_proc = NUM_Points / size;
    int start = rank * points_per_proc;
    int end = (rank == size - 1) ? NUM_Points : start + points_per_proc;

    #pragma omp parallel for num_threads(nthreads)
    for (int p = start; p < end; p++) {

        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;

        double lx = x - (i * dx);
        double ly = y - (j * dy);

        double wx_m = dx - lx;
        double wy_m = dy - ly;

        int base_idx = j * GRID_X + i;

        double Fi =
              wx_m * wy_m * mesh_value[base_idx]
            + lx   * wy_m * mesh_value[base_idx + 1]
            + wx_m * ly   * mesh_value[base_idx + GRID_X]
            + lx   * ly   * mesh_value[base_idx + GRID_X + 1];

        points[p].x += Fi * dx;
        points[p].y += Fi * dy;

        if (points[p].x < 0.0 || points[p].x > 1.0 ||
            points[p].y < 0.0 || points[p].y > 1.0) {
            points[p].is_void = true;
        }
    }

    return omp_get_wtime() - start_time;
}

// ======================================================
// DENORMALIZATION
// ======================================================
double denormalization(double *mesh_value, int nthreads) {

    int total_size = GRID_X * GRID_Y;

    double start_time = omp_get_wtime();

    double range = max_val - min_val;

    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < total_size; i++) {
        mesh_value[i] = ((mesh_value[i] + 1.0) * 0.5) * range + min_val;
    }

    return omp_get_wtime() - start_time;
}

// ======================================================
// COUNT VOID PARTICLES
// ======================================================
long long int void_count(Points *points) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int points_per_proc = NUM_Points / size;
    int start = rank * points_per_proc;
    int end = (rank == size - 1) ? NUM_Points : start + points_per_proc;

    long long int local_voids = 0;

    #pragma omp parallel for reduction(+:local_voids)
    for (int i = start; i < end; i++) {
        local_voids += (int)points[i].is_void;
    }

    long long int global_voids = 0;

    MPI_Reduce(&local_voids, &global_voids, 1,
               MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    return global_voids;
}

// ======================================================
// SAVE MESH
// ======================================================
void save_mesh(double *mesh_value) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) return;

    FILE *fd = fopen("Mesh.out", "w");

    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}
