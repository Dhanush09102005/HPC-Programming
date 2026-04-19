#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

double min_val, max_val;



void interpolation(double *mesh_value, Points *points, int nthreads) {

    double inv_dx = (double)NX;
    double inv_dy = (double)NY;

    int total_size = GRID_X * GRID_Y;

 

    memset(mesh_value, 0, total_size * sizeof(double));

    // Allocate per-thread local meshes
    double **all_meshes = (double**)malloc(nthreads * sizeof(double*));

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();

        all_meshes[tid] = (double*)calloc(total_size, sizeof(double));

        #pragma omp for schedule(static)
        for (int p = 0; p < NUM_Points; p++) {

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

    // Reduction of thread-local meshes
    for (int t = 0; t < nthreads; t++) {
        for (int i = 0; i < total_size; i++) {
            mesh_value[i] += all_meshes[t][i];
        }
        free(all_meshes[t]);
    }

    free(all_meshes);

  
}


// ======================================================
// NORMALIZATION (Grid -> [-1,1])
// ======================================================
double normalization(double *mesh_value, int nthreads) {

    int total_size = GRID_X * GRID_Y;

    double start = omp_get_wtime();

    min_val = mesh_value[0];
    max_val = mesh_value[0];

    #pragma omp parallel num_threads(nthreads)
    {
        double local_min = min_val;
        double local_max = max_val;

        #pragma omp for nowait
        for (int i = 0; i < total_size; i++) {
            if (mesh_value[i] < local_min) local_min = mesh_value[i];
            if (mesh_value[i] > local_max) local_max = mesh_value[i];
        }

        #pragma omp critical
        {
            if (local_min < min_val) min_val = local_min;
            if (local_max > max_val) max_val = local_max;
        }
    }

    double range = max_val - min_val;

    if (range == 0.0) {
        #pragma omp parallel for num_threads(nthreads)
        for (int i = 0; i < total_size; i++) {
            mesh_value[i] = 0.0;
        }

        return omp_get_wtime() - start;
    }

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int i = 0; i < total_size; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - min_val) / range - 1.0;
    }

    return omp_get_wtime() - start;
}


// ======================================================
// REVERSE INTERPOLATION / MOVER (Grid -> Particle)
// ======================================================
double mover(double *mesh_value, Points *points, int nthreads) {

    double inv_dx = (double)NX;
    double inv_dy = (double)NY;

    double start = omp_get_wtime();

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int p = 0; p < NUM_Points; p++) {

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

    return omp_get_wtime() - start;
}


// ======================================================
// DENORMALIZATION ([-1,1] -> Original Range)
// ======================================================
double denormalization(double *mesh_value, int nthreads) {

    int total_size = GRID_X * GRID_Y;

    double start = omp_get_wtime();

    double range = max_val - min_val;

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int i = 0; i < total_size; i++) {
        mesh_value[i] = ((mesh_value[i] + 1.0) * 0.5) * range + min_val;
    }

    return omp_get_wtime() - start;
}


// ======================================================
// COUNT VOID PARTICLES
// ======================================================
long long int void_count(Points *points) {

    long long int voids = 0;

    #pragma omp parallel for reduction(+:voids)
    for (int i = 0; i < NUM_Points; i++) {
        voids += (int)points[i].is_void;
    }

    return voids;
}


// ======================================================
// SAVE MESH TO FILE
// ======================================================
void save_mesh(double *mesh_value) {

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
