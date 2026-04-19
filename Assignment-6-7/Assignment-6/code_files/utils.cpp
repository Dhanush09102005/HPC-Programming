#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include <omp.h>
#include <stdlib.h>

double interpolation(double *mesh_value, Points *points, int nthreads) {

    double inv_dx = (double)NX;
    double inv_dy = (double)NY;

    int total_size = GRID_X * GRID_Y;

    double start = omp_get_wtime();

    // Allocate per-thread meshes
    double **all_meshes = (double**)malloc(nthreads * sizeof(double*));

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();

        all_meshes[tid] = (double*)calloc(total_size, sizeof(double));

        #pragma omp for
        for (int p = 0; p < NUM_Points; p++) {

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

    // Reduction (combine all thread-local meshes)
    for (int t = 0; t < nthreads; t++) {
        for (int i = 0; i < total_size; i++) {
            mesh_value[i] += all_meshes[t][i];
        }
        free(all_meshes[t]);
    }

    free(all_meshes);

    double end = omp_get_wtime();

    return (end - start);  // return execution time
}
// Write mesh to file
void save_mesh(double *mesh_value) {

    FILE *fd = fopen("Mesh1.out", "w");
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
