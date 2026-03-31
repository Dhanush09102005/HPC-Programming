#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Interpolation (Serial Code)
void interpolation(double *mesh_value, Points *points) {
    double inv_dx = (double)NX; 
    double inv_dy = (double)NY;

    for (int p = 0; p < NUM_Points; p++) {
        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        if (i >= NX - 1) i = NX - 2;
        if (j >= NY - 1) j = NY - 2;

        double lx = x - (i * dx);
        double ly = y - (j * dy);

        double wx_m = dx - lx;
        double wy_m = dy - ly;

        int base_idx = j * GRID_X + i;

        mesh_value[base_idx]                 += wx_m * wy_m;
        mesh_value[base_idx + 1]             += lx * wy_m;
        mesh_value[base_idx + GRID_X]        += wx_m * ly;
        mesh_value[base_idx + GRID_X + 1]    += lx * ly;
    }
}
// Stochastic Mover (Serial Code) 
// Stochastic Mover (Serial Code)
// ---------------------------------
// Mover: Immediate Replacement (Serial)
// ---------------------------------
// --------------------------------------
// Mover: Deferred Insertion (Serial)
// --------------------------------------
void mover_serial_deferred(Points *points, double deltaX, double deltaY)
{
    int write_index = 0;

    for (int p = 0; p < NUM_Points; p++)
    {
        unsigned int seed = 1234 + p;  // unique per particle

        double rx = ((double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0) * deltaX;
        double ry = ((double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0) * deltaY;

        double x = points[p].x + rx;
        double y = points[p].y + ry;

        if (x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0)
        {
            points[write_index].x = x;
            points[write_index].y = y;
            write_index++;
        }
    }

    for (int p = write_index; p < NUM_Points; p++)
    {
        unsigned int seed = 5678 + p;
        points[p].x = (double)rand_r(&seed) / RAND_MAX;
        points[p].y = (double)rand_r(&seed) / RAND_MAX;
    }
}
// Stochastic Mover (Parallel Safe)
// -----------------------------

void mover_parallel_deferred(Points *points, double deltaX, double deltaY)
{
    double *new_x = (double *)malloc(NUM_Points * sizeof(double));
    double *new_y = (double *)malloc(NUM_Points * sizeof(double));
    int *valid = (int *)malloc(NUM_Points * sizeof(int));
    int *prefix = (int *)malloc(NUM_Points * sizeof(int));

    // -------------------------
    // Phase 1: Move + mark valid
    // -------------------------
    #pragma omp parallel for
    for (int p = 0; p < NUM_Points; p++)
    {
        unsigned int seed = 1234 + p;

        double rx = ((double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0) * deltaX;
        double ry = ((double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0) * deltaY;

        double x = points[p].x + rx;
        double y = points[p].y + ry;

        new_x[p] = x;
        new_y[p] = y;

        valid[p] = (x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0);
    }

    // -------------------------
    // Phase 2: Prefix sum (serial is fine, cheap)
    // -------------------------
    prefix[0] = valid[0];
    for (int i = 1; i < NUM_Points; i++)
    {
        prefix[i] = prefix[i - 1] + valid[i];
    }

    int total_valid = prefix[NUM_Points - 1];
    int deleted_count = NUM_Points - total_valid;

    // -------------------------
    // Phase 3: Scatter valid particles
    // -------------------------
    #pragma omp parallel for
    for (int p = 0; p < NUM_Points; p++)
    {
        if (valid[p])
        {
            int index = prefix[p] - 1;

            points[index].x = new_x[p];
            points[index].y = new_y[p];
        }
    }

    // -------------------------
    // Phase 4: Fill voids
    // -------------------------
    #pragma omp parallel for
    for (int p = total_valid; p < NUM_Points; p++)
    {
        unsigned int seed = 5678 + p;

        points[p].x = (double)rand_r(&seed) / RAND_MAX;
        points[p].y = (double)rand_r(&seed) / RAND_MAX;
    }

    free(new_x);
    free(new_y);
    free(valid);
    free(prefix);
}
// Write mesh to file
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
