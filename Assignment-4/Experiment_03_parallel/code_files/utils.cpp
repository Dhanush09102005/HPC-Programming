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
	    i = (i >= NX) ? NX - 1 : i;
	    j = (j >= NY) ? NY - 1 : j;

	    double lx = x - (i * dx);
	    double ly = y - (j * dy);

	    double wx_m = dx - lx;
	    double wy_m = dy - ly;

	    int base_idx = j * GRID_X + i;

	    mesh_value[base_idx]              += wx_m * wy_m; 
	    mesh_value[base_idx + 1]          += lx * wy_m; 
	    mesh_value[base_idx + GRID_X]     += wx_m * ly; 
	    mesh_value[base_idx + GRID_X + 1] += lx * ly; 
	}
	}

// Stochastic Mover (Serial Code) 
// Stochastic Mover (Serial Code)
void mover_serial(Points *points, double deltaX, double deltaY)
{
    unsigned int seed = 1234;

    for (int p = 0; p < NUM_Points; p++) {

        double rx = ((double)rand_r(&seed) / RAND_MAX - 0.5) * deltaX;
        double ry = ((double)rand_r(&seed) / RAND_MAX - 0.5) * deltaY;

        points[p].x += rx;
        points[p].y += ry;

        if (points[p].x < 0.0) points[p].x += 1.0;
        else if (points[p].x >= 1.0) points[p].x -= 1.0;

        if (points[p].y < 0.0) points[p].y += 1.0;
        else if (points[p].y >= 1.0) points[p].y -= 1.0;
    }
}
// -----------------------------
// Stochastic Mover (Parallel Safe)
// -----------------------------
void mover_parallel(Points *points, double deltaX, double deltaY)
{
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < NUM_Points; p++)
    {
        unsigned int seed = 1234 + p;

        double rx = ((double)rand_r(&seed) / RAND_MAX - 0.5) * deltaX;
        double ry = ((double)rand_r(&seed) / RAND_MAX - 0.5) * deltaY;

        double x = points[p].x + rx;
        double y = points[p].y + ry;

        if (x < 0.0) x += 1.0;
        else if (x >= 1.0) x -= 1.0;

        if (y < 0.0) y += 1.0;
        else if (y >= 1.0) y -= 1.0;

        points[p].x = x;
        points[p].y = y;
    }
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
