#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <omp.h>

#include "utils.h"

using namespace std;

void interpolation(double *mesh_value, Points *points)
{
    memset(mesh_value, 0, (size_t)GRID_X * GRID_Y * sizeof(double));

    for (int p = 0; p < NUM_Points; p++)
    {
        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x / dx);
        int j = (int)(y / dy);

        if (i >= NX)
            i = NX - 1;
        if (j >= NY)
            j = NY - 1;
        if (i < 0)
            i = 0;
        if (j < 0)
            j = 0;

        double wx = (x - (double)i * dx) / dx;
        double wy = (y - (double)j * dy) / dy;

        mesh_value[j * GRID_X + i] += (1.0 - wx) * (1.0 - wy);
        mesh_value[j * GRID_X + i + 1] += wx * (1.0 - wy);
        mesh_value[(j + 1) * GRID_X + i] += (1.0 - wx) * wy;
        mesh_value[(j + 1) * GRID_X + i + 1] += wx * wy;
    }
}

void mover_immediate_serial(Points *points, double deltaX, double deltaY)
{
    const double inverse_rand = 1.0 / RAND_MAX;
    for (int p = 0; p < NUM_Points; p++)
    {
        double rx = (rand() * inverse_rand * 2.0 - 1.0) * deltaX;
        double ry = (rand() * inverse_rand * 2.0 - 1.0) * deltaY;

        points[p].x += rx;
        points[p].y += ry;

        if (points[p].x < 0.0 || points[p].x > 1.0 || points[p].y < 0.0 || points[p].y > 1.0)
        {
            points[p].x = rand() * inverse_rand;
            points[p].y = rand() * inverse_rand;
        }
    }
}

void mover_serial(Points *points, double deltaX, double deltaY)
{
    mover_immediate_serial(points, deltaX, deltaY);
}
void save_mesh(double *mesh_value)
{
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd)
    {
        fprintf(stderr, "Error creating Mesh.out\n");
        exit(1);
    }
    for (int i = 0; i < GRID_Y; i++)
    {
        for (int j = 0; j < GRID_X; j++)
        {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}
