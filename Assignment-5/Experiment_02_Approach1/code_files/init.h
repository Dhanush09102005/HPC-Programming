#ifndef INIT_H
#define INIT_H

#include <stdio.h>

// Point structure
typedef struct {
    double x, y;
} Points;

// Global simulation parameters
extern int NX, NY;
extern int GRID_X, GRID_Y;
extern int NUM_Points;
extern int Maxiter;
extern double dx, dy;

// Initialization & I/O
void initializepoints(Points *points);

#endif
