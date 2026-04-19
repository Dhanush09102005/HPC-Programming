#ifndef UTILS_H
#define UTILS_H

#include <time.h>
#include "init.h"

extern double min_val, max_val;

// PIC Operations
void interpolation(double *mesh_value, Points *points, int nthreads);
double normalization(double *mesh_value, int nthreads);
double mover(double *mesh_value, Points *points, int nthreads);
double denormalization(double *mesh_value, int nthreads);

// Utility Functions
long long int void_count(Points *points);
void save_mesh(double *mesh_value);

#endif
