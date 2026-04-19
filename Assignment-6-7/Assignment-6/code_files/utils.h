#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"

// PIC operations
double interpolation(double *mesh_value, Points *points, int nthreads);
void save_mesh(double *mesh_value);

#endif
