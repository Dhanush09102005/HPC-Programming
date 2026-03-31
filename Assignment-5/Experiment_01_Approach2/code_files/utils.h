#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"

void interpolation(double *mesh_value, Points *points);
void mover_serial(Points *points, double deltaX, double deltaY);
void mover_parallel(Points *points, double deltaX, double deltaY);
void save_mesh(double *mesh_value);
void interpolation(double *mesh_value, Points *points);
void mover_deferred_serial(Points *points, double deltaX, double deltaY);
void mover_deferred_parallel(Points *points, double deltaX, double deltaY);
void mover_simple_parallel(Points *points, double deltaX, double deltaY);
#endif
