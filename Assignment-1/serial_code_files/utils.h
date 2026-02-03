#ifndef UTILS_H
#define UTILS_H
#include <time.h>

void vector_triad_operation(double *x, double *y, double *v, double *S, int Np);

void vector_add_operation(double *x, double *y, double *S, int Np);

void vector_copy_operation(double *x, double *S, int Np);

void vector_scale_operation(double *x, double *S, int Np);

void dummy(int x);

#endif
