#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <ctime>
#include "init.h"
#include "utils.h"

using namespace std;

struct Grid {
    int Nx, Ny;
};

Grid grids[3] = {
    {250, 100},
    {500, 200},
    {1000, 400}
};

long long particle_counts[5] = {
    100LL, 10000LL, 1000000LL, 100000000LL, 1000000000LL
};

int main() {
    srand(1234);

    Maxiter = 10;

    ofstream file("results_serial.csv");
    file << "Nx,Ny,Particles,TotalTime\n";

    for (int g = 0; g < 3; g++) {
        Grid grid = grids[g];
        cout << "\n===== Grid: " << grid.Nx << " x " << grid.Ny << " =====\n";

        NX = grid.Nx;
        NY = grid.Ny;
        GRID_X = grid.Nx;
        GRID_Y = grid.Ny;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        for (int n = 0; n < 5; n++) {
            long long N = particle_counts[n];
            cout << "Particles: " << N << endl;
            NUM_Points = (int)N;

            Points *points = new Points[NUM_Points];
            double *mesh = new double[NX * NY];

            initializepoints(points);

            memset(mesh, 0, NX * NY * sizeof(double));

            double total_time = 0.0;

            for (int iter = 0; iter < Maxiter; iter++) {
                clock_t start = clock();

                interpolation(mesh, points);
                mover_serial_deferred(points, 0.01, 0.01);

                clock_t end = clock();
                total_time += double(end - start) / CLOCKS_PER_SEC;
            }

            cout << "Total Time: " << total_time << " sec\n";

            file << NX << ","
                 << NY << ","
                 << N << ","
                 << total_time << "\n";

            delete[] points;
            delete[] mesh;
        }
    }

    file.close();
    cout << "\nResults saved to results_serial.csv\n";

    return 0;
}
