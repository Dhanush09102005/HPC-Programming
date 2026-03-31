#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <ctime>
#include <omp.h>
#include "init.h"
#include "utils.h"

using namespace std;

struct Grid {
    int Nx, Ny;
};

// Grid configurations
Grid grids[3] = {
    {250, 100},
    {500, 200},
    {1000, 400}
};

// Thread counts for scalability
int thread_counts[4] = {2, 4, 8, 16};

int main() {
    srand(1234);

    Maxiter = 10;
    NUM_Points = 14000000; // 14 million particles

    ofstream file("results_parallel.csv");
    file << "Grid_Nx,Grid_Ny,Threads,InterpTime,MoverTime,TotalTime\n";

    for (int g = 0; g < 3; g++) {
        Grid grid = grids[g];

        cout << "\n===== Grid: " << grid.Nx << " x " << grid.Ny << " =====\n";

        NX = grid.Nx;
        NY = grid.Ny;
        GRID_X = grid.Nx;
        GRID_Y = grid.Ny;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        // Allocate particles and mesh once per grid
        Points *points = new Points[NUM_Points];
        double *mesh = new double[NX * NY];

        initializepoints(points);

        for (int t = 0; t < 4; t++) {
            int num_threads = thread_counts[t];
            cout << "\nRunning with " << num_threads << " threads...\n";

            omp_set_num_threads(num_threads);

            // Reset mesh
            memset(mesh, 0, NX * NY * sizeof(double));

            double total_interp_time = 0.0;
            double total_mover_time = 0.0;
            double total_time = 0.0;

            for (int iter = 0; iter < Maxiter; iter++) {
                clock_t start_interp = clock();
                interpolation(mesh, points);
                clock_t end_interp = clock();
                double interp_time = double(end_interp - start_interp) / CLOCKS_PER_SEC;

                clock_t start_mover = clock();
                mover_parallel_deferred(points, 0.01, 0.01);
                clock_t end_mover = clock();
                double mover_time = double(end_mover - start_mover) / CLOCKS_PER_SEC;

                total_interp_time += interp_time;
                total_mover_time += mover_time;
                total_time += interp_time + mover_time;
            }

            cout << "Interp Time: " << total_interp_time << " sec, "
                 << "Mover Time: " << total_mover_time << " sec, "
                 << "Total Time: " << total_time << " sec\n";

            file << NX << "," << NY << "," << num_threads << ","
                 << total_interp_time << "," << total_mover_time << "," << total_time << "\n";
        }

        delete[] points;
        delete[] mesh;
    }

    file.close();
    cout << "\nResults saved to results_parallel.csv\n";

    return 0;
}
