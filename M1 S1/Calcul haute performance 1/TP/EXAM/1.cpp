#include <iostream>
#include <mpi.h>
#include "randomnumber.hpp"

int main() {
    MPI_Init( nullptr, nullptr );
    int worldSize, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &worldSize );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    RandomNumberNormal<double> rnd(12.34,3.14);
    int p = 20000;
    int q = p*worldSize;
    double * u = new double[p];
    for (int k = 0; k < p; ++k)
        u[k] = rnd();

    // TODO : compute v and w, then processus 0 print results

    double local_sum = 0.0;
    for (int k = 0; k < p; ++k)
        local_sum += u[k];

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double v = 0.0;
    if (rank == 0) {
        v = global_sum / q;
    }
    MPI_Bcast(&v, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_sq_sum = 0.0;
    for (int k = 0; k < p; ++k) {
        double diff = u[k] - v;
        local_sq_sum += diff * diff;
    }

    double global_sq_sum = 0.0;
    MPI_Reduce(&local_sq_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double w = sqrt(global_sq_sum / q);
        std::cout << "v = " << v << std::endl;
        std::cout << "w = " << w << std::endl;
    }

    MPI_Finalize();
    return 0;
}