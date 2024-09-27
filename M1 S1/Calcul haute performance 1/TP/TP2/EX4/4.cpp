#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 100;
    int local_size = N / nproc;

    std::vector<double> local_vector(local_size);

    std::srand(time(0) + rank);
    for (int i = 0; i < local_size; ++i) {
        local_vector[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    double local_sum = 0.0;
    for (int i = 0; i < local_size; ++i) {
        local_sum += std::fabs(local_vector[i]);
    }

    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < local_size; ++i) {
        local_vector[i] /= global_sum;
    }

    double local_norm_check = 0.0;
    for (int i = 0; i < local_size; ++i) {
        local_norm_check += std::fabs(local_vector[i]);
    }

    double global_norm_check;
    MPI_Allreduce(&local_norm_check, &global_norm_check, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "The 1-norm of the normalized vector is: " << global_norm_check << std::endl;
    }

    MPI_Finalize();
    return 0;
}
