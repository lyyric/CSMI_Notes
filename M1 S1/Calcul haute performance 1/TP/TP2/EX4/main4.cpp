#include <mpi.h>
#include <iostream>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include <cmath>    // For fabs()

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);  // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get the current process rank

    // Step 1: Define the size of the vector and distribute among processes
    int N = 100;  // Total size of the vector
    int local_size = N / nproc;  // Each process handles a part of the vector

    // Allocate array for local part of the vector
    double* local_vector = new double[local_size];

    // Initialize the vector with random values (each process generates its part)
    std::srand(time(0) + rank);
    for (int i = 0; i < local_size; ++i) {
        local_vector[i] = static_cast<double>(rand()) / RAND_MAX;  // Random values between 0 and 1
    }

    // Step 2: Calculate the local 1-norm
    double local_sum = 0.0;
    for (int i = 0; i < local_size; ++i) {
        local_sum += std::fabs(local_vector[i]);
    }

    // Step 3: Reduce to find the global 1-norm (sum over all processes)
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Step 4: Normalize the local part of the vector
    for (int i = 0; i < local_size; ++i) {
        local_vector[i] /= global_sum;
    }

    // Step 5: Verify the normalization (calculate the global 1-norm again)
    double local_norm_check = 0.0;
    for (int i = 0; i < local_size; ++i) {
        local_norm_check += std::fabs(local_vector[i]);
    }

    double global_norm_check;
    MPI_Allreduce(&local_norm_check, &global_norm_check, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Only one process (e.g., rank 0) outputs the results
    if (rank == 0) {
        std::cout << "The 1-norm of the normalized vector is: " << global_norm_check << std::endl;
    }

    // Free allocated memory
    delete[] local_vector;

    MPI_Finalize();
    return 0;
}
