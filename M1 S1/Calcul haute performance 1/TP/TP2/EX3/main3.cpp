#include <mpi.h>
#include <iostream>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);  // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get the current process rank

    // Step 1: Define intervals
    int p = rank;  // Current process index
    int Ip_start = p;
    int Ip_end = p + 1;

    // Step 2: Generate M random numbers in the interval and calculate the average
    int M = 100;  // You can adjust M to see the effect on convergence
    double* random_numbers = new double[M];  // Allocate array for random numbers
    std::srand(time(0) + rank);  // Seed the random number generator uniquely for each process

    double sum = 0.0;
    for (int i = 0; i < M; ++i) {
        random_numbers[i] = Ip_start + static_cast<double>(rand()) / RAND_MAX * (Ip_end - Ip_start);
        sum += random_numbers[i];
    }

    double local_average = sum / M;

    // Step 3: Each process sends its average to process 0
    double* averages = nullptr;
    if (rank == 0) {
        averages = new double[nproc];  // Allocate array to gather averages in process 0
    }
    MPI_Gather(&local_average, 1, MPI_DOUBLE, averages, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 4: Display results (only process 0 will gather and print results)
    if (rank == 0) {
        std::cout << "Received averages from all processes:" << std::endl;
        for (int i = 0; i < nproc; ++i) {
            std::cout << "Process " << i << " average: " << averages[i] << std::endl;
        }
        delete[] averages;  // Free the allocated memory for averages
    }

    // Free the allocated memory for random numbers
    delete[] random_numbers;

    MPI_Finalize();
    return 0;
}
