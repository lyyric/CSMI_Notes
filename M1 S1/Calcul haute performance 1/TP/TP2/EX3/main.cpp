# include <iostream>
# include <mpi.h>
# include <vector>
# include <cstdlib>  // For rand() and srand()
# include <ctime>    // For time()

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
    std::srand(time(0) + rank);  // Seed the random number generator uniquely for each process

    std::vector<double> random_numbers(M);
    double sum = 0.0;
    for (int i = 0; i < M; ++i) {
        random_numbers[i] = Ip_start + static_cast<double>(rand()) / RAND_MAX * (Ip_end - Ip_start);
        sum += random_numbers[i];
    }

    double local_average = sum / M;

    // Step 3: Each process sends its average to process p (itself in this case)
    std::vector<double> averages(nproc);
    MPI_Gather(&local_average, 1, MPI_DOUBLE, averages.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 4: Display results (only process 0 will gather and print results)
    if (rank == 0) {
        std::cout << "Received averages from all processes:" << std::endl;
        for (int i = 0; i < nproc; ++i) {
            std::cout << "Process " << i << " average: " << averages[i] << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
