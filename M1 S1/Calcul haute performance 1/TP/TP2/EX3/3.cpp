#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime> 

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int p = rank;
    int Ip_start = p;
    int Ip_end = p + 1;

    int M = 1000;
    if ( argc>1 )
        M = std::stoi(argv[1]);
    double* random_numbers = new double[M];
    std::srand(time(0) + rank);

    double sum = 0.0;
    for (int i = 0; i < M; ++i) {
        random_numbers[i] = Ip_start + static_cast<double>(rand()) / RAND_MAX * (Ip_end - Ip_start);
        sum += random_numbers[i];
    }

    double local_average = sum / M;

    double* averages = nullptr;
    if (rank == 0) {
        averages = new double[nproc];
    }

    MPI_Gather(&local_average, 1, MPI_DOUBLE, averages, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
    std::cout << "Received averages from all processes:" << std::endl;
        for (int i = 0; i < nproc; ++i) {
            std::cout << "Process " << i << " average: " << averages[i] << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
