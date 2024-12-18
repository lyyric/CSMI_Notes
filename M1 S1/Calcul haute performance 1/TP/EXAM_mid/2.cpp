// mpi_program.cpp
#include <mpi.h>
#include <iostream>
#include <cstdlib> // For atoi
#include "RandomNumber.h"

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure at least 2 processes
    if (size < 2) {
        if (rank == 0) {
            std::cerr << "At least 2 processes are required." << std::endl;
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Get L from command line arguments
    int L = 5000; // default value

    if (argc > 1) {
        L = std::atoi(argv[1]);
    }

    // Check that L is strictly positive and even
    if (L <= 0 || L % 2 != 0) {
        if (rank == 0) {
            std::cerr << "L must be a strictly positive even integer." << std::endl;
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Each process creates an integer array 'tab' of size L
    int* tab = new int[L];

    // Initialize 'tab' with random values
    RandomNumber<int> randEven(0, 100);
    RandomNumber<int> randOdd(-100, 0);

    for (int i = 0; i < L; ++i) {
        if (i % 2 == 0) {
            tab[i] = randEven();
        } else {
            tab[i] = randOdd();
        }
    }

    // For debugging: print initial 'tab' when L is small
    if (L <= 10) {
        std::cout << "Process " << rank << " initial tab: ";
        for (int i = 0; i < L; ++i) {
            std::cout << tab[i] << " ";
        }
        std::cout << std::endl;
    }

    // Prepare communication
    int dest = (rank + 1) % size;
    int source = (rank - 1 + size) % size;

    // Create derived types for even and odd indices
    MPI_Datatype even_indices_type, odd_indices_type;
    MPI_Type_vector(L / 2, 1, 2, MPI_INT, &even_indices_type);
    MPI_Type_commit(&even_indices_type);

    MPI_Type_vector(L / 2, 1, 2, MPI_INT, &odd_indices_type);
    MPI_Type_commit(&odd_indices_type);

    // Allocate temporary array for receiving data
    int* tab_recv = new int[L];
    std::fill(tab_recv, tab_recv + L, 0);

    MPI_Status status;

    // Communication using derived types
    if (rank % 2 == 0) {
        // Even-ranked process
        MPI_Sendrecv(tab, 1, even_indices_type, dest, 0,
                     tab_recv, 1, even_indices_type, source, 0,
                     MPI_COMM_WORLD, &status);

        // Update 'tab' according to specification
        for (int i = 0; i < L / 2; ++i) {
            int k = 2 * i;
            tab[(k + 1) % L] = tab_recv[k];
        }
    } else {
        // Odd-ranked process
        MPI_Sendrecv(tab, 1, odd_indices_type, dest, 0,
                     tab_recv, 1, odd_indices_type, source, 0,
                     MPI_COMM_WORLD, &status);

        // Update 'tab' according to specification
        for (int i = 0; i < L / 2; ++i) {
            int k = 2 * i + 1;
            tab[(k - 1 + L) % L] = tab_recv[k];
        }
    }

    // For debugging: print 'tab' after communication when L is small
    if (L <= 10) {
        std::cout << "Process " << rank << " tab after communication: ";
        for (int i = 0; i < L; ++i) {
            std::cout << tab[i] << " ";
        }
        std::cout << std::endl;
    }

    // Compute the average of the modified 'tab' array
    long long sum = 0;
    for (int i = 0; i < L; ++i) {
        sum += tab[i];
    }
    double average = sum / (double)L;

    // Use MPI_Gather to collect the averages to process 0
    double* averages = nullptr;
    if (rank == 0) {
        averages = new double[size];
    }

    MPI_Gather(&average, 1, MPI_DOUBLE, averages, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Process 0 displays the averages
    if (rank == 0) {
        std::cout << "Averages after communication:" << std::endl;
        for (int i = 0; i < size; ++i) {
            std::cout << "Process " << i << ": " << averages[i] << std::endl;
        }
        delete[] averages;
    }

    // Clean up
    delete[] tab;
    delete[] tab_recv;
    MPI_Type_free(&even_indices_type);
    MPI_Type_free(&odd_indices_type);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
