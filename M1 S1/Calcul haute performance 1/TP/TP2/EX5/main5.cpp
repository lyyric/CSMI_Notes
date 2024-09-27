#include <mpi.h>
#include <iostream>
#include <cmath>   // For mathematical functions like sin
#include <cstdlib> // For atof()

// Function to evaluate: f(x) = (x^6 + x^4 - 2*x^2) * sin(8*x)
double func(double x) {
    return (std::pow(x, 6) + std::pow(x, 4) - 2 * std::pow(x, 2)) * std::sin(8 * x);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);  // Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get the current process rank

    // Define the interval [a, b]
    double a = -10.0, b = 10.0;  // Change these values as needed
    int num_points = 100000;     // Number of points to sample within [a, b]

    // Divide the interval among the processes
    double local_a = a + rank * (b - a) / nproc;
    double local_b = a + (rank + 1) * (b - a) / nproc;
    double step = (local_b - local_a) / num_points;

    // Find zeros in the local interval
    double previous_value = func(local_a);
    double zero_points[100]; // Assuming at most 100 zeros per process (adjust as needed)
    int zero_count = 0;

    for (int i = 1; i <= num_points; ++i) {
        double x = local_a + i * step;
        double current_value = func(x);
        
        // Check for sign change indicating a zero crossing
        if (previous_value * current_value <= 0 && zero_count < 100) {
            zero_points[zero_count++] = x;
        }
        previous_value = current_value;
    }

    // Gather all zero counts to process 0
    int* all_counts = nullptr;
    if (rank == 0) {
        all_counts = new int[nproc];
    }
    MPI_Gather(&zero_count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather all zeros to process 0
    double* all_zeros = nullptr;
    if (rank == 0) {
        int total_zeros = 0;
        for (int i = 0; i < nproc; ++i) {
            total_zeros += all_counts[i];
        }
        all_zeros = new double[total_zeros];
    }

    // Use MPI_Gatherv to collect all zero points to process 0
    int* displs = nullptr;
    if (rank == 0) {
        displs = new int[nproc];
        displs[0] = 0;
        for (int i = 1; i < nproc; ++i) {
            displs[i] = displs[i - 1] + all_counts[i - 1];
        }
    }

    MPI_Gatherv(zero_points, zero_count, MPI_DOUBLE, all_zeros, all_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Display results in process 0
    if (rank == 0) {
        std::cout << "Zero points found in the interval [" << a << ", " << b << "]:" << std::endl;
        int total_zeros = 0;
        for (int i = 0; i < nproc; ++i) {
            total_zeros += all_counts[i];
        }
        for (int i = 0; i < total_zeros; ++i) {
            std::cout << all_zeros[i] << std::endl;
        }

        // Free allocated memory
        delete[] all_counts;
        delete[] all_zeros;
        delete[] displs;
    }

    MPI_Finalize();
    return 0;
}
