#include <iostream>
#include <cmath>
#include <mpi.h>

double f(double x) {
    return (pow(x, 6) + pow(x, 4) - 2 * x * x) * sin(8 * x);
}

int main(int argc, char *argv[]) {
    int rank, size;
    double a, b;
    int num_steps = 1000000;  // Number of steps for scanning
    double *zeros = nullptr;
    int zero_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Master process reads the interval [a, b]
    if (rank == 0) {
        std::cout << "Enter the interval [a, b]: ";
        std::cin >> a >> b;
    }

    // Broadcast the interval to all processes
    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Divide the interval among processes
    double interval_length = (b - a) / size;
    double local_a = a + rank * interval_length;
    double local_b = local_a + interval_length;

    double h = (local_b - local_a) / num_steps;
    double x1 = local_a;
    double f1 = f(x1);

    // Array to store zeros found by this process
    const int max_zeros = 1000;  // Assume at most 1000 zeros per process
    double local_zeros[max_zeros];
    int local_zero_count = 0;

    for (int i = 1; i <= num_steps; i++) {
        double x2 = local_a + i * h;
        double f2 = f(x2);

        if (f1 * f2 <= 0) {
            // Possible zero crossing between x1 and x2
            // Use bisection method to find zero more precisely
            double xa = x1;
            double xb = x2;
            double fa = f1;
            double fb = f2;
            double xm, fm;
            int iter;
            for (iter = 0; iter < 100; iter++) {  // Max 100 iterations
                xm = (xa + xb) / 2;
                fm = f(xm);
                if (fabs(fm) < 1e-10) {
                    break;  // Found zero
                }
                if (fa * fm < 0) {
                    xb = xm;
                    fb = fm;
                } else {
                    xa = xm;
                    fa = fm;
                }
            }
            if (local_zero_count < max_zeros) {
                local_zeros[local_zero_count++] = xm;
            }
        }

        x1 = x2;
        f1 = f2;
    }

    // Gather zeros from all processes to the master process
    if (rank == 0) {
        zeros = new double[size * max_zeros];
    }
    int *recvcounts = nullptr;
    int *displs = nullptr;
    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];
    }
    MPI_Gather(&local_zero_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
        zero_count = displs[size - 1] + recvcounts[size - 1];
    }

    MPI_Gatherv(local_zeros, local_zero_count, MPI_DOUBLE,
                zeros, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Master process prints all zeros found
    if (rank == 0) {
        std::cout << "Zeros of the function in the interval [" << a << ", " << b << "]:\n";
        for (int i = 0; i < zero_count; i++) {
            std::cout.precision(10);
            std::cout << zeros[i] << std::endl;
        }
        delete[] zeros;
        delete[] recvcounts;
        delete[] displs;
    }

    MPI_Finalize();
    return 0;
}
