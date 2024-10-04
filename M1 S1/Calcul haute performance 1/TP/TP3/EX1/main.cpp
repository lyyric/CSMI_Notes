#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

void sorrrt(std::vector<int>& row) {
    for (size_t i = 1; i < row.size(); ++i) {
        int key = row[i];
        int j = i - 1;
        while (j >= 0 && row[j] > key) {
            row[j + 1] = row[j];
            j--;
        }
        row[j + 1] = key;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = 2;
    int N = M * size;
    std::vector<int> matrix;

    if (rank == 0) {
        srand(time(0));
        matrix.resize(N * N);
        for (int i = 0; i < N * N; ++i) {
            matrix[i] = rand() % 100;
        }

        std::cout << "Original matrix:" << std::endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << matrix[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Datatype row_type;
    MPI_Type_vector(1, N, N, MPI_INT, &row_type);
    MPI_Type_commit(&row_type);

    std::vector<int> local_matrix(M * N);

    MPI_Scatter(matrix.data(), M, row_type, local_matrix.data(), M * N, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < M; ++i) {
        std::vector<int> row(local_matrix.begin() + i * N, local_matrix.begin() + (i + 1) * N);
        sorrrt(row);
        std::copy(row.begin(), row.end(), local_matrix.begin() + i * N);
    }

    MPI_Gather(local_matrix.data(), M * N, MPI_INT, matrix.data(), M, row_type, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Sorted matrix:" << std::endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << matrix[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Type_free(&row_type);
    MPI_Finalize();
    return 0;
}
