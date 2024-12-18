#include <iostream>
#include <mpi.h>

int main() {
    MPI_Init(nullptr, nullptr);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int valeurSend = 1000 + world_rank; // 例如：rank 0 -> 1000, rank 1 -> 1001
    if (world_rank == 0 || world_rank == 1) {
        int rankToComm = (world_rank + 1) % 2; // 0 -> 1, 1 -> 0
        int tag = 123, valeurRecv = 0;
        MPI_Sendrecv(&valeurSend, 1, MPI_INT, rankToComm, tag,
                     &valeurRecv, 1, MPI_INT, rankToComm, tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "rank=" << world_rank
                  << " valeurSend=" << valeurSend
                  << " valeurRecv=" << valeurRecv << "\n";
    }
    MPI_Finalize();
    return 0;
}