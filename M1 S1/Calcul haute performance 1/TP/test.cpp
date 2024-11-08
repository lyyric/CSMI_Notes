#include <iostream>
#include <mpi.h>

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "MPI rank:" << world_rank << std::endl;
    if(world_rank == 0){
        int data = 100;
        MPI_Send(&data, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        std::cout << "data:" << data << std::endl;
    } else {
        int data = 200;
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "data:" << data << std::endl;
    }
    MPI_Finalize();
}