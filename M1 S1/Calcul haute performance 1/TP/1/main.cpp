#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Request reqs[2];
    MPI_Status stats[2];

    int num_proc = (world_rank + 1) % 2;
    int nData = 1000;
    double sent_message[nData], recv_message[nData];
    sent_message[0] = 123 + world_rank; // 等等...

    MPI_Isend(sent_message, nData, MPI_DOUBLE, num_proc, 110, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(recv_message, nData, MPI_DOUBLE, num_proc, 110, MPI_COMM_WORLD, &reqs[1]);

    std::cout << "Before Wait(), got from processor " << num_proc << " message " << recv_message[0] << std::endl;

    MPI_Wait(&reqs[0], &stats[0]);
    MPI_Wait(&reqs[1], &stats[1]);

    std::cout << "After Wait(), got from processor " << num_proc << " message " << recv_message[0] << std::endl;

    MPI_Finalize();
    return 0;
}