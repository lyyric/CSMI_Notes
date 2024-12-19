#include <mpi.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include "randomnumber.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int I[8] = {1,2,3,5,8,13,21,34};
    if(rank == 0){
        RandomNumber<double> rnd(1.0,100.0);
        double P[40];
        for(int i = 0; i<40; ++i) P[i] = rnd();
    }
    if(rank == 1){
        double Q[8];
        for(int i = 0; i<8; ++i) Q[i] = 0;         
    }
    double recv_msg[40];
    double sum = 0;
    MPI_Datatype newtype;
    MPI_Type_contiguous (40 , MPI_INTEGER ,& newtype );
    MPI_Type_commit (&newtype);
    MPI_Sendrecv(P,1,newtype,0,110,recv_msg,1,newtype,0,100,MPI_COMM_WORLD , MPI_STATUS_IGNORE);
    if(rank == 1){
        for(int i = 0; i<8; ++i){
            Q[i] = P[I[i]];
            sum += Q[i];
        }

    }
    MPI_Finalize();
    return 0;
}
