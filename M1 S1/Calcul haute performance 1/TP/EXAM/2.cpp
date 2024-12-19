#include <mpi.h>
#include <iostream>
#include <ctime>
#include <vector>


int marque(int* listn, int n){
    int k = 2;
    while (k*k < n){
        for(int i = k*k; i< n; ++i){
            if(listn[i]%k==0) listn[i]=0;
        }
        for(int i = 0; i< n; ++i)
            if(listn[i]>k && listn[i]!=0) k = listn[i];
    }
    int li=0;
    for(int i = 0; i< n; ++i) if(listn[i]!=0) ++li;
    return li;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc > 1) {
        int n = argv[1];
    }
    int base = (n-1) / size;
    int rest = (n-1) % size;
    int list = base + (rank < rest)?1:0;
    int listn[list];
    for(int i = 0; i < list; ++i) listn[i] = i + rank*base + (rank < rest)?rank:rest;
    int li = marque(listn, list);
    int result = 0;
    MPI_Reduce(&li, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "n = " << n << ", nombre = " << result << std::endl;
    }
    MPI_Finalize();
    return 0;
}