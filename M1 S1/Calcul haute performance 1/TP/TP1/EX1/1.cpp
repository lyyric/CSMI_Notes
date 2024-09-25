# include <iostream>
# include <mpi.h>
int main () {
    MPI_Init ( nullptr , nullptr );
    int world_rank ;
    MPI_Comm_rank ( MPI_COMM_WORLD , & world_rank );
    float tableau[5];
    if ( world_rank == 0 )
    {
        float tableau[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        MPI_Send(tableau, 5, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(tableau, 5, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        bool correct = true;
        for (int i = 0; i < 5; i++) {
            if (tableau[i] != (i + 1) / 2.0) {
                correct = false;
                break;
            }
        }
        if (correct) {
            std::cout << "Le tableau est correct." << std::endl;
        } else {
            std::cout << "Erreur tableau." << std::endl;
        }
    }
    else if ( world_rank == 1)
    {
        MPI_Recv(tableau, 5, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < 5; i++) {
            tableau[i] /= 2.0;
        }
        MPI_Send(tableau, 5, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize ();
    return 0;
}