# include <iostream>
# include <mpi.h>
int main () {
    MPI_Init ( nullptr , nullptr );
    int world_rank ;
    MPI_Comm_rank ( MPI_COMM_WORLD , & world_rank );
    int valeur = 10 * world_rank ;
    int valeur_recue;

    if (world_rank == 0) {
        MPI_Sendrecv(&valeur, 1, MPI_INT, 1, 0, 
                     &valeur_recue, 1, MPI_INT, 3, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Proc 0 : envoie " << valeur << " à Proc 1, reçoit " << valeur_recue << " de Proc 3\n";
    } else if (world_rank == 1) {
        MPI_Sendrecv(&valeur, 1, MPI_INT, 2, 0, 
                     &valeur_recue, 1, MPI_INT, 0, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Proc 1 : envoie " << valeur << " à Proc 2, reçoit " << valeur_recue << " de Proc 0\n";
    } else if (world_rank == 2) {
        MPI_Sendrecv(&valeur, 1, MPI_INT, 3, 0, 
                     &valeur_recue, 1, MPI_INT, 1, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Proc 2 : envoie " << valeur << " à Proc 3, reçoit " << valeur_recue << " de Proc 1\n";
    } else if (world_rank == 3) {
        MPI_Sendrecv(&valeur, 1, MPI_INT, 0, 0, 
                     &valeur_recue, 1, MPI_INT, 2, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Proc 3 : envoie " << valeur << " à Proc 0, reçoit " << valeur_recue << " de Proc 2\n";
    }
    MPI_Finalize ();
    return 0;
}