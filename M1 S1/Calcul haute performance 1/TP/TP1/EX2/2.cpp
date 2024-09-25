#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
    // Initialisation de MPI
    MPI_Init(&argc, &argv);

    // Récupération du rang du processus
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Création de la valeur propre à chaque processus
    int valeur = 10 * rank;
    int valeur_recue;  // Pour stocker la valeur reçue

    // Établissement des communications selon les instructions
    if (rank == 0) {
        // Proc 0 envoie sa valeur au proc 1 et reçoit la valeur du proc 3
        MPI_Sendrecv(&valeur, 1, MPI_INT, 1, 0, 
                     &valeur_recue, 1, MPI_INT, 3, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Proc 0 : envoie " << valeur << " à Proc 1, reçoit " << valeur_recue << " de Proc 3\n";
    } else if (rank == 1) {
        // Proc 1 envoie sa valeur au proc 2 et reçoit la valeur du proc 0
        MPI_Sendrecv(&valeur, 1, MPI_INT, 2, 0, 
                     &valeur_recue, 1, MPI_INT, 0, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Proc 1 : envoie " << valeur << " à Proc 2, reçoit " << valeur_recue << " de Proc 0\n";
    } else if (rank == 2) {
        // Proc 2 envoie sa valeur au proc 3 et reçoit la valeur du proc 1
        MPI_Sendrecv(&valeur, 1, MPI_INT, 3, 0, 
                     &valeur_recue, 1, MPI_INT, 1, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Proc 2 : envoie " << valeur << " à Proc 3, reçoit " << valeur_recue << " de Proc 1\n";
    } else if (rank == 3) {
        // Proc 3 envoie sa valeur au proc 0 et reçoit la valeur du proc 2
        MPI_Sendrecv(&valeur, 1, MPI_INT, 0, 0, 
                     &valeur_recue, 1, MPI_INT, 2, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Proc 3 : envoie " << valeur << " à Proc 0, reçoit " << valeur_recue << " de Proc 2\n";
    }

    // Finalisation de MPI
    MPI_Finalize();
    return 0;
}
