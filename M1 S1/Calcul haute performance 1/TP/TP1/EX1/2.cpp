#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    // Initialisation de MPI
    MPI_Init(&argc, &argv);

    // Récupération du rang du processus
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int taille = 5;  // Taille du tableau de réels
    std::vector<double> tableau(taille);  // Tableau de réels

    if (rank == 0) {
        // Processus 0 initialise le tableau
        tableau = {1.0, 2.0, 3.0, 4.0, 5.0};
        std::cout << "Processus 0 : Tableau initial = ";
        for (double val : tableau) std::cout << val << " ";
        std::cout << std::endl;

        // Envoie du tableau au processus 1
        MPI_Send(tableau.data(), taille, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

        // Réception du tableau modifié par le processus 1
        MPI_Recv(tableau.data(), taille, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Vérification du tableau modifié
        std::cout << "Processus 0 : Tableau reçu modifié = ";
        for (double val : tableau) std::cout << val << " ";
        std::cout << std::endl;

        // Vérification des résultats
        bool correct = true;
        for (int i = 0; i < taille; i++) {
            if (tableau[i] != (i + 1) / 2.0) {
                correct = false;
                break;
            }
        }
        if (correct) {
            std::cout << "Le tableau reçu est correct." << std::endl;
        } else {
            std::cout << "Erreur dans le tableau reçu." << std::endl;
        }
    }
    else if (rank == 1) {
        // Processus 1 reçoit le tableau du processus 0
        MPI_Recv(tableau.data(), taille, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Processus 1 : Tableau reçu = ";
        for (double val : tableau) std::cout << val << " ";
        std::cout << std::endl;

        // Division par 2 des éléments du tableau
        for (int i = 0; i < taille; i++) {
            tableau[i] /= 2.0;
        }

        std::cout << "Processus 1 : Tableau modifié = ";
        for (double val : tableau) std::cout << val << " ";
        std::cout << std::endl;

        // Envoie du tableau modifié au processus 0
        MPI_Send(tableau.data(), taille, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // Finalisation de MPI
    MPI_Finalize();
    return 0;
}
