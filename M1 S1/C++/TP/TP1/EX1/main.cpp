#include <iostream>

int main() {
    double largeur, longueur; 

    std::cout << "Entrez la largeur du champ: ";
    std::cin >> largeur;

    std::cout << "Entrez la longueur du champ: ";
    std::cin >> longueur;

    double perimetre = 2 * (largeur + longueur);
    double surface = largeur * longueur;

    std::cout << "Le périmètre du champ est: " << perimetre << std::endl;
    std::cout << "La surface du champ est: " << surface << std::endl;

    return 0;
}
