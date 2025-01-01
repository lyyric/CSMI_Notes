#include <iostream>

int main()
{
    double largeur, longueur, surface, perimetre;

    std::cout << "Tapez la largeur du champ : ";
    std::cin >> largeur;
    std::cout << "Tapez la longueur du champ : ";
    std::cin >> longueur;

    surface = largeur * longueur;
    perimetre = 2 * (largeur + longueur);

    std::cout << "La surface vaut : " << surface << std::endl;
    std::cout << "Le perimetre vaut : " << perimetre << std::endl;

    std::cout << "Appuyez sur une touche pour continuer." << std::endl;
    std::cin.ignore();
    std::cin.get();

    return EXIT_SUCCESS;
}
