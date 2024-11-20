#include <iostream>
#include <vector>
#include <numeric>
#include "randomnumber.hpp"

int main() {
    int size;
    std::cout << "Entrez la taille du tableau : ";
    std::cin >> size;

    RandomNumber<double> rnd(0.0, 500.0); 

    std::vector<double> tableau(size);

    std::vector<int>::iterator it = tableau.begin();
    std::vector<int>::iterator en = tableau.end();
    for ( ; it!=en; ++it )
        *it = rnd();

    double somme = std::accumulate(tableau.begin(), tableau.end(), 0.0);
    double moyenne = somme / size;

    std::cout << "La moyenne des valeurs dans le tableau est : " << moyenne << std::endl;

    return 0;
}