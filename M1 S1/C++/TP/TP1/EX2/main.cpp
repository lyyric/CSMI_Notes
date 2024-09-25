#include <iostream>

int main() {
    double somme = 0;
    double nombre;

    for (int i = 0; i < 5; ++i) {
        std::cout << "Entrez un entier (" << i + 1 << "/5): ";
        std::cin >> nombre;
        somme += nombre;
    }

    std::cout << "La moyenne des 5 entiers est: " << somme / 5 << std::endl;

    return 0;
}
