#include <iostream>

int main() {
    int A;
    int B;
    int C;

    std::cout << "Entrez un entier A: ";
        std::cin >> A;
    std::cout << "Entrez un entier B: ";
        std::cin >> B;
    C = A;
    A = B;
    B = C;
    std::cout << "A:"<< A << std::endl;
    std::cout << "B:"<< B << std::endl;
    return 0;
}