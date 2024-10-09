#include <iostream>
#include <iomanip>

int main () {
    std::cout << "entrer un réel : ";
    double val;
    std::cin >> val;

    double val2 = val*val;
    for (int i=0;i<10;++i){
        std::cout << "précision de " << i << " chiffres : "  << std::fixed << std::setw(15) << std::setprecision(i) << val2 << std :: endl;
    }
    for (int i=0;i<10;++i){
        std::cout << "précision de " << i << " chiffres : "  << std::scientific << std::setw(15) << std::setprecision(i) << val2 << std :: endl;
    }
}
