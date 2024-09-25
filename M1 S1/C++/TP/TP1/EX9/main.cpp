#include <iostream>
#include <cmath> 

double distance(double xa, double ya, double xb, double yb) {
    return sqrt(pow(xb - xa, 2) + pow(yb - ya, 2));
}

int main() {
    double xa, ya, xb, yb;

    std::cout << "Entrez les coordonnées du point A (xa, ya): ";
    std::cin >> xa >> ya;

    std::cout << "Entrez les coordonnées du point B (xb, yb): ";
    std::cin >> xb >> yb;

    double resultat = distance(xa, ya, xb, yb);
    std::cout << "La distance entre A et B est: " << resultat << std::endl;

    return 0;
}
