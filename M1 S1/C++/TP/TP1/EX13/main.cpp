#include <iostream>
#include <cmath>

double* echantillonner(double min, double max, int n_points, double (*func)(double)) {
    if (n_points <= 2) {
        std::cerr << "Le nombre de points doit être supérieur à 2." << std::endl;
        return nullptr;
    }

    double* resultat = new double[n_points];
    double pas = (max - min) / (n_points - 1);

    for (int i = 0; i < n_points; ++i) {
        double x = min + i * pas;
        resultat[i] = func(x);
    }

    return resultat;
}

int main() {
    double min = 0.0, max = M_PI;
    int n_points = 10;

    double* result_cos = echantillonner(min, max, n_points, cos);

    if (result_cos != nullptr) {
        std::cout << "Échantillonnage de la fonction cosinus:" << std::endl;
        for (int i = 0; i < n_points; ++i) {
            std::cout << "cos(" << min + i * ((max - min) / (n_points - 1)) << ") = " << result_cos[i] << std::endl;
        }
        delete[] result_cos;
    }

    double* result_sin = echantillonner(min, max, n_points, sin);

    if (result_sin != nullptr) {
        std::cout << "\nÉchantillonnage de la fonction sinus:" << std::endl;
        for (int i = 0; i < n_points; ++i) {
            std::cout << "sin(" << min + i * ((max - min) / (n_points - 1)) << ") = " << result_sin[i] << std::endl;
        }
        delete[] result_sin;
    }

    return 0;
}
