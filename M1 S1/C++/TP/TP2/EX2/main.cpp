#include <iostream>
#include <cmath>

class Complexe {
private:
    double re; // Partie réelle
    double im; // Partie imaginaire

public:
    // Constructeur par défaut (initialise à 0 + 0i)
    Complexe() : re(0.0), im(0.0) {}

    // Constructeur avec partie réelle et imaginaire
    Complexe(double real, double imag) : re(real), im(imag) {}

    // Constructeur prenant uniquement la partie réelle (pour interactions avec double)
    Complexe(double real) : re(real), im(0.0) {}

    // Accesseurs
    double getRe() const { return re; }
    double getIm() const { return im; }

    // Mutateurs
    void setRe(double real) { re = real; }
    void setIm(double imag) { im = imag; }

    // Surcharge de l'opérateur de flux de sortie
    friend std::ostream& operator<<(std::ostream& os, const Complexe& c) {
        os << c.re;
        if (c.im >= 0)
            os << " + " << c.im << "i";
        else
            os << " - " << -c.im << "i";
        return os;
    }

    // Opérateurs de comparaison
    bool operator==(const Complexe& other) const {
        return (std::abs(re - other.re) < 1e-10) && (std::abs(im - other.im) < 1e-10);
    }

    bool operator!=(const Complexe& other) const {
        return !(*this == other);
    }

    // Opérateurs d'affectation composés
    Complexe& operator+=(const Complexe& other) {
        re += other.re;
        im += other.im;
        return *this;
    }

    Complexe& operator-=(const Complexe& other) {
        re -= other.re;
        im -= other.im;
        return *this;
    }

    Complexe& operator*=(const Complexe& other) {
        double tempRe = re * other.re - im * other.im;
        double tempIm = re * other.im + im * other.re;
        re = tempRe;
        im = tempIm;
        return *this;
    }

    Complexe& operator/=(const Complexe& other) {
        double denom = other.re * other.re + other.im * other.im;
        if (denom == 0.0) {
            std::cerr << "Division par zéro dans les nombres complexes. Résultat défini à 0 + 0i." << std::endl;
            re = 0.0;
            im = 0.0;
            return *this;
        }
        double tempRe = (re * other.re + im * other.im) / denom;
        double tempIm = (im * other.re - re * other.im) / denom;
        re = tempRe;
        im = tempIm;
        return *this;
    }

    // Opérateurs binaires
    friend Complexe operator+(Complexe lhs, const Complexe& rhs) {
        lhs += rhs;
        return lhs;
    }

    friend Complexe operator-(Complexe lhs, const Complexe& rhs) {
        lhs -= rhs;
        return lhs;
    }

    friend Complexe operator*(Complexe lhs, const Complexe& rhs) {
        lhs *= rhs;
        return lhs;
    }

    friend Complexe operator/(Complexe lhs, const Complexe& rhs) {
        lhs /= rhs;
        return lhs;
    }

    // Opérateurs avec double (à gauche)
    friend Complexe operator+(double lhs, const Complexe& rhs) {
        return Complexe(lhs + rhs.re, rhs.im);
    }

    friend Complexe operator-(double lhs, const Complexe& rhs) {
        return Complexe(lhs - rhs.re, -rhs.im);
    }

    friend Complexe operator*(double lhs, const Complexe& rhs) {
        return Complexe(lhs * rhs.re, lhs * rhs.im);
    }

    friend Complexe operator/(double lhs, const Complexe& rhs) {
        double denom = rhs.re * rhs.re + rhs.im * rhs.im;
        if (denom == 0.0) {
            std::cerr << "Division par zéro dans les nombres complexes. Résultat défini à 0 + 0i." << std::endl;
            return Complexe(0.0, 0.0);
        }
        return Complexe((lhs * rhs.re) / denom, (-lhs * rhs.im) / denom);
    }

    // Opérateurs avec double (à droite)
    friend Complexe operator+(const Complexe& lhs, double rhs) {
        return Complexe(lhs.re + rhs, lhs.im);
    }

    friend Complexe operator-(const Complexe& lhs, double rhs) {
        return Complexe(lhs.re - rhs, lhs.im);
    }

    friend Complexe operator*(const Complexe& lhs, double rhs) {
        return Complexe(lhs.re * rhs, lhs.im * rhs);
    }

    friend Complexe operator/(const Complexe& lhs, double rhs) {
        if (rhs == 0.0) {
            std::cerr << "Division par zéro. Résultat défini à 0 + 0i." << std::endl;
            return Complexe(0.0, 0.0);
        }
        return Complexe(lhs.re / rhs, lhs.im / rhs);
    }
};

// Fonction de test
int main() {
    // Test des constructeurs
    Complexe c1; // 0 + 0i
    Complexe c2(3.4, 2.3); // 3.4 + 2.3i
    Complexe c3(5.0); // 5.0 + 0i

    std::cout << "Constructeurs:" << std::endl;
    std::cout << "c1 = " << c1 << " (Expected: 0 + 0i)" << std::endl;
    std::cout << "c2 = " << c2 << " (Expected: 3.4 + 2.3i)" << std::endl;
    std::cout << "c3 = " << c3 << " (Expected: 5 + 0i)" << std::endl;
    std::cout << std::endl;

    // Test des accesseurs et mutateurs
    c1.setRe(1.1);
    c1.setIm(-1.1);
    std::cout << "Après setRe et setIm sur c1: " << c1 << " (Expected: 1.1 - 1.1i)" << std::endl;
    std::cout << "Accesseurs c1: Re = " << c1.getRe() << ", Im = " << c1.getIm() << std::endl;
    std::cout << std::endl;

    // Test des opérateurs == et !=
    Complexe c4(1.1, -1.1);
    std::cout << "c1 == c4: " << (c1 == c4 ? "true" : "false") << " (Expected: true)" << std::endl;
    std::cout << "c1 != c2: " << (c1 != c2 ? "true" : "false") << " (Expected: true)" << std::endl;
    std::cout << std::endl;

    // Test des opérateurs +=, -=, *=, /=
    Complexe c5(2.0, 3.0);
    Complexe c6(1.0, -4.0);
    std::cout << "Avant opérations:" << std::endl;
    std::cout << "c5 = " << c5 << std::endl;
    std::cout << "c6 = " << c6 << std::endl;

    c5 += c6;
    std::cout << "Après c5 += c6: " << c5 << " (Expected: 3 + -1i)" << std::endl;

    c5 -= Complexe(1.0, 1.0);
    std::cout << "Après c5 -= (1 + 1i): " << c5 << " (Expected: 2 + -2i)" << std::endl;

    c5 *= Complexe(0.0, 1.0); // (2 - 2i) * i = 2i - 2i² = 2i + 2 = 2 + 2i
    std::cout << "Après c5 *= i: " << c5 << " (Expected: 2 + 2i)" << std::endl;

    c5 /= Complexe(1.0, 1.0); // (2 + 2i) / (1 + 1i) = (4 + 0i)/2 = 2 + 0i
    std::cout << "Après c5 /= (1 + 1i): " << c5 << " (Expected: 2 + 0i)" << std::endl;
    std::cout << std::endl;

    // Test des opérateurs +, -, *, /
    Complexe c7 = c2 + c3; // (3.4 + 2.3i) + (5 + 0i) = 8.4 + 2.3i
    Complexe c8 = c2 - c3; // (3.4 + 2.3i) - (5 + 0i) = -1.6 + 2.3i
    Complexe c9 = c2 * c3; // (3.4 + 2.3i) * 5 = 17 + 11.5i
    Complexe c10 = c2 / c3; // (3.4 + 2.3i) / 5 = 0.68 + 0.46i

    std::cout << "Opérateurs binaires:" << std::endl;
    std::cout << "c7 = c2 + c3 = " << c7 << " (Expected: 8.4 + 2.3i)" << std::endl;
    std::cout << "c8 = c2 - c3 = " << c8 << " (Expected: -1.6 + 2.3i)" << std::endl;
    std::cout << "c9 = c2 * c3 = " << c9 << " (Expected: 17 + 11.5i)" << std::endl;
    std::cout << "c10 = c2 / c3 = " << c10 << " (Expected: 0.68 + 0.46i)" << std::endl;
    std::cout << std::endl;

    // Test des opérateurs avec double
    double d = 2.5;
    Complexe c11 = d + c2; // 2.5 + (3.4 + 2.3i) = 5.9 + 2.3i
    Complexe c12 = c2 - d; // (3.4 + 2.3i) - 2.5 = 0.9 + 2.3i
    Complexe c13 = d * c2; // 2.5 * (3.4 + 2.3i) = 8.5 + 5.75i
    Complexe c14 = c2 / d; // (3.4 + 2.3i) / 2.5 = 1.36 + 0.92i

    std::cout << "Opérateurs avec double:" << std::endl;
    std::cout << "c11 = 2.5 + c2 = " << c11 << " (Expected: 5.9 + 2.3i)" << std::endl;
    std::cout << "c12 = c2 - 2.5 = " << c12 << " (Expected: 0.9 + 2.3i)" << std::endl;
    std::cout << "c13 = 2.5 * c2 = " << c13 << " (Expected: 8.5 + 5.75i)" << std::endl;
    std::cout << "c14 = c2 / 2.5 = " << c14 << " (Expected: 1.36 + 0.92i)" << std::endl;
    std::cout << std::endl;

    // Test des opérateurs avec double à gauche
    Complexe c15 = 3.0 + Complexe(1.0, 1.0); // 4.0 + 1.0i
    Complexe c16 = 4.0 - Complexe(1.0, 1.0); // 3.0 - 1.0i
    Complexe c17 = 2.0 * Complexe(1.5, -0.5); // 3.0 - 1.0i
    Complexe c18 = 5.0 / Complexe(1.0, -1.0); // (5*(1) + 5*(1)i) / 2 = 2.5 + 2.5i

    std::cout << "Opérateurs avec double à gauche:" << std::endl;
    std::cout << "c15 = 3.0 + (1 + 1i) = " << c15 << " (Expected: 4 + 1i)" << std::endl;
    std::cout << "c16 = 4.0 - (1 + 1i) = " << c16 << " (Expected: 3 - 1i)" << std::endl;
    std::cout << "c17 = 2.0 * (1.5 - 0.5i) = " << c17 << " (Expected: 3 - 1i)" << std::endl;
    std::cout << "c18 = 5.0 / (1 - 1i) = " << c18 << " (Expected: 2.5 + 2.5i)" << std::endl;
    std::cout << std::endl;

    // Test des erreurs (division par zéro)
    Complexe c19(1.0, 1.0);
    Complexe c20(0.0, 0.0);
    Complexe c21 = c19 / c20; // Devrait afficher une erreur et définir à 0 + 0i
    std::cout << "c21 = " << c21 << " (Expected: 0 + 0i)" << std::endl;

    return 0;
}
