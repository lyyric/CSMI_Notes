#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>

class Complexe{
private:
    double re;
    double im;

public:
    Complexe() : re(0.0), im(0.0) {}

    Complexe(double real, double imag) : re(real), im(imag) {}

    double module() const {
        return sqrt(re * re + im * im);
    }

    friend std::ostream& operator<<(std::ostream& os, const Complexe& c) {
        os << "(" << c.re << "," << c.im << ")";
        return os;
    }

    friend std::istream& operator>>(std::istream& is, Complexe& c) {
        char ch;
        is >> ch >> c.re >> ch >> c.im >> ch;
        return is;
    }
}

std::vector<Complexe> lireNombresComplexes(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<Complexe> nombresComplexes;
    if (file.is_open()) {
        int taille;
        file >> taille;
        
    }
    return nombresComplexes;
}

int main (){

    return 0;
}