#include <iostream>
#include <cmath>

class Complexe {
private:
    double re;
    double im;

public:
    Complexe() : re(0.0), im(0.0) {}

    Complexe(double real, double imag) : re(real), im(imag) {}

    Complexe(double real) : re(real), im(0.0) {}

    // Accesseurs
    double getRe() const { return re; }
    double getIm() const { return im; }

    // Mutateurs
    void setRe(double real) { re = real; }
    void setIm(double imag) { im = imag; }

    friend std::ostream& operator<<(std::ostream& os, const Complexe& c) {
        os << c.re;
        if (c.im >= 0)
            os << " + " << c.im << "i";
        else
            os << " - " << -c.im << "i";
        return os;
    }

    bool operator==(const Complexe& other) const {
        return (std::abs(re - other.re) < 1e-10) && (std::abs(im - other.im) < 1e-10);
    }

    bool operator!=(const Complexe& other) const {
        return !(*this == other);
    }

    Complexe& operator+=(const Complexe& other) {
        re += other.re;
        im += other.im;
        return *this;
    }

    friend Complexe operator+(Complexe lhs, const Complexe& rhs) {
        lhs += rhs;
        return lhs;
    }

    friend Complexe operator+(double lhs, const Complexe& rhs) {
        return Complexe(lhs + rhs.re, rhs.im);
    }

};

int main() {
    Complexe c1;
    Complexe c2(3.4, 2.3);
    Complexe c3(5.0);

    std::cout << "Constructeurs:" << std::endl;
    std::cout << "c1 = " << c1 << std::endl;
    std::cout << "c2 = " << c2 << std::endl;
    std::cout << "c3 = " << c3 << std::endl;
    std::cout << std::endl;

    Complexe c4(1.1, -1.1);
    std::cout << "c1 == c4: " << (c1 == c4 ? "true" : "false") << std::endl;
    std::cout << "c1 != c2: " << (c1 != c2 ? "true" : "false") << std::endl;
    std::cout << std::endl;

    Complexe c5(2.0, 3.0);
    Complexe c6(1.0, -4.0);
    std::cout << "c5 = " << c5 << std::endl;
    std::cout << "c6 = " << c6 << std::endl;

    c5 += c6;
    std::cout << "c5 += c6: " << c5 << std::endl;
    std::cout << std::endl;

    Complexe c7(2.0, 3.0);
    Complexe c8(1.0, -4.0);
    std::cout << "c7 = " << c7 << std::endl;
    std::cout << "c8 = " << c8 << std::endl;
    Complexe c9;
    c9 = c7 + c8;
    std::cout << "c7 + c8 = " << c9 << std::endl;
    std::cout << std::endl;

    Complexe c10(2.0, 3.0);
    double d = 2.5;
    std::cout << "c10 = " << c7 << std::endl;
    std::cout << "d = " << d << std::endl;
    Complexe c11;
    c11 = c10 + d;
    std::cout << "c10 + d = " << c11 << std::endl;
    std::cout << std::endl;    

    return 0;
}