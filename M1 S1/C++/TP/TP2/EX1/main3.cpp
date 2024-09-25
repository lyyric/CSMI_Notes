#include <iostream>

class Date {
private:
    int jour;
    int mois;
    int annee;

    static const int joursParMois[12];

public:
    Date(int jour1 = 1, int mois1 = 1, int annee1 = 0) {
        if (estValide(jour1, mois1, annee1)) {
            jour = jour1;
            mois = mois1;
            annee = annee1;
        } else {
            jour = 1;
            mois = 1;
            annee = 0;
        }
    }

    int getJour() const { return jour; }
    int getMois() const { return mois; }
    int getAnnee() const { return annee; }

    static bool estValide(int j, int m, int a) {
        if (m < 1 || m > 12 || j < 1) return false;

        if (j > joursParMois[m - 1]) return false;

        return true;
    }

    void setJour(int jour1) {
        if (estValide(jour1, mois, annee)) {
            jour = jour1;
        } else {
            std::cerr << "Jour incorrect: " << jour1 << std::endl;
        }
    }

    void setMois(int mois1) {
        if (estValide(jour, mois1, annee)) {
            mois = mois1;
        } else {
            std::cerr << "Mois incorrect: " << mois1 << std::endl;
        }
    }

    void setAnnee(int annee1) {
        if (estValide(jour, mois, annee1)) {
            annee = annee1;
        } else {
            std::cerr << "Annee incorrect: " << annee1 << std::endl;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Date& d) {
        os << d.jour << "/" << d.mois << "/" << d.annee;
        return os;
    }

    static Date nJourEnDate(int njour) {
        int jour1 = 1;
        int mois1 = 1;
        int annee1 = 0;

        njour--;

        while (njour >= 365) {
            njour -= 365;
            annee1++;
        }

        while (njour >= joursParMois[mois1 - 1]) {
            njour -= joursParMois[mois1 - 1];
            mois1++;
            if (mois1 > 12) {
                mois1 = 1;
                annee1++;
            }
        }

        jour1 += njour;
        return Date(jour1, mois1, annee1);
    }
};

const int Date::joursParMois[12] = {31,28,31,30,31,30,31,31,30,31,30,31};

int main() {
    std::cout << "nJourEnDate(31) = " << Date::nJourEnDate(31) << std::endl;
    std::cout << "nJourEnDate(32) = " << Date::nJourEnDate(32) << std::endl;
    std::cout << "nJourEnDate(31+28) = " << Date::nJourEnDate(31 + 28) << std::endl;
    std::cout << "nJourEnDate(31+29) = " << Date::nJourEnDate(31 + 29) << std::endl;
    std::cout << "nJourEnDate(365) = " << Date::nJourEnDate(365) << std::endl;
    std::cout << "nJourEnDate(366) = " << Date::nJourEnDate(366) << std::endl;
    std::cout << "nJourEnDate(365*2017+293) = " << Date::nJourEnDate(365 * 2017 + 293) << std::endl;

    return 0;
}