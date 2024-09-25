#include <iostream>

class Date {
private:
    int jour;
    int mois;
    int annee;

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

    int jour() const { return jour; }
    int mois() const { return mois; }
    int annee() const { return annee; }



    static bool estValide(int j, int m, int a) {
        if (m < 1 || m > 12 || j < 1) return false;

        int joursParMois[12] = {31,28,31,30,31,30,31,31,30,31,30,31};

        if (j > joursParMois[m - 1]) return false;

        return true;
    }


    void setJour(int jour1) {
        if (estValide(jour1, mois, annee)) {
            jour = jour1;
        } else {
            std::cerr << "Jour incorrect: " << jour << std::endl;
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
        int joursParMois[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
        
        while (njour >= 365) {
            njour = njour - 365;
            annee1 = annee1 + 1;
        }

        while (njour >= joursParMois[mois1 - 1]) {
            njour = njour - joursParMois[mois1 - 1];
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