#include <iostream>

class Date {
private:
    int jour_;
    int mois_;
    int annee_;

public:
    // Constructeur initialisant les attributs
    Date(int jour = 1, int mois = 1, int annee = 0) {
        if (estValide(jour, mois, annee)) {
            jour_ = jour;
            mois_ = mois;
            annee_ = annee;
        } else {
            std::cerr << "Date incorrecte dans le constructeur: "
                      << jour << "/" << mois << "/" << annee << std::endl;
            jour_ = 1;
            mois_ = 1;
            annee_ = 0;
        }
    }

    // Accesseurs
    int jour() const { return jour_; }
    int mois() const { return mois_; }
    int annee() const { return annee_; }

    // Mutateurs
    void setJour(int jour) {
        if (estValide(jour, mois_, annee_)) {
            jour_ = jour;
        } else {
            std::cerr << "Jour incorrect: " << jour << std::endl;
        }
    }

    void setMois(int mois) {
        if (estValide(jour_, mois, annee_)) {
            mois_ = mois;
        } else {
            std::cerr << "Mois incorrect: " << mois << std::endl;
        }
    }

    void setAnnee(int annee) {
        if (estValide(jour_, mois_, annee)) {
            annee_ = annee;
        } else {
            std::cerr << "Année incorrecte: " << annee << std::endl;
        }
    }

    // Surcharge de l'opérateur <<
    friend std::ostream& operator<<(std::ostream& os, const Date& d) {
        os << d.jour_ << "/" << d.mois_ << "/" << d.annee_;
        return os;
    }

    // Méthode statique vérifiant la validité d'une date
    static bool estValide(int j, int m, int a) {
        if (m < 1 || m > 12 || j < 1) return false;

        int joursParMois[12] = {31,28,31,30,31,30,31,31,30,31,30,31};

        if (j > joursParMois[m - 1]) return false;

        return true;
    }

    // Méthode statique convertissant un nombre de jours en une date
    static Date nJourEnDate(int njour) {
        int jour = 1;
        int mois = 1;
        int annee = 0;

        int joursParMois[12] = {31,28,31,30,31,30,31,31,30,31,30,31};

        njour--;  // Le jour 1 correspond à 1/1/0

        while (njour >= 365) {
            njour -= 365;
            annee++;
        }

        while (njour >= joursParMois[mois - 1]) {
            njour -= joursParMois[mois - 1];
            mois++;
            if (mois > 12) {
                mois = 1;
                annee++;
            }
        }

        jour += njour;

        return Date(jour, mois, annee);
    }
};

int main() {
    // Vérification de la méthode nJourEnDate avec les exemples donnés
    std::cout << "nJourEnDate(31) = " << Date::nJourEnDate(31) << std::endl;
    std::cout << "nJourEnDate(32) = " << Date::nJourEnDate(32) << std::endl;
    std::cout << "nJourEnDate(31+28) = " << Date::nJourEnDate(31 + 28) << std::endl;
    std::cout << "nJourEnDate(31+29) = " << Date::nJourEnDate(31 + 29) << std::endl;
    std::cout << "nJourEnDate(365) = " << Date::nJourEnDate(365) << std::endl;
    std::cout << "nJourEnDate(366) = " << Date::nJourEnDate(366) << std::endl;
    std::cout << "nJourEnDate(365*2017+293) = " << Date::nJourEnDate(365 * 2017 + 293) << std::endl;

    // Test des mutateurs avec des valeurs incorrectes
    Date d(31, 12, 2020);
    std::cout << "Date initiale: " << d << std::endl;
    d.setJour(32);  // Devrait afficher une erreur
    d.setMois(13);  // Devrait afficher une erreur
    d.setAnnee(-1); // Devrait afficher une erreur
    d.setJour(30);
    std::cout << "Après setJour(30): " << d << std::endl;

    return 0;
}
