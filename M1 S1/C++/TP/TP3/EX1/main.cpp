#include <iostream>

class Vehicule {
protected:
    int total_places;
    int occupees;
    double poids;

public:
    Vehicule(int places, double p) : total_places(places), occupees(0), poids(p) {}

    void setOccupees(int nb) {
        if (nb > total_places) {
            std::cout << "Erreur : Places insuffisantes." << std::endl;
        } else {
            occupees = nb;
        }
    }

    int placesRestantes() const {
        return total_places - occupees;
    }

    double poidsTotal() const {
        double poids_passagers = occupees * 75.0;
        return poids + poids_passagers;
    }
};

class Voiture : public Vehicule {
public:
    Voiture() : Vehicule(5, 1000.0) {}
};

class Moto : public Vehicule {
public:
    Moto() : Vehicule(2, 500.0) {}
};

class Camion : public Vehicule {
public:
    Camion() : Vehicule(3, 4000.0) {}
};

class Bus : public Vehicule {
public:
    Bus() : Vehicule(40, 5000.0) {}
};

int main() {
    Bus monBus;
    monBus.setOccupees(35);
    std::cout << "Places restantes dans le bus : " << monBus.placesRestantes() << std::endl;
    std::cout << "Poids total du bus : " << monBus.poidsTotal() << " kg" << std::endl;

    monBus.setOccupees(45);

    return 0;
}
