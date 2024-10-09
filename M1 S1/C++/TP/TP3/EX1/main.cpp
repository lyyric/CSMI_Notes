#include <iostream>

class Vehicule {
protected:
    int nombre_de_place_totale;
    int nombre_de_place_occupee;
    double poids_du_vehicule; // 单位：公斤

public:
    Vehicule(int total_places, double poids)
        : nombre_de_place_totale(total_places), poids_du_vehicule(poids), nombre_de_place_occupee(0) {}

    virtual ~Vehicule() {}

    // 设置已占用的座位数
    void setNombreDePlaceOccupee(int n) {
        if (n > nombre_de_place_totale) {
            std::cout << "Erreur: les sièges occupés dépassent le nombre total de sièges." << std::endl;
        } else {
            nombre_de_place_occupee = n;
        }
    }

    // 获取剩余的座位数
    int getNombreDePlaceRestante() const {
        return nombre_de_place_totale - nombre_de_place_occupee;
    }

    // 获取总重量（车辆 + 乘客）
    double getPoidsTotal() const {
        double poids_passagers = nombre_de_place_occupee * 75.0;
        return poids_du_vehicule + poids_passagers;
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
    Voiture maVoiture;
    maVoiture.setNombreDePlaceOccupee(3);
    std::cout << "Places restantes dans le Voiture: " << maVoiture.getNombreDePlaceRestante() << std::endl;
    std::cout << "Poids total du Voiture: " << maVoiture.getPoidsTotal() << " kg" << std::endl;

    Moto maMoto;
    maMoto.setNombreDePlaceOccupee(2);
    std::cout << "Places restantes dans le Moto: " << maMoto.getNombreDePlaceRestante() << std::endl;
    std::cout << "Poids total du Moto: " << maMoto.getPoidsTotal() << " kg" << std::endl;

    Bus monBus;
    monBus.setNombreDePlaceOccupee(45); // 这将触发错误消息
    std::cout << "Places restantes dans le Bus: " << monBus.getNombreDePlaceRestante() << std::endl;
    std::cout << "Poids total du Bus: " << monBus.getPoidsTotal() << " kg" << std::endl;

    return 0;
}