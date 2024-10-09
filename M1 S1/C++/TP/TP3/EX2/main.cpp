#include <iostream>
#include <vector>
#include <cstdlib>  // rand(), srand()
#include <ctime>    // time()

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
    srand(static_cast<unsigned int>(time(0))); // 初始化随机数种子

    const int taille = 10;
    Vehicule* vehicules[taille];

    // 初始化包含各种类型车辆的数组
    for (int i = 0; i < taille; ++i) {
        int type = rand() % 4; // 随机选择车辆类型
        switch (type) {
            case 0:
                vehicules[i] = new Voiture();
                break;
            case 1:
                vehicules[i] = new Moto();
                break;
            case 2:
                vehicules[i] = new Camion();
                break;
            case 3:
                vehicules[i] = new Bus();
                break;
        }

        // 为每个车辆随机分配乘客数量
        int max_places = vehicules[i]->getNombreDePlaceRestante() + vehicules[i]->nombre_de_place_occupee;
        int passagers = rand() % (max_places + 1); // 随机乘客数，可能为0
        vehicules[i]->setNombreDePlaceOccupee(passagers);
    }

    // 示例：输出每个车辆的信息
    for (int i = 0; i < taille; ++i) {
        std::cout << "Vehicule " << i + 1 << ": ";
        if (dynamic_cast<Voiture*>(vehicules[i])) {
            std::cout << "Voiture, ";
        } else if (dynamic_cast<Moto*>(vehicules[i])) {
            std::cout << "Moto, ";
        } else if (dynamic_cast<Camion*>(vehicules[i])) {
            std::cout << "Camion, ";
        } else if (dynamic_cast<Bus*>(vehicules[i])) {
            std::cout << "Bus, ";
        }
        std::cout << "乘客数: " << vehicules[i]->nombre_de_place_occupee
                  << ", 总重量: " << vehicules[i]->getPoidsTotal() << " 公斤" << std::endl;
    }

    // 释放内存
    for (int i = 0; i < taille; ++i) {
        delete vehicules[i];
    }

    return 0;
}