## 练习1

考虑类汽车 (Voiture) 、摩托车 (Moto) 、卡车 (Camion) 和公交车 (Bus) ，它们都继承自类车辆 (Véhicule) 。这些类有以下共同的属性：

- 总座位数
- 已占用的座位数
- 车辆重量

假设每个乘客的重量为75公斤，并且对于没有乘客的车辆，以下是数据：

- 汽车：5个座位，1000公斤
- 摩托车：2个座位，500公斤
- 卡车：3个座位，4000公斤
- 公交车：40个座位，5000公斤

使用继承，实现这5个类。每个类还应提供：

- 一个方法，用于设置已占用的座位数。如果座位数量不足，则显示错误消息。
- 一个方法，返回剩余的座位数。
- 一个方法，返回总重量（车辆 + 乘客）。

以下是使用C++编写的代码：

```cpp
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
            std::cerr << "错误：已占用的座位数超过了总座位数。\n";
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
    std::cout << "汽车剩余座位数: " << maVoiture.getNombreDePlaceRestante() << std::endl;
    std::cout << "汽车总重量: " << maVoiture.getPoidsTotal() << " 公斤" << std::endl;

    Moto maMoto;
    maMoto.setNombreDePlaceOccupee(2);
    std::cout << "摩托车剩余座位数: " << maMoto.getNombreDePlaceRestante() << std::endl;
    std::cout << "摩托车总重量: " << maMoto.getPoidsTotal() << " 公斤" << std::endl;

    Bus monBus;
    monBus.setNombreDePlaceOccupee(45); // 这将触发错误消息
    std::cout << "公交车剩余座位数: " << monBus.getNombreDePlaceRestante() << std::endl;
    std::cout << "公交车总重量: " << monBus.getPoidsTotal() << " 公斤" << std::endl;

    return 0;
}
```

**代码说明：**

- **Vehicule**：基类，包含公共属性和方法。
  - **setNombreDePlaceOccupee**：设置已占用的座位数，如果超过总座位数，输出错误信息。
  - **getNombreDePlaceRestante**：返回剩余的座位数。
  - **getPoidsTotal**：计算并返回总重量（车辆重量 + 乘客重量）。

- **Voiture、Moto、Camion、Bus**：从Vehicule继承，每个类在构造函数中初始化各自的总座位数和车辆重量。

- **main函数**：创建各个车辆对象，演示方法的使用，并测试错误处理。

**注意事项：**

- 当试图设置已占用的座位数超过总座位数时，程序会输出错误消息而不会更改已占用的座位数。
- 乘客的重量被假设为75公斤，计算总重量时乘以已占用的座位数。

## 练习2

在您的程序中初始化一个包含大约十个Voiture（汽车）、Moto（摩托车）、Camion（卡车）和Bus（公交车）对象的数组。对于每个车辆，乘客数量将随机选择。提示：需要使用一个指向基类的指针数组。

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>  // rand(), srand()
#include <ctime>    // time()

// 假设前面的Vehicule、Voiture、Moto、Camion、Bus类已经定义
// 这里直接继续使用

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
```

**代码说明：**

- **随机数生成**：使用`rand()`和`srand()`函数生成随机数，`srand()`用`time(0)`作为种子。
- **车辆数组**：创建了一个`Vehicule*`类型的数组，大小为10。
- **车辆初始化**：在循环中，随机选择车辆类型（0到3），并使用`new`创建相应的对象。
- **随机乘客数量**：为每个车辆分配随机的乘客数量，范围从0到车辆的最大座位数。
- **信息输出**：使用`dynamic_cast`确定车辆的实际类型，并输出相关信息。
- **内存管理**：使用`delete`释放动态分配的内存，防止内存泄漏。

## 练习3

编写两个函数，它们各自以车辆数组及其大小为参数，并分别返回：

- 总重量。
- 未被占用的座位数。

```cpp
// 计算总重量的函数
double calculerPoidsTotal(Vehicule* vehicules[], int taille) {
    double poids_total = 0.0;
    for (int i = 0; i < taille; ++i) {
        poids_total += vehicules[i]->getPoidsTotal();
    }
    return poids_total;
}

// 计算未被占用的座位数的函数
int calculerPlacesNonOccupees(Vehicule* vehicules[], int taille) {
    int places_non_occupees = 0;
    for (int i = 0; i < taille; ++i) {
        places_non_occupees += vehicules[i]->getNombreDePlaceRestante();
    }
    return places_non_occupees;
}

// 在main函数中调用示例
int main() {
    // ...（前面的车辆初始化代码）

    double poidsTotal = calculerPoidsTotal(vehicules, taille);
    int placesLibres = calculerPlacesNonOccupees(vehicules, taille);

    std::cout << "车辆总重量: " << poidsTotal << " 公斤" << std::endl;
    std::cout << "未被占用的座位总数: " << placesLibres << std::endl;

    // ...（释放内存等）
    return 0;
}
```

**代码说明：**

- **calculerPoidsTotal**：遍历车辆数组，累加每个车辆的总重量。
- **calculerPlacesNonOccupees**：遍历车辆数组，累加每个车辆的剩余座位数。
- 在`main`函数中调用这两个函数，并输出结果。

## 练习4

编写一个函数，它以车辆数组为参数，计算Voiture（汽车）的数量。提示：使用`dynamic_cast`进行类型转换。

```cpp
// 计算Voiture数量的函数
int compterVoitures(Vehicule* vehicules[], int taille) {
    int nombre_de_voitures = 0;
    for (int i = 0; i < taille; ++i) {
        if (dynamic_cast<Voiture*>(vehicules[i]) != nullptr) {
            ++nombre_de_voitures;
        }
    }
    return nombre_de_voitures;
}

// 在main函数中调用示例
int main() {
    // ...（前面的车辆初始化代码）

    int nombreVoitures = compterVoitures(vehicules, taille);
    std::cout << "车辆数组中Voiture的数量: " << nombreVoitures << std::endl;

    // ...（释放内存等）
    return 0;
}
```

**代码说明：**

- **compterVoitures**：遍历车辆数组，使用`dynamic_cast`将`Vehicule*`转换为`Voiture*`。如果转换成功（即指针不为`nullptr`），则计数器加一。
- 在`main`函数中调用`compterVoitures`函数，并输出结果。

**总体说明：**

- **dynamic_cast**：一种安全的类型转换方式，适用于含有虚函数的多态基类体系。它可以在运行时检查类型安全，防止不正确的类型转换。
- **内存管理**：由于使用了`new`运算符分配内存，必须确保在程序结束前调用`delete`来释放内存，防止内存泄漏。
- **面向对象编程**：通过使用继承和多态，我们可以方便地管理不同类型的车辆，同时利用基类指针数组来统一处理。

**注意事项：**

- 确保在编译时包含必要的头文件，例如`<cstdlib>`和`<ctime>`。
- 在使用`dynamic_cast`时，基类应至少有一个虚函数（在我们的`Vehicule`类中可以确保析构函数为虚函数）。
- 为了提高代码的健壮性，可以在各个函数中添加参数检查和错误处理。
