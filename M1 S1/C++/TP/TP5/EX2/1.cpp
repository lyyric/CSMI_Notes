//
//**练习 2**

//(a) 编写一个模板类 `Point`，该类由两个模板参数定义：用于坐标的类型和维度。类中包含一个（静态分配的）数组来表示坐标。

//(b) 我们希望创建一个模板类 `Cercle`，用于操作定义为“中心点”（一个二维的 `Point` 对象）和“半径”的圆。模板参数包括用于坐标的类型和用于半径的类型。我们需要提供一个构造函数以及一个 `affiche()` 方法，该方法仅显示圆心坐标和半径的值。

//要求实现这个类：

//- 通过继承实现：一个 `Cercle` 是一个带有半径的 `Point`
//- 通过组合实现：一个 `Cercle` 拥有一个 `Point` 和一个半径

#include <iostream>
#include <array>

template <typename T, size_t Dimension>
class Point {
private:
    std::array<T, Dimension> coordinates; // 使用静态分配的数组存储坐标

public:
    // 默认构造函数
    Point() {
        coordinates.fill(T()); // 初始化所有坐标为类型T的默认值
    }

    // 设置坐标
    void setCoordinate(size_t index, T value) {
        if (index < Dimension) {
            coordinates[index] = value;
        }
    }

    // 获取坐标
    T getCoordinate(size_t index) const {
        if (index < Dimension) {
            return coordinates[index];
        }
        return T();
    }

    // 打印坐标
    void print() const {
        std::cout << "(";
        for (size_t i = 0; i < Dimension; ++i) {
            std::cout << coordinates[i];
            if (i < Dimension - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")";
    }
};
