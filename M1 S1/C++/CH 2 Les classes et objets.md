# C++ 面向对象编程：类与对象

## 目录

1. 面向对象编程的基本原则
2. C++ 中的类与对象
3. 类的封装
4. 构造函数与析构函数
5. 成员方法
6. 运算符重载
7. 静态成员与方法
8. 友元函数
9. `this` 指针
10. 类图
11. 进一步学习

---

## 1. 面向对象编程的基本原则

### 1.1 编程的开发流程

开发一个计算机程序通常涉及多个阶段，每个阶段都有其特定的目标和任务：

1. **分析阶段**：明确程序的目标，正式化问题，并描述其规格（包括输入和输出数据）。
2. **设计阶段**：识别解决方案中的关键概念，描述所采用的方法（算法和编程模型）。
3. **编程阶段**：在特定编程语言中实现设计方案。
4. **交付阶段**：提供经过验证和验证的程序。

### 1.2 编程范式

主要的编程范式包括：

- **命令式编程**（Imperative Programming）
- **过程式编程**（Procedural Programming）
- **面向对象编程**（Object-Oriented Programming，OOP）

每种范式都有其独特的方式来组织和编写代码，面向对象编程通过类和对象来模拟现实世界的实体和行为，提供更高的代码复用性和可维护性。

---

## 2. C++ 中的类与对象

### 2.1 类的定义

在 C++ 中，类是一个用户定义的数据类型，它描述了一组对象的共有特性（属性）和行为（方法）。类是面向对象编程的核心，通过类可以创建多个具有相同属性和行为的对象。

#### 类的基本结构

```cpp
class Point {
public:
    // 构造函数
    Point(double x, double y);

    // 访问器（获取属性）
    double getX() const;
    double getY() const;

    // 修改器（设置属性）
    void setX(double x);
    void setY(double y);

    // 其他方法
    void print() const;

private:
    double M_x, M_y; // 属性
};
```

### 2.2 对象的创建与使用

对象是类的实例，每个对象拥有类中定义的属性和方法。创建对象的过程称为**实例化**。

#### 示例代码

```cpp
#include <iostream>

// 类的定义
class Point {
public:
    Point(double x, double y) : M_x(x), M_y(y) {}
    
    double getX() const { return M_x; }
    double getY() const { return M_y; }
    
    void setX(double x) { M_x = x; }
    void setY(double y) { M_y = y; }
    
    void print() const {
        std::cout << M_x << ", " << M_y << "\n";
    }
    
private:
    double M_x, M_y;
};

int main() {
    Point p1(3.2, 6.1);        // 使用构造函数创建对象
    Point p2 = p1;              // 使用拷贝构造函数创建对象
    p1.setX(4.5);               // 修改对象的属性
    p1.print();                 // 调用对象的方法
    std::cout << "p2.x = " << p2.getX() << "\n";
    return 0;
}
```

#### 输出结果

```
4.5, 6.1
p2.x = 3.2
```

---

## 3. 类的封装

### 3.1 封装的概念

**封装**是面向对象编程的一个基本原则，指的是将数据（属性）和操作数据的方法（行为）绑定在一起，并限制对数据的直接访问，只允许通过公开的方法来访问和修改数据。

### 3.2 访问控制

在 C++ 中，类成员（属性和方法）的访问权限由以下三个关键字控制：

- **public**：公有成员，可以被类的外部访问。
- **private**：私有成员，只能在类的内部访问。
- **protected**：受保护成员，可以在类的内部及其派生类中访问。

#### 示例代码

```cpp
class Point {
public:
    // 公有方法，可以在类外部访问
    Point(double x, double y) : M_x(x), M_y(y) {}
    
    double getX() const { return M_x; }
    double getY() const { return M_y; }
    
    void setX(double x) { M_x = x; }
    void setY(double y) { M_y = y; }
    
    void print() const {
        std::cout << "(" << M_x << ", " << M_y << ")\n";
    }

private:
    double M_x, M_y; // 私有属性，只能通过公有方法访问
};
```

#### 使用访问控制的示例

```cpp
int main() {
    Point p(3.0, 4.0);
    p.print();                 // 正常调用公有方法
    std::cout << p.getX();     // 正常访问公有方法获取属性
    p.setY(5.0);               // 正常通过公有方法修改属性
    
    // 以下访问将导致编译错误，因为属性是私有的
    // std::cout << p.M_x;     
    // p.M_y = 10.0;
    
    return 0;
}
```

### 3.3 封装的优势

- **数据隐藏**：隐藏类的内部实现细节，保护数据不被意外或恶意修改。
- **提高代码安全性**：通过控制访问权限，减少潜在的错误和漏洞。
- **增强代码可维护性**：更改类的内部实现不会影响类的外部接口，降低了维护成本。
- **促进代码复用**：通过清晰的接口，类可以更容易地在不同的项目中复用。

---

## 4. 构造函数与析构函数

### 4.1 构造函数

**构造函数**是类的一种特殊方法，用于在创建对象时初始化对象的属性。构造函数的名称与类名相同，没有返回类型。

#### 构造函数的特点

- **名称与类名相同**。
- **没有返回类型**。
- **可以有多个构造函数（构造函数重载）**。
- **如果没有定义构造函数，编译器将自动生成一个默认构造函数**。

#### 示例代码

```cpp
class Point {
public:
    // 默认构造函数
    Point() : M_x(0), M_y(0) {
        std::cout << "默认构造函数被调用\n";
    }
    
    // 带参数的构造函数
    Point(double x, double y) : M_x(x), M_y(y) {
        std::cout << "带参数的构造函数被调用\n";
    }

private:
    double M_x, M_y;
};

int main() {
    Point p1;            // 调用默认构造函数
    Point p2(3.2, 6.1);  // 调用带参数的构造函数
    return 0;
}
```

#### 输出结果

```
默认构造函数被调用
带参数的构造函数被调用
```

### 4.2 拷贝构造函数

**拷贝构造函数**用于通过另一个同类型的对象来初始化新对象。默认情况下，编译器会生成一个拷贝构造函数，该函数执行逐成员拷贝。

#### 示例代码

```cpp
class Point {
public:
    Point(double x, double y) : M_x(x), M_y(y) {}
    
    // 自定义拷贝构造函数
    Point(const Point& pt) : M_x(pt.M_x), M_y(pt.M_y) {
        std::cout << "拷贝构造函数被调用\n";
    }

private:
    double M_x, M_y;
};

int main() {
    Point p1(3.0, 4.0);
    Point p2 = p1; // 调用拷贝构造函数
    return 0;
}
```

#### 输出结果

```
拷贝构造函数被调用
```

### 4.3 移动构造函数（C++11）

**移动构造函数**允许资源（如动态分配的内存）从一个临时对象“移动”到另一个对象，而不是进行拷贝。这可以显著提高性能，特别是在处理大对象或资源密集型对象时。

#### 示例代码

```cpp
#include <iostream>
#include <utility>

class Point {
public:
    Point(double x = 0, double y = 0) : M_x(x), M_y(y) {}
    
    // 拷贝构造函数
    Point(const Point& pt) : M_x(pt.M_x), M_y(pt.M_y) {
        std::cout << "拷贝构造函数被调用\n";
    }
    
    // 移动构造函数
    Point(Point&& pt) noexcept : M_x(std::move(pt.M_x)), M_y(std::move(pt.M_y)) {
        std::cout << "移动构造函数被调用\n";
    }

private:
    double M_x, M_y;
};

int main() {
    Point p1(3.0, 4.0);
    Point p2 = std::move(p1); // 调用移动构造函数
    return 0;
}
```

#### 输出结果

```
移动构造函数被调用
```

### 4.4 析构函数

**析构函数**是类的一种特殊方法，用于在对象生命周期结束时执行清理工作，如释放动态分配的内存。析构函数的名称与类名相同，前面加上波浪号（`~`），没有返回类型和参数。

#### 析构函数的特点

- **名称与类名相同，前面加波浪号**。
- **没有返回类型和参数**。
- **一个类只能有一个析构函数**。
- **当对象被销毁时，析构函数会自动调用**。

#### 示例代码

```cpp
#include <iostream>

class Vecteur {
public:
    Vecteur(int s) : M_size(s), M_tab(new double[s]) {
        std::cout << "构造函数被调用，大小为 " << M_size << "\n";
    }
    
    // 析构函数
    ~Vecteur() {
        std::cout << "析构函数被调用，释放大小为 " << M_size << " 的内存\n";
        delete[] M_tab;
    }

private:
    int M_size;
    double* M_tab;
};

int main() {
    Vecteur v1(15);           // 调用构造函数
    Vecteur* v2 = new Vecteur(23); // 调用构造函数
    delete v2;                // 调用析构函数
    return 0;
}
```

#### 输出结果

```
构造函数被调用，大小为 15
构造函数被调用，大小为 23
析构函数被调用，释放大小为 23 的内存
析构函数被调用，释放大小为 15 的内存
```

### 4.5 构造函数的委托（C++11）

**构造函数的委托**允许一个构造函数调用同一类中的另一个构造函数，以减少重复代码并提高代码的可维护性。

#### 示例代码

```cpp
#include <iostream>

class Point {
public:
    // 委托默认构造函数调用带参数的构造函数
    Point() : Point(0, 0) {
        std::cout << "默认构造函数被调用\n";
    }
    
    // 带参数的构造函数
    Point(double x, double y) : M_x(x), M_y(y) {
        std::cout << "带参数的构造函数被调用\n";
    }

private:
    double M_x, M_y;
};

int main() {
    Point p1;        // 调用默认构造函数，实际调用带参数的构造函数
    Point p2(3.0, 4.0); // 调用带参数的构造函数
    return 0;
}
```

#### 输出结果

```
带参数的构造函数被调用
默认构造函数被调用
带参数的构造函数被调用
```

### 4.6 拷贝赋值操作符

**拷贝赋值操作符**用于将一个对象的值赋给另一个已有的对象。默认情况下，编译器会生成一个逐成员拷贝的赋值操作符。

#### 自定义拷贝赋值操作符

当类中包含指针或需要特殊的拷贝逻辑时，建议自定义拷贝赋值操作符以避免浅拷贝带来的问题。

#### 示例代码

```cpp
#include <iostream>

class Point {
public:
    Point(double x = 0, double y = 0) : M_x(x), M_y(y) {}
    
    // 拷贝赋值操作符
    Point& operator=(const Point& pt) {
        if (this != &pt) { // 避免自赋值
            M_x = pt.M_x;
            M_y = pt.M_y;
            std::cout << "拷贝赋值操作符被调用\n";
        }
        return *this;
    }

private:
    double M_x, M_y;
};

int main() {
    Point p1(3.0, 4.0);
    Point p2;
    p2 = p1;          // 调用拷贝赋值操作符
    p2 = p2;          // 自赋值，避免不必要的赋值
    Point p3 = p1;    // 调用拷贝构造函数
    return 0;
}
```

#### 输出结果

```
拷贝赋值操作符被调用
```

---

## 5. 成员方法

### 5.1 成员方法的定义与调用

**成员方法**是定义在类内部的函数，用于操作对象的属性或执行特定的行为。成员方法可以访问类的私有成员，并通过对象调用。

#### 示例代码

```cpp
#include <iostream>

class Point {
public:
    Point(double x, double y) : M_x(x), M_y(y) {}
    
    // 获取属性
    double getX() const { return M_x; }
    double getY() const { return M_y; }
    
    // 设置属性
    void setX(double x) { M_x = x; }
    void setY(double y) { M_y = y; }
    
    // 打印坐标
    void print() const {
        std::cout << "(" << M_x << ", " << M_y << ")\n";
    }

private:
    double M_x, M_y;
};

int main() {
    Point p(3.0, 4.0);
    p.print();            // 调用成员方法
    p.setX(5.0);          // 修改属性
    p.print();
    std::cout << "X坐标：" << p.getX() << "\n";
    return 0;
}
```

#### 输出结果

```
(3, 4)
(5, 4)
X坐标：5
```

### 5.2 常量成员方法

**常量成员方法**是指在方法声明后添加 `const` 关键字的方法，这意味着该方法不会修改对象的任何属性，并且可以在常量对象上调用。

#### 示例代码

```cpp
class Point {
public:
    Point(double x, double y) : M_x(x), M_y(y) {}
    
    double getX() const { return M_x; }
    double getY() const { return M_y; }
    
    void setX(double x) { M_x = x; }
    void setY(double y) { M_y = y; }
    
    void print() const {
        std::cout << "(" << M_x << ", " << M_y << ")\n";
    }

private:
    double M_x, M_y;
};

int main() {
    const Point p(3.0, 4.0);
    p.print();                // 可以调用常量成员方法
    std::cout << p.getX();    // 可以调用常量成员方法
    
    // p.setX(5.0);           // 错误：不能在常量对象上调用非常量方法
    return 0;
}
```

#### 输出结果

```
(3, 4)
3
```

### 5.3 分离类的声明与定义

在大型项目中，通常将类的声明和定义分离到不同的文件中，以提高代码的可维护性和可读性。

#### 示例：Point 类

**point.hpp**（头文件）

```cpp
#ifndef POINT_HPP
#define POINT_HPP

#include <iostream>

class Point {
public:
    // 构造函数
    Point(double x, double y);
    
    // 访问器
    double getX() const;
    double getY() const;
    
    // 修改器
    void setX(double x);
    void setY(double y);
    
    // 打印坐标
    void print() const;

private:
    double M_x, M_y;
};

#endif // POINT_HPP
```

**point.cpp**（实现文件）

```cpp
#include "point.hpp"

// 构造函数的定义
Point::Point(double x, double y) : M_x(x), M_y(y) {}

// 访问器的定义
double Point::getX() const { return M_x; }
double Point::getY() const { return M_y; }

// 修改器的定义
void Point::setX(double x) { M_x = x; }
void Point::setY(double y) { M_y = y; }

// 打印坐标的方法定义
void Point::print() const {
    std::cout << "(" << M_x << ", " << M_y << ")\n";
}
```

**main.cpp**（主程序文件）

```cpp
#include "point.hpp"

int main() {
    Point p1(3.0, 4.0);
    p1.print();
    p1.setX(5.0);
    p1.print();
    return 0;
}
```

#### 编译与链接

```bash
g++ -c point.cpp -o point.o
g++ -c main.cpp -o main.o
g++ point.o main.o -o myprog
```

#### 执行

```bash
./myprog
```

#### 输出结果

```
(3, 4)
(5, 4)
```

---

## 6. 运算符重载

### 6.1 运算符重载的概念

**运算符重载**（Operator Overloading）允许开发者为自定义类定义或修改运算符的行为，使得自定义类型的对象可以像基本类型一样进行运算。这提高了代码的可读性和可维护性。

### 6.2 方法重载与运算符重载

- **方法重载**：在同一类中，可以定义多个同名方法，只要它们的参数列表不同。
- **运算符重载**：允许为自定义类型定义运算符的行为，可以通过成员方法或友元函数实现。

### 6.3 运算符重载的规则

- **运算符的优先级和结合性不变**。
- **运算符的重载不能改变运算符的优先级或结合性**。
- **不能重载某些运算符**，如 `::`、`.`、`.*`、`?:`、`sizeof`、`typeid` 以及类型转换运算符。
- **重载后的运算符必须保持其原有的用途和语义**，以避免代码的混乱和错误。

### 6.4 运算符重载的实现方式

运算符重载可以通过成员方法或友元函数来实现：

1. **成员方法**：适用于运算符的左操作数是类的对象。
2. **友元函数或外部函数**：适用于运算符的左操作数不是类的对象。

#### 示例：内部重载运算符 `+` 和 `/`

```cpp
#include <iostream>

class Point {
public:
    Point(double x, double y) : M_x(x), M_y(y) {}
    
    // 重载 + 运算符（成员方法）
    Point operator+(const Point& p) const {
        return Point(M_x + p.M_x, M_y + p.M_y);
    }
    
    // 重载 / 运算符（成员方法）
    Point operator/(double val) const {
        return Point(M_x / val, M_y / val);
    }
    
    // 重载 += 运算符（成员方法）
    Point& operator+=(const Point& p) {
        M_x += p.M_x;
        M_y += p.M_y;
        return *this;
    }
    
    void print() const {
        std::cout << "(" << M_x << ", " << M_y << ")\n";
    }

private:
    double M_x, M_y;
};

int main() {
    Point p1(3, 8);
    p1.print(); // 输出 (3, 8)
    
    Point p2(5, 12);
    p2.print(); // 输出 (5, 12)
    
    Point p3 = (p1 + p2) / 2;
    p3.print(); // 输出 (4, 10)
    
    p1 += p3;
    p1.print(); // 输出 (7, 18)
    
    return 0;
}
```

#### 输出结果

```
(3, 8)
(5, 12)
(4, 10)
(7, 18)
```

### 6.5 外部重载运算符

当需要重载的运算符的左操作数不是类的对象时，必须通过外部函数或友元函数来实现运算符重载。

#### 示例：外部重载运算符 `+` 和 `/`

```cpp
#include <iostream>

// 类的定义
class Point {
public:
    Point(double x, double y) : M_x(x), M_y(y) {}
    
    double getX() const { return M_x; }
    double getY() const { return M_y; }
    
    void setX(double x) { M_x = x; }
    void setY(double y) { M_y = y; }
    
    void print() const {
        std::cout << "(" << M_x << ", " << M_y << ")\n";
    }

private:
    double M_x, M_y;
};

// 重载 + 运算符（外部函数）
Point operator+(const Point& p1, const Point& p2) {
    return Point(p1.getX() + p2.getX(), p1.getY() + p2.getY());
}

// 重载 / 运算符（外部函数）
Point operator/(const Point& p, double val) {
    return Point(p.getX() / val, p.getY() / val);
}

// 重载 += 运算符（外部函数）
Point& operator+=(Point& p1, const Point& p2) {
    p1.setX(p1.getX() + p2.getX());
    p1.setY(p1.getY() + p2.getY());
    return p1;
}

int main() {
    Point p1(3, 8);
    p1.print(); // 输出 (3, 8)
    
    Point p2(5, 12);
    p2.print(); // 输出 (5, 12)
    
    Point p3 = (p1 + p2) / 2;
    p3.print(); // 输出 (4, 10)
    
    p1 += p3;
    p1.print(); // 输出 (7, 18)
    
    return 0;
}
```

#### 输出结果

```
(3, 8)
(5, 12)
(4, 10)
(7, 18)
```

### 6.6 重载 `<<` 运算符用于输出流

重载 `<<` 运算符可以使自定义类的对象能够直接通过 `std::cout` 输出其内容，提升代码的可读性和易用性。

#### 示例代码

```cpp
#include <iostream>

// 类的定义
class Point {
public:
    Point(double x, double y) : M_x(x), M_y(y) {}
    
    double getX() const { return M_x; }
    double getY() const { return M_y; }
    
    // 重载 + 运算符（成员方法）
    Point operator+(const Point& p) const {
        return Point(M_x + p.M_x, M_y + p.M_y);
    }
    
    // 重载 / 运算符（成员方法）
    Point operator/(double val) const {
        return Point(M_x / val, M_y / val);
    }

private:
    double M_x, M_y;

    // 声明友元函数以访问私有成员
    friend std::ostream& operator<<(std::ostream& os, const Point& p);
};

// 重载 << 运算符（友元函数）
std::ostream& operator<<(std::ostream& os, const Point& p) {
    os << "(" << p.M_x << ", " << p.M_y << ")";
    return os;
}

int main() {
    Point p1(3, 8);
    Point p2(5, 12);
    
    std::cout << "p1 = " << p1 << "\n"; // 输出 p1 = (3, 8)
    std::cout << "p2 = " << p2 << "\n"; // 输出 p2 = (5, 12)
    
    Point p3 = (p1 + p2) / 2;
    std::cout << "(p1 + p2) / 2 = " << p3 << "\n"; // 输出 (p1 + p2) / 2 = (4, 10)
    
    return 0;
}
```

#### 输出结果

```
p1 = (3, 8)
p2 = (5, 12)
(p1 + p2) / 2 = (4, 10)
```

---

## 7. 静态成员与方法

### 7.1 静态属性

**静态属性**（Static Members）属于类本身，而不是某个具体的对象。所有对象共享同一个静态属性。

#### 示例代码

```cpp
#include <iostream>
#include <string>

class Poly3d {
public:
    static const int nDim = 3; // 静态常量属性
    static int nOrder;         // 静态属性

    // 静态方法
    static void info() {
        std::cout << "[Poly3d] nOrder = " << nOrder << "\n";
    }
};

// 静态属性的初始化
int Poly3d::nOrder = 2;

int main() {
    std::cout << "维度: " << Poly3d::nDim << "\n"; // 输出 3
    std::cout << "阶数: " << Poly3d::nOrder << "\n"; // 输出 2
    
    Poly3d::info(); // 调用静态方法，输出 [Poly3d] nOrder = 2
    
    // 修改静态属性
    Poly3d::nOrder = 5;
    Poly3d::info(); // 输出 [Poly3d] nOrder = 5
    
    // 创建对象并访问静态属性
    Poly3d poly, poly2;
    std::cout << "阶数: " << poly.nOrder << "\n";   // 输出 5
    poly.nOrder = 9;
    std::cout << "阶数: " << poly2.nOrder << "\n";  // 输出 9
    
    return 0;
}
```

#### 输出结果

```
维度: 3
阶数: 2
[Poly3d] nOrder = 2
[Poly3d] nOrder = 5
阶数: 5
阶数: 9
```

### 7.2 静态方法

**静态方法**（Static Methods）属于类本身，可以在没有创建对象的情况下调用。静态方法只能访问静态属性，不能访问非静态属性。

#### 示例代码

```cpp
#include <iostream>

class Poly3d {
public:
    static const int nDim = 3;
    static int nOrder;

    static void info() {
        std::cout << "[Poly3d] nOrder = " << nOrder << "\n";
    }
};

// 静态属性的初始化
int Poly3d::nOrder = 2;

int main() {
    Poly3d::info(); // 调用静态方法，输出 [Poly3d] nOrder = 2
    
    Poly3d::nOrder = 4;
    Poly3d::info(); // 输出 [Poly3d] nOrder = 4
    
    return 0;
}
```

#### 输出结果

```
[Poly3d] nOrder = 2
[Poly3d] nOrder = 4
```

### 7.3 静态成员的注意事项

- 静态成员在所有对象之间共享，因此需要谨慎管理，以避免不必要的依赖和冲突。
- 静态成员必须在类外部进行初始化，除非它们是 `const` 并且在类内初始化。

---

## 8. 友元函数

### 8.1 友元函数的概念

**友元函数**（Friend Functions）是被声明为友元的外部函数，允许它们访问类的私有和受保护成员。友元函数不是类的成员函数，但它们拥有特殊的访问权限。

### 8.2 友元函数的声明与定义

在类的声明中使用 `friend` 关键字声明友元函数，然后在类外部定义该函数。

#### 示例代码

```cpp
#include <iostream>

class Point {
public:
    Point(double x, double y) : M_x(x), M_y(y) {}
    
    // 声明友元函数
    friend void modify(Point& a, double x, double y);

private:
    double M_x, M_y;
};

// 友元函数的定义
void modify(Point& a, double x, double y) {
    a.M_x = x;
    a.M_y = y;
}

int main() {
    Point p(2, 3);
    modify(p, 6, 7); // 直接修改私有成员
    // p.print(); // 如果有 print 方法，可以调用查看修改后的值
    return 0;
}
```

#### 友元函数的用途

- **运算符重载**：如 `<<` 运算符通常作为友元函数实现，以访问类的私有成员。
- **外部辅助函数**：需要访问类的私有成员以完成特定任务的函数。

### 8.3 一个类可以有多个友元

一个类可以声明多个友元函数，甚至其他类作为友元，以允许其成员函数访问该类的私有成员。

#### 示例代码

```cpp
class ClassA {
public:
    ClassA(int val) : M_val(val) {}
    
    friend void display(const ClassA& a);
    friend class ClassB; // ClassB 是 ClassA 的友元类

private:
    int M_val;
};

void display(const ClassA& a) {
    std::cout << "ClassA::M_val = " << a.M_val << "\n";
}

class ClassB {
public:
    void show(const ClassA& a) {
        std::cout << "ClassB accessing ClassA::M_val = " << a.M_val << "\n";
    }
};

int main() {
    ClassA a(10);
    display(a);
    
    ClassB b;
    b.show(a);
    
    return 0;
}
```

#### 输出结果

```
ClassA::M_val = 10
ClassB accessing ClassA::M_val = 10
```

---

## 9. `this` 指针

### 9.1 `this` 指针的概念

在类的成员方法内部，`this` 是一个隐式的指针，指向调用该方法的对象本身。通过 `this` 指针，可以访问对象的成员变量和成员方法。

### 9.2 `this` 指针的用途

- **区分成员变量和局部变量**：当成员变量和参数名相同时，可以使用 `this` 指针来区分。
- **返回当前对象的引用**：允许链式调用。
- **在方法中获取对象的地址**。

#### 示例代码

```cpp
#include <iostream>

class Point {
public:
    Point(double x, double y) : M_x(x), M_y(y) {}
    
    // 方法中使用 this 指针
    bool isSameObject(const Point& p) const {
        return this == &p;
    }
    
    bool isEqual(const Point& p) const {
        return (this->getX() == p.getX()) && (this->getY() == p.getY());
    }

    double getX() const { return M_x; }
    double getY() const { return M_y; }

private:
    double M_x, M_y;
};

int main() {
    Point p1(2, 3);
    Point p2(5, 8);
    Point p3 = p1;
    
    std::cout << std::boolalpha;
    std::cout << "p1 与 p2 是同一个对象吗？ " << p1.isSameObject(p2) << "\n"; // false
    std::cout << "p1 与 p1 是同一个对象吗？ " << p1.isSameObject(p1) << "\n"; // true
    
    std::cout << "p1 与 p3 的坐标相等吗？ " << p1.isEqual(p3) << "\n"; // true
    
    return 0;
}
```

#### 输出结果

```
p1 与 p2 是同一个对象吗？ false
p1 与 p1 是同一个对象吗？ true
p1 与 p3 的坐标相等吗？ true
```

### 9.3 使用 `this` 指针返回对象自身

通过 `this` 指针，可以在成员方法中返回对象自身的引用，从而支持链式调用。

#### 示例代码

```cpp
#include <iostream>

class Point {
public:
    Point(double x = 0, double y = 0) : M_x(x), M_y(y) {}
    
    // 设置 X 坐标并返回对象自身
    Point& setX(double x) {
        M_x = x;
        return *this;
    }
    
    // 设置 Y 坐标并返回对象自身
    Point& setY(double y) {
        M_y = y;
        return *this;
    }
    
    void print() const {
        std::cout << "(" << M_x << ", " << M_y << ")\n";
    }

private:
    double M_x, M_y;
};

int main() {
    Point p;
    p.setX(5).setY(10).setX(7); // 链式调用
    p.print(); // 输出 (7, 10)
    return 0;
}
```

#### 输出结果

```
(7, 10)
```

---

## 10. 类图

### 10.1 类图的概念

**类图**（Class Diagram）是用来描述类及其之间关系的图形化表示。类图通常使用统一建模语言（UML）来绘制，展示类的属性、方法及其访问权限，以及类之间的继承、关联等关系。

### 10.2 类图的组成部分

- **类名**：位于类图的顶部。
- **属性**：列在类名下方，标明属性的名称和类型，前面有 `+`（公有）或 `-`（私有）符号表示访问权限。
- **方法**：列在属性下方，标明方法的名称、参数和返回类型，前面有 `+`（公有）或 `-`（私有）符号表示访问权限。

#### 示例：Point 类的类图

```
+-----------------+
|      Point      |
+-----------------+
| - M_x : double  |
| - M_y : double  |
+-----------------+
| + Point(x, y)   |
| + getX() : double |
| + getY() : double |
| + setX(x)       |
| + setY(y)       |
| + print()       |
+-----------------+
```

### 10.3 类图示例

#### Point 类的类图

```plaintext
+-----------------+
|      Point      |
+-----------------+
| - M_x : double  |
| - M_y : double  |
+-----------------+
| + Point(x, y)   |
| + getX() : double |
| + getY() : double |
| + setX(x)       |
| + setY(y)       |
| + print()       |
+-----------------+
```

#### 多类关系的类图

假设有两个类 `Point` 和 `Line`，其中 `Line` 包含两个 `Point` 对象作为端点。

```plaintext
+-----------------+          +-----------------+
|      Point      |          |      Line       |
+-----------------+          +-----------------+
| - M_x : double  |          | - start : Point |
| - M_y : double  |<>--------| - end : Point   |
+-----------------+          +-----------------+
| + Point(x, y)   |          | + Line(start, end)|
| + getX() : double |        | + length() : double|
| + getY() : double |        | + print()       |
| + setX(x)       |          +-----------------+
| + setY(y)       |
| + print()       |
+-----------------+
```

- **<>** 表示 `Line` 类与 `Point` 类之间的关联关系，即 `Line` 类包含两个 `Point` 对象。

---

## 11. 进一步学习

以上内容涵盖了 C++ 面向对象编程的基础知识，包括类与对象的定义、封装、构造函数与析构函数、成员方法、运算符重载、静态成员与方法、友元函数以及 `this` 指针等。掌握这些基础知识后，您可以开始编写更复杂的 C++ 程序，并进一步学习以下高级主题：

- **继承与多态**：学习如何通过继承机制实现类之间的关系，以及多态如何提高代码的灵活性。
- **抽象类与接口**：理解如何通过抽象类定义接口，提升代码的可扩展性。
- **模板编程**：掌握泛型编程，编写更通用和可复用的代码。
- **标准模板库（STL）**：深入学习 C++ 提供的各种容器、算法和迭代器，提升编程效率。
- **异常处理**：学习如何处理程序运行中的错误，提高程序的健壮性。
- **现代 C++ 特性**：如智能指针、Lambda 表达式、并行编程等，掌握最新的 C++ 标准（如 C++11、C++14、C++17、C++20）提供的新功能。

### 推荐资源

- **书籍**：
    
    - 《C++ Primer》
    - 《Effective C++》
    - 《The C++ Programming Language》 by Bjarne Stroustrup
- **在线教程**：
    
    - [cplusplus.com](http://www.cplusplus.com/)
    - [Learn C++](https://www.learncpp.com/)
- **视频课程**：
    
    - [Coursera - C++ For C Programmers](https://www.coursera.org/learn/c-plus-plus-a)
    - [edX - Introduction to C++](https://www.edx.org/course/introduction-to-c-plus-plus)

---

希望以上内容能够帮助您更好地理解 C++ 面向对象编程的基础知识。如有任何疑问，请随时联系我。

祝学习愉快！