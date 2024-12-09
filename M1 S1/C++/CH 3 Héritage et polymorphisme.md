# C++ 面向对象编程：继承与多态

## 目录

1. 继承
    - 继承的概念
    - 类图示例
    - C++ 中的继承示例
    - 访问权限在继承中的作用
    - 继承方式（public、protected、private）
    - 构造函数在继承中的调用
    - 拷贝构造函数与赋值操作符在继承中的处理
    - 析构函数与继承
    - 方法重写与方法重载
2. 多态
    - 多态的概念
    - 静态绑定与动态绑定
    - 多态的实现（虚函数）
    - 多态的优势与注意事项
    - 多态与析构函数
    - 多态在集合中的应用
    - 类型转换（dynamic_cast、static_cast）
3. 多重继承
    - 多重继承的概念
    - 多重继承中的问题
    - 菱形继承问题
    - 虚拟继承的解决方案
4. 抽象类
    - 抽象类的概念
    - 纯虚函数
    - 抽象类的用途
5. 进一步学习

---

## 1. 继承

### 1.1 继承的概念

在面向对象编程中，**继承**是一种机制，允许我们基于已有的类（称为**基类**或**父类**）创建新类（称为**派生类**或**子类**）。派生类可以继承基类的属性和方法，同时还可以增加新的属性和方法，或修改（重写）基类的方法。

**继承的优点**：

- **代码复用**：通过继承，可以复用基类中已经编写的代码，减少重复劳动。
- **分类管理**：通过类的层次结构，可以更清晰地组织和管理代码，实现概念的泛化与特殊化。

**术语**：

- **基类（Superclass、Base Class、Parent Class）**：被继承的类。
- **派生类（Subclass、Derived Class、Child Class）**：继承自基类的新类。

### 1.2 类图示例

类图是用来描述类及其之间关系的图形化工具，通常使用统一建模语言（UML）绘制。以下是一个简单的类图示例，展示了继承关系：

```
+------------+
|  Polygon   |
+------------+
      |
      |
+------------+------------+
| Quadrilateral |  Triangle |
+------------+------------+
     / \              / \
    /   \            /   \
Rectangle  Rhombus  EquilateralTriangle RightTriangle
```

**说明**：

- **Polygon** 是基类，**Quadrilateral** 和 **Triangle** 是其派生类。
- **Quadrilateral** 进一步派生出 **Rectangle** 和 **Rhombus**。
- **Triangle** 进一步派生出 **EquilateralTriangle**（等边三角形）和 **RightTriangle**（直角三角形）。

### 1.3 C++ 中的继承示例

#### 基类的定义

```cpp
#include <iostream>
#include <string>

// 基类 Triangle
class Triangle {
public:
    // 构造函数
    Triangle(const Point& a, const Point& b, const Point& c)
        : M_name("Triangle"), M_a(a), M_b(b), M_c(c) {}
    
    // 访问器
    const std::string& name() const { return M_name; }
    
    // 修改器
    void setName(const std::string& name) { M_name = name; }
    
    // 计算面积的方法
    virtual double area() const { 
        // 计算三角形面积的代码
        return 0.0; // 示例
    }
    
protected:
    Point M_a, M_b, M_c; // 三角形的三个顶点

private:
    std::string M_name; // 三角形的名称
};
```

#### 派生类的定义

```cpp
// 派生类 TriangleRect（直角三角形）
class TriangleRect : public Triangle {
public:
    // 构造函数
    TriangleRect(const Point& a, const Point& b, const Point& c)
        : Triangle(a, b, c) // 调用基类构造函数
    {
        this->setName("Right Triangle"); // 修改基类的名称属性
    }
    
    // 重写计算面积的方法
    double area() const override {
        // 直角三角形的面积计算方法
        return M_a.distance(M_b) * M_a.distance(M_c) / 2.0;
    }
};
```

#### 主函数示例

```cpp
int main() {
    Point a(0, 0);
    Point b(0, 3);
    Point c(3, 0);
    
    TriangleRect tr(a, b, c);
    
    std::cout << "Name: " << tr.name() << "\n";       // 输出名称
    std::cout << "Area: " << tr.area() << "\n";       // 输出面积
    
    return 0;
}
```

#### 输出结果

```
Name: Right Triangle
Area: 4.5
```

### 1.4 访问权限在继承中的作用

在 C++ 中，类成员（属性和方法）的访问权限由以下三个关键字控制：

- **public**：公有成员，可以被类的外部访问。
- **protected**：受保护成员，只能在类内部及其派生类中访问。
- **private**：私有成员，只能在类内部访问。

继承时，基类成员在派生类中的访问权限会根据继承方式（public、protected、private）进行调整。

#### 示例

```cpp
class A {
public:
    int publicA;
protected:
    int protectedA;
private:
    int privateA;
};

class B : public A {
    // publicA 依然是 public
    // protectedA 依然是 protected
    // privateA 不可访问
};

class C : protected A {
    // publicA 和 protectedA 都变为 protected
    // privateA 不可访问
};

class D : private A {
    // publicA 和 protectedA 都变为 private
    // privateA 不可访问
};
```

**说明**：

- **public 继承**：基类的 `public` 成员在派生类中保持 `public`，`protected` 成员保持 `protected`，`private` 成员不可访问。
- **protected 继承**：基类的 `public` 和 `protected` 成员在派生类中都变为 `protected`，`private` 成员不可访问。
- **private 继承**：基类的 `public` 和 `protected` 成员在派生类中都变为 `private`，`private` 成员不可访问。

### 1.5 构造函数在继承中的调用

当创建一个派生类对象时，基类的构造函数会被自动调用。派生类的构造函数必须明确调用基类的构造函数，如果不调用，编译器会默认调用基类的默认构造函数。

#### 示例

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) { std::cout << "A 的构造函数被调用，x = " << M_x << "\n"; }
private:
    int M_x;
};

class B : public A {
public:
    B(int x, int y) : A(x), M_y(y) { std::cout << "B 的构造函数被调用，y = " << M_y << "\n"; }
private:
    int M_y;
};

int main() {
    B b(10, 20);
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用，x = 10
B 的构造函数被调用，y = 20
```

**说明**：

- 类 `B` 的构造函数通过初始化列表调用了基类 `A` 的构造函数，传递参数 `x`。
- 基类的构造函数在派生类的构造函数之前被调用。

### 1.6 拷贝构造函数与赋值操作符在继承中的处理

#### 拷贝构造函数

如果派生类没有定义自己的拷贝构造函数，编译器会自动生成一个，它会依次调用基类的拷贝构造函数，并逐成员拷贝派生类的成员。

如果派生类自定义了拷贝构造函数，必须显式调用基类的拷贝构造函数，否则基类部分将调用基类的默认拷贝构造函数。

#### 示例

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) {}
    A(const A& a) : M_x(a.M_x) { std::cout << "A 的拷贝构造函数被调用\n"; }
private:
    int M_x;
};

class B : public A {
public:
    B(int x, int y) : A(x), M_y(y) {}
    // 自定义拷贝构造函数
    B(const B& b) : A(b), M_y(b.M_y) { std::cout << "B 的拷贝构造函数被调用\n"; }
private:
    int M_y;
};

int main() {
    B b1(10, 20);
    B b2 = b1; // 调用 B 的拷贝构造函数
    return 0;
}
```

#### 输出结果

```
A 的拷贝构造函数被调用
B 的拷贝构造函数被调用
```

#### 赋值操作符

如果派生类没有定义自己的赋值操作符，编译器会自动生成一个，它会依次调用基类的赋值操作符，并逐成员赋值派生类的成员。

如果派生类自定义了赋值操作符，必须显式调用基类的赋值操作符，以确保基类部分正确赋值。

#### 示例

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) {}
    A& operator=(const A& a) {
        if (this != &a) {
            M_x = a.M_x;
            std::cout << "A 的赋值操作符被调用\n";
        }
        return *this;
    }
private:
    int M_x;
};

class B : public A {
public:
    B(int x, int y) : A(x), M_y(y) {}
    B& operator=(const B& b) {
        if (this != &b) {
            A::operator=(b); // 调用基类的赋值操作符
            M_y = b.M_y;
            std::cout << "B 的赋值操作符被调用\n";
        }
        return *this;
    }
private:
    int M_y;
};

int main() {
    B b1(10, 20);
    B b2(30, 40);
    b2 = b1; // 调用 B 的赋值操作符
    return 0;
}
```

#### 输出结果

```
A 的赋值操作符被调用
B 的赋值操作符被调用
```

**注意**：

- 在赋值操作符中，通常需要检查自赋值（`if (this != &b)`），以避免不必要的操作。
- 赋值操作符的返回类型通常是自身的引用类型（如 `B&`），以支持链式赋值（如 `b1 = b2 = b3`）。

### 1.7 析构函数与继承

在继承关系中，当销毁派生类对象时，首先调用派生类的析构函数，然后自动调用基类的析构函数。为了确保通过基类指针删除派生类对象时能够正确调用派生类的析构函数，基类的析构函数应当声明为 `virtual`。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    virtual ~A() { std::cout << "A 的析构函数被调用\n"; }
};

class B : public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
    ~B() { std::cout << "B 的析构函数被调用\n"; }
};

int main() {
    A* a = new B();
    delete a; // 先调用 B 的析构函数，再调用 A 的析构函数
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
B 的析构函数被调用
A 的析构函数被调用
```

**说明**：

- 如果基类的析构函数不是 `virtual`，则通过基类指针删除派生类对象时，只会调用基类的析构函数，派生类的析构函数不会被调用，可能导致资源泄漏。

#### 非虚析构函数的示例

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    ~A() { std::cout << "A 的析构函数被调用\n"; }
};

class B : public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
    ~B() { std::cout << "B 的析构函数被调用\n"; }
};

int main() {
    A* a = new B();
    delete a; // 只调用 A 的析构函数，B 的析构函数不会被调用
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
A 的析构函数被调用
```

**说明**：

- 由于基类 `A` 的析构函数不是虚函数，通过基类指针删除派生类对象时，只会调用基类的析构函数，派生类的析构函数不会被调用，可能导致资源泄漏。

### 1.8 方法重写与方法重载

#### 方法重写（Overriding）

**方法重写**指的是在派生类中重新定义基类的虚函数，以提供不同的实现。重写的方法必须与基类的方法具有相同的签名（名称、参数列表、`const` 修饰符等）。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    virtual void affiche() const {
        std::cout << "class A\n";
    }
    
    void afficheAutre() const {
        std::cout << "Autre\n";
    }
};

class B : public A {
public:
    void affiche() const override { // 重写基类的虚函数
        std::cout << "class B\n";
    }
    
    void afficheBase() const {
        A::affiche(); // 调用基类的方法
    }
};

int main() {
    B b;
    b.affiche();         // 输出 class B
    b.afficheBase();     // 输出 class A
    b.afficheAutre();    // 输出 Autre
    return 0;
}
```

#### 输出结果

```
class B
class A
Autre
```

#### 方法重载（Overloading）

**方法重载**指的是在同一个类中定义多个同名方法，只要它们的参数列表不同（参数类型、数量或顺序不同）。编译器根据调用时传递的参数自动选择合适的方法。

#### 示例代码

```cpp
#include <iostream>

class Point {
public:
    void print() const {
        std::cout << "Point\n";
    }
    
    void print(int a) const {
        std::cout << "Point with int: " << a << "\n";
    }
    
    void print(double a, double b) const {
        std::cout << "Point with two doubles: " << a << ", " << b << "\n";
    }
};

int main() {
    Point p;
    p.print();            // 调用无参数的 print
    p.print(5);           // 调用带一个 int 参数的 print
    p.print(3.14, 2.71);  // 调用带两个 double 参数的 print
    return 0;
}
```

#### 输出结果

```
Point
Point with int: 5
Point with two doubles: 3.14, 2.71
```

**注意**：

- 方法重载仅根据参数列表进行区分，与返回类型无关。
- 在继承中，如果派生类中重载了基类的方法，会隐藏基类中所有同名的方法，除非显式指定调用基类的方法。

#### 示例：继承中的方法重载

```cpp
#include <iostream>

class A {
public:
    virtual void affiche() const {
        std::cout << "class A\n";
    }
    
    void affiche(int a) const {
        std::cout << "A::affiche(int): " << a << "\n";
    }
};

class B : public A {
public:
    void affiche() const override {
        std::cout << "class B\n";
    }
    
    // 重载方法
    void affiche(double a) const {
        std::cout << "B::affiche(double): " << a << "\n";
    }
};

int main() {
    B b;
    b.affiche();        // 调用 B::affiche()
    // b.affiche(5);     // 编译错误：A::affiche(int) 被隐藏
    b.A::affiche(5);    // 显式调用 A::affiche(int)
    b.affiche(3.14);    // 调用 B::affiche(double)
    return 0;
}
```

#### 输出结果

```
class B
A::affiche(int): 5
B::affiche(double): 3.14
```

**说明**：

- 在类 `B` 中重载了 `affiche` 方法，增加了一个接受 `double` 参数的方法。
- 由于方法重载会隐藏基类中所有同名的方法，直接调用 `b.affiche(5)` 会导致编译错误。
- 通过 `b.A::affiche(5)` 可以显式调用基类的方法。

---

## 2. 多态

### 2.1 多态的概念

**多态性**（Polymorphism）是面向对象编程的核心概念之一，指的是同一个操作作用于不同的对象，可以产生不同的行为。多态性使得程序在运行时能够根据对象的实际类型来决定调用哪个方法，实现更灵活和可扩展的代码。

### 2.2 静态绑定与动态绑定

- **静态绑定**（Static Binding）：在编译时决定调用哪个方法，通常发生在非虚函数或通过对象直接调用方法时。
    
- **动态绑定**（Dynamic Binding）：在运行时决定调用哪个方法，通常发生在通过基类指针或引用调用虚函数时。
    

#### 静态绑定示例

```cpp
#include <iostream>

class A {
public:
    void affiche() const {
        std::cout << "class A\n";
    }
};

class B : public A {
public:
    void affiche() const {
        std::cout << "class B\n";
    }
};

void myfunc(const A& a) {
    a.affiche(); // 静态绑定，调用 A::affiche
}

int main() {
    B b;
    myfunc(b); // 输出 class A
    return 0;
}
```

#### 输出结果

```
class A
```

**说明**：

- 由于 `affiche` 方法不是虚函数，调用 `a.affiche()` 时，编译器根据引用类型 `A` 决定调用基类的方法，即使实际对象是 `B`，也不会调用派生类的方法。

#### 动态绑定示例

```cpp
#include <iostream>

class A {
public:
    virtual void affiche() const {
        std::cout << "class A\n";
    }
};

class B : public A {
public:
    void affiche() const override {
        std::cout << "class B\n";
    }
};

void myfunc(const A& a) {
    a.affiche(); // 动态绑定，调用实际类型的方法
}

int main() {
    B b;
    myfunc(b); // 输出 class B
    return 0;
}
```

#### 输出结果

```
class B
```

**说明**：

- 由于 `affiche` 方法被声明为 `virtual`，调用 `a.affiche()` 时，编译器根据实际对象类型决定调用哪个方法，实现动态绑定。

### 2.3 多态的实现（虚函数）

在 C++ 中，通过使用 `virtual` 关键字，可以将基类的方法声明为虚函数，从而实现动态绑定。派生类可以重写（override）基类的虚函数，以提供不同的实现。

#### 示例代码

```cpp
#include <iostream>
#include <string>

class A {
public:
    virtual void affiche() const {
        std::cout << "class A\n";
    }
    
    void afficheAutre() const {
        std::cout << "Autre\n";
    }
};

class B : public A {
public:
    void affiche() const override { // 重写基类的虚函数
        std::cout << "class B\n";
    }
    
    void afficheBase() const {
        A::affiche(); // 调用基类的方法
    }
};

int main() {
    B b;
    A* aPtr = new B();
    
    aPtr->affiche();       // 动态绑定，输出 class B
    aPtr->afficheAutre();  // 静态绑定，输出 Autre
    
    delete aPtr;
    return 0;
}
```

#### 输出结果

```
class B
Autre
```

**说明**：

- `affiche` 方法被声明为 `virtual`，所以通过基类指针调用时，会根据实际对象类型调用派生类的方法。
- `afficheAutre` 方法不是虚函数，调用时基于引用类型 `A` 进行静态绑定，始终调用基类的方法。

### 2.4 多态的优势与注意事项

**优势**：

- **灵活性**：允许使用基类指针或引用指向派生类对象，实现统一接口。
- **可扩展性**：新增派生类时，不需要修改现有代码，只需实现基类接口即可。
- **代码复用**：基类中实现的通用功能可以被所有派生类共享。

**注意事项**：

- **基类析构函数必须为虚函数**，以确保通过基类指针删除派生类对象时能正确调用派生类的析构函数，避免资源泄漏。
- **虚函数的使用**会引入一定的性能开销**，因为需要在运行时进行动态绑定。
- **避免过度使用**：过度依赖多态可能导致设计复杂，需要合理设计类的继承关系。

### 2.5 多态与析构函数

在继承关系中，如果基类的析构函数不是虚函数，通过基类指针删除派生类对象时，只会调用基类的析构函数，派生类的析构函数不会被调用，可能导致资源泄漏。

#### 示例：非虚析构函数

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    ~A() { std::cout << "A 的析构函数被调用\n"; }
};

class B : public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
    ~B() { std::cout << "B 的析构函数被调用\n"; }
};

int main() {
    A* a = new B();
    delete a; // 只调用 A 的析构函数，B 的析构函数不会被调用
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
A 的析构函数被调用
```

#### 示例：虚析构函数

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    virtual ~A() { std::cout << "A 的析构函数被调用\n"; }
};

class B : public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
    ~B() override { std::cout << "B 的析构函数被调用\n"; }
};

int main() {
    A* a = new B();
    delete a; // 先调用 B 的析构函数，再调用 A 的析构函数
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
B 的析构函数被调用
A 的析构函数被调用
```

**说明**：

- 当基类的析构函数为 `virtual` 时，通过基类指针删除派生类对象，会正确调用派生类的析构函数，实现完整的资源释放。
- 如果基类的析构函数不是 `virtual`，则只能调用基类的析构函数，派生类的析构函数不会被调用，可能导致资源泄漏。

### 2.6 多态在集合中的应用

通过多态，可以创建一个基类指针数组，用来存储不同派生类的对象，实现多态集合。这在需要处理不同类型对象时非常有用，例如图形系统中处理不同形状的对象。

#### 示例代码

```cpp
#include <iostream>
#include <string>

// 基类 Triangle
class Triangle {
public:
    virtual std::string name() const { return "Triangle"; }
    virtual ~Triangle() {}
};

// 派生类 TriangleEquilateral（等边三角形）
class TriangleEquilateral : public Triangle {
public:
    std::string name() const override { return "TriangleEquilateral"; }
};

// 派生类 TriangleIsoceles（等腰三角形）
class TriangleIsoceles : public Triangle {
public:
    std::string name() const override { return "TriangleIsoceles"; }
};

// 派生类 TriangleRectangle（直角三角形）
class TriangleRectangle : public Triangle {
public:
    std::string name() const override { return "TriangleRectangle"; }
    double lengthHypotenuse() const { return 1.23; }
};

int main() {
    // 创建基类指针数组，指向不同派生类对象
    Triangle** listTriangle = new Triangle*[10];
    for (int k = 0; k < 10; ++k) {
        if (k % 3 == 0)
            listTriangle[k] = new TriangleRectangle();
        else if (k % 3 == 1)
            listTriangle[k] = new TriangleEquilateral();
        else
            listTriangle[k] = new TriangleIsoceles();
    }
    
    // 调用各个对象的方法
    for (int k = 0; k < 10; ++k)
        std::cout << listTriangle[k]->name() << std::endl;
    
    // 释放内存
    for (int k = 0; k < 10; ++k)
        delete listTriangle[k];
    delete[] listTriangle;
    
    return 0;
}
```

#### 输出结果

```
TriangleRectangle
TriangleEquilateral
TriangleIsoceles
TriangleRectangle
TriangleEquilateral
TriangleIsoceles
TriangleRectangle
TriangleEquilateral
TriangleIsoceles
TriangleRectangle
```

### 2.7 类型转换（dynamic_cast 和 static_cast）

在多态编程中，常常需要将基类指针或引用转换为派生类类型，以访问派生类特有的方法或属性。C++ 提供了几种类型转换运算符：

- **dynamic_cast**：用于安全的向下转换（将基类指针或引用转换为派生类类型），需要基类有虚函数。运行时会检查转换是否合法，如果不合法，返回 `nullptr`（指针）或抛出异常（引用）。
- **static_cast**：用于编译时类型转换，不进行运行时检查。适用于确定转换合法的情况。

#### dynamic_cast 示例

```cpp
#include <iostream>
#include <string>

class Triangle {
public:
    virtual std::string name() const { return "Triangle"; }
    virtual ~Triangle() {}
};

class TriangleRectangle : public Triangle {
public:
    std::string name() const override { return "TriangleRectangle"; }
    double lengthHypotenuse() const { return 1.23; }
};

int main() {
    Triangle* t = new TriangleRectangle();
    
    // 尝试将基类指针转换为派生类指针
    TriangleRectangle* tr = dynamic_cast<TriangleRectangle*>(t);
    if (tr) {
        std::cout << "长度斜边: " << tr->lengthHypotenuse() << "\n";
    } else {
        std::cout << "转换失败\n";
    }
    
    delete t;
    return 0;
}
```

#### 输出结果

```
长度斜边: 1.23
```

#### static_cast 示例

```cpp
#include <iostream>
#include <string>

class Triangle {
public:
    virtual std::string name() const { return "Triangle"; }
    virtual ~Triangle() {}
};

class TriangleRectangle : public Triangle {
public:
    std::string name() const override { return "TriangleRectangle"; }
    double lengthHypotenuse() const { return 1.23; }
};

int main() {
    Triangle* t = new TriangleRectangle();
    
    // 使用 static_cast 进行类型转换
    TriangleRectangle* tr = static_cast<TriangleRectangle*>(t);
    std::cout << "长度斜边: " << tr->lengthHypotenuse() << "\n";
    
    delete t;
    return 0;
}
```

#### 输出结果

```
长度斜边: 1.23
```

**注意**：

- 使用 `dynamic_cast` 时，如果转换失败，返回 `nullptr`（对于指针）或抛出 `std::bad_cast` 异常（对于引用），更安全。
- 使用 `static_cast` 时，不会进行运行时检查，如果类型不匹配，可能导致未定义行为。

---

## 3. 多重继承

### 3.1 多重继承的概念

C++ 支持**多重继承**（Multiple Inheritance），即一个派生类可以同时继承多个基类。这允许派生类组合多个基类的特性和行为，实现更加复杂的类层次结构。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) { std::cout << "A 的构造函数被调用，x = " << M_x << "\n"; }
private:
    int M_x;
};

class B {
public:
    B(int y) : M_y(y) { std::cout << "B 的构造函数被调用，y = " << M_y << "\n"; }
private:
    int M_y;
};

class C : public A, public B {
public:
    C(int x, int y, int z) : A(x), B(y), M_z(z) { std::cout << "C 的构造函数被调用，z = " << M_z << "\n"; }
private:
    int M_z;
};

int main() {
    C c(1, 2, 3);
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用，x = 1
B 的构造函数被调用，y = 2
C 的构造函数被调用，z = 3
```

**说明**：

- 类 `C` 同时继承自类 `A` 和类 `B`，因此在创建 `C` 的对象时，首先调用基类 `A` 的构造函数，然后调用基类 `B` 的构造函数，最后调用派生类 `C` 的构造函数。

### 3.2 多重继承中的问题

多重继承虽然强大，但也带来了一些问题，最典型的是**菱形继承**（Diamond Inheritance）问题。

#### 菱形继承问题

菱形继承指的是当两个基类继承自同一个基类，而一个派生类同时继承自这两个基类时，会导致基类成员出现多份副本，产生二义性和资源浪费。

#### 类图示例

```
      A
     / \
    B   C
     \ /
      D
```

在这个结构中，类 `D` 继承自类 `B` 和类 `C`，而类 `B` 和类 `C` 都继承自类 `A`。这样，类 `D` 中会有两份类 `A` 的成员。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    void display() const { std::cout << "A::display()\n"; }
};

class B : public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
};

class C : public A {
public:
    C() { std::cout << "C 的构造函数被调用\n"; }
};

class D : public B, public C {
public:
    D() { std::cout << "D 的构造函数被调用\n"; }
};

int main() {
    D d;
    // d.display(); // 编译错误，二义性：B::A::display 或 C::A::display
    d.B::display(); // 指定调用 B 继承的 A::display
    d.C::display(); // 指定调用 C 继承的 A::display
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
A 的构造函数被调用
C 的构造函数被调用
D 的构造函数被调用
A::display()
A::display()
```

**问题分析**：

- 类 `D` 中有两份类 `A` 的成员，一份来自类 `B`，另一份来自类 `C`。这会导致资源浪费和二义性问题。
- 调用 `d.display()` 会导致编译错误，因为编译器无法确定调用哪一份 `A::display` 方法。

### 3.3 虚继承解决菱形继承问题

为了避免菱形继承带来的问题，C++ 提供了**虚继承**（Virtual Inheritance）。通过将继承声明为虚继承，派生类共享同一份基类成员，从而避免重复。

#### 虚继承示例

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    void display() const { std::cout << "A::display()\n"; }
};

class B : virtual public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
};

class C : virtual public A {
public:
    C() { std::cout << "C 的构造函数被调用\n"; }
};

class D : public B, public C {
public:
    D() { std::cout << "D 的构造函数被调用\n"; }
};

int main() {
    D d;
    d.display(); // 不再有二义性
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
C 的构造函数被调用
D 的构造函数被调用
A::display()
```

**说明**：

- 通过在类 `B` 和类 `C` 的继承声明中加上 `virtual`，类 `D` 中只有一份类 `A` 的成员。
- 调用 `d.display()` 不再有二义性，因为只有一份类 `A` 的成员。

### 3.4 虚继承中的构造函数调用

在虚继承中，基类的构造函数由最底层的派生类负责调用，而其他派生类不会再调用基类的构造函数。这确保了基类成员只被初始化一次，避免了重复。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) { std::cout << "A 的构造函数被调用，x = " << M_x << "\n"; }
private:
    int M_x;
};

class B : virtual public A {
public:
    B(int x, int y) : A(x), M_y(y) { std::cout << "B 的构造函数被调用，y = " << M_y << "\n"; }
private:
    int M_y;
};

class C : virtual public A {
public:
    C(int x, int z) : A(x), M_z(z) { std::cout << "C 的构造函数被调用，z = " << M_z << "\n"; }
private:
    int M_z;
};

class D : public B, public C {
public:
    D(int x, int y, int z, int w) : A(x), B(x, y), C(x, z), M_w(w) { std::cout << "D 的构造函数被调用，w = " << M_w << "\n"; }
private:
    int M_w;
};

int main() {
    D d(1, 2, 3, 4);
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用，x = 1
B 的构造函数被调用，y = 2
C 的构造函数被调用，z = 3
D 的构造函数被调用，w = 4
```

**说明**：

- 类 `B` 和类 `C` 虚继承自类 `A`。
- 类 `D` 的构造函数通过初始化列表调用基类 `A` 的构造函数，仅调用一次。
- 类 `B` 和类 `C` 的构造函数也会调用基类 `A` 的构造函数，但由于虚继承，基类 `A` 的构造函数只会被调用一次，由最底层的派生类 `D` 负责调用。

---

## 4. 抽象类

### 4.1 抽象类的概念

**抽象类**（Abstract Class）是一种包含至少一个纯虚函数（Pure Virtual Function）的类。抽象类不能被实例化，只能作为基类使用，定义接口供派生类实现。

**纯虚函数**：在类中声明但不定义的虚函数，使用 `= 0` 语法表示。

#### 示例代码

```cpp
#include <iostream>
#include <string>

// 抽象基类 A
class A {
public:
    virtual void f(int i) = 0; // 纯虚函数
    virtual ~A() {}
};

// 派生类 B，实现纯虚函数
class B : public A {
public:
    void f(int i) override {
        std::cout << "B::f(" << i << ")\n";
    }
};

int main() {
    // A a; // 错误：无法实例化抽象类
    B b;
    b.f(10); // 正常调用
    return 0;
}
```

#### 输出结果

```
B::f(10)
```

### 4.2 抽象类的用途

- **定义接口**：抽象类可以用来定义接口，确保所有派生类实现特定的方法。
- **实现多态**：通过抽象类的指针或引用，可以实现对派生类对象的多态操作。

#### 示例：接口定义

```cpp
#include <iostream>
#include <string>

// 抽象基类 Shape
class Shape {
public:
    virtual double area() const = 0; // 纯虚函数
    virtual void display() const = 0; // 纯虚函数
    virtual ~Shape() {}
};

// 派生类 Circle
class Circle : public Shape {
public:
    Circle(double r) : radius(r) {}
    double area() const override { return 3.14159 * radius * radius; }
    void display() const override { std::cout << "Circle with radius " << radius << "\n"; }
private:
    double radius;
};

// 派生类 Rectangle
class Rectangle : public Shape {
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const override { return width * height; }
    void display() const override { std::cout << "Rectangle with width " << width << " and height " << height << "\n"; }
private:
    double width, height;
};

int main() {
    Shape* s1 = new Circle(5.0);
    Shape* s2 = new Rectangle(4.0, 6.0);
    
    s1->display(); // 输出 Circle 的信息
    std::cout << "Area: " << s1->area() << "\n";
    
    s2->display(); // 输出 Rectangle 的信息
    std::cout << "Area: " << s2->area() << "\n";
    
    delete s1;
    delete s2;
    return 0;
}
```

#### 输出结果

```
Circle with radius 5
Area: 78.5398
Rectangle with width 4 and height 6
Area: 24
```

**说明**：

- 类 `Shape` 是一个抽象类，定义了 `area` 和 `display` 两个纯虚函数。
- 类 `Circle` 和类 `Rectangle` 实现了这些纯虚函数，成为具体的类，可以实例化。
- 通过基类指针 `Shape*`，可以指向不同派生类的对象，实现多态。

---

## 5. 多重继承

### 5.1 多重继承的概念

C++ 支持**多重继承**（Multiple Inheritance），即一个派生类可以同时继承多个基类。这允许派生类组合多个基类的特性和行为，实现更加复杂的类层次结构。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) { std::cout << "A 的构造函数被调用，x = " << M_x << "\n"; }
private:
    int M_x;
};

class B {
public:
    B(int y) : M_y(y) { std::cout << "B 的构造函数被调用，y = " << M_y << "\n"; }
private:
    int M_y;
};

class C : public A, public B {
public:
    C(int x, int y, int z) : A(x), B(y), M_z(z) { std::cout << "C 的构造函数被调用，z = " << M_z << "\n"; }
private:
    int M_z;
};

int main() {
    C c(1, 2, 3);
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用，x = 1
B 的构造函数被调用，y = 2
C 的构造函数被调用，z = 3
```

**说明**：

- 类 `C` 同时继承自类 `A` 和类 `B`，因此在创建 `C` 的对象时，首先调用基类 `A` 的构造函数，然后调用基类 `B` 的构造函数，最后调用派生类 `C` 的构造函数。

### 5.2 多重继承中的问题

多重继承虽然强大，但也带来了一些问题，最典型的是**菱形继承**（Diamond Inheritance）问题。

#### 菱形继承问题

菱形继承指的是当两个基类继承自同一个基类，而一个派生类同时继承自这两个基类时，会导致基类成员出现多份副本，产生二义性和资源浪费。

#### 类图示例

```
      A
     / \
    B   C
     \ /
      D
```

在这个结构中，类 `D` 继承自类 `B` 和类 `C`，而类 `B` 和类 `C` 都继承自类 `A`。这样，类 `D` 中会有两份类 `A` 的成员。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    void display() const { std::cout << "A::display()\n"; }
};

class B : public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
};

class C : public A {
public:
    C() { std::cout << "C 的构造函数被调用\n"; }
};

class D : public B, public C {
public:
    D() { std::cout << "D 的构造函数被调用\n"; }
};

int main() {
    D d;
    // d.display(); // 编译错误，二义性：B::A::display 或 C::A::display
    d.B::display(); // 指定调用 B 继承的 A::display
    d.C::display(); // 指定调用 C 继承的 A::display
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
A 的构造函数被调用
C 的构造函数被调用
D 的构造函数被调用
A::display()
A::display()
```

**问题分析**：

- 类 `D` 中有两份类 `A` 的成员，一份来自类 `B`，另一份来自类 `C`。这会导致资源浪费和二义性问题。
- 调用 `d.display()` 会导致编译错误，因为编译器无法确定调用哪一份 `A::display` 方法。

### 5.3 虚继承解决菱形继承问题

为了避免菱形继承带来的问题，C++ 提供了**虚继承**（Virtual Inheritance）。通过将继承声明为虚继承，派生类共享同一份基类成员，从而避免重复。

#### 虚继承示例

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    void display() const { std::cout << "A::display()\n"; }
};

class B : virtual public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
};

class C : virtual public A {
public:
    C() { std::cout << "C 的构造函数被调用\n"; }
};

class D : public B, public C {
public:
    D() { std::cout << "D 的构造函数被调用\n"; }
};

int main() {
    D d;
    d.display(); // 不再有二义性
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
C 的构造函数被调用
D 的构造函数被调用
A::display()
```

**说明**：

- 通过在类 `B` 和类 `C` 的继承声明中加上 `virtual`，类 `D` 中只有一份类 `A` 的成员。
- 调用 `d.display()` 不再有二义性，因为只有一份类 `A` 的成员。

### 5.4 虚继承中的构造函数调用

在虚继承中，基类的构造函数由最底层的派生类负责调用，而其他派生类不会再调用基类的构造函数。这确保了基类成员只被初始化一次，避免了重复。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) { std::cout << "A 的构造函数被调用，x = " << M_x << "\n"; }
private:
    int M_x;
};

class B : virtual public A {
public:
    B(int x, int y) : A(x), M_y(y) { std::cout << "B 的构造函数被调用，y = " << M_y << "\n"; }
private:
    int M_y;
};

class C : virtual public A {
public:
    C(int x, int z) : A(x), M_z(z) { std::cout << "C 的构造函数被调用，z = " << M_z << "\n"; }
private:
    int M_z;
};

class D : public B, public C {
public:
    D(int x, int y, int z, int w) : A(x), B(x, y), C(x, z), M_w(w) { std::cout << "D 的构造函数被调用，w = " << M_w << "\n"; }
private:
    int M_w;
};

int main() {
    D d(1, 2, 3, 4);
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用，x = 1
B 的构造函数被调用，y = 2
C 的构造函数被调用，z = 3
D 的构造函数被调用，w = 4
```

**说明**：

- 类 `B` 和类 `C` 虚继承自类 `A`。
- 类 `D` 的构造函数通过初始化列表调用基类 `A` 的构造函数，仅调用一次。
- 类 `B` 和类 `C` 的构造函数也会调用基类 `A` 的构造函数，但由于虚继承，基类 `A` 的构造函数只会被调用一次，由最底层的派生类 `D` 负责调用。

---

## 4. 抽象类

### 4.1 抽象类的概念

**抽象类**（Abstract Class）是一种包含至少一个纯虚函数（Pure Virtual Function）的类。抽象类不能被实例化，只能作为基类使用，定义接口供派生类实现。

**纯虚函数**：在类中声明但不定义的虚函数，使用 `= 0` 语法表示。

#### 示例代码

```cpp
#include <iostream>
#include <string>

// 抽象基类 A
class A {
public:
    virtual void f(int i) = 0; // 纯虚函数
    virtual ~A() {}
};

// 派生类 B，实现纯虚函数
class B : public A {
public:
    void f(int i) override {
        std::cout << "B::f(" << i << ")\n";
    }
};

int main() {
    // A a; // 错误：无法实例化抽象类
    B b;
    b.f(10); // 正常调用
    return 0;
}
```

#### 输出结果

```
B::f(10)
```

### 4.2 抽象类的用途

- **定义接口**：抽象类可以用来定义接口，确保所有派生类实现特定的方法。
- **实现多态**：通过抽象类的指针或引用，可以实现对派生类对象的多态操作。

#### 示例：接口定义

```cpp
#include <iostream>
#include <string>

// 抽象基类 Shape
class Shape {
public:
    virtual double area() const = 0; // 纯虚函数
    virtual void display() const = 0; // 纯虚函数
    virtual ~Shape() {}
};

// 派生类 Circle
class Circle : public Shape {
public:
    Circle(double r) : radius(r) {}
    double area() const override { return 3.14159 * radius * radius; }
    void display() const override { std::cout << "Circle with radius " << radius << "\n"; }
private:
    double radius;
};

// 派生类 Rectangle
class Rectangle : public Shape {
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const override { return width * height; }
    void display() const override { std::cout << "Rectangle with width " << width << " and height " << height << "\n"; }
private:
    double width, height;
};

int main() {
    Shape* s1 = new Circle(5.0);
    Shape* s2 = new Rectangle(4.0, 6.0);
    
    s1->display(); // 输出 Circle 的信息
    std::cout << "Area: " << s1->area() << "\n";
    
    s2->display(); // 输出 Rectangle 的信息
    std::cout << "Area: " << s2->area() << "\n";
    
    delete s1;
    delete s2;
    return 0;
}
```

#### 输出结果

```
Circle with radius 5
Area: 78.5398
Rectangle with width 4 and height 6
Area: 24
```

**说明**：

- 类 `Shape` 是一个抽象类，定义了 `area` 和 `display` 两个纯虚函数。
- 类 `Circle` 和类 `Rectangle` 实现了这些纯虚函数，成为具体的类，可以实例化。
- 通过基类指针 `Shape*`，可以指向不同派生类的对象，实现多态。

---

## 3. 多重继承

### 3.1 多重继承的概念

C++ 支持**多重继承**（Multiple Inheritance），即一个派生类可以同时继承多个基类。这允许派生类组合多个基类的特性和行为，实现更加复杂的类层次结构。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) { std::cout << "A 的构造函数被调用，x = " << M_x << "\n"; }
private:
    int M_x;
};

class B {
public:
    B(int y) : M_y(y) { std::cout << "B 的构造函数被调用，y = " << M_y << "\n"; }
private:
    int M_y;
};

class C : public A, public B {
public:
    C(int x, int y, int z) : A(x), B(y), M_z(z) { std::cout << "C 的构造函数被调用，z = " << M_z << "\n"; }
private:
    int M_z;
};

int main() {
    C c(1, 2, 3);
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用，x = 1
B 的构造函数被调用，y = 2
C 的构造函数被调用，z = 3
```

**说明**：

- 类 `C` 同时继承自类 `A` 和类 `B`，因此在创建 `C` 的对象时，首先调用基类 `A` 的构造函数，然后调用基类 `B` 的构造函数，最后调用派生类 `C` 的构造函数。

### 3.2 多重继承中的问题

多重继承虽然强大，但也带来了一些问题，最典型的是**菱形继承**（Diamond Inheritance）问题。

#### 菱形继承问题

菱形继承指的是当两个基类继承自同一个基类，而一个派生类同时继承自这两个基类时，会导致基类成员出现多份副本，产生二义性和资源浪费。

#### 类图示例

```
      A
     / \
    B   C
     \ /
      D
```

在这个结构中，类 `D` 继承自类 `B` 和类 `C`，而类 `B` 和类 `C` 都继承自类 `A`。这样，类 `D` 中会有两份类 `A` 的成员。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    void display() const { std::cout << "A::display()\n"; }
};

class B : public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
};

class C : public A {
public:
    C() { std::cout << "C 的构造函数被调用\n"; }
};

class D : public B, public C {
public:
    D() { std::cout << "D 的构造函数被调用\n"; }
};

int main() {
    D d;
    // d.display(); // 编译错误，二义性：B::A::display 或 C::A::display
    d.B::display(); // 指定调用 B 继承的 A::display
    d.C::display(); // 指定调用 C 继承的 A::display
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
A 的构造函数被调用
C 的构造函数被调用
D 的构造函数被调用
A::display()
A::display()
```

**问题分析**：

- 类 `D` 中有两份类 `A` 的成员，一份来自类 `B`，另一份来自类 `C`。这会导致资源浪费和二义性问题。
- 调用 `d.display()` 会导致编译错误，因为编译器无法确定调用哪一份 `A::display` 方法。

### 3.3 虚继承解决菱形继承问题

为了避免菱形继承带来的问题，C++ 提供了**虚继承**（Virtual Inheritance）。通过将继承声明为虚继承，派生类共享同一份基类成员，从而避免重复。

#### 虚继承示例

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    void display() const { std::cout << "A::display()\n"; }
};

class B : virtual public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
};

class C : virtual public A {
public:
    C() { std::cout << "C 的构造函数被调用\n"; }
};

class D : public B, public C {
public:
    D() { std::cout << "D 的构造函数被调用\n"; }
};

int main() {
    D d;
    d.display(); // 不再有二义性
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
C 的构造函数被调用
D 的构造函数被调用
A::display()
```

**说明**：

- 通过在类 `B` 和类 `C` 的继承声明中加上 `virtual`，类 `D` 中只有一份类 `A` 的成员。
- 调用 `d.display()` 不再有二义性，因为只有一份类 `A` 的成员。

### 3.4 虚继承中的构造函数调用

在虚继承中，基类的构造函数由最底层的派生类负责调用，而其他派生类不会再调用基类的构造函数。这确保了基类成员只被初始化一次，避免了重复。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) { std::cout << "A 的构造函数被调用，x = " << M_x << "\n"; }
private:
    int M_x;
};

class B : virtual public A {
public:
    B(int x, int y) : A(x), M_y(y) { std::cout << "B 的构造函数被调用，y = " << M_y << "\n"; }
private:
    int M_y;
};

class C : virtual public A {
public:
    C(int x, int z) : A(x), M_z(z) { std::cout << "C 的构造函数被调用，z = " << M_z << "\n"; }
private:
    int M_z;
};

class D : public B, public C {
public:
    D(int x, int y, int z, int w) : A(x), B(x, y), C(x, z), M_w(w) { std::cout << "D 的构造函数被调用，w = " << M_w << "\n"; }
private:
    int M_w;
};

int main() {
    D d(1, 2, 3, 4);
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用，x = 1
B 的构造函数被调用，y = 2
C 的构造函数被调用，z = 3
D 的构造函数被调用，w = 4
```

**说明**：

- 类 `B` 和类 `C` 虚继承自类 `A`。
- 类 `D` 的构造函数通过初始化列表调用基类 `A` 的构造函数，仅调用一次。
- 类 `B` 和类 `C` 的构造函数也会调用基类 `A` 的构造函数，但由于虚继承，基类 `A` 的构造函数只会被调用一次，由最底层的派生类 `D` 负责调用。

---

## 4. 抽象类

### 4.1 抽象类的概念

**抽象类**（Abstract Class）是一种包含至少一个纯虚函数（Pure Virtual Function）的类。抽象类不能被实例化，只能作为基类使用，定义接口供派生类实现。

**纯虚函数**：在类中声明但不定义的虚函数，使用 `= 0` 语法表示。

#### 示例代码

```cpp
#include <iostream>
#include <string>

// 抽象基类 A
class A {
public:
    virtual void f(int i) = 0; // 纯虚函数
    virtual ~A() {}
};

// 派生类 B，实现纯虚函数
class B : public A {
public:
    void f(int i) override {
        std::cout << "B::f(" << i << ")\n";
    }
};

int main() {
    // A a; // 错误：无法实例化抽象类
    B b;
    b.f(10); // 正常调用
    return 0;
}
```

#### 输出结果

```
B::f(10)
```

### 4.2 抽象类的用途

- **定义接口**：抽象类可以用来定义接口，确保所有派生类实现特定的方法。
- **实现多态**：通过抽象类的指针或引用，可以实现对派生类对象的多态操作。

#### 示例：接口定义

```cpp
#include <iostream>
#include <string>

// 抽象基类 Shape
class Shape {
public:
    virtual double area() const = 0; // 纯虚函数
    virtual void display() const = 0; // 纯虚函数
    virtual ~Shape() {}
};

// 派生类 Circle
class Circle : public Shape {
public:
    Circle(double r) : radius(r) {}
    double area() const override { return 3.14159 * radius * radius; }
    void display() const override { std::cout << "Circle with radius " << radius << "\n"; }
private:
    double radius;
};

// 派生类 Rectangle
class Rectangle : public Shape {
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const override { return width * height; }
    void display() const override { std::cout << "Rectangle with width " << width << " and height " << height << "\n"; }
private:
    double width, height;
};

int main() {
    Shape* s1 = new Circle(5.0);
    Shape* s2 = new Rectangle(4.0, 6.0);
    
    s1->display(); // 输出 Circle 的信息
    std::cout << "Area: " << s1->area() << "\n";
    
    s2->display(); // 输出 Rectangle 的信息
    std::cout << "Area: " << s2->area() << "\n";
    
    delete s1;
    delete s2;
    return 0;
}
```

#### 输出结果

```
Circle with radius 5
Area: 78.5398
Rectangle with width 4 and height 6
Area: 24
```

**说明**：

- 类 `Shape` 是一个抽象类，定义了 `area` 和 `display` 两个纯虚函数。
- 类 `Circle` 和类 `Rectangle` 实现了这些纯虚函数，成为具体的类，可以实例化。
- 通过基类指针 `Shape*`，可以指向不同派生类的对象，实现多态。

---

## 5. 多重继承

### 5.1 多重继承的概念

C++ 支持**多重继承**（Multiple Inheritance），即一个派生类可以同时继承多个基类。这允许派生类组合多个基类的特性和行为，实现更加复杂的类层次结构。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) { std::cout << "A 的构造函数被调用，x = " << M_x << "\n"; }
private:
    int M_x;
};

class B {
public:
    B(int y) : M_y(y) { std::cout << "B 的构造函数被调用，y = " << M_y << "\n"; }
private:
    int M_y;
};

class C : public A, public B {
public:
    C(int x, int y, int z) : A(x), B(y), M_z(z) { std::cout << "C 的构造函数被调用，z = " << M_z << "\n"; }
private:
    int M_z;
};

int main() {
    C c(1, 2, 3);
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用，x = 1
B 的构造函数被调用，y = 2
C 的构造函数被调用，z = 3
```

**说明**：

- 类 `C` 同时继承自类 `A` 和类 `B`，因此在创建 `C` 的对象时，首先调用基类 `A` 的构造函数，然后调用基类 `B` 的构造函数，最后调用派生类 `C` 的构造函数。

### 5.2 多重继承中的问题

多重继承虽然强大，但也带来了一些问题，最典型的是**菱形继承**（Diamond Inheritance）问题。

#### 菱形继承问题

菱形继承指的是当两个基类继承自同一个基类，而一个派生类同时继承自这两个基类时，会导致基类成员出现多份副本，产生二义性和资源浪费。

#### 类图示例

```
      A
     / \
    B   C
     \ /
      D
```

在这个结构中，类 `D` 继承自类 `B` 和类 `C`，而类 `B` 和类 `C` 都继承自类 `A`。这样，类 `D` 中会有两份类 `A` 的成员。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    void display() const { std::cout << "A::display()\n"; }
};

class B : public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
};

class C : public A {
public:
    C() { std::cout << "C 的构造函数被调用\n"; }
};

class D : public B, public C {
public:
    D() { std::cout << "D 的构造函数被调用\n"; }
};

int main() {
    D d;
    // d.display(); // 编译错误，二义性：B::A::display 或 C::A::display
    d.B::display(); // 指定调用 B 继承的 A::display
    d.C::display(); // 指定调用 C 继承的 A::display
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
A 的构造函数被调用
C 的构造函数被调用
D 的构造函数被调用
A::display()
A::display()
```

**问题分析**：

- 类 `D` 中有两份类 `A` 的成员，一份来自类 `B`，另一份来自类 `C`。这会导致资源浪费和二义性问题。
- 调用 `d.display()` 会导致编译错误，因为编译器无法确定调用哪一份 `A::display` 方法。

### 5.3 虚继承解决菱形继承问题

为了避免菱形继承带来的问题，C++ 提供了**虚继承**（Virtual Inheritance）。通过将继承声明为虚继承，派生类共享同一份基类成员，从而避免重复。

#### 虚继承示例

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    void display() const { std::cout << "A::display()\n"; }
};

class B : virtual public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
};

class C : virtual public A {
public:
    C() { std::cout << "C 的构造函数被调用\n"; }
};

class D : public B, public C {
public:
    D() { std::cout << "D 的构造函数被调用\n"; }
};

int main() {
    D d;
    d.display(); // 不再有二义性
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
C 的构造函数被调用
D 的构造函数被调用
A::display()
```

**说明**：

- 通过在类 `B` 和类 `C` 的继承声明中加上 `virtual`，类 `D` 中只有一份类 `A` 的成员。
- 调用 `d.display()` 不再有二义性，因为只有一份类 `A` 的成员。

### 5.4 虚继承中的构造函数调用

在虚继承中，基类的构造函数由最底层的派生类负责调用，而其他派生类不会再调用基类的构造函数。这确保了基类成员只被初始化一次，避免了重复。

#### 示例代码

```cpp
#include <iostream>

class A {
public:
    A(int x) : M_x(x) { std::cout << "A 的构造函数被调用，x = " << M_x << "\n"; }
private:
    int M_x;
};

class B : virtual public A {
public:
    B(int x, int y) : A(x), M_y(y) { std::cout << "B 的构造函数被调用，y = " << M_y << "\n"; }
private:
    int M_y;
};

class C : virtual public A {
public:
    C(int x, int z) : A(x), M_z(z) { std::cout << "C 的构造函数被调用，z = " << M_z << "\n"; }
private:
    int M_z;
};

class D : public B, public C {
public:
    D(int x, int y, int z, int w) : A(x), B(x, y), C(x, z), M_w(w) { std::cout << "D 的构造函数被调用，w = " << M_w << "\n"; }
private:
    int M_w;
};

int main() {
    D d(1, 2, 3, 4);
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用，x = 1
B 的构造函数被调用，y = 2
C 的构造函数被调用，z = 3
D 的构造函数被调用，w = 4
```

**说明**：

- 类 `B` 和类 `C` 虚继承自类 `A`。
- 类 `D` 的构造函数通过初始化列表调用基类 `A` 的构造函数，仅调用一次。
- 类 `B` 和类 `C` 的构造函数也会调用基类 `A` 的构造函数，但由于虚继承，基类 `A` 的构造函数只会被调用一次，由最底层的派生类 `D` 负责调用。

---

## 6. 多态

### 6.1 多态的概念

**多态性**（Polymorphism）是面向对象编程的核心概念之一，指的是同一个操作作用于不同的对象，可以产生不同的行为。多态性使得程序在运行时能够根据对象的实际类型来决定调用哪个方法，实现更灵活和可扩展的代码。

### 6.2 静态绑定与动态绑定

- **静态绑定**（Static Binding）：在编译时决定调用哪个方法，通常发生在非虚函数或通过对象直接调用方法时。
    
- **动态绑定**（Dynamic Binding）：在运行时决定调用哪个方法，通常发生在通过基类指针或引用调用虚函数时。
    

#### 静态绑定示例

```cpp
#include <iostream>

class A {
public:
    void affiche() const {
        std::cout << "class A\n";
    }
};

class B : public A {
public:
    void affiche() const {
        std::cout << "class B\n";
    }
};

void myfunc(const A& a) {
    a.affiche(); // 静态绑定，调用 A::affiche
}

int main() {
    B b;
    myfunc(b); // 输出 class A
    return 0;
}
```

#### 输出结果

```
class A
```

**说明**：

- 由于 `affiche` 方法不是虚函数，调用 `a.affiche()` 时，编译器根据引用类型 `A` 决定调用基类的方法，即使实际对象是 `B`，也不会调用派生类的方法。

#### 动态绑定示例

```cpp
#include <iostream>

class A {
public:
    virtual void affiche() const {
        std::cout << "class A\n";
    }
};

class B : public A {
public:
    void affiche() const override {
        std::cout << "class B\n";
    }
};

void myfunc(const A& a) {
    a.affiche(); // 动态绑定，调用实际类型的方法
}

int main() {
    B b;
    myfunc(b); // 输出 class B
    return 0;
}
```

#### 输出结果

```
class B
```

**说明**：

- 由于 `affiche` 方法被声明为 `virtual`，调用 `a.affiche()` 时，编译器根据实际对象类型决定调用派生类的方法，实现动态绑定。

### 6.3 多态的实现（虚函数）

在 C++ 中，通过使用 `virtual` 关键字，可以将基类的方法声明为虚函数，从而实现动态绑定。派生类可以重写（override）基类的虚函数，以提供不同的实现。

#### 示例代码

```cpp
#include <iostream>
#include <string>

class A {
public:
    virtual void affiche() const {
        std::cout << "class A\n";
    }
    
    void afficheAutre() const {
        std::cout << "Autre\n";
    }
};

class B : public A {
public:
    void affiche() const override { // 重写基类的虚函数
        std::cout << "class B\n";
    }
    
    void afficheBase() const {
        A::affiche(); // 调用基类的方法
    }
};

int main() {
    B b;
    A* aPtr = new B();
    
    aPtr->affiche();       // 动态绑定，输出 class B
    aPtr->afficheAutre();  // 静态绑定，输出 Autre
    
    delete aPtr;
    return 0;
}
```

#### 输出结果

```
class B
Autre
```

**说明**：

- `affiche` 方法被声明为 `virtual`，所以通过基类指针调用时，会根据实际对象类型调用派生类的方法。
- `afficheAutre` 方法不是虚函数，调用时基于引用类型 `A` 进行静态绑定，始终调用基类的方法。

### 6.4 多态的优势与注意事项

**优势**：

- **灵活性**：允许使用基类指针或引用指向派生类对象，实现统一接口。
- **可扩展性**：新增派生类时，不需要修改现有代码，只需实现基类接口即可。
- **代码复用**：基类中实现的通用功能可以被所有派生类共享。

**注意事项**：

- **基类析构函数必须为虚函数**，以确保通过基类指针删除派生类对象时能正确调用派生类的析构函数，避免资源泄漏。
- **虚函数的使用**会引入一定的性能开销，因为需要在运行时进行动态绑定。
- **避免过度使用**：过度依赖多态可能导致设计复杂，需要合理设计类的继承关系。

### 6.5 多态与析构函数

在继承关系中，如果基类的析构函数不是虚函数，通过基类指针删除派生类对象时，只会调用基类的析构函数，派生类的析构函数不会被调用，可能导致资源泄漏。

#### 示例：非虚析构函数

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    ~A() { std::cout << "A 的析构函数被调用\n"; }
};

class B : public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
    ~B() { std::cout << "B 的析构函数被调用\n"; }
};

int main() {
    A* a = new B();
    delete a; // 只调用 A 的析构函数，B 的析构函数不会被调用
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
A 的析构函数被调用
```

#### 示例：虚析构函数

```cpp
#include <iostream>

class A {
public:
    A() { std::cout << "A 的构造函数被调用\n"; }
    virtual ~A() { std::cout << "A 的析构函数被调用\n"; }
};

class B : public A {
public:
    B() { std::cout << "B 的构造函数被调用\n"; }
    ~B() override { std::cout << "B 的析构函数被调用\n"; }
};

int main() {
    A* a = new B();
    delete a; // 先调用 B 的析构函数，再调用 A 的析构函数
    return 0;
}
```

#### 输出结果

```
A 的构造函数被调用
B 的构造函数被调用
B 的析构函数被调用
A 的析构函数被调用
```

**说明**：

- 当基类的析构函数为 `virtual` 时，通过基类指针删除派生类对象，会正确调用派生类的析构函数，实现完整的资源释放。
- 如果基类的析构函数不是 `virtual`，则只能调用基类的析构函数，派生类的析构函数不会被调用，可能导致资源泄漏。

### 6.6 多态在集合中的应用

通过多态，可以创建一个基类指针数组，用来存储不同派生类的对象，实现多态集合。这在需要处理不同类型对象时非常有用，例如图形系统中处理不同形状的对象。

#### 示例代码

```cpp
#include <iostream>
#include <string>

// 基类 Triangle
class Triangle {
public:
    virtual std::string name() const { return "Triangle"; }
    virtual ~Triangle() {}
};

// 派生类 TriangleEquilateral（等边三角形）
class TriangleEquilateral : public Triangle {
public:
    std::string name() const override { return "TriangleEquilateral"; }
};

// 派生类 TriangleIsoceles（等腰三角形）
class TriangleIsoceles : public Triangle {
public:
    std::string name() const override { return "TriangleIsoceles"; }
};

// 派生类 TriangleRectangle（直角三角形）
class TriangleRectangle : public Triangle {
public:
    std::string name() const override { return "TriangleRectangle"; }
    double lengthHypotenuse() const { return 1.23; }
};

int main() {
    // 创建基类指针数组，指向不同派生类对象
    Triangle** listTriangle = new Triangle*[10];
    for (int k = 0; k < 10; ++k) {
        if (k % 3 == 0)
            listTriangle[k] = new TriangleRectangle();
        else if (k % 3 == 1)
            listTriangle[k] = new TriangleEquilateral();
        else
            listTriangle[k] = new TriangleIsoceles();
    }
    
    // 调用各个对象的方法
    for (int k = 0; k < 10; ++k)
        std::cout << listTriangle[k]->name() << std::endl;
    
    // 释放内存
    for (int k = 0; k < 10; ++k)
        delete listTriangle[k];
    delete[] listTriangle;
    
    return 0;
}
```

#### 输出结果

```
TriangleRectangle
TriangleEquilateral
TriangleIsoceles
TriangleRectangle
TriangleEquilateral
TriangleIsoceles
TriangleRectangle
TriangleEquilateral
TriangleIsoceles
TriangleRectangle
```

### 6.7 类型转换（dynamic_cast 和 static_cast）

在多态编程中，常常需要将基类指针或引用转换为派生类类型，以访问派生类特有的方法或属性。C++ 提供了几种类型转换运算符：

- **dynamic_cast**：用于安全的向下转换（将基类指针或引用转换为派生类类型），需要基类有虚函数。运行时会检查转换是否合法，如果不合法，返回 `nullptr`（指针）或抛出异常（引用）。
- **static_cast**：用于编译时类型转换，不进行运行时检查。适用于确定转换合法的情况。

#### dynamic_cast 示例

```cpp
#include <iostream>
#include <string>

class Triangle {
public:
    virtual std::string name() const { return "Triangle"; }
    virtual ~Triangle() {}
};

class TriangleRectangle : public Triangle {
public:
    std::string name() const override { return "TriangleRectangle"; }
    double lengthHypotenuse() const { return 1.23; }
};

int main() {
    Triangle* t = new TriangleRectangle();
    
    // 尝试将基类指针转换为派生类指针
    TriangleRectangle* tr = dynamic_cast<TriangleRectangle*>(t);
    if (tr) {
        std::cout << "长度斜边: " << tr->lengthHypotenuse() << "\n";
    } else {
        std::cout << "转换失败\n";
    }
    
    delete t;
    return 0;
}
```

#### 输出结果

```
长度斜边: 1.23
```

#### static_cast 示例

```cpp
#include <iostream>
#include <string>

class Triangle {
public:
    virtual std::string name() const { return "Triangle"; }
    virtual ~Triangle() {}
};

class TriangleRectangle : public Triangle {
public:
    std::string name() const override { return "TriangleRectangle"; }
    double lengthHypotenuse() const { return 1.23; }
};

int main() {
    Triangle* t = new TriangleRectangle();
    
    // 使用 static_cast 进行类型转换
    TriangleRectangle* tr = static_cast<TriangleRectangle*>(t);
    std::cout << "长度斜边: " << tr->lengthHypotenuse() << "\n";
    
    delete t;
    return 0;
}
```

#### 输出结果

```
长度斜边: 1.23
```

**注意**：

- 使用 `dynamic_cast` 时，如果转换失败，返回 `nullptr`（对于指针）或抛出 `std::bad_cast` 异常（对于引用），更加安全。
- 使用 `static_cast` 时，不会进行运行时检查，如果类型不匹配，可能导致未定义行为。

#### dynamic_cast 与 static_cast 的区别

- **dynamic_cast**：
    - 适用于多态类型（基类含有虚函数）。
    - 运行时类型检查，确保类型转换的合法性。
    - 成本较高，涉及运行时检查。
- **static_cast**：
    - 适用于确定类型转换合法的情况。
    - 编译时类型检查，不进行运行时检查。
    - 成本较低，但不安全。

#### 示例代码：动态转换

```cpp
#include <iostream>
#include <string>
#include <typeinfo>

class A {
public:
    virtual void foo() const { std::cout << "A::foo()\n"; }
    virtual ~A() {}
};

class B : public A {
public:
    void foo() const override { std::cout << "B::foo()\n"; }
    void bar() const { std::cout << "B::bar()\n"; }
};

int main() {
    A* a = new B();
    
    // 使用 dynamic_cast 进行类型转换
    B* b = dynamic_cast<B*>(a);
    if (b) {
        b->bar(); // 成功转换，调用 B::bar()
    } else {
        std::cout << "转换失败\n";
    }
    
    // 错误的转换
    A a2;
    B* b2 = dynamic_cast<B*>(&a2);
    if (b2) {
        b2->bar();
    } else {
        std::cout << "转换失败\n";
    }
    
    delete a;
    return 0;
}
```

#### 输出结果

```
B::bar()
转换失败
```

**说明**：

- 第一个 `dynamic_cast` 成功，将基类指针 `a` 转换为派生类指针 `b`，可以调用 `B::bar()`。
- 第二个 `dynamic_cast` 失败，因为对象 `a2` 实际上是基类 `A` 的实例，不是 `B` 的实例。

---

## 6. 进一步学习

以上内容涵盖了 C++ 面向对象编程中继承与多态的基本知识，包括继承的概念、类图示例、C++ 中的继承示例、访问权限在继承中的作用、继承方式（public、protected、private）、构造函数在继承中的调用、拷贝构造函数与赋值操作符在继承中的处理、析构函数与继承、方法重写与方法重载、多态的概念、静态绑定与动态绑定、多态的实现（虚函数）、多态的优势与注意事项、多态与析构函数、多态在集合中的应用以及类型转换（dynamic_cast、static_cast）。

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

### 进一步的学习方向

- **继承与多态的高级应用**：
    
    - 了解虚继承的更多细节和使用场景。
    - 学习如何设计类的继承体系，避免不必要的复杂性。
- **抽象类与接口**：
    
    - 掌握如何通过抽象类定义接口，设计可扩展的系统。
- **模板编程**：
    
    - 学习 C++ 模板的使用，编写通用和可复用的代码。
- **标准模板库（STL）**：
    
    - 深入学习 C++ 提供的各种容器（如 `vector`、`list`、`map` 等）、算法和迭代器，提升编程效率。
- **异常处理**：
    
    - 学习如何在 C++ 中处理异常，提高程序的健壮性。
- **现代 C++ 特性**：
    
    - 掌握 C++11、C++14、C++17、C++20 等新标准提供的新功能，如智能指针、Lambda 表达式、并行编程等。

**祝您在 C++ 面向对象编程的学习中取得更大的进步！如果有任何疑问，请随时联系我。**
