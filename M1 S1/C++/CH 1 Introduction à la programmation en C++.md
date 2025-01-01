# C++ 编程入门

## 课程大纲

1. C++ 编程简介
2. 类和对象
3. 继承与多态
4. 输入输出流
5. 泛型编程与元编程
6. 标准模板库（STL）
7. 异常处理、现代 C++ 和 CMake

## 评估方式

- 两次考试
- 项目

---

## 1. C++ 编程简介

### 编程的基本概念

计算机程序是一系列由计算机执行的指令。然而，计算机只能理解二进制代码，即由0和1组成的序列。为了让人类能够更方便地与计算机交流，计算机科学家们开发了各种编程语言。这些编程语言比二进制更接近人类的思维方式，易于理解和使用。目前有数百种编程语言，每种语言都有其特定的用途和优势。

### 编程的两大步骤

编写一个程序通常包括以下两个步骤：

1. **编写源代码**：使用编程语言编写指令和逻辑。
2. **编译源代码**：使用编译器将源代码转换为计算机可以执行的二进制代码。

### 编程语言的分类

#### 高级语言

高级编程语言更接近人类的自然语言，使用常用词汇和数学符号，使编程更加直观。C++ 就是一个结合了高级和低级语言特性的语言。

- **高级特性**：如标准模板库（STL）、丰富的外部库、现代 C++ 标准（如 C++20）等。
- **低级特性**：如内存管理、指针、类型控制等。

#### 低级语言

低级编程语言更接近计算机的硬件，允许更精细地控制计算机资源。这类语言的执行速度通常非常快，但编写起来较为复杂。

### 解释型语言与编译型语言

- **解释型语言**：源代码由解释器逐行解释执行，无需事先编译。优点是跨平台，但执行速度较慢。示例：Python、Ruby、PHP。
- **编译型语言**：源代码由编译器一次性编译成二进制代码，再执行。通常执行速度较快。示例：C、C++、Fortran。
- **中间语言**：如 Java，源代码先编译成中间字节码，再由虚拟机解释执行或即时编译。

### 为什么选择 C++

C++ 是一种多范式的编程语言，结合了过程式编程、面向对象编程和泛型编程等多种编程方式。它具有高性能，广泛应用于游戏引擎、桌面应用、浏览器、科学计算等领域。此外，C++ 标准和库不断发展，增加了更多功能和优化，使其具有很强的扩展性和适应性。

### 现代 C++ 的优势

现代 C++（从 C++11 开始）引入了许多新特性，提升了语言的表达力和性能：

- **更简洁的语法**：如自动类型推断（auto）。
- **性能优化**：如移动语义（move semantics）。
- **更强大的泛型编程**：如概念（concepts）。
- **增强的安全性**：如智能指针（smart pointers）。
- **并行与异步编程**：简化多线程编程。
- **向后兼容**：保留了与旧版本 C++ 的兼容性。

### 编译器简介

C++ 程序需要通过编译器将源代码转换为可执行的二进制代码。常用的 C++ 编译器包括：

- **GCC（GNU Compiler Collection）**：支持 Windows、Linux、MacOS。
- **Clang**：支持 Windows、Linux、MacOS。
- **Microsoft Visual C++**：专为 Windows 设计。
- **Borland C++ Builder**：专为 Windows 设计。
- **Intel C++ Compiler**：支持 Windows、Linux、MacOS。

---

## 2. 示例程序

### 经典的“Hello, World”程序

这是一个最简单的 C++ 程序，目的是在屏幕上显示“Hello, World”消息。

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello, world" << std::endl;
    return 0;
}
```

- **第1行**：包含输入输出库，允许使用 `std::cout` 进行输出。
- **第2行**：定义主函数，程序的入口点。
- **第4行**：使用 `std::cout` 输出字符串到屏幕，并换行。
- **第5行**：返回 0，表示程序正常结束。

#### 编译与执行

- **编译**：

    ```bash
    g++ hello.cpp -o hello
    ```

- **执行**：

    ```bash
    ./hello
    ```

    输出：

    ```
    Hello, world
    ```

### 编译过程详解

C++ 程序的编译过程包括以下几个步骤：

1. **预处理（Preprocessing）**：处理以 `#` 开头的指令，如 `#include`。
2. **编译（Compilation）**：将预处理后的代码转换为汇编代码。
3. **汇编（Assembly）**：将汇编代码转换为机器码（目标文件）。
4. **链接（Linking）**：将目标文件与库文件链接，生成最终的可执行文件。

#### 查看编译的每个阶段结果

- **预处理**：

    ```bash
    g++ -E hello.cpp > hello.E
    ```

- **生成汇编代码**：

    ```bash
    g++ -S hello.cpp -o hello.s
    ```

- **生成目标文件**：

    ```bash
    g++ -c hello.cpp -o hello.o
    ```

- **链接生成可执行文件**：

    ```bash
    g++ hello.o -o hello
    ```

### 模块化编程

当程序规模较大时，需要将代码分割成多个模块，分别存放在不同的文件中（头文件和源文件），以便于管理和复用。

#### 代码组织方式

- **头文件（Headers）**：通常以 `.h` 或 `.hpp` 结尾，包含函数声明、类定义等。
- **源文件（Sources）**：通常以 `.cpp` 或 `.cxx` 结尾，包含函数定义、类实现等。

### 示例：多个文件

#### `myfunc.hpp`

```cpp
#ifndef __MYFUNC_HPP__
#define __MYFUNC_HPP__

void mymsg();

#endif
```

#### `myfunc.cpp`

```cpp
#include "myfunc.hpp"
#include <iostream>

void mymsg()
{
    std::cout << "Hello, world\n";
}
```

#### `mymain.cpp`

```cpp
#include "myfunc.hpp"

int main()
{
    mymsg();
    return 0;
}
```

#### 编译与链接

```bash
g++ -c myfunc.cpp -o myfunc.o
g++ -c mymain.cpp -o mymain.o
g++ myfunc.o mymain.o -o myprog
```

#### 执行

```bash
./myprog
```

输出：

```
Hello, world
```

### 使用类

#### 示例类 `Point`

```cpp
#include <iostream>

class Point {
public:
    Point(double x, double y, double z)
    : M_x(x), M_y(y), M_z(z)
    {}
    
    double x() const { return M_x; }
    double y() const { return M_y; }
    double z() const { return M_z; }
    
    Point& operator+=(Point const& pt)
    {
        M_x += pt.x();
        M_y += pt.y();
        M_z += pt.z();
        return *this;
    }
    
private:
    double M_x, M_y, M_z;
};

int main()
{
    Point p(5.2, 10.1, 30);
    Point q(10.5, 20, 36.3);
    p += q;
    std::cout << "x=" << p.x() << " y=" << p.y() << " z=" << p.z() << "\n";
    return 0;
}
```

#### 输出

```
x=15.7 y=30.1 z=66.3
```

### 使用模板

#### 示例模板函数 `max`

```cpp
#include <iostream>

template <typename T>
T max(T const& A, T const& B)
{
    return (A >= B) ? A : B;
}

int main()
{
    double x = 3.12, y = 6.3;
    double dmax = max(x, y);
    std::cout << "dmax=" << dmax << "\n";
    
    int a = 2, b = 3;
    int imax = max(a, b);
    std::cout << "imax=" << imax << "\n";
    
    return 0;
}
```

#### 输出

```
dmax=6.3
imax=3
```

### 使用 Lambda 表达式

#### 示例

```cpp
#include <iostream>

int main()
{
    auto q = [] () { std::cout << "我是一个lambda表达式\n"; };
    q();
    
    int a = 2;
    auto p = [&a] (double d) { return a * a + d; };
    double res1 = p(3.14);
    std::cout << "res1=" << res1 << "\n";
    double res2 = p(5.36);
    std::cout << "res2=" << res2 << "\n";
    
    return 0;
}
```

#### 输出

```
我是一个lambda表达式
res1=7.14
res2=9.36
```

### 编译选项

- **-std=c++XX**：指定 C++ 标准（如 C++11、C++14、C++17、C++20）。
- **-Wall -Wextra -Wpedantic**：启用所有警告信息。
- **-O0, -O1, -O2, -O3**：优化级别，`-O0` 无优化，`-O3` 最高优化。
- **-g**：包含调试信息。
- **-l**：链接特定的库。
- **-I**：指定头文件搜索路径。
- **-fsanitize=address,undefined**：启用地址检查器和未定义行为检查器。

#### 示例命令

```bash
g++ -std=c++17 -O3 hello.cpp -o hello.o
g++ -std=c++17 -O3 -I/路径/到/include -lfeelpp hello.cpp -o hello.o
```

### C++11 示例

#### 示例代码

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    std::vector<char> nums = { 'h','e','l','l','o',',',' ','w','o','r','l','d' };
    bool useUpper = false;
    if (argc > 1)
        useUpper = true;
    
    // 使用lambda表达式
    std::for_each(nums.begin(), nums.end(),
        [useUpper](char& n) {
            if (useUpper)
                n = std::toupper(n);
        });
    
    // 基于范围的for循环
    for (auto const& n : nums)
        std::cout << n;
    std::cout << std::endl;
    
    return 0;
}
```

#### 新特性

- **自动类型推断（auto）**：自动推断变量类型，减少显式类型声明。
- **Lambda 表达式**：定义匿名函数，提高代码灵活性。
- **基于范围的 for 循环**：更简洁地遍历容器元素。

### C++14 示例

#### 示例代码

```cpp
#include <iostream>
#include <memory>

int main () {
    // 使用auto作为函数返回类型
    auto get_hello = []() -> auto { return "Hello, C++14 World!"; };
    std::cout << get_hello() << std::endl;
    
    // 泛型lambda表达式
    auto generic_print = [](auto&& x) { std::cout << x << std::endl; };
    generic_print("使用泛型lambda");
    generic_print(42);
    
    // std::make_unique用于创建唯一指针
    auto ptr = std::make_unique<std::string>("C++14中的唯一指针");
    std::cout << *ptr << std::endl;
    
    return 0;
}
```

#### 新特性

- **自动返回类型（auto return type）**：允许函数返回类型自动推断。
- **泛型 Lambda**：Lambda 表达式支持泛型参数，提高代码复用性。
- **std::make_unique**：安全地创建唯一指针，管理动态内存。

### C++17 示例

#### 示例代码

```cpp
#include <iostream>
#include <string>
#include <optional>

std::optional<std::string> get_hello(bool include_world) {
    if (include_world) return "Hello, C++17 World!";
    return std::nullopt;
}

int main () {
    if (auto result = get_hello(false); result)
        std::cout << *result << std::endl;
    return 0;
}
```

#### 新特性

- **std::optional**：用于表示可选值，可能有值也可能没有，避免使用裸指针。
- **if 语句中的初始化**：在 `if` 语句中直接初始化变量，简化代码结构。

### C++20 示例

#### 示例代码

```cpp
#include <iostream>
#include <string>
#include <ranges>
#include <algorithm>
#include <concepts>

template <typename T>
concept Printable = requires(T t) {
    { std::cout << t } -> std::convertible_to<std::ostream&>;
};

template <Printable T>
void print(const T& value) {
    std::cout << value << std::endl;
}

template <typename T>
concept StringLike = std::same_as<T, std::string> || std::convertible_to<T, std::string>;

void printInUpperCase(StringLike auto message) {
    auto toUpper = [](char c) { return std::toupper(c); };
    for (char c : message | std::ranges::views::transform(toUpper)) {
        std::cout << c;
    }
    std::cout << std::endl;
}

int main () {
    std::string message = "Hello, World!";
    print(message);
    printInUpperCase(message); // 输出：HELLO, WORLD!
    return 0;
}
```

#### 新特性

- **概念（Concepts）**：用于在模板中约束类型，增强类型安全性和可读性。
- **std::ranges**：用于处理序列的强大工具，简化操作。
- **std::format**：用于格式化字符串，替代传统的 `printf` 风格函数。
- **std::span**：表示连续内存区域的视图，方便处理数组和容器。

---

## 3. 语言基础元素

### 基本数据类型

#### 整数类型

- `int`：通常为32位，存储整数。
- `short int`：通常为16位，存储较小的整数。
- `long int`：在不同平台上可能为32位或64位，存储较大的整数。
- `long long int`：通常为64位，存储更大的整数。
- `unsigned int`：无符号整数，只能存储非负数。

#### 浮点类型

- `float`：单精度浮点数，约7位有效数字。
- `double`：双精度浮点数，约15位有效数字。
- `long double`：扩展精度浮点数，具体精度依编译器和平台而定。

#### 字符类型

- `char`：存储单个字符，基于ASCII编码。
- `signed char`：有符号字符，范围通常为-128到127。
- `unsigned char`：无符号字符，范围通常为0到255。

#### 布尔类型

- `bool`：存储布尔值，`true` 或 `false`。

### 变量

变量是通过名称引用的内存位置，可以存储数据并在程序运行期间修改其值。C++ 中的变量是有类型的，即每个变量都有一个数据类型，决定了它所能存储的数据类型和占用的内存大小。

#### 示例

```cpp
int a;
char b;
a = 3;
b = 'c';
double pi = 3.1415926;
```

#### 常量

常量是值不可更改的变量，使用 `const` 关键字定义。

```cpp
const int A = 26;
const double pi = 3.1415926;
```

### 变量的作用域（可见性）

变量的作用域决定了变量在程序中的可见范围。C++ 中主要有以下几种作用域：

- **全局作用域**：在所有函数外部定义的变量，对整个程序可见。
- **命名空间作用域**：在命名空间内部定义的变量，只在该命名空间内可见。
- **局部作用域**：在函数或代码块内部定义的变量，只在其定义的区域内可见。

#### 示例

```cpp
const int varGlobalConst = 1;
int varGlobal;

namespace SomeVariables {
    const int varGlobalConstNS = 2;
    int varGlobalNS;
}

int main(int argc, char* argv[])
{
    int j = 2;
    if(1 < argc)
    {
        int i = 1;
        j = i;
    }

    SomeVariables::varGlobalNS = varGlobalConst;
    varGlobal = SomeVariables::varGlobalConstNS;
    return 0;
}
```

### 运算符

#### 算术运算符

- **加法**：`+`
- **减法**：`-`
- **乘法**：`*`
- **除法**：`/`（整数除法会舍弃小数部分）
- **取模**：`%`（返回整数除法的余数）

#### 逻辑运算符

- **逻辑非**：`!`
- **逻辑与**：`&&`
- **逻辑或**：`||`

#### 关系运算符

- **小于**：`<`
- **大于**：`>`
- **小于等于**：`<=`
- **大于等于**：`>=`
- **等于**：`==`
- **不等于**：`!=`

#### 赋值运算符

- **基本赋值**：`=`
- **复合赋值**：`+=`、`-=`、`*=`、`/=`、`%=`

#### 自增自减运算符

- **后置自增**：`a++`（先使用变量，再自增）
- **前置自增**：`++a`（先自增，再使用变量）
- **后置自减**：`a--`
- **前置自减**：`--a`

#### 位运算符

- **按位与**：`&`
- **按位或**：`|`
- **按位异或**：`^`

### 枚举类型

#### 传统枚举

枚举（Enumeration）是一种用户定义的类型，包含一组命名的整型常量。

```cpp
enum couleurs { noir, bleu, vert, rouge, blanc, jaune };
// noir == 0, bleu == 1, vert == 2, rouge == 3, blanc == 4, jaune == 5
```

可以手动指定枚举值：

```cpp
enum couleurs { noir = -2, bleu, vert, rouge = 5, blanc, jaune };
// noir == -2, bleu == -1, vert == 0, rouge == 5, blanc == 6, jaune == 7
```

#### 使用枚举

```cpp
couleurs a = couleurs::vert;
std::cout << "couleur vert :" << a << "\n";
```

#### 传统枚举的问题

- **作用域不受控**：枚举值在全局作用域中可见，容易命名冲突。
- **类型不安全**：不同枚举类型之间可以相互比较，可能导致逻辑错误。
- **隐式转换为整数**：枚举值可以自动转换为整数，可能引发错误。

#### 强类型枚举（enum class）

C++11 引入的 `enum class` 解决了传统枚举的上述问题。

```cpp
enum class Fruit { Apple, Banana, Cherry };

int main ()
{
    enum class Color { Red, Green, Blue };

    Color c = Color::Red;
    Fruit f = Fruit::Apple;

    // if (c == f) { // 编译错误：不同类型的枚举不能比较
    // }

    // int i = Color::Red; // 编译错误：不能隐式转换为int
    int i = static_cast<int>(Color::Red); // 需要显式转换

    return 0;
}
```

### 指针

指针用于存储变量的内存地址。一个指针包含两个信息：所指向地址的值和所指类型的大小。声明指向类型 `T` 的指针方式为 `T*`。

#### 基本示例

```cpp
char x = 'A'; // 创建一个char类型的变量
char* p = &x; // 指向x的指针
```

#### 指针的解引用

通过指针可以访问其指向的变量。

```cpp
char y = *p; // y == 'A'
```

#### 空指针

表示不指向任何对象的指针。

```cpp
int *a = nullptr; // 推荐的空指针写法，从C++11开始
```

#### `void*` 指针

`void*` 是通用指针类型，可以存储任何类型变量的地址，但不能解引用。

```cpp
void *ptr = nullptr;
int a = 26;
ptr = &a;
char b = 'b';
ptr = &b;
```

### 引用

引用是某个变量的别名，可以通过引用来修改原变量。引用必须在声明时初始化，并且一旦绑定到某个变量后，不能再指向其他变量。

#### 基本示例

```cpp
int i = 26;
int &j = i;
j = 33; // i == 33, j == 33

int* pi = &i;
int* pj = &j; // pi == pj
```

### 左值引用与右值引用

#### 左值引用（Lvalue References）

引用一个具有持久身份的对象（如命名变量）。使用单个 `&`。

```cpp
int x = 10;
int& lref = x;
```

特点：

- 绑定到持久对象。
- 可以通过引用修改对象。
- 不能绑定到临时对象。

#### 右值引用（Rvalue References）

引用一个临时对象或即将被销毁的值。使用双 `&&`。

```cpp
int&& rref = 20;
```

特点：

- 绑定到临时对象或将被销毁的对象。
- 主要用于实现移动语义和完美转发。
- 优化性能，避免不必要的复制。

#### 右值引用示例

```cpp
#include <iostream>
#include <utility>

int bar() { return 6; }

void foo(int& x) { std::cout << "lvalue reference\n"; }
void foo(int&& x) { std::cout << "rvalue reference\n"; }

int main()
{
    int y = 10;
    foo(y); // 调用 foo(int&)
    foo(20); // 调用 foo(int&&)
    foo(bar()); // 调用 foo(int&&)
    foo(std::move(y)); // 调用 foo(int&&)
    return 0;
}
```

#### 输出

```
lvalue reference
rvalue reference
rvalue reference
rvalue reference
```

`std::move` 用于将对象转换为右值引用，即使它是一个命名变量。

### 数组

数组是一组相同类型的数据，存储在内存中连续的位置。

#### 定义和访问

```cpp
int tab[10]; // 分配10个整数空间

tab[0] = 1;
tab[1] = 3;
int val = tab[3];
```

#### 数组初始化

```cpp
int tab[10] = {1,2,3,4,5,6,7,8,9,10};
```

#### 多维数组

```cpp
int multitab[10][3][5]; // 定义一个三维数组
multitab[0][1][3] = 4; // 初始化
```

#### 数组名作为指针

数组名等同于指向第一个元素的指针。

```cpp
int val1 = tab[5];
int val2 = *(tab + 5);
```

### 内存管理

计算机内存分为栈（stack）和堆（heap）两部分。

- **栈（Stack）**：用于存储自动变量，生命周期由作用域决定，采用后进先出（LIFO）原则。
- **堆（Heap）**：用于动态分配内存，程序员需要手动管理内存，内存容量几乎无限。

#### 动态内存分配

C++ 提供 `new`、`new[]`、`delete` 和 `delete[]` 操作符来管理动态内存。

```cpp
// 单个变量
int* var = new int(10); // 在堆上分配一个整数，并初始化为10
*var = 13; // 修改值为13
delete var; // 释放内存

// 数组
int* tab = new int[10]; // 在堆上分配一个包含10个整数的数组
tab[3] = 13; // 修改第4个元素为13
delete[] tab; // 释放数组内存
```

### 字符串

#### C风格字符串

以字符数组表示，末尾以 `\0` 结束。

```cpp
char hola[5] = {'h','o','l','a','\0'};
char toto[] = "toto";
```

#### `std::string`

C++ 标准模板库提供了 `std::string` 类，方便操作字符串。

```cpp
#include <string>

std::string s1; // 空字符串
std::string s2 = "hello, world"; // 包含12个字符
std::string s3(60, '*'); // 包含60个星号
std::string s4 = s2 + s3; // 字符串连接
std::string s5(s2, 3, 2); // s5包含"lo"
bool isEqual = (s2 == s3); // 比较字符串
const char* s6 = s2.c_str(); // 获取底层字符数组
```

### 条件语句

#### if-else

根据条件执行不同的代码块。

```cpp
int a = 8, b = 0;
if(a > 0)
{
    int c = 8;
    b = 2 * c;
}
else
{
    b = 9;
}

// 三目运算符
int b = (a > 0) ? 3 : 6;
int c = (a > 0) ? myfunc1(3) : myfunc2(6);
```

#### switch

根据变量的值执行不同的代码块。

```cpp
#include <iostream>

int main()
{
    int a;
    std::cout << "请输入a的值：";
    std::cin >> a;

    switch(a)
    {
        case 1:
            std::cout << "a的值是1" << std::endl;
            break;
        case 2:
            std::cout << "a的值是2" << std::endl;
            break;
        case 3:
            std::cout << "a的值是3" << std::endl;
            break;
        default:
            std::cout << "a的值不是1, 2, 或3" << std::endl;
            break;
    }
    return 0;
}
```

### 循环语句

#### for 循环

用于重复执行代码块。

```cpp
// 计数器变量在for中声明
for (int i = 0; i < 2; i++) {
    std::cout << i;
}

// 计数器变量在for外声明
int i;
for (i = 0; i < 2; i++) {
    std::cout << i;
}

// 初始化在for外
int j = 0, k = 10;
for (; j < k; j++) {
    std::cout << j;
}
```

#### 基于范围的 for 循环（C++11）

简化遍历容器或数组的操作。

```cpp
int tableau[] = {2, 3, 5, 8, 13};
for(int i : tableau)
    std::cout << i;

// 部分遍历
for (int m : tableau)
{
    if (m > 5) break;
    std::cout << m;
}
```

#### while 循环

在条件为真时重复执行代码块。

```cpp
int i = 0;
while (i < 10)
{
    std::cout << "i的值是：" << i << std::endl;
    i++;
}
std::cout << "最终i的值是：" << i << std::endl;
```

#### do-while 循环

至少执行一次代码块，然后根据条件决定是否继续。

```cpp
int i = 0;
do {
    std::cout << "i的值是：" << i << std::endl;
    i++;
} while (i < 10);
std::cout << "最终i的值是：" << i << std::endl;
```

#### break 和 continue

- **break**：退出当前循环或 `switch` 语句。

    ```cpp
    for (int i = 0; i < 10; i++)
    {
        std::cout << "i的值是: " << i << std::endl;
        if (i == 5)
            break;
    }
    ```

- **continue**：跳过当前循环的剩余部分，继续下一次迭代。

    ```cpp
    for (int i = 0; i < 10; i++)
    {
        if (i == 5) continue;
        std::cout << "i的值是: " << i << std::endl;
    }
    ```

### 输入输出流

C++ 提供了 `iostream` 库，用于与外部设备（如控制台、文件）进行交互。

- **std::cin**：标准输入（通常是键盘）
- **std::cout**：标准输出（通常是屏幕）
- **std::cerr**：标准错误输出（通常是屏幕）

#### 示例

```cpp
#include <iostream>

int main ()
{
    int n;
    std::cout << "请输入一个正数：";
    std::cin >> n;
    if (n < 0)
        std::cerr << "错误：数字 " << n << " 不是正数\n";
    else
        std::cout << "您输入的数字是 " << n << "\n";
    return 0;
}
```

### 函数

#### 函数的定义

函数是执行特定任务的代码块，可以接受参数并返回值。使用函数有助于代码的组织、复用和维护。

#### 示例

```cpp
#include <iostream>

// 计算a的n次方
double my_pow(double a, unsigned int n)
{
    double res = 1;
    for (int i = 0; i < n; ++i)
        res *= a;
    return res;
}

int main()
{
    std::cout << "2的6次方 = " << my_pow(2.0, 6) << "\n";
    return 0;
}
```

#### 函数语法

```cpp
返回类型 函数名(参数类型1 参数1, 参数类型2 参数2, ...) {
    // 函数体
}
```

- **返回类型**：函数返回值的类型，若不返回值则使用 `void`。
- **函数名**：遵循变量命名规则，区分大小写。
- **参数**：传递给函数的数据，可以有多个参数。

#### 声明与定义

- **声明（Declaration）**：告诉编译器函数的存在和其签名。

    ```cpp
    int min(int, int); // 函数声明
    ```

- **定义（Definition）**：提供函数的具体实现。

    ```cpp
    int min(int i, int j)
    {
        return (i < j) ? i : j;
    }
    ```

### 参数传递方式

#### 按值传递

参数值被复制到函数的局部变量中，函数内部的修改不会影响原变量。

```cpp
#include <iostream>

void myfunc(int n)
{
    n = 3;
}

int main()
{
    int k = 6;
    myfunc(k);
    std::cout << "k = " << k << "\n"; // 输出 k = 6
    return 0;
}
```

#### 按引用传递

通过引用传递参数，函数内部对参数的修改会影响原变量。

```cpp
#include <iostream>

void myfunc(int& n)
{
    n = 3;
}

int main()
{
    int k = 6;
    myfunc(k);
    std::cout << "k = " << k << "\n"; // 输出 k = 3
    return 0;
}
```

#### 右值引用传递

用于传递临时对象或可被移动的对象，提高性能。

```cpp
#include <iostream>
#include <utility>

int bar() { return 6; }

void foo(int& x) { std::cout << "lvalue reference\n"; }
void foo(int&& x) { std::cout << "rvalue reference\n"; }

int main()
{
    int y = 10;
    foo(y); // 调用 foo(int&)
    foo(20); // 调用 foo(int&&)
    foo(bar()); // 调用 foo(int&&)
    foo(std::move(y)); // 调用 foo(int&&)
    return 0;
}
```

#### 输出

```
lvalue reference
rvalue reference
rvalue reference
rvalue reference
```

`std::move` 用于将对象转换为右值引用，即使它是一个命名变量。

### 指针和数组作为参数

函数参数可以是指针或数组，数组参数实际上是指针。

```cpp
#include <iostream>

void f(char* c);
void f(char c[]);
void f(char c[16]);

int main()
{
    char arr[10];
    f(arr);
    return 0;
}
```

### main 函数的参数

`main` 函数可以接受命令行参数：

```cpp
#include <iostream>

int main(int argc, char* argv[])
{
    std::cout << "argc = " << argc << "\n";
    for(int k = 0; k < argc; ++k)
        std::cout << "argv[" << k << "] : " << argv[k] << "\n";
    return 0;
}
```

#### 执行示例

```bash
./myprog abc toto
```

#### 输出

```
argc = 3
argv[0] : ./myprog
argv[1] : abc
argv[2] : toto
```

### 可选参数

函数可以为参数指定默认值，默认参数必须从右到左排列。

```cpp
int myfunc(int a, int b = 1) { return a * b; }

int main()
{
    int res1 = myfunc(2); // res1 = 2
    int res2 = myfunc(3, 5); // res2 = 15
    return 0;
}
```

#### 注意事项

- 默认参数只能在函数声明中指定，定义中不应重复指定。
- 默认参数必须从右到左排列，不能在中间有非默认参数。

```cpp
// 声明
int nombreDeSecondes(int heures, int minutes = 0, int secondes = 0);

// 定义
int nombreDeSecondes(int heures, int minutes, int secondes)
{
    int total = heures * 3600 + minutes * 60 + secondes;
    return total;
}
```

### 函数重载

允许多个同名函数，前提是参数列表不同。编译器根据参数类型和数量决定调用哪个函数。

```cpp
#include <iostream>

void showMe(int a) {
    std::cout << "我是一个int: " << a << "\n";
}

void showMe(double a) {
    std::cout << "我是一个double: " << a << "\n";
}

int main()
{
    showMe(3); // 调用 showMe(int)
    showMe(5.3); // 调用 showMe(double)
    return 0;
}
```

#### 输出

```
我是一个int: 3
我是一个double: 5.3
```

### 递归函数

递归函数是指在函数体内调用自身的函数。递归常用于解决分治问题，但需要注意防止无限递归和栈溢出。

#### 示例：斐波那契数列

```cpp
#include <iostream>

int fibo(int n)
{
    if (n <= 2)
        return 1;
    else
        return fibo(n - 1) + fibo(n - 2);
}

int main()
{
    std::cout << "fibo(10) = " << fibo(10) << "\n"; // 输出 55
    return 0;
}
```

#### 输出

```
fibo(10) = 55
```

#### 注意事项

- **终止条件**：递归必须有终止条件，否则会导致无限递归。
- **空间复杂度**：每次递归调用都会占用栈空间，过深的递归可能导致栈溢出。
- **时间复杂度**：某些递归算法（如斐波那契数列）可能比迭代算法效率低。

### 函数指针

函数指针是指向函数的指针，可以通过指针调用函数。

#### 基本示例

```cpp
#include <iostream>

int f(int x, int y) { return x + y; }

int main()
{
    int (*pf)(int, int) = f; // 定义函数指针
    std::cout << "pf(2, 3) = " << pf(2, 3) << "\n"; // 输出 5
    return 0;
}
```

#### 函数指针作为参数

```cpp
#include <iostream>
#include <cmath>

double carre(double x) { return x * x; }
double inverse(double x) { return 1 / x; }
double racine(double x) { return sqrt(x); }

double minimum(double a, double b, double(*f)(double))
{
    double minVal = 100000;
    for(double x = a; x < b; x += 0.01)
        minVal = (minVal < f(x)) ? minVal : f(x);
    return minVal;
}

int main()
{
    int type;
    std::cout << "输入函数类型: 1=平方, 2=倒数, 3=平方根\n";
    std::cin >> type;
    double (*monPointeur)(double);
    switch(type) {
        case 1: monPointeur = carre; break;
        case 2: monPointeur = inverse; break;
        case 3: monPointeur = racine; break;
    }
    std::cout << "函数在2到3之间的最小值是: " << minimum(2, 3, monPointeur) << "\n";
    return 0;
}
```

#### 输出示例

```
输入函数类型: 1=平方, 2=倒数, 3=平方根
2
函数在2到3之间的最小值是: 0.333333
```

### 内联函数

`inline` 关键字建议编译器将函数体嵌入到调用处，减少函数调用开销，提高性能。

#### 示例

```cpp
inline int max(int i, int j)
{
    return (i > j) ? i : j;
}
```

#### 注意事项

- **递归函数不能内联**。
- **内联函数不能使用函数指针**。
- **内联只是建议**，编译器可能会忽略。

---

## 4. 进一步学习

以上内容涵盖了C++编程的基础知识，包括基本数据类型、变量、运算符、条件语句、循环、函数、指针和引用等。通过理解和掌握这些基础知识，您可以开始编写简单的C++程序，并为学习更高级的编程概念打下坚实的基础。随着课程的深入，您将学习到更多关于类和对象、继承与多态、泛型编程、标准模板库（STL）以及现代C++的新特性。

---

希望以上内容能够帮助您更好地理解C++编程的基础知识。如有任何疑问，请随时联系我。

祝学习愉快！