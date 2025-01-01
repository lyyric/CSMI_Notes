好的，我已经删除了重复的内容，并对章节进行了重新编号。以下是整理后的《C++ 输入输出流详解》课程内容：

---

# C++ 输入输出流详解

## 目录

1. 输入输出流的概念
2. 类层次结构
3. 输出流：`std::ostream` 类
4. 输入流：`std::istream` 类
5. 标准输入输出流
6. 文件流
7. 字符串流
8. 流的状态
9. 格式化输出
10. 在流中使用自定义类

---

## 1. 输入输出流的概念

### 1.1 流（Stream）的概念

在编程中，**流**（或称为**流对象**）是用于处理数据输入和输出的媒介。流可以看作是数据的通道，负责接收（输入流）或发送（输出流）信息。通过流，程序可以与外部环境交互，如控制台、文件或字符串。

C++ 标准库提供了一系列的流类，形成了一个类层次结构，用于不同类型的输入输出操作。

### 1.2 单向流

**单向流**只能进行单一方向的数据传输，即只能输入或输出。常见的单向流包括：

|流类型|输入流类|输出流类|需要包含的头文件|
|---|---|---|---|
|控制台|`std::istream`|`std::ostream`|`<iostream>`|
|文件|`std::ifstream`|`std::ofstream`|`<fstream>`|
|字符串|`std::istringstream`|`std::ostringstream`|`<sstream>`|

### 1.3 双向流

**双向流**既可以进行输入也可以进行输出的数据传输。这类流适用于需要同时读取和写入数据的场景，如文件的读写操作。

|流类型|流类|需要包含的头文件|
|---|---|---|
|文件|`std::fstream`|`<fstream>`|
|字符串|`std::stringstream`|`<sstream>`|

---

## 2. 类层次结构

C++ 的流类层次结构如下所示：

```
                 ios
          /              \
    istream                 ostream
     /    \                 /    \
ifstream istringstream ofstream ostringstream
                  \
                  fstream stringstream
```

**说明**：

- `ios` 是所有流类的基类，提供了基本的流功能。
- `istream` 和 `ostream` 分别是输入流和输出流的基类。
- `ifstream`、`istringstream`、`ofstream`、`ostringstream` 分别用于文件和字符串的输入输出操作。
- `fstream` 和 `stringstream` 是双向流类，支持读写操作。

---

## 3. 输出流：`std::ostream` 类

`std::ostream` 是所有输出流的基类，提供了将数据输出到不同媒介的功能。它重载了插入运算符 `<<`，用于将各种类型的数据发送到流中。

### 3.1 插入运算符 `<<`

`std::ostream` 类重载了 `<<` 运算符，使其能够接受多种类型的表达式。插入运算符的基本形式如下：

```cpp
ostream & operator << (表达式);
```

**表达式类型可以包括**：

- 基本数据类型（如 `int`、`double`、`char` 等）
- 指针（除了 `char*` 类型，指针会输出其地址）
- `char*` 类型（会输出指向的字符串内容）
- 自定义类（前提是已经重载了 `<<` 运算符）

**注意**：如果需要输出指针的地址，而不是指针指向的内容，可以将指针转换为 `void*` 类型。

### 3.2 成员函数

`std::ostream` 提供了一些成员函数，用于更精细地控制输出：

- `ostream & put(char c)`：输出单个字符 `c`。
- `ostream & write(const char* ptr, std::streamsize n)`：输出 `ptr` 指向的内存区域的前 `n` 个字符。

### 3.3 示例代码

```cpp
#include <iostream>
#include <fstream>
#include <sstream>

int main() {
    // 输出基本数据类型
    int a = 42;
    double pi = 3.14159;
    std::cout << "整数 a = " << a << std::endl;
    std::cout << "浮点数 pi = " << pi << std::endl;

    // 输出字符串
    std::string str = "Hello, World!";
    std::cout << "字符串: " << str << std::endl;

    // 输出指针地址
    int* ptr = &a;
    std::cout << "指针 ptr 的地址: " << ptr << std::endl;
    std::cout << "指针 ptr 指向的值: " << static_cast<void*>(ptr) << std::endl;

    return 0;
}
```

**输出结果**：

```
整数 a = 42
浮点数 pi = 3.14159
字符串: Hello, World!
指针 ptr 的地址: 0x7ffee4bff5ac
指针 ptr 指向的值: 0x7ffee4bff5ac
```

---

## 4. 输入流：`std::istream` 类

`std::istream` 是所有输入流的基类，提供了从不同媒介读取数据的功能。它重载了提取运算符 `>>`，用于从流中提取各种类型的数据。

### 4.1 提取运算符 `>>`

`std::istream` 类重载了 `>>` 运算符，使其能够接受多种类型的表达式。提取运算符的基本形式如下：

```cpp
istream & operator >> (表达式 &);
```

**表达式类型可以包括**：

- 基本数据类型（如 `int`、`double`、`char` 等）
- `char*` 类型（会读取一个单词或由空白字符分隔的字符串）
- 自定义类（前提是已经重载了 `>>` 运算符）

**注意**：指针类型（除了 `char*`）不被支持，因为直接读取指针地址没有实际意义。

### 4.2 分隔符

提取操作默认使用空格、制表符 `\t`、垂直制表符 `\v` 和换行符 `\n` 作为分隔符。这意味着在使用 `>>` 运算符时，输入会被这些字符分割。

### 4.3 成员函数

`std::istream` 提供了一些成员函数，用于更精细地控制输入：

- `istream & get(char &c)`：从输入流中提取一个字符，并存储到变量 `c` 中。
- `int get()`：从输入流中提取一个字符，并以整数形式返回。如果到达文件末尾（EOF），则返回 EOF。
- `istream & read(char* ptr, std::streamsize n)`：从输入流中读取 `n` 个字符，并存储到 `ptr` 指向的内存区域。

### 4.4 示例代码

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main() {
    // 从标准输入读取数据
    int num;
    std::cout << "请输入一个整数: ";
    std::cin >> num;
    std::cout << "您输入的整数是: " << num << std::endl;

    // 使用 get() 函数读取字符
    char ch;
    std::cout << "请输入一个字符: ";
    std::cin.get(ch); // 读取上一个输入后的换行符
    std::cin.get(ch); // 读取实际字符
    std::cout << "您输入的字符是: " << ch << std::endl;

    // 从文件中读取数据
    std::ifstream infile("data.txt");
    if (infile) {
        double d;
        std::string s;
        infile >> d >> s;
        std::cout << "读取到的浮点数: " << d << std::endl;
        std::cout << "读取到的字符串: " << s << std::endl;
        infile.close();
    } else {
        std::cout << "无法打开文件 data.txt" << std::endl;
    }

    // 从字符串流中读取数据
    std::string input = "123 456";
    std::istringstream iss(input);
    int x, y;
    iss >> x >> y;
    std::cout << "从字符串流读取到的 x = " << x << ", y = " << y << std::endl;

    return 0;
}
```

**假设 `data.txt` 文件内容如下**：

```
3.14 Hello
```

**输出结果**：

```
请输入一个整数: 25
您输入的整数是: 25
请输入一个字符: A
您输入的字符是: A
读取到的浮点数: 3.14
读取到的字符串: Hello
从字符串流读取到的 x = 123, y = 456
```

---

## 5. 标准输入输出流

C++ 提供了四个预定义的流对象，用于与控制台进行交互。这些流对象分别对应输入和输出的不同用途。

|流名称|流对象|描述|
|---|---|---|
|标准输入流|`std::cin`|从控制台接收输入数据|
|标准输出流|`std::cout`|将输出数据显示到控制台|
|标准错误输出|`std::cerr`|输出错误信息到控制台（不经过缓冲区）|
|标准日志输出|`std::clog`|输出日志信息到控制台（经过缓冲区）|

### 5.1 示例代码

```cpp
#include <iostream>

int main() {
    int a = 36;
    std::cout << "a = " << a << std::endl; // 输出 a 的值

    int b;
    std::cout << "请输入一个整数: ";
    std::cin >> b; // 从控制台读取一个整数
    std::cout << "您输入的整数是: " << b << std::endl;

    // 使用 std::cerr 输出错误信息
    if (b < 0) {
        std::cerr << "错误: 输入的整数为负数!" << std::endl;
    }

    // 使用 std::clog 输出日志信息
    std::clog << "日志: 程序执行完毕。" << std::endl;

    return 0;
}
```

**运行示例**：

```
a = 36
请输入一个整数: 10
您输入的整数是: 10
日志: 程序执行完毕。
```

**说明**：

- `std::cout` 用于正常的输出操作。
- `std::cin` 用于从控制台接收用户输入。
- `std::cerr` 用于输出错误信息，通常不经过缓冲区，适合即时输出错误。
- `std::clog` 用于输出日志信息，经过缓冲区，适合较大量或不需要即时显示的信息。

---

## 6. 文件流

文件流允许程序与文件进行读写操作。C++ 提供了三种文件流类型：

- `std::ifstream`：用于从文件中读取数据。
- `std::ofstream`：用于向文件中写入数据。
- `std::fstream`：用于同时读取和写入文件。

### 6.1 打开文件的语法

```cpp
std::ofstream outfile("filename.txt", std::ios::out | std::ios::app);
std::ifstream infile("filename.txt", std::ios::in);
std::fstream file("filename.txt", std::ios::in | std::ios::out);
```

### 6.2 打开模式说明

|模式|描述|
|---|---|
|`std::ios::in`|以读取模式打开文件（对于 `ifstream` 是必须的）|
|`std::ios::out`|以写入模式打开文件（对于 `ofstream` 是必须的）|
|`std::ios::app`|以追加模式打开文件，写入内容会添加到文件末尾|
|`std::ios::ate`|打开文件后，将文件指针移动到文件末尾|
|`std::ios::trunc`|如果文件存在，打开时会清空文件内容|
|`std::ios::binary`|以二进制模式打开文件|

### 6.3 关闭文件

使用流对象的 `close()` 方法可以手动关闭文件流：

```cpp
outfile.close();
infile.close();
file.close();
```

**注意**：在流对象销毁时（例如，作用域结束），文件会自动关闭。但在某些情况下，如需要在文件操作完成后立即释放资源，手动调用 `close()` 是有用的。

### 6.4 写入文件示例

```cpp
#include <iostream>
#include <fstream>

int main() {
    // 打开文件进行写入，使用追加模式
    std::ofstream outfile("myfile.txt", std::ios::out | std::ios::app);
    if (outfile) { // 检查文件是否成功打开
        double d = 3.14;
        outfile << d << std::endl;          // 写入浮点数
        outfile << "一段文本" << std::endl;   // 写入字符串
        outfile.close();                     // 关闭文件流
        std::cout << "数据已成功写入文件。" << std::endl;
    } else {
        std::cerr << "无法打开文件进行写入。" << std::endl;
    }
    return 0;
}
```

**假设 `myfile.txt` 文件内容如下**：

```
3.14
一段文本
```

### 6.5 读取文件示例

```cpp
#include <iostream>
#include <fstream>
#include <string>

int main() {
    // 打开文件进行读取
    std::ifstream infile("myfile.txt", std::ios::in);
    if (infile) { // 检查文件是否成功打开
        double d;
        std::string s1, s2;
        infile >> d >> s1 >> s2; // 读取浮点数和两个字符串
        std::cout << "读取到的浮点数: " << d << std::endl;
        std::cout << "读取到的字符串: " << s1 << " " << s2 << std::endl;
        infile.close(); // 关闭文件流
    } else {
        std::cerr << "无法打开文件进行读取。" << std::endl;
    }
    return 0;
}
```

**输出结果**：

```
读取到的浮点数: 3.14
读取到的字符串: 一段 文本
```

**说明**：

- 使用 `std::ifstream` 打开文件并读取数据。
- 读取操作会根据空白字符（空格、换行符等）分割输入。
- 确保在读取完成后关闭文件流。

---

## 7. 字符串流

**字符串流**允许程序在内存中对字符串进行输入输出操作，类似于文件流和控制台流，但不涉及文件系统。

C++ 提供了两种字符串流类：

- `std::istringstream`：用于从字符串中读取数据。
- `std::ostringstream`：用于向字符串中写入数据。
- `std::stringstream`：用于同时进行输入和输出操作。

### 7.1 写入字符串流

```cpp
#include <iostream>
#include <sstream>
#include <string>

int main() {
    double px = 3.14, py = 6.32;
    std::ostringstream ostr; // 创建输出字符串流对象
    ostr << "点的坐标是 (" << px << ", " << py << ")\n"; // 向流中写入数据
    std::cout << ostr.str(); // 获取并输出流中的字符串内容
    return 0;
}
```

**输出结果**：

```
点的坐标是 (3.14, 6.32)
```

### 7.2 从字符串流读取数据

```cpp
#include <iostream>
#include <sstream>
#include <string>

int main() {
    int tab[5];
    std::string stringvalues = "125 320 512 750 333";
    std::istringstream istr(stringvalues); // 创建输入字符串流对象

    for (int k = 0; k < 5; ++k) {
        istr >> tab[k]; // 从流中读取数据到数组
    }

    std::cout << "tab[1] = " << tab[1] << std::endl; // 输出读取到的数据
    return 0;
}
```

**输出结果**：

```
tab[1] = 320
```

**说明**：

- 使用 `std::istringstream` 从字符串中提取数据，类似于从文件或控制台读取。
- `str()` 方法可以获取 `std::ostringstream` 中的字符串内容。

---

## 8. 流的状态

每个流对象都有一个内部状态，用于描述流当前的状态。这些状态可以通过以下成员函数查询：

- `bool eof() const`：如果已到达文件末尾（EOF），返回 `true`。
- `bool bad() const`：如果发生了严重的错误，返回 `true`。
- `bool fail() const`：如果前一次操作失败（如格式错误），返回 `true`。
- `bool good() const`：如果流处于良好状态（没有错误），返回 `true`。

### 8.1 使用流状态

可以将流对象当作布尔值使用，表示流是否处于良好状态。这是通过重载 `operator bool()` 实现的。

### 8.2 示例代码

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main() {
    // 从文件中读取数据并输出到字符串流
    std::ifstream rfile("myfile.txt");
    if (rfile) { // 检查文件是否成功打开
        std::ostringstream ostr; // 创建输出字符串流对象
        ostr << "读取的值: ";
        while (true) {
            int a;
            rfile >> a; // 尝试读取一个整数
            if (rfile.eof()) // 如果到达文件末尾，退出循环
                break;
            ostr << a << " "; // 将读取到的整数写入字符串流
        }
        std::cout << ostr.str() << "\n"; // 输出字符串流中的内容
        rfile.close(); // 关闭文件流
    } else {
        std::cerr << "无法打开文件 myfile.txt 进行读取。" << std::endl;
    }

    return 0;
}
```

**假设 `myfile.txt` 文件内容如下**：

```
1 2 3 4 5 6 7 8 9 10 11 12
```

**输出结果**：

```
读取的值: 1 2 3 4 5 6 7 8 9 10 11 12 
```

**说明**：

- 使用 `eof()` 检查是否到达文件末尾。
- 在循环中不断读取整数，直到到达文件末尾。

### 8.3 流的布尔转换

可以像使用布尔变量一样使用流对象。流对象会根据其内部状态返回 `true` 或 `false`。

```cpp
#include <iostream>
#include <fstream>

int main() {
    std::ifstream file("myfile.txt");
    if (file) { // 流对象转换为布尔值，检查是否成功打开
        int num;
        while (file >> num) { // 每次提取操作后，流状态会更新
            std::cout << "读取到的数字: " << num << std::endl;
        }
    } else {
        std::cerr << "无法打开文件 myfile.txt" << std::endl;
    }
    return 0;
}
```

**说明**：

- `if (file)` 判断文件是否成功打开。
- `while (file >> num)` 在每次提取操作后，检查流是否处于良好状态。

---

## 9. 格式化输出

C++ 允许通过操纵符（Manipulators）来控制输出流的格式。这些操纵符可以调整布尔值的显示方式、数字的进制、浮点数的表示形式等。

### 9.1 常用操纵符

|操纵符|描述|
|---|---|
|`std::boolalpha`|将布尔值以 `true` 或 `false` 显示|
|`std::noboolalpha`|将布尔值以 `1` 或 `0` 显示（默认设置）|
|`std::dec`|将整数以十进制显示（默认设置）|
|`std::oct`|将整数以八进制显示|
|`std::hex`|将整数以十六进制显示|
|`std::scientific`|将浮点数以科学计数法显示|
|`std::fixed`|将浮点数以定点表示法显示|
|`std::showpos`|在正数前显示加号 `+`|
|`std::noshowpos`|不显示正数的加号（默认设置）|
|`std::setprecision`|设置浮点数的小数位数，需要包含 `<iomanip>` 头文件|
|`std::setw`|设置下一个输出项的宽度，需要包含 `<iomanip>` 头文件|
|`std::left`|将输出项左对齐，需要包含 `<iomanip>` 头文件|
|`std::right`|将输出项右对齐，需要包含 `<iomanip>` 头文件|

### 9.2 示例代码

```cpp
#include <iostream>
#include <iomanip> // 包含操纵符所需的头文件

int main() {
    bool b = true;
    int i = 43;
    double d = 31223.565465;

    // 布尔值默认以 1 或 0 显示
    std::cout << "默认布尔值显示: " << b << std::endl;

    // 使用 boolalpha 以 true 或 false 显示
    std::cout << "使用 boolalpha 显示: " << std::boolalpha << b << std::endl;

    // 整数默认以十进制显示
    std::cout << "整数默认显示: " << i << std::endl;

    // 使用十六进制显示整数
    std::cout << "整数以十六进制显示: " << std::hex << i << std::endl;

    // 浮点数默认显示
    std::cout << "浮点数默认显示: " << d << std::endl;

    // 使用科学计数法显示浮点数
    std::cout << "浮点数以科学计数法显示: " << std::scientific << d << std::endl;

    // 使用定点表示法显示浮点数
    std::cout << "浮点数以定点表示法显示: " << std::fixed << d << std::endl;

    return 0;
}
```

**输出结果**：

```
默认布尔值显示: 1
使用 boolalpha 显示: true
整数默认显示: 43
整数以十六进制显示: 2b
浮点数默认显示: 31223.6
浮点数以科学计数法显示: 3.122357e+04
浮点数以定点表示法显示: 31223.565465
```

### 9.3 设置浮点数精度

可以使用 `std::setprecision(int n)` 来设置浮点数的小数位数：

```cpp
#include <iostream>
#include <iomanip>

int main() {
    double d = 31223.565465;

    // 使用科学计数法并设置精度
    std::cout << "科学计数法，精度 12: " << std::setprecision(12) << std::scientific << d << std::endl;

    // 使用定点表示法
    std::cout << "定点表示法: " << std::fixed << d << std::endl;

    return 0;
}
```

**输出结果**：

```
科学计数法，精度 12: 3.122356546500e+04
定点表示法: 31223.565465000000
```

**说明**：

- `std::setprecision(12)` 设置浮点数的总有效数字位数为 12 位。
- 使用 `std::fixed` 和 `std::scientific` 改变浮点数的表示方式。

### 9.4 设置输出项宽度和对齐方式

可以使用 `std::setw(int n)` 设置下一个输出项的最小宽度，结合 `std::left` 和 `std::right` 设置对齐方式：

```cpp
#include <iostream>
#include <iomanip>

int main() {
    std::cout << ":) 示例未格式化: " << "exemple" << " de gabarit" << std::endl;

    // 设置下一个输出项的宽度为 15，左对齐
    std::cout << ":)" << std::setw(15) << std::left << "exemple" << " de gabarit" << std::endl;

    // 设置下一个输出项的宽度为 15，右对齐
    std::cout << ":)" << std::setw(15) << std::right << "exemple" << " de gabarit" << std::endl;

    return 0;
}
```

**输出结果**：

```
:) 示例未格式化: exemple de gabarit
:)exemple          de gabarit
:)         exemple de gabarit
```

**说明**：

- `std::setw(15)` 设置下一个输出项的最小宽度为 15 个字符。
- `std::left` 和 `std::right` 分别设置左对齐和右对齐。
- 注意：`std::setw` 只对下一个输出项有效，需要每次使用时重新设置。

---

## 10. 在流中使用自定义类

为了能够将自定义类的对象与流进行交互，需要重载插入运算符 `<<` 和提取运算符 `>>`。

### 10.1 自定义类示例：`Point` 类

```cpp
#include <iostream>
#include <string>
#include <fstream>

// 定义 Point 类
class Point {
public:
    Point(double x = 0, double y = 0) : M_x(x), M_y(y) {}
    
    double x() const { return M_x; }
    double y() const { return M_y; }
    
    void setXY(double x, double y) { M_x = x; M_y = y; }
    
private:
    double M_x, M_y;
};

// 重载插入运算符 <<，用于输出 Point 对象
std::ostream& operator<<(std::ostream& os, const Point& p) {
    return os << p.x() << "," << p.y();
}

// 重载提取运算符 >>，用于输入 Point 对象
std::istream& operator>>(std::istream& is, Point& p) {
    char delimiter;
    double x, y;
    is >> x >> delimiter >> y; // 假设输入格式为 "x,y"
    p.setXY(x, y);
    return is;
}

int main() {
    // 写入文件
    Point a(2, 3);
    Point b(4, 5);
    std::ofstream ofile("myfile.txt");
    if (ofile) {
        ofile << a << "\n" << b << "\n"; // 使用重载的 << 运算符
        ofile.close();
        std::cout << "数据已成功写入文件。" << std::endl;
    } else {
        std::cerr << "无法打开文件进行写入。" << std::endl;
    }

    // 从文件读取
    Point c, d;
    std::ifstream ifile("myfile.txt");
    if (ifile) {
        ifile >> c >> d; // 使用重载的 >> 运算符
        std::cout << "c = " << c << "\n";
        std::cout << "d = " << d << "\n";
        ifile.close();
    } else {
        std::cerr << "无法打开文件进行读取。" << std::endl;
    }

    return 0;
}
```

**假设 `myfile.txt` 文件内容如下**：

```
2,3
4,5
```

**输出结果**：

```
数据已成功写入文件。
c = 2,3
d = 4,5
```

**说明**：

- 重载 `<<` 运算符使得可以直接将 `Point` 对象输出到流中，例如文件或控制台。
- 重载 `>>` 运算符使得可以从流中读取数据并存储到 `Point` 对象中。
- 输入和输出操作依赖于自定义的格式，这里假设输入格式为 `x,y`。

**注意**：

- 在重载 `>>` 运算符时，需要确保输入格式与预期一致，否则可能导致提取失败。
- 可以在重载运算符中添加更多的错误检查和处理，以提高程序的健壮性。

---
