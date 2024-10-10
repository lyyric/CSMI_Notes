## C++ 示例
### Hello, world

```cpp
# include <iostream >
int main (){
	std :: cout << "Hello , world" << std :: endl;
	return 0;
}
```

```
user $ g++ hello.cpp -o hello
```

```
user $ ./hello
Hello, world
```

### C++ 示例

以一个简单的C++文件为例来展示整个过程：

```cpp
// test.cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

#### 1. **预处理 Le préprocesseur**：

```
g++ -E test.cpp -o test.i
```

这一步输出预处理后的文件 `test.i`，其中所有的 `#include` 和 `#define` 都已经被处理。

#### 2. **编译 La compilation**：

```
g++ -S test.i -o test.s
```

这一步输出汇编代码文件 `test.s`，编译器将C++源代码转换为汇编代码。

#### 3. **汇编 L’assemblage**：

```
g++ -c test.s -o test.o
```

这一步生成目标文件 `test.o`，汇编器将汇编代码转换为机器码（但还不是可执行文件）。

#### 4. **链接 L’édition des liens**：

```
g++ test.o -o test
```

这一步生成可执行文件 `test`，链接器将目标文件和标准库（如 `iostream`）进行链接。

#### 最后，运行可执行文件：

```
./test
```

输出：

```
Hello, World!
```

### C++多文件结构示例

假设我们有以下文件：

#### `math.h`（声明函数）

```cpp
#ifndef MATH_H
#define MATH_H

int add(int a, int b);

#endif
```

#### `math.cpp`（定义函数）

```cpp
#include "math.h"

int add(int a, int b) {
    return a + b;
}
```

#### `main.cpp`（调用函数）

```cpp
#include <iostream>
#include "math.h"

int main() {
    int result = add(3, 4);
    std::cout << "Result of 3 + 4 is: " << result << std::endl;
    return 0;
}
```

#### 编译步骤

1. **编译 `main.cpp`**：

```bash
g++ -c main.cpp -o main.o
```
   
这会生成一个 `main.o` 目标文件，其中包含对 `add` 函数的未解析引用。

2. **编译 `math.cpp`**：

```bash
g++ -c math.cpp -o math.o
```

这会生成一个 `math.o` 目标文件，其中包含 `add` 函数的实际实现。

3. **链接 `main.o` 和 `math.o`**：

```bash
g++ main.o math.o -o my_program
```

这一步链接器会将 `main.o` 中对 `add` 的引用与 `math.o` 中 `add` 函数的定义相对应，生成最终的可执行文件 `my_program`。

#### 一步编译方法

```bash
g++ main.cpp math.cpp -o my_program
```

### 主函数

- **`int main()`**：不接受命令行参数，通常用于不需要与用户交互的程序或不关心命令行输入的简单程序。
- **`int main(int argc, char* argv[])`**：接受命令行参数，适合需要根据用户输入来处理数据的程序，允许通过命令行传递参数给程序。

`argc` 和 `argv` 分别是以下英文术语的缩写：

1. **`argc`**: **argument count**  
   - 表示命令行参数的数量（count）。它是命令行中传递给程序的参数个数，包括程序名本身。

2. **`argv`**: **argument vector**  
   - 表示命令行参数的向量（vector）。它是一个指针数组，其中每个元素都是指向一个命令行参数的指针。`argv[0]` 通常是程序名，`argv[1]` 到 `argv[argc-1]` 是传递给程序的实际参数。

## Éléments de base du langage

### Les types élémentaires

#### Les entiers :
- `int` : précision 32 bits
- `short int` : précision 16 bits
- `long int` : précision 32 ou 64 bits (dépend de la plateforme)
- `long long int` : précision 64 bits
- `unsigned <type>` : version non signée (entiers positifs ou nuls)

#### Les réels (nombres à virgule flottante) :
- `float` : simple précision (32 bits)
- `double` : double précision (64 bits)
- `long double` : précision étendue (dépend de la plateforme, souvent > 64 bits)

#### Les caractères :
- `char` : représente un caractère (codé sur 8 bits, dans la table ASCII)
- `signed char` : entier entre -127 et 128
- `unsigned char` : entier entre 0 et 255

#### Le type booléen (logique) :
- `bool` : valeurs possibles `true` ou `false` (représentées respectivement par 1 et 0 en interne)

### Les variables

### Les opérateurs

### Les énumérations

```cpp
enum Status {
    OK = 0,
    ERROR = 1,
    TIMEOUT = 2
};

// 使用枚举常量比直接使用数字更清晰
Status systemStatus = OK;
if (systemStatus == ERROR) {
    std::cout << "An error occurred" << std::endl;
}
```

```cpp
enum class Color { Red, Green, Blue };

Color myColor = Color::Red;  // 必须使用作用域
if (myColor == Color::Green) {
    std::cout << "The color is Green" << std::endl;
}
```

### Les pointeurs

### Les références

### Le concept des références Lvalue et Rvalue

### Les tableaux

### Gestion de la mémoire

### Les chaines de caractères

### Les instructions conditionnelles

1. if - else 
2. switch

### La boucle

1. for
2. for (c++11)
3. while
4. do while

### break - continue

### Les flux d'entrées/sorties

## Les fonctions

Les fonctions
Un exemple de fonction
Syntaxe d’une fonction
Déclaration et définition de fonction
Passage de paramètres : par valeur
Passage de paramètres : par référence
Passage de paramètres : par Rvalue
Passage de paramètres : pointeurs et tableaux
La fonction main avec paramètres
Paramètres optionnels
Surcharge de fonctions
Fonctions récursives
Pointeurs de fonctions
Fonctions inline
Les fonctions mathématiques