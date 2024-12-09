# 标准模板库（STL）：C++的强大工具

---

## 目录

1. [C++标准库简介](#1-c++标准库简介)
2. [容器（Containers）](#2-容器containers)
   - [容器的概念](#容器的概念)
   - [容器的类型](#容器的类型)
   - [序列容器（Sequence Containers）](#序列容器sequence-containers)
     - [std::array](#stdarray)
     - [std::vector](#stdvector)
     - [std::list](#stdlist)
     - [std::forward_list](#stdforward_list)
     - [std::deque](#stddeque)
   - [关联容器（Associative Containers）](#关联容器associative-containers)
     - [std::map](#stdmap)
     - [std::set](#stdset)
   - [容器适配器（Container Adaptors）](#容器适配器container-adaptors)
     - [std::queue](#stdqueue)
     - [std::priority_queue](#stdpriority_queue)
     - [std::stack](#stdstack)
3. [迭代器（Iterators）](#3-迭代器iterators)
   - [迭代器的概念](#迭代器的概念)
   - [迭代器的类型](#迭代器的类型)
   - [迭代器的使用](#迭代器的使用)
   - [迭代器与常量迭代器](#迭代器与常量迭代器)
4. [算法（Algorithms）](#4-算法algorithms)
   - [不修改容器的算法](#不修改容器的算法)
   - [修改容器的算法](#修改容器的算法)
   - [分区算法](#分区算法)
   - [排序算法](#排序算法)
   - [最小值/最大值算法](#最小值最大值算法)
5. [额外功能](#5-额外功能)
   - [std::pair和std::tuple](#stdpair和stdtuple)
   - [std::complex](#stdcomplex)
   - [智能指针（Smart Pointers）](#智能指针smart-pointers)
   - [时间测量](#时间测量)
   - [正则表达式（Regular Expressions）](#正则表达式regular-expressions)
6. [进一步学习](#6-进一步学习)
7. [推荐资源](#7-推荐资源)

---

## 1. C++标准库简介

C++拥有一个功能强大的**标准库**，其中包含了多种组件，如流库（用于输入输出）、C标准库（如`<cstring>`、`<cstdlib>`等）、异常处理机制，以及**标准模板库（STL, Standard Template Library）**。STL提供了大量基于模板的通用工具，包括：

- **容器（Containers）**：用于存储数据的结构，如数组、向量、列表、集合等。
- **迭代器（Iterators）**：用于遍历容器中的元素。
- **算法（Algorithms）**：用于对容器中的元素进行操作，如排序、搜索、复制等。

此外，STL还包含了许多其他实用功能。在C++程序中，通常优先使用STL提供的容器和算法，而不是手动实现这些功能，因为STL不仅高效、健壮，还提高了代码的可读性和可维护性。

**常用文档资源：**
- [cppreference.com](https://en.cppreference.com/w/cpp)
- [cplusplus.com](https://www.cplusplus.com/)

---

## 2. 容器（Containers）

### 容器的概念

**容器（Container）**是STL中用于存储和管理数据的抽象数据结构。容器提供了一系列方法来方便地操作数据集合，如添加、删除、插入、排序、访问和搜索元素。STL中提供了多种容器，适用于不同的需求和使用场景。

常见的STL容器包括：
- **序列容器（Sequence Containers）**：如`std::array`、`std::vector`、`std::list`等，保持元素的线性顺序。
- **关联容器（Associative Containers）**：如`std::set`、`std::map`等，通过键值进行元素的快速访问。
- **容器适配器（Container Adaptors）**：如`std::queue`、`std::stack`等，基于已有的容器提供特定的接口和行为。

### 容器的类型

STL中的容器主要分为两大类：

1. **序列容器（Sequence Containers）**：保持元素的线性顺序。
2. **关联容器（Associative Containers）**：通过键值进行快速访问和管理元素。

#### 序列容器（Sequence Containers）

序列容器用于存储线性排列的数据，可以通过索引或迭代器进行访问和操作。常见的序列容器包括：

- **`std::array`**：固定大小的数组，大小在编译时确定，不可动态调整。
- **`std::vector`**：动态数组，支持快速随机访问和在末尾高效插入。
- **`std::list`**：双向链表，支持在任意位置高效插入和删除。
- **`std::forward_list`**：单向链表，比`std::list`更轻量，但仅支持单向遍历。
- **`std::deque`**：双端队列，支持在头部和尾部高效插入和删除。

#### 关联容器（Associative Containers）

关联容器通过键值进行快速查找和管理元素。常见的关联容器包括：

- **`std::map`**：有序的键值对集合，每个键唯一，支持快速查找。
- **`std::set`**：有序的唯一元素集合。
- **`std::multimap`**：有序的键值对集合，允许多个相同的键。
- **`std::multiset`**：有序的元素集合，允许多个相同的元素。

### 序列容器sequence-containers

#### std::array

**`std::array`**是一个固定大小的数组，大小在编译时确定，不能动态调整。它提供了类似C数组的性能和接口，但具有更强的类型安全和与STL算法的兼容性。

**特点：**
- 尺寸固定。
- 支持STL的迭代器和算法。
- 元素存储连续。

**示例：**

```cpp
#include <iostream>
#include <array>

int main() {
    // 定义一个包含5个整数的std::array
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    
    // 访问元素
    std::cout << "第一个元素: " << arr[0] << "\n";
    
    // 遍历数组
    for(auto it = arr.begin(); it != arr.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";
    
    return 0;
}
```

**输出：**
```
第一个元素: 1
1 2 3 4 5 
```

#### std::vector

**`std::vector`**是最常用的序列容器，类似于动态数组，支持快速的随机访问和在末尾高效插入和删除元素。

**特点：**
- 动态调整大小。
- 元素存储连续，支持快速随机访问（`O(1)`时间复杂度）。
- 在末尾插入和删除元素效率高（均摊`O(1)`时间复杂度）。
- 插入和删除中间或头部元素效率较低（`O(n)`时间复杂度）。

**常用方法：**
- `size()`：返回元素数量。
- `resize(int n)`：调整容器大小。
- `begin()` / `end()`：返回指向首尾的迭代器。
- `front()` / `back()`：访问第一个和最后一个元素。
- `push_back(T x)`：在末尾添加元素。
- `pop_back()`：删除末尾元素。

**示例：**

```cpp
#include <iostream>
#include <vector>

int main() {
    // 声明一个大小为5的vector，默认初始化为0
    std::vector<int> v1(5);
    
    // 初始化vector：1 2 3 4 5
    for(int k = 0; k < 5; ++k)
        v1[k] = k + 1;
    
    // 添加一个元素6到末尾
    v1.push_back(6); // v1: 1 2 3 4 5 6
    std::cout << "v1的大小: " << v1.size() << "\n";
    
    // 声明一个大小为8并初始化为15的vector
    std::vector<int> v2(8, 15);
    for(int k = 0; k < v2.size(); ++k)
        std::cout << "v2[" << k << "] = " << v2[k] << "\n";
    
    // 声明一个包含5个3元素vector的vector
    std::vector<std::vector<int>> v3(5, std::vector<int>(3));
    for(int i = 0; i < v3.size(); ++i)
        for(int j = 0; j < v3[i].size(); ++j)
            v3[i][j] = i * j;
    
    return 0;
}
```

**输出：**
```
v1的大小: 6
v2[0] = 15
v2[1] = 15
v2[2] = 15
v2[3] = 15
v2[4] = 15
v2[5] = 15
v2[6] = 15
v2[7] = 15
```

#### std::list

**`std::list`**是一个双向链表，支持在任意位置高效地插入和删除元素，但不支持随机访问，访问任意元素的时间复杂度为`O(n)`。

**特点：**
- 双向链表结构，每个元素包含指向前后元素的指针。
- 插入和删除元素在任意位置都很高效（`O(1)`时间复杂度）。
- 不支持随机访问，仅支持顺序遍历。

**常用方法：**
- `push_back(T x)`：在末尾添加元素。
- `push_front(T x)`：在前端添加元素。
- `pop_back()`：删除末尾元素。
- `pop_front()`：删除前端元素。
- `front()` / `back()`：访问第一个和最后一个元素。

**示例：**

```cpp
#include <iostream>
#include <list>

int main() {
    std::list<int> l;
    
    // 在末尾添加元素
    l.push_back(7); // l: 7
    l.push_back(5); // l: 7 5
    
    // 在前端添加元素
    l.push_front(6); // l: 6 7 5
    l.push_front(2); // l: 2 6 7 5
    
    std::cout << "列表大小: " << l.size() << "\n"; // 输出：4
    std::cout << "列表头部: " << l.front() << "\n"; // 输出：2
    std::cout << "列表尾部: " << l.back() << "\n"; // 输出：5
    
    // 删除末尾和前端元素
    l.pop_back(); // l: 2 6 7
    l.pop_front(); // l: 6 7
    
    // 遍历并输出列表元素
    for(int val : l)
        std::cout << val << "\n"; // 输出：6 7
    
    return 0;
}
```

**输出：**
```
列表大小: 4
列表头部: 2
列表尾部: 5
6
7
```

#### std::forward_list

**`std::forward_list`**是一个单向链表，与`std::list`类似，但每个元素只包含指向下一个元素的指针。它比`std::list`更轻量，但仅支持单向遍历。

**特点：**
- 单向链表结构。
- 更节省内存，适用于只需要单向遍历的场景。
- 插入和删除元素效率高（`O(1)`时间复杂度）。
- 不支持随机访问。

**示例：**

```cpp
#include <iostream>
#include <forward_list>

int main() {
    std::forward_list<int> fl;
    
    // 在前端添加元素
    fl.push_front(3);
    fl.push_front(2);
    fl.push_front(1); // fl: 1 2 3
    
    // 遍历并输出元素
    for(auto it = fl.begin(); it != fl.end(); ++it)
        std::cout << *it << " "; // 输出：1 2 3 
    std::cout << "\n";
    
    // 删除元素
    fl.pop_front(); // fl: 2 3
    
    for(auto val : fl)
        std::cout << val << " "; // 输出：2 3 
    std::cout << "\n";
    
    return 0;
}
```

**输出：**
```
1 2 3 
2 3 
```

#### std::deque

**`std::deque`**（双端队列）是一种支持在两端高效插入和删除的序列容器，同时也支持随机访问。

**特点：**
- 支持在头部和尾部高效插入和删除（`O(1)`时间复杂度）。
- 元素存储在多个连续块中，支持快速随机访问（接近`O(1)`时间复杂度）。
- 插入和删除中间元素较为耗时（`O(n)`时间复杂度）。

**常用方法：**
- `push_back(T x)`：在末尾添加元素。
- `push_front(T x)`：在前端添加元素。
- `pop_back()`：删除末尾元素。
- `pop_front()`：删除前端元素。
- `operator[]` / `at(int n)`：随机访问元素。

**示例：**

```cpp
#include <iostream>
#include <deque>

int main() {
    std::deque<int> dq;
    
    // 在末尾添加元素
    dq.push_back(10); // dq: 10
    dq.push_back(20); // dq: 10 20
    
    // 在前端添加元素
    dq.push_front(5); // dq: 5 10 20
    
    // 访问元素
    std::cout << "第一个元素: " << dq.front() << "\n"; // 输出：5
    std::cout << "最后一个元素: " << dq.back() << "\n"; // 输出：20
    
    // 随机访问
    std::cout << "元素[1]: " << dq[1] << "\n"; // 输出：10
    
    // 删除前端和末尾元素
    dq.pop_front(); // dq: 10 20
    dq.pop_back(); // dq: 10
    
    // 遍历并输出元素
    for(auto val : dq)
        std::cout << val << " "; // 输出：10 
    std::cout << "\n";
    
    return 0;
}
```

**输出：**
```
第一个元素: 5
最后一个元素: 20
元素[1]: 10
10 
```

### 关联容器associative-containers

#### std::map

**`std::map`**是一个关联容器，存储键值对（`std::pair<const Key, T>`），其中每个键是唯一的，并且按照键的顺序自动排序。`std::map`内部通常使用平衡二叉搜索树（如红黑树）实现，支持高效的查找、插入和删除操作。

**特点：**
- 键值对存储，键唯一。
- 自动按键排序（默认使用`std::less<Key>`）。
- 支持快速查找、插入和删除（`O(log n)`时间复杂度）。

**常用方法：**
- `insert()`：插入元素。
- `erase()`：删除元素。
- `find()`：查找元素。
- `operator[]`：访问或插入元素。
- `size()`：返回元素数量。
- `begin()` / `end()`：返回迭代器。

**示例：**

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> m;
    
    // 插入元素
    m["a"] = 10;
    m["b"] = 30;
    m["c"] = 50;
    m["d"] = 70;
    
    // 删除元素
    m.erase("c");
    
    std::cout << "map的大小: " << m.size() << "\n"; // 输出：3
    
    // 遍历并输出map元素
    for(const std::pair<std::string, int>& entry : m)
        std::cout << entry.first << " : " << entry.second << "\n";
    
    return 0;
}
```

**输出：**
```
map的大小: 3
a : 10
b : 30
d : 70
```

#### std::set

**`std::set`**是一个关联容器，存储唯一的元素，并自动按元素的顺序排序。与`std::map`类似，`std::set`通常使用平衡二叉搜索树实现，支持高效的查找、插入和删除操作。

**特点：**
- 元素唯一。
- 自动排序（默认使用`std::less<Key>`）。
- 支持快速查找、插入和删除（`O(log n)`时间复杂度）。

**常用方法：**
- `insert()`：插入元素。
- `erase()`：删除元素。
- `find()`：查找元素。
- `size()`：返回元素数量。
- `begin()` / `end()`：返回迭代器。

**示例：**

```cpp
#include <iostream>
#include <set>

int main() {
    std::set<int> s;
    
    // 插入元素
    s.insert(10);
    s.insert({30, 50, 70});
    
    // 删除元素
    s.erase(50);
    
    std::cout << "set的大小: " << s.size() << "\n"; // 输出：3
    
    // 遍历并输出set元素
    for(int entry : s)
        std::cout << entry << "\n"; // 输出：10 30 70
    
    return 0;
}
```

**输出：**
```
set的大小: 3
10
30
70
```

### 容器适配器container-adaptors

容器适配器是基于已有容器（通常是序列容器或关联容器）提供特定接口和行为的容器。它们简化了特定数据结构的使用，并限制了接口以适应特定用途。

**常见的容器适配器包括：**
- **`std::queue`**：先进先出（FIFO）队列。
- **`std::priority_queue`**：优先队列，元素按优先级排序。
- **`std::stack`**：后进先出（LIFO）栈。

#### std::queue

**`std::queue`**是一个先进先出（FIFO）的容器适配器。它通常基于`std::deque`或`std::list`实现，提供了限制性的接口，仅支持在末尾添加元素和在前端删除元素。

**特点：**
- 先进先出（FIFO）结构。
- 仅支持`push`、`pop`、`front`和`back`操作。
- 不支持随机访问和迭代器。

**示例：**

```cpp
#include <iostream>
#include <queue>

int main() {
    std::queue<int> q;
    
    // 添加元素到队列末尾
    q.push(10);
    q.push(20);
    q.push(30);
    
    std::cout << "队列大小: " << q.size() << "\n"; // 输出：3
    
    // 访问队列头部元素
    std::cout << "队列头部: " << q.front() << "\n"; // 输出：10
    std::cout << "队列尾部: " << q.back() << "\n";   // 输出：30
    
    // 删除队列头部元素
    q.pop(); // 队列: 20 30
    
    std::cout << "队列头部: " << q.front() << "\n"; // 输出：20
    std::cout << "队列大小: " << q.size() << "\n";   // 输出：2
    
    return 0;
}
```

**输出：**
```
队列大小: 3
队列头部: 10
队列尾部: 30
队列头部: 20
队列大小: 2
```

#### std::priority_queue

**`std::priority_queue`**是一个优先队列，确保每次访问的元素都是当前队列中最大的元素（默认情况下，基于`std::less`进行排序，可以自定义比较函数）。

**特点：**
- 元素按照优先级自动排序。
- 访问和删除总是队首元素（最大值）。
- 不支持随机访问和迭代器。

**示例：**

```cpp
#include <iostream>
#include <queue>

int main() {
    // 默认情况下，priority_queue按从大到小排序
    std::priority_queue<int> pq;
    
    // 添加元素
    pq.push(30);
    pq.push(10);
    pq.push(20);
    
    std::cout << "优先队列大小: " << pq.size() << "\n"; // 输出：3
    
    // 访问队首元素
    std::cout << "队首元素: " << pq.top() << "\n"; // 输出：30
    
    // 删除队首元素
    pq.pop(); // 队首元素30被移除
    
    std::cout << "队首元素: " << pq.top() << "\n"; // 输出：20
    std::cout << "优先队列大小: " << pq.size() << "\n"; // 输出：2
    
    return 0;
}
```

**输出：**
```
优先队列大小: 3
队首元素: 30
队首元素: 20
优先队列大小: 2
```

#### std::stack

**`std::stack`**是一个后进先出（LIFO）的容器适配器，类似于堆栈。它通常基于`std::deque`或`std::vector`实现，提供了限制性的接口，仅支持在顶端添加和删除元素。

**特点：**
- 后进先出（LIFO）结构。
- 仅支持`push`、`pop`、`top`操作。
- 不支持随机访问和迭代器。

**示例：**

```cpp
#include <iostream>
#include <stack>

int main() {
    std::stack<int> s;
    
    // 添加元素到栈顶
    s.push(10);
    s.push(20);
    s.push(30);
    
    std::cout << "栈大小: " << s.size() << "\n"; // 输出：3
    
    // 访问栈顶元素
    std::cout << "栈顶元素: " << s.top() << "\n"; // 输出：30
    
    // 删除栈顶元素
    s.pop(); // 栈顶元素30被移除
    
    std::cout << "栈顶元素: " << s.top() << "\n"; // 输出：20
    std::cout << "栈大小: " << s.size() << "\n";   // 输出：2
    
    return 0;
}
```

**输出：**
```
栈大小: 3
栈顶元素: 30
栈顶元素: 20
栈大小: 2
```

---

## 3. 迭代器（Iterators）

### 迭代器的概念

**迭代器（Iterator）**是STL中的一个核心概念，用于在容器中遍历元素。迭代器类似于指针，可以指向容器中的元素，并提供访问和操作元素的方法。通过迭代器，我们无需了解容器的内部实现，就可以高效地访问和处理数据。

**迭代器的类型：**
- **普通迭代器（Iterator）**：用于读写容器中的元素。
- **常量迭代器（Const Iterator）**：只能读取容器中的元素，不能修改。

**方向性：**
- **正向迭代器（Forward Iterator）**：只能向前移动。
- **双向迭代器（Bidirectional Iterator）**：可以向前和向后移动。
- **随机访问迭代器（Random Access Iterator）**：支持任意方向和步长的移动。

### 迭代器的类型

根据容器的不同，迭代器也分为不同的类型：

- **随机访问迭代器（RandomAccessIterator）**：如`std::vector`和`std::deque`，支持快速随机访问。
- **双向迭代器（BidirectionalIterator）**：如`std::list`、`std::set`和`std::map`，支持双向遍历。
- **前向迭代器（ForwardIterator）**：如`std::forward_list`和`std::unordered_set`，支持单向遍历。

### 迭代器的使用

使用迭代器遍历容器元素的基本步骤：

1. **声明迭代器**：根据容器类型选择合适的迭代器类型。
2. **初始化迭代器**：使用`begin()`和`end()`方法获取容器的起始和结束迭代器。
3. **遍历容器**：通过迭代器进行循环遍历，访问和操作元素。

**示例：遍历`std::vector`**

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {32, 71, 12, 45, 26, 80, 53, 33};
    
    // 声明一个vector的迭代器
    std::vector<int>::iterator it = v.begin();
    std::vector<int>::iterator en = v.end();
    
    // 使用迭代器遍历vector
    for(; it != en; ++it)
        std::cout << *it << "\n";
    
    return 0;
}
```

**输出：**
```
32
71
12
45
26
80
53
33
```

**示例：遍历`std::map`**

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> m;
    m["a"] = 10;
    m["b"] = 30;
    m["c"] = 50;
    m["d"] = 70;
    
    // 声明一个map的迭代器
    std::map<std::string, int>::iterator itm = m.begin();
    std::map<std::string, int>::iterator enm = m.end();
    
    // 使用迭代器遍历map
    for(; itm != enm; ++itm)
        std::cout << itm->first << " : " << itm->second << "\n";
    
    return 0;
}
```

**输出：**
```
a : 10
b : 30
c : 50
d : 70
```

### 迭代器与常量迭代器

- **迭代器（Iterator）**：允许读写容器中的元素。
- **常量迭代器（Const Iterator）**：只能读取容器中的元素，不能修改。

**示例：使用常量迭代器**

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {32, 71, 12, 45, 26, 80, 53, 33};
    
    // 声明一个常量迭代器
    std::vector<int>::const_iterator it = v.begin();
    std::vector<int>::const_iterator en = v.end();
    
    // 使用常量迭代器遍历vector
    for(; it != en; ++it)
        std::cout << *it << "\n";
    
    // 尝试修改元素（会编译错误）
    // *it = 100; // 错误：无法修改
     
    return 0;
}
```

**输出：**
```
32
71
12
45
26
80
53
33
```

**示例：使用迭代器修改元素**

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {32, 71, 12, 45, 26, 80, 53, 33};
    
    // 声明一个迭代器
    std::vector<int>::iterator it = v.begin();
    std::vector<int>::iterator en = v.end();
    
    // 使用迭代器遍历并修改元素
    for(; it != en; ++it)
        *it = 23;
    
    // 输出修改后的vector
    for(int val : v)
        std::cout << val << " "; // 输出：23 23 23 23 23 23 23 23 
    std::cout << "\n";
    
    return 0;
}
```

**输出：**
```
23 23 23 23 23 23 23 23 
```

---

## 4. 算法（Algorithms）

STL提供了大量的算法，这些算法可以作用于容器的元素，通过迭代器进行操作。算法大致分为以下几类：

- **不修改容器的算法**：如查找、计数、验证条件等。
- **修改容器的算法**：如填充、反转、删除元素等。
- **分区算法**：将容器分成满足条件和不满足条件的两部分。
- **排序算法**：对容器中的元素进行排序。
- **最小值/最大值算法**：查找容器中的最小或最大元素。

所有这些算法都包含在头文件`<algorithm>`中。

**重要链接：**
- [cppreference.com - STL算法](https://en.cppreference.com/w/cpp/algorithm)

### 不修改容器的算法

这些算法用于查找、计数或验证容器中的元素，但不改变容器本身。

**常用算法：**
- `std::find`：查找指定元素。
- `std::find_if`：查找满足条件的元素。
- `std::count_if`：计算满足条件的元素数量。
- `std::all_of`：检查所有元素是否满足条件。
- `std::any_of`：检查是否存在满足条件的元素。
- `std::none_of`：检查是否所有元素都不满足条件。

**示例：使用`std::find`和`std::find_if`**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// 判断一个数是否为奇数
bool IsOdd(int i) { return ((i % 2) == 1); }

int main() {
    std::vector<int> v = {1, 3, 2, 5, 4};
    
    // 使用 std::find 查找元素5
    std::vector<int>::iterator it = std::find(v.begin(), v.end(), 5);
    if(it != v.end())
        std::cout << "在vector中找到元素: " << *it << "\n";
    else
        std::cout << "在vector中未找到元素\n";
    
    // 使用 std::find_if 查找第一个奇数
    it = std::find_if(v.begin(), v.end(), IsOdd);
    if(it != v.end())
        std::cout << "找到一个奇数: " << *it << "\n";
    else
        std::cout << "未找到奇数\n";
    
    // 使用 std::count_if 计算奇数的数量
    int mycount = std::count_if(v.begin(), v.end(), IsOdd);
    std::cout << "vector中有 " << mycount << " 个奇数\n";
    
    // 使用 std::all_of 检查所有元素是否为奇数
    if(std::all_of(v.begin(), v.end(), IsOdd))
        std::cout << "所有元素都是奇数\n";
    else
        std::cout << "存在非奇数元素\n";
    
    return 0;
}
```

**输出：**
```
在vector中找到元素: 5
找到一个奇数: 1
vector中有 3 个奇数
存在非奇数元素
```

### 修改容器的算法

这些算法用于填充、反转、删除或修改容器中的元素。

**常用算法：**
- `std::fill`：用指定值填充容器中的部分或全部元素。
- `std::reverse`：反转容器中的元素顺序。
- `std::remove_if`：移除满足条件的元素，并返回新的结束迭代器。
- `std::erase`：根据迭代器范围删除元素。

**示例：使用`std::fill`和`std::reverse`**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// 判断一个数是否为偶数
bool IsEven(int i) { return ((i % 2) == 0); }

int main() {
    std::vector<int> v(8, 0); // v: 0 0 0 0 0 0 0 0
    
    // 使用 std::fill 填充前4个元素为5
    std::fill(v.begin(), v.begin() + 4, 5); // v: 5 5 5 5 0 0 0 0
    
    // 使用 std::fill 填充第4到第6个元素为8
    std::fill(v.begin() + 3, v.end() - 2, 8); // v: 5 5 5 8 8 8 0 0
    
    // 使用 std::fill 填充最后两个元素为1
    std::fill(v.begin() + 6, v.end(), 1); // v: 5 5 5 8 8 8 1 1
    
    // 使用 std::reverse 反转整个vector
    std::reverse(v.begin(), v.end()); // v: 1 1 8 8 8 5 5 5
    
    // 使用 std::remove_if 移除所有偶数元素
    std::vector<int>::iterator newend = std::remove_if(v.begin(), v.end(), IsEven);
    // 此时v: 1 1 5 5 5 ? ? ?
    
    // 输出移除后的部分
    for(std::vector<int>::iterator it = v.begin(); it != newend; ++it)
        std::cout << *it << " "; // 输出：1 1 5 5 5 
    std::cout << "\n";
    
    // 使用 erase 删除被移除的元素
    v.erase(newend, v.end()); // v: 1 1 5 5 5
    
    return 0;
}
```

**输出：**
```
1 1 5 5 5 
```

### 分区算法

**分区算法**用于将容器分成满足某个条件和不满足条件的两部分。

**常用算法：**
- `std::partition`：重新排列容器，使得满足条件的元素位于前半部分，不满足的元素位于后半部分。

**示例：使用`std::partition`**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// 判断一个数是否为奇数
bool IsOdd(int i) { return ((i % 2) == 1); }

int main() {
    // 初始化一个vector：1 2 3 4 5 6 7 8 9
    std::vector<int> myvector;
    for(int i = 1; i < 10; ++i)
        myvector.push_back(i);
    
    // 使用 std::partition 将奇数放前面，偶数放后面
    std::vector<int>::iterator bound = std::partition(myvector.begin(), myvector.end(), IsOdd);
    
    // 输出奇数部分
    std::cout << "奇数元素:";
    for(std::vector<int>::iterator it = myvector.begin(); it != bound; ++it)
        std::cout << " " << *it;
    std::cout << "\n";
    
    // 输出偶数部分
    std::cout << "偶数元素:";
    for(std::vector<int>::iterator it = bound; it != myvector.end(); ++it)
        std::cout << " " << *it;
    std::cout << "\n";
    
    return 0;
}
```

**输出：**
```
奇数元素: 1 3 5 7 9
偶数元素: 2 4 6 8
```

### 排序算法

**排序算法**用于对容器中的元素进行排序。STL提供了多种排序算法，支持不同的比较策略。

**常用算法：**
- `std::sort`：对随机访问迭代器支持的容器进行排序。
- `std::list::sort`：对链表进行排序，支持自定义比较函数。

**比较函数：**
- 可以使用默认的`operator<`进行升序排序。
- 可以提供自定义的比较函数或函数对象，实现降序或其他排序规则。

**示例：使用`std::sort`对`std::vector`排序**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

// 自定义比较函数，降序排序
bool myfunc(int i, int j) { return (i > j); }

// 自定义比较类，降序排序
struct myclass {
    bool operator() (int i, int j) { return (i > j); }
};

int main() {
    std::vector<int> v = {32, 71, 12, 45, 26, 80, 53, 33};
    
    // 使用默认排序（升序）
    std::sort(v.begin(), v.begin() + 4);
    // v: 12 32 45 71 26 80 53 33
    
    // 使用自定义函数排序（降序）对后4个元素排序
    std::sort(v.begin() + 4, v.end(), myfunc);
    // v: 12 32 45 71 80 53 33 26
    
    // 使用自定义比较对象排序（降序）对所有元素排序
    std::sort(v.begin(), v.end(), myclass());
    // v: 80 71 53 45 33 32 26 12
    
    // 输出排序后的vector
    for(int val : v)
        std::cout << val << " ";
    std::cout << "\n";
    
    return 0;
}
```

**输出：**
```
80 71 53 45 33 32 26 12 
```

**示例：使用`std::list::sort`对`std::list`排序**

```cpp
#include <iostream>
#include <list>
#include <algorithm>

// 自定义比较类，降序排序
struct opSup {
    bool operator() (int i, int j) { return (i > j); }
};

int main() {
    std::list<int> l = {5, 2, 8, 3};
    
    // 使用默认排序（升序）
    l.sort(); // l: 2 3 5 8
    
    // 使用自定义比较对象排序（降序）
    l.sort(opSup()); // l: 8 5 3 2
    
    // 输出排序后的list
    for(int val : l)
        std::cout << val << " "; // 输出：8 5 3 2 
    std::cout << "\n";
    
    return 0;
}
```

**输出：**
```
8 5 3 2 
```

### 最小值/最大值算法

STL提供了多种算法用于查找容器中的最小值、最大值或同时查找两者。

**常用算法：**
- `std::min_element`：查找容器中的最小元素。
- `std::max_element`：查找容器中的最大元素。
- `std::minmax_element`：同时查找容器中的最小和最大元素。

**示例：使用`std::min_element`和`std::max_element`**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {32, 71, 12, 45, 26, 80, 53, 33};
    
    // 使用 std::min_element 查找最小元素
    std::vector<int>::iterator itmin = std::min_element(v.begin(), v.end());
    std::cout << "最小值 = " << *itmin << "\n"; // 输出：12
    
    // 使用 std::max_element 查找最大元素
    std::vector<int>::iterator itmax = std::max_element(v.begin(), v.end());
    std::cout << "最大值 = " << *itmax << "\n"; // 输出：80
    
    // 使用 std::minmax_element 同时查找最小和最大元素
    std::pair<std::vector<int>::iterator, std::vector<int>::iterator> itminmax;
    itminmax = std::minmax_element(v.begin(), v.end());
    std::cout << "最小值 = " << *itminmax.first << "\n";  // 输出：12
    std::cout << "最大值 = " << *itminmax.second << "\n"; // 输出：80
    
    return 0;
}
```

**输出：**
```
最小值 = 12
最大值 = 80
最小值 = 12
最大值 = 80
```

---

## 5. 额外功能

### std::pair 和 std::tuple

**`std::pair`**和**`std::tuple`**是STL提供的结构，用于存储不同类型的多个数据项。

#### std::pair

**`std::pair`**用于存储两个相关联的数据项，可能类型不同。它在`<utility>`头文件中定义。

**特点：**
- 存储两个元素，类型可以不同。
- 使用`first`和`second`访问元素。

**示例：**

```cpp
#include <iostream>
#include <utility>
#include <string>

int main() {
    // 创建一个pair，包含一个int和一个string
    std::pair<int, std::string> p = std::make_pair(5, "hello");
    
    // 访问并修改元素
    p.first = 7;
    p.second = "toto";
    
    std::cout << p.first << " " << p.second << std::endl; // 输出：7 toto
    
    return 0;
}
```

**输出：**
```
7 toto
```

**相关链接：**
- [cppreference.com - std::pair](https://en.cppreference.com/w/cpp/utility/pair)

#### std::tuple

**`std::tuple`**用于存储多个不同类型的数据项，数量不限。它在`<tuple>`头文件中定义。

**特点：**
- 存储多个元素，类型可以不同。
- 使用`std::get<index>(tuple)`访问元素。

**示例：**

```cpp
#include <iostream>
#include <tuple>
#include <string>

int main() {
    // 创建一个tuple，包含一个int、一个string和一个double
    std::tuple<int, std::string, double> p = std::make_tuple(5, "hello", 3.14);
    
    // 修改元素
    std::get<0>(p) = 7;
    std::get<1>(p) = "toto";
    std::get<2>(p) = 2.36;
    
    std::cout << std::get<0>(p) << " " 
              << std::get<1>(p) << " " 
              << std::get<2>(p) << std::endl; // 输出：7 toto 2.36
    
    return 0;
}
```

**输出：**
```
7 toto 2.36
```

### std::complex

**`std::complex`**是一个用于表示和操作复数的模板类，定义在`<complex>`头文件中。

**特点：**
- 模板参数指定复数的底层数值类型（如`double`、`float`）。
- 提供了丰富的数学运算符和函数支持，如加法、减法、乘法、除法、模长、共轭等。

**示例：**

```cpp
#include <iostream>
#include <complex>

int main() {
    // 创建一个double类型的复数，实部为2.3，虚部为5.2
    std::complex<double> c(2.3, 5.2);
    std::cout << "实部 = " << c.real() << " 虚部 = " << c.imag() << "\n"; // 输出：实部 = 2.3 虚部 = 5.2
    
    // 修改实部和虚部
    c.real(3.2); // 实部改为3.2
    c.imag(8.1); // 虚部改为8.1
    std::cout << "实部 = " << c.real() << " 虚部 = " << c.imag() << "\n"; // 输出：实部 = 3.2 虚部 = 8.1
    
    // 创建另一个复数
    std::complex<double> d(5.1, 6.9);
    
    // 复数相加
    c += d;
    std::cout << "c = " << c << "\n"; // 输出：c = (8.3, 15)
    
    return 0;
}
```

**输出：**
```
实部 = 2.3 虚部 = 5.2
实部 = 3.2 虚部 = 8.1
c = (8.3,15)
```

**相关链接：**
- [cppreference.com - std::complex](https://en.cppreference.com/w/cpp/numeric/complex)

### 智能指针Smart Pointers

**智能指针（Smart Pointers）**是C++11引入的一组模板类，用于管理动态分配的内存，自动处理资源释放，避免内存泄漏和其他资源管理问题。智能指针封装了原始指针，并通过RAII（资源获取即初始化）机制在对象生命周期结束时自动释放资源。

**主要类型：**
- **`std::unique_ptr`**：独占所有权，不可共享，不能复制，只能移动。
- **`std::shared_ptr`**：共享所有权，通过引用计数管理资源，多个`shared_ptr`可以指向同一个对象。
- **`std::weak_ptr`**：与`std::shared_ptr`配合使用，提供对对象的非拥有引用，避免循环引用。

#### std::unique_ptr

**`std::unique_ptr`**是一个独占所有权的智能指针，不能被复制，只能被移动。它适用于需要唯一所有权的场景。

**示例：**

```cpp
#include <iostream>
#include <memory>

int main() {
    // 创建一个unique_ptr，指向一个int
    std::unique_ptr<int> up1 = std::make_unique<int>(10);
    
    std::cout << "up1指向的值: " << *up1 << "\n"; // 输出：10
    
    // 转移所有权到up2
    std::unique_ptr<int> up2 = std::move(up1);
    
    if(up1)
        std::cout << "up1仍然指向某个值\n";
    else
        std::cout << "up1为空\n"; // 输出：up1为空
    
    std::cout << "up2指向的值: " << *up2 << "\n"; // 输出：10
    
    return 0;
}
```

**输出：**
```
up1指向的值: 10
up1为空
up2指向的值: 10
```

#### std::shared_ptr

**`std::shared_ptr`**允许多个指针共享对同一对象的所有权，通过内部的引用计数机制来管理资源的释放。当最后一个`shared_ptr`被销毁或重置时，管理的对象才会被删除。

**示例：**

```cpp
#include <iostream>
#include <memory>

struct A {
    A(int i, double d) : M_i(i), M_d(d) {}
    int i() const { return M_i; }
    double d() const { return M_d; }
private:
    int M_i;
    double M_d;
};

int main() {
    // 创建一个shared_ptr，指向A对象
    std::shared_ptr<A> a1 = std::make_shared<A>(12, 3.14);
    
    // 输出引用计数
    std::cout << "a1.use_count() = " << a1.use_count() << "\n"; // 输出：1
    
    // 访问A对象的成员
    std::cout << "a1: i = " << a1->i() << ", d = " << (*a1).d() << "\n"; // 输出：12 3.14
    
    // 通过解引用赋值创建A对象的副本
    A a2 = *a1;
    
    // 共享所有权
    std::shared_ptr<A> a3 = a1;
    
    // 检查引用计数
    if(a3) { // 检查a3是否指向有效对象
        std::cout << "a1.use_count() = " << a1.use_count() << " and a3.use_count() = " << a3.use_count() << "\n"; // 输出：2 and 2
        std::cout << "a1地址: " << a1.get() << ", a3地址: " << a3.get() << "\n"; // 输出：相同的地址
    }
    
    return 0;
}
```

**输出：**
```
a1.use_count() = 1
a1: i = 12, d = 3.14
a1.use_count() = 2 and a3.use_count() = 2
a1地址: 0x7ffeb3c2c970, a3地址: 0x7ffeb3c2c970
```

#### std::weak_ptr

**`std::weak_ptr`**是一种辅助智能指针，用于观察由`std::shared_ptr`管理的对象，但不拥有该对象。`weak_ptr`主要用于打破`shared_ptr`之间的循环引用，避免内存泄漏。

**示例：循环引用问题与解决**

```cpp
#include <memory>
#include <iostream>

struct B;

struct A {
    std::shared_ptr<B> b;
    ~A() { std::cout << "~A()\n"; }
};

struct B {
    std::shared_ptr<A> a;
    ~B() { std::cout << "~B()\n"; }
};

void useAnB() {
    auto a = std::make_shared<A>();
    auto b = std::make_shared<B>();
    std::cout << "state1: " << a.use_count() << " and " << b.use_count() << "\n"; // 输出：1 and 1
    
    a->b = b;
    b->a = a;
    std::cout << "state2: " << a.use_count() << " and " << b.use_count() << "\n"; // 输出：2 and 2
}

int main() {
    useAnB();
    std::cout << "Finished using A and B\n";
    return 0;
}
```

**输出：**
```
state1: 1 and 1
state2: 2 and 2
Finished using A and B
```

**问题分析：**
- 由于`A`和`B`互相持有`shared_ptr`，导致它们的引用计数永远不为0，析构函数不会被调用，造成内存泄漏。

**解决方案：使用`std::weak_ptr`打破循环引用**

```cpp
#include <memory>
#include <iostream>

struct B;

struct A {
    std::shared_ptr<B> b;
    ~A() { std::cout << "~A()\n"; }
};

struct B {
    std::weak_ptr<A> a; // 使用weak_ptr避免循环引用
    ~B() { std::cout << "~B()\n"; }
};

void useAnB() {
    auto a = std::make_shared<A>();
    auto b = std::make_shared<B>();
    std::cout << "state1: " << a.use_count() << " and " << b.use_count() << "\n"; // 输出：1 and 1
    
    a->b = b;
    b->a = a;
    std::cout << "state2: " << a.use_count() << " and " << b.use_count() << "\n"; // 输出：2 and 1
}

int main() {
    useAnB();
    std::cout << "Finished using A and B\n";
    return 0;
}
```

**输出：**
```
state1: 1 and 1
state2: 2 and 1
~A()
~B()
Finished using A and B
```

**说明：**
- 通过将`B`中的`std::shared_ptr<A>`改为`std::weak_ptr<A>`，打破了循环引用，确保对象能够正确析构。

**相关链接：**
- [cppreference.com - Smart Pointers](https://en.cppreference.com/w/cpp/memory)

### 时间测量

STL提供了`<chrono>`头文件，用于高精度的时间测量，方便我们评估代码的性能。

**常用组件：**
- `std::chrono::time_point`：表示一个时间点。
- `std::chrono::duration`：表示时间间隔。
- `std::chrono::system_clock`：系统时间，适用于普通时间测量。
- `std::chrono::high_resolution_clock`：高精度时间，适用于需要高精度的测量。

**示例：测量代码执行时间**

```cpp
#include <iostream>
#include <chrono>

int main() {
    // 定义时间点
    std::chrono::time_point<std::chrono::system_clock> start, end;
    
    // 记录开始时间
    start = std::chrono::system_clock::now();
    
    // 需要测量的代码部分
    // -------------------------------------------
    for(int i = 0; i < 1000000; ++i);
    // -------------------------------------------
    
    // 记录结束时间
    end = std::chrono::system_clock::now();
    
    // 计算时间差并转换为毫秒
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "耗时 (毫秒): " << elapsed_ms << "\n";
    
    // 计算时间差并转换为秒
    double elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "耗时 (秒): " << elapsed_s << "\n";
    
    return 0;
}
```

**输出：**
```
耗时 (毫秒): 5
耗时 (秒): 0
```

**说明：**
- 使用`std::chrono::system_clock`记录代码执行前后的时间点，通过`std::chrono::duration_cast`计算时间差，并转换为毫秒或秒。

**相关链接：**
- [cppreference.com - std::chrono](https://en.cppreference.com/w/cpp/chrono)

### 正则表达式regular-expressions

**`std::regex`**类用于表示和操作正则表达式，支持模式匹配、替换和提取等功能。它在`<regex>`头文件中定义。

**特点：**
- 支持复杂的字符串匹配和操作。
- 提供了丰富的匹配和替换功能。
- 支持多种正则表达式语法（如ECMAScript标准）。

**常用方法：**
- `std::regex_search`：在字符串中搜索匹配的模式。
- `std::regex_match`：检查整个字符串是否匹配模式。
- `std::regex_replace`：替换字符串中匹配的模式。

**示例：使用`std::regex`进行匹配和替换**

```cpp
#include <iostream>
#include <regex>
#include <string>

int main() {
    // 定义一个正则表达式模式 "abc"
    std::regex pattern("abc");
    
    // 定义一个目标字符串
    std::string target = "abcdefabc rtyabc";
    
    // 使用 regex_search 查找是否存在匹配的模式
    bool hasPattern = std::regex_search(target, pattern);
    std::cout << "找到模式 abc: " << (hasPattern ? "是" : "否") << "\n"; // 输出：是
    
    // 使用 regex_replace 将所有匹配的 "abc" 替换为 "123"
    std::string replacement = "123";
    std::string result = std::regex_replace(target, pattern, replacement);
    std::cout << "替换后的字符串: " << result << "\n"; // 输出：123def123 rty123
    
    // 定义一个复杂的正则表达式，用于匹配日期格式 xx/yy/zzzz 或 xx-yy-zzzz
    std::regex datePattern(R"((\d{2})[-/](\d{2})[-/](\d{4}))");
    std::string target2 = "abcdefabc22/11/2017rtyabc";
    
    // 使用 regex_search 查找并提取匹配的部分
    std::smatch m;
    bool hasPattern2 = std::regex_search(target2, m, datePattern);
    if(hasPattern2)
        std::cout << "找到日期模式: " << m.str() << "\n"; // 输出：22/11/2017
    
    return 0;
}
```

**输出：**
```
找到模式 abc: 是
替换后的字符串: 123def123 rty123
找到日期模式: 22/11/2017
```

**说明：**
- `std::regex_search`用于在字符串中查找是否存在匹配的模式，并可以提取匹配结果。
- `std::regex_replace`用于将匹配的模式替换为指定的字符串。
- 正则表达式模式可以使用原始字符串字面量（如`R"(...)"`）来提高可读性。

**相关链接：**
- [cppreference.com - std::regex](https://en.cppreference.com/w/cpp/regex)

---

## 6. 进一步学习

掌握了STL的基础知识后，可以进一步深入学习以下高级主题，以提升C++编程能力和代码效率：

### 高级模板元编程

- **模板偏特化与部分特化**：深入理解如何为复杂的模板参数组合进行偏特化，提供更灵活的模板实现。
- **SFINAE（Substitution Failure Is Not An Error）**：学习模板替换失败时的错误处理机制，利用它实现更智能的模板选择。
- **C++11/14/17新特性**：利用C++11及更高版本的特性，如`constexpr`、`auto`、`decltype`等，增强模板的功能和表达能力。

### 类型萃取（Type Traits）

- **类型特性检测**：利用标准库中的类型特性，如`std::is_integral`、`std::is_same`等，进行类型检测和选择。
- **条件编译**：根据类型特性选择不同的实现路径，提高代码的泛用性和效率。

### 现代C++模板编程

- **概念（Concepts）**：C++20引入的概念，用于定义模板参数的约束，增强模板的可读性和错误信息。
- **模板别名（Alias Templates）**：通过`using`关键字定义模板别名，简化复杂模板类型的使用。
- **变参模板（Variadic Templates）**：处理不定数量的模板参数，实现更灵活的模板结构。

### 模板与多态

- **模板与继承结合**：在模板类中实现多态，理解虚函数与模板的结合使用。
- **混合编程**：结合模板与传统的继承和多态，实现高效且灵活的代码结构。

### 模板优化与编译时间管理

- **模板实例化优化**：减少模板实例化带来的代码膨胀，提高编译效率。
- **显式实例化与模板编译**：通过显式实例化管理模板代码的编译过程，避免重复编译和链接错误。

### 实践项目与代码分析

- **开源项目学习**：通过分析开源项目中复杂的模板使用，理解实际应用中的模板编程技巧。
- **代码重构与优化**：将现有代码转换为模板化，实现代码复用和性能提升。

---

## 7. 推荐资源

### 书籍

- **《C++ Primer》**：全面介绍C++基础知识和编程技巧，适合初学者。
- **《Effective C++》**：深入探讨C++编程的最佳实践，适合有一定基础的开发者。
- **《The C++ Programming Language》 by Bjarne Stroustrup**：由C++语言的创建者编写，详尽介绍C++的各个方面。
- **《Modern C++ Design》 by Andrei Alexandrescu**：深入研究C++模板编程和设计模式的高级书籍。

### 在线教程

- **[cplusplus.com](https://www.cplusplus.com/)**：提供C++标准库和语言特性的详细文档，适合查阅语法和函数。
- **[LearnCpp](https://www.learncpp.com/)**：系统化的C++学习资源，适合初学者和进阶者，包含大量示例和练习。
- **[C++ Templates: The Complete Guide](https://www.stroustrup.com/)**：虽然是书籍，但在网络上有相关资源和讨论，可以作为参考。

### 视频课程

- **[Coursera - C++ For C Programmers](https://www.coursera.org/learn/c-plus-plus-a)**：适合有C语言基础的学习者，深入学习C++，包括模板编程。
- **[edX - Introduction to C++](https://www.edx.org/course/introduction-to-c-plus-plus)**：全面的C++入门课程，涵盖语言基础和面向对象编程。
- **[YouTube - TheCherno C++ Series](https://www.youtube.com/user/TheCherno)**：高质量的C++教学视频，涵盖基础到高级主题，包括模板。

### 其他资源

- **[Stack Overflow](https://stackoverflow.com/)**：遇到问题时，可以在此平台搜索或提问，获得社区的帮助。
- **[GitHub](https://github.com/)**：浏览和分析开源项目中的模板使用，学习实际应用中的技巧和最佳实践。
- **[cppreference.com](https://en.cppreference.com/w/cpp/)**：C++标准库和语言特性的详细参考资料，适合查阅标准和实现细节。

---

**祝您在学习C++标准模板库的过程中取得更大的进步！如果有任何疑问，请随时联系我。**