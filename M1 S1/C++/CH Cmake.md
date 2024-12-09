# CMake：多平台软件构建自动化


---

## 目录

1. [CMake 概述](#1-cmake-概述)
2. [CMake 示例：Hello World](#2-cmake-示例hello-world)
3. [CMake 项目构建流程](#3-cmake-项目构建流程)
4. [CMake 配置阶段](#4-cmake-配置阶段)
5. [CMake 生成阶段](#5-cmake-生成阶段)
6. [CMake 构建阶段](#6-cmake-构建阶段)
7. [CMake 安装阶段](#7-cmake-安装阶段)
8. [定义 C++ 编译器](#8-定义-c++-编译器)
9. [CMake 创建库](#9-cmake-创建库)
10. [CMake 文件层次结构](#10-cmake-文件层次结构)
11. [CMake 变量](#11-cmake-变量)
12. [CMake 目标属性](#12-cmake-目标属性)
13. [CMake 编程语言](#13-cmake-编程语言)
14. [CMake 依赖管理](#14-cmake-依赖管理)
15. [使用 CTest 进行测试](#15-使用-ctest-进行测试)

---

## 1. CMake 概述

### 1.1 什么是 CMake？

**CMake** 是一个多平台的软件构建系统。它的主要功能包括：

- **检测构建所需的前提条件**：检查编译器、库和工具是否满足项目的需求。
- **确定项目中各组件的依赖关系**：分析源代码文件之间的依赖，确保正确的编译顺序。
- **生成适合当前平台的构建文件**：根据不同的编译环境（如 Makefile、Ninja、Visual Studio 项目文件等）生成相应的构建配置文件。
- **自动化编译和安装过程**：简化编译、链接和安装软件的步骤，减少手动操作。

### 1.2 CMake 的优势

- **跨平台支持**：支持多种操作系统和编译器，如 Windows、Linux、macOS 等。
- **灵活性**：能够处理复杂的项目结构和依赖关系。
- **易于集成**：与许多集成开发环境（IDE）兼容，如 Visual Studio、CLion 等。
- **社区支持**：拥有广泛的用户基础和丰富的文档资源。

---

## 2. CMake 示例：Hello World

### 2.1 最小化 CMake 配置

以下示例展示了如何使用 CMake 配置一个简单的 C++ 项目，该项目包含一个源文件 `hello.cpp`，并生成一个名为 `Hello` 的可执行文件。

#### 项目目录结构

```
cpp_project
├── CMakeLists.txt
└── src
    └── hello.cpp
```

#### CMakeLists.txt 内容

```cmake
cmake_minimum_required(VERSION 3.21)

# 设置项目名称
project(MonProjet)

# 添加可执行文件
add_executable(Hello src/hello.cpp)
```

#### hello.cpp 内容

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

### 2.2 构建步骤

在 Unix 环境下，通过以下命令生成并编译项目：

```bash
mkdir build && cd build
cmake ..
make
```

**解释**：

1. `mkdir build && cd build`：创建并进入构建目录。
2. `cmake ..`：运行 CMake，配置项目并生成 Makefile。
3. `make`：使用生成的 Makefile 编译项目，生成可执行文件 `Hello`。

---

## 3. CMake 项目构建流程

CMake 的构建过程通常分为四个阶段：

1. **配置阶段（Configuration）**
2. **生成阶段（Generation）**
3. **构建阶段（Build）**
4. **安装阶段（Installation）**

### 3.1 配置阶段

在配置阶段，CMake 执行以下操作：

- **解析所有的 CMakeLists.txt 文件**：读取项目的配置文件，理解项目结构和依赖关系。
- **检测编译器和工具链**：确定使用的编译器版本及其特性。
- **查找并配置所需的库和包**：确保项目所依赖的外部库已经安装并配置正确。

**输出文件**：

- `CMakeCache.txt`：包含所有配置变量及其值。
- `CMakeFiles/` 目录：包含临时文件和日志信息。
- `install.cmake`：安装脚本。

### 3.2 生成阶段

在生成阶段，CMake 根据配置阶段的结果，生成适用于当前平台的构建文件，如 Makefile、Ninja 构建文件或 IDE 项目文件。

**命令示例**：

```bash
cmake . -B build
```

**参数说明**：

- `.`：指定源代码目录（即包含 CMakeLists.txt 的目录）。
- `-B build`：指定构建目录为 `build`。

### 3.3 构建阶段

在构建阶段，使用生成的构建文件进行实际的编译和链接。

**命令示例**：

```bash
cmake --build build
```

**常用选项**：

- `--target <target>`：构建指定的目标。
- `--clean-first`：在构建前清理之前的构建产物。
- `--parallel <num>`：并行构建，指定并行任务数。
- `--verbose`：显示详细的构建过程。

**示例**：

```bash
cmake --build build --target Hello --verbose
```

### 3.4 安装阶段

安装阶段负责将构建生成的可执行文件、库文件等安装到指定的位置。

#### 修改 CMakeLists.txt 以支持安装

```cmake
cmake_minimum_required(VERSION 3.21)

# 设置项目名称
project(MonProjet)

# 添加可执行文件
add_executable(Hello src/hello.cpp)

# 包含 GNU 安装目录变量
include(GNUInstallDirs)

# 安装可执行文件到指定目录
install(TARGETS Hello DESTINATION ${CMAKE_INSTALL_BINDIR})
```

#### 安装命令示例

```bash
cmake --install build --prefix install
```

**参数说明**：

- `--install build`：指定构建目录。
- `--prefix install`：指定安装前缀目录为 `install`。

**效果**：

可执行文件 `Hello` 将被安装到 `install/bin` 目录下（假设 `CMAKE_INSTALL_BINDIR` 设置为 `bin`）。

---

## 4. CMake 配置阶段

### 4.1 配置阶段详解

配置阶段是 CMake 构建流程的第一步，主要完成以下任务：

1. **解析 CMakeLists.txt 文件**：理解项目结构和构建需求。
2. **检测编译器和工具链**：确定使用的编译器版本及其特性。
3. **查找所需的库和包**：确保项目所依赖的外部库已经安装并配置正确。

### 4.2 配置阶段输出

- **CMakeCache.txt**：存储所有配置变量及其值，方便后续使用。
- **CMakeFiles/** 目录：包含临时文件、生成的构建规则和日志信息。
- **install.cmake**：安装脚本，用于安装阶段将构建产物复制到指定位置。

---

## 5. CMake 生成阶段

### 5.1 生成构建文件

生成阶段根据配置阶段的结果，创建适用于当前平台的构建文件。这些文件可以是 Makefile、Ninja 构建文件，或者是特定 IDE（如 Visual Studio、Xcode）的项目文件。

### 5.2 选择生成器

CMake 支持多种生成器，每种生成器对应不同的构建工具。常见的生成器包括：

- **Unix Makefiles**：适用于 Unix/Linux 平台，生成标准的 Makefile。
- **Ninja**：高效的构建系统，适用于大规模项目。
- **Visual Studio**：生成 Visual Studio 的解决方案和项目文件。
- **Xcode**：生成 Xcode 的项目文件，适用于 macOS。

### 5.3 选择生成器示例

使用 `-G` 选项可以显式指定生成器：

```bash
cmake . -B build -G Ninja
```

**说明**：

- `-G Ninja`：指定使用 Ninja 作为构建工具。

**查看可用生成器**：

```bash
cmake --help
```

**输出示例**：

```
Generators
  Visual Studio 16 2019        = Generates Visual Studio 2019 project files.
  Ninja                        = Generates build.ninja files.
  Unix Makefiles               = Generates standard UNIX makefiles.
  ...
```

---

## 6. CMake 构建阶段

### 6.1 构建阶段目标

构建阶段的主要目标是将源代码编译为目标文件（如 `.o`、`.obj`），并链接生成最终的可执行文件或库文件。

### 6.2 构建步骤

在构建阶段，CMake 执行以下操作：

1. **编译**：将源代码文件编译为中间目标文件。
2. **链接**：将目标文件链接生成最终的可执行文件或库文件。
3. **处理依赖关系**：确保按照正确的顺序编译和链接，避免依赖问题。

### 6.3 构建命令示例

```bash
cmake --build build
```

**常用选项**：

- `--target <target>`：构建指定的目标（如 `Hello`）。
- `--clean-first`：在构建前清理之前的构建产物。
- `--parallel <num>`：并行构建，指定并行任务数。
- `--verbose`：显示详细的构建过程。

**示例**：

```bash
cmake --build build --target Hello --verbose
```

**输出示例**：

```
[ 50%] Building CXX object CMakeFiles/Hello.dir/src/hello.cpp.o
[100%] Linking CXX executable Hello
[100%] Built target Hello
```

### 6.4 构建选项解释

- `--target Hello`：仅构建 `Hello` 这个目标。
- `--verbose`：显示详细的构建命令，有助于调试。

---

## 7. CMake 安装阶段

### 7.1 安装阶段目标

安装阶段将构建生成的可执行文件、库文件及其他资源文件复制到预定的安装目录，方便用户使用或部署。

### 7.2 修改 CMakeLists.txt 以支持安装

```cmake
cmake_minimum_required(VERSION 3.21)

# 设置项目名称
project(MonProjet)

# 添加可执行文件
add_executable(Hello src/hello.cpp)

# 包含 GNU 安装目录变量
include(GNUInstallDirs)

# 安装可执行文件到指定目录
install(TARGETS Hello DESTINATION ${CMAKE_INSTALL_BINDIR})
```

**解释**：

- `include(GNUInstallDirs)`：引入 GNU 安装目录的变量，如 `CMAKE_INSTALL_BINDIR`。
- `install(TARGETS Hello DESTINATION ${CMAKE_INSTALL_BINDIR})`：指定将 `Hello` 可执行文件安装到 `bin` 目录。

### 7.3 安装命令示例

```bash
cmake --install build --prefix install
```

**参数说明**：

- `--install build`：指定构建目录。
- `--prefix install`：指定安装前缀目录为 `install`。

**效果**：

可执行文件 `Hello` 将被安装到 `install/bin` 目录下。

### 7.4 手动调用安装

虽然 CMake 在销毁构建对象时会自动关闭文件流，但有时需要手动调用 `close()` 方法来释放资源或确保文件被正确关闭。

```cpp
std::ofstream ofile("myfile.txt");
if (ofile) {
    ofile << "Hello, World!" << std::endl;
    ofile.close(); // 手动关闭文件流
}
```

**说明**：

- `ofile.close()`：手动关闭文件流，确保数据写入文件。
- 手动关闭文件流在需要立即释放资源或继续对文件进行其他操作时非常有用。

---

## 8. 定义 C++ 编译器

### 8.1 指定编译器

CMake 默认会选择系统上的默认编译器，但可以通过以下方式显式指定使用的编译器。

#### 方法一：通过命令行参数

```bash
cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++ . -B build
```

**解释**：

- `-DCMAKE_CXX_COMPILER=/usr/bin/g++`：指定使用 `/usr/bin/g++` 作为 C++ 编译器。
- `.`：源代码目录。
- `-B build`：构建目录为 `build`。

#### 方法二：通过环境变量

```bash
export CXX=/usr/bin/g++
cmake . -B build
```

**解释**：

- `export CXX=/usr/bin/g++`：设置环境变量 `CXX` 为指定的编译器。
- `cmake . -B build`：运行 CMake，使用环境变量中指定的编译器。

### 8.2 指定 C++ 标准

可以在 `CMakeLists.txt` 中指定使用的 C++ 标准版本，确保代码按照特定的标准进行编译。

#### 示例：使用 C++17 标准

```cmake
cmake_minimum_required(VERSION 3.21)

# 设置项目名称
project(MonProjet)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加可执行文件
add_executable(Hello hello.cpp)
```

**解释**：

- `set(CMAKE_CXX_STANDARD 17)`：指定使用 C++17 标准。
- `set(CMAKE_CXX_STANDARD_REQUIRED True)`：确保编译器严格遵守指定的 C++ 标准。

### 8.3 优化编译

在编译过程中，可以通过设置构建类型来启用不同级别的优化。

#### 常见的构建类型

- **Debug**：
  - **用途**：开发阶段，包含调试符号，关闭优化。
  - **编译选项**：`-g`（包含调试信息）、`-O0`（关闭优化）。
  
- **Release**：
  - **用途**：生产环境，开启优化，关闭调试符号。
  - **编译选项**：`-O3`（最高级别的优化）、`-DNDEBUG`（关闭断言）。
  
- **RelWithDebInfo**：
  - **用途**：发布版本，同时保留调试信息。
  - **编译选项**：`-O2`（优化级别 2）、`-g`（包含调试信息）、`-DNDEBUG`。
  
- **MinSizeRel**：
  - **用途**：优化可执行文件大小。
  - **编译选项**：`-Os`（优化代码大小）、`-DNDEBUG`。

#### 设置构建类型示例

```bash
cmake -DCMAKE_BUILD_TYPE=Release . -B build
```

**说明**：

- `-DCMAKE_BUILD_TYPE=Release`：设置构建类型为 Release，启用优化。

#### 在 CMakeLists.txt 中设置默认构建类型

```cmake
cmake_minimum_required(VERSION 3.21)

# 设置项目名称
project(MonProjet)

# 设置默认构建类型为 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "选择构建类型" FORCE)
endif()

# 添加可执行文件
add_executable(Hello hello.cpp)
```

**解释**：

- `if(NOT CMAKE_BUILD_TYPE)`：如果未指定构建类型，则设置默认值。
- `set(CMAKE_BUILD_TYPE Release CACHE STRING "选择构建类型" FORCE)`：将默认构建类型设置为 Release。

---

## 9. CMake 创建库

### 9.1 创建静态库和共享库

CMake 允许创建静态库（`.a` 或 `.lib`）和共享库（`.so` 或 `.dll`）。

#### 示例：创建静态库

```cmake
add_library(MyLib STATIC func.cpp)
target_include_directories(MyLib PUBLIC "${PROJECT_SOURCE_DIR}/include")
```

#### 示例：创建共享库

```cmake
add_library(MyLib SHARED func.cpp)
target_include_directories(MyLib PUBLIC "${PROJECT_SOURCE_DIR}/include")
```

**解释**：

- `add_library(MyLib STATIC func.cpp)`：创建一个名为 `MyLib` 的静态库，包含 `func.cpp`。
- `target_include_directories(MyLib PUBLIC "${PROJECT_SOURCE_DIR}/include")`：指定包含目录，`PUBLIC` 表示依赖该库的目标也会使用这些包含目录。

### 9.2 使用库

#### 示例：链接库到可执行文件

```cmake
add_executable(Hello2 hello2.cpp func.cpp)
target_include_directories(Hello2 PUBLIC "${PROJECT_SOURCE_DIR}")
```

**解释**：

- `add_executable(Hello2 hello2.cpp func.cpp)`：创建可执行文件 `Hello2`，包含 `hello2.cpp` 和 `func.cpp`。
- `target_include_directories(Hello2 PUBLIC "${PROJECT_SOURCE_DIR}")`：指定包含目录，确保 `hello2.cpp` 能找到 `func.hpp`。

#### 示例：使用目标链接库

```cmake
# 创建共享库
add_library(mylib SHARED func.cpp)
target_include_directories(mylib PUBLIC "${PROJECT_SOURCE_DIR}")

# 创建可执行文件并链接库
add_executable(Hello3 hello2.cpp)
target_link_libraries(Hello3 mylib)
```

**解释**：

- `add_library(mylib SHARED func.cpp)`：创建共享库 `mylib`。
- `target_link_libraries(Hello3 mylib)`：将 `mylib` 链接到可执行文件 `Hello3`。

### 9.3 示例代码

#### hello2.cpp

```cpp
#include <iostream>
#include "func.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;
    toto();
    return 0;
}
```

#### func.hpp

```cpp
#ifndef FUNC_HPP
#define FUNC_HPP

int toto();

#endif
```

#### func.cpp

```cpp
#include <iostream>
#include "func.hpp"

int toto() {
    std::cout << "Hola" << std::endl;
    return 3;
}
```

### 9.4 解释

- **函数声明与定义**：
  - `func.hpp` 声明了函数 `toto()`。
  - `func.cpp` 定义了函数 `toto()`，并在其中输出 `"Hola"`。
  
- **主程序**：
  - `hello2.cpp` 中调用了 `toto()` 函数，输出 `"Hello, World!"` 和 `"Hola"`。

---

## 10. CMake 文件层次结构

### 10.1 多目录项目结构

当项目包含多个子目录时，需要在每个子目录中添加 `CMakeLists.txt` 文件，并在主 `CMakeLists.txt` 中使用 `add_subdirectory()` 指定子目录。

#### 示例项目结构

```
cpp_project
├── CMakeLists.txt
└── myapp
    ├── CMakeLists.txt
    └── test.cpp
```

#### 根目录的 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.21)

# 设置项目名称
project(MonProjet)

# 添加子目录
add_subdirectory(myapp)
```

#### 子目录 myapp 的 CMakeLists.txt

```cmake
# 添加可执行文件
add_executable(MyApp test.cpp)
```

**说明**：

- **主 CMakeLists.txt**：
  - 使用 `add_subdirectory(myapp)` 将 `myapp` 目录包含进来。
  
- **子目录 CMakeLists.txt**：
  - 添加可执行文件 `MyApp`，包含 `test.cpp`。

**注意**：

- 子目录中的 `CMakeLists.txt` 不需要重新定义项目名称、CMake 最低版本或 C++ 标准。这些信息在主目录中已经定义，并会自动继承到子目录。

---

## 11. CMake 变量

### 11.1 变量类型

- **本地变量（Local Variables）**：
  - 使用 `set()` 命令定义，仅在当前作用域内有效。
  
  ```cmake
  set(MY_VARIABLE "value")
  set(MY_LIST "one" "two")
  set(MY_LISTB "one;two") # 等同于前一行
  ```

- **缓存变量（Cache Variables）**：
  - 用于在配置阶段存储全局变量，可以通过命令行参数传递或持久化保存。
  
  ```cmake
  set(MY_CACHE_VARIABLE "VALUE" CACHE STRING "Description")
  ```

  **说明**：
  
  - `CACHE STRING "Description"`：定义一个字符串类型的缓存变量，并添加描述。
  - 使用 `FORCE` 选项可以强制覆盖已有的缓存变量，但通常不推荐使用，除非必要。

### 11.2 访问变量

- 使用 `${}` 语法访问变量值。

```cmake
message(STATUS "MY_VARIABLE = ${MY_VARIABLE}")
```

### 11.3 预定义变量

CMake 提供了许多预定义变量，方便项目配置和管理：

- `CMAKE_SOURCE_DIR`：项目的根源代码目录。
- `CMAKE_BINARY_DIR`：项目的根构建目录。
- `PROJECT_SOURCE_DIR`：项目的源代码目录（即 `project()` 被调用的位置）。
- `PROJECT_BINARY_DIR`：项目的构建目录。
- `CMAKE_CURRENT_SOURCE_DIR`：当前 CMakeLists.txt 文件所在的源代码目录。
- `CMAKE_CURRENT_BINARY_DIR`：当前 CMakeLists.txt 文件所在的构建目录。

**示例**：

```cmake
message(STATUS "项目源代码目录: ${CMAKE_SOURCE_DIR}")
message(STATUS "项目构建目录: ${CMAKE_BINARY_DIR}")
```

---

## 12. CMake 目标属性

在 CMake 中，**目标**（Target）是构建过程中的基本单元，如可执行文件、库文件等。通过设置目标属性，可以定义目标的编译选项、包含目录、链接库等。

### 12.1 目标使用要求（Usage Requirements）

**Usage Requirements** 指定目标的使用属性，包括：

- **包含目录（Include Directories）**：编译时需要搜索的头文件目录。
- **编译定义（Compile Definitions）**：预处理器定义。
- **编译选项（Compile Options）**：编译器选项，如警告级别、优化级别等。
- **链接库（Link Libraries）**：需要链接的库文件。

### 12.2 目标属性命令

#### 12.2.1 `target_compile_definitions`

为目标添加预处理器定义，相当于在编译时使用 `-D` 选项。

**语法**：

```cmake
target_compile_definitions(<target> [INTERFACE|PUBLIC|PRIVATE] [definitions...])
```

**示例**：

```cmake
add_library(MyLib STATIC mylib.cpp)

target_compile_definitions(MyLib
    PRIVATE MYLIB_INTERNAL=1            # 仅对 MyLib 本身生效
    PUBLIC MYLIB_API=__declspec(dllexport) # 对 MyLib 及其依赖生效
)
```

**说明**：

- `PRIVATE`：仅对目标本身生效。
- `PUBLIC`：对目标及依赖于该目标的其他目标生效。
- `INTERFACE`：仅对依赖于该目标的其他目标生效，不影响目标本身。

#### 12.2.2 `target_compile_options`

为目标添加编译器选项，如警告级别、优化选项等。

**语法**：

```cmake
target_compile_options(<target> [BEFORE] [INTERFACE|PUBLIC|PRIVATE] [options...])
```

**示例**：

```cmake
add_library(MyLib STATIC mylib.cpp)

target_compile_options(MyLib
    PRIVATE -Wall -Wextra  # 仅对 MyLib 本身生效
    PUBLIC -O2             # 对 MyLib 及其依赖生效
)
```

**说明**：

- `BEFORE`：将选项添加到编译器选项的前面。
- `PRIVATE`、`PUBLIC`、`INTERFACE` 的含义与 `target_compile_definitions` 相同。

#### 12.2.3 `target_include_directories`

指定目标的包含目录，相当于编译时使用 `-I` 选项。

**语法**：

```cmake
target_include_directories(<target> [SYSTEM] [BEFORE] [INTERFACE|PUBLIC|PRIVATE] [directories...])
```

**示例**：

```cmake
add_library(MyLib STATIC mylib.cpp)

target_include_directories(MyLib
    PUBLIC ${PROJECT_SOURCE_DIR}/include  # 对 MyLib 及其依赖生效
    PRIVATE ${PROJECT_SOURCE_DIR}/src      # 仅对 MyLib 本身生效
)
```

**说明**：

- `SYSTEM`：将指定的目录作为系统包含目录，通常用于第三方库，避免产生编译警告。
- `PRIVATE`、`PUBLIC`、`INTERFACE` 的含义与前述相同。

#### 12.2.4 `target_link_libraries`

指定目标需要链接的库，相当于编译时使用 `-l` 选项。

**语法**：

```cmake
target_link_libraries(<target> [INTERFACE|PUBLIC|PRIVATE] [libraries...])
```

**示例**：

```cmake
add_library(MyLib STATIC mylib.cpp)

target_link_libraries(MyLib
    PUBLIC AnotherLib        # 对 MyLib 及其依赖生效
    PRIVATE SomeInternalLib  # 仅对 MyLib 本身生效
)
```

**说明**：

- `PRIVATE`：仅对目标本身生效。
- `PUBLIC`：对目标及依赖于该目标的其他目标生效。
- `INTERFACE`：仅对依赖于该目标的其他目标生效，不影响目标本身。

---

## 13. CMake 编程语言

### 13.1 CMake 语言特点

CMake 是一种动态类型的编程语言，提供了多种编程结构和功能，增强了构建过程的灵活性和模块化。

### 13.2 条件语句

#### 13.2.1 `if` 语句

用于根据条件执行不同的代码块。

**语法**：

```cmake
if(<condition>)
    # 条件为真时执行的命令
else()
    # 条件为假时执行的命令
endif()
```

**示例**：

```cmake
if(EXISTS ${CMAKE_SOURCE_DIR}/config.txt)
    message(STATUS "config.txt exists!")
else()
    message(STATUS "config.txt not found.")
endif()
```

**说明**：

- `EXISTS` 检查指定文件是否存在。
- `message(STATUS ...)` 用于输出状态信息。

#### 13.2.2 变量条件判断

```cmake
if(myvariable)
    # 如果 myvariable 是 "ON", "YES", "TRUE", "Y" 或非零数值
else()
    # 如果 myvariable 是 "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND", "" 或以 "-NOTFOUND" 结尾
endif()

if("${variable}")
    # 如果 variable 不为假（false-like）
else()
    # 如果 variable 为假，通常是空字符串
endif()
```

### 13.3 循环语句

#### 13.3.1 `foreach` 循环

用于遍历列表或范围内的值。

**语法**：

```cmake
foreach(<variable> [values...])
    # 循环体
endforeach()
```

**示例**：

```cmake
foreach(var a b c d)
    message(STATUS "当前值: ${var}")
endforeach()

foreach(i RANGE 1 5)
    message(STATUS "i 的值: ${i}")
endforeach()
```

**输出结果**：

```
当前值: a
当前值: b
当前值: c
当前值: d
i 的值: 1
i 的值: 2
i 的值: 3
i 的值: 4
i 的值: 5
```

### 13.4 生成器表达式

生成器表达式用于根据特定条件动态设置属性或选项。

**示例**：仅在 Debug 配置下添加编译选项

```cmake
target_compile_options(MyTarget PRIVATE "$<$<CONFIG:Debug>:-O0>")
```

**说明**：

- `$<$<CONFIG:Debug>:-O0>`：如果当前配置是 Debug，则添加 `-O0` 编译选项。

### 13.5 函数定义

#### 13.5.1 定义函数

```cmake
function(add_two_numbers a b)
    math(EXPR result "${a} + ${b}")
    message(STATUS "Sum of ${a} and ${b} is: ${result}")
endfunction()
```

#### 13.5.2 调用函数

```cmake
add_two_numbers(5 10)
```

**输出**：

```
Sum of 5 and 10 is: 15
```

#### 13.5.3 具名参数

使用 `cmake_parse_arguments` 处理具名参数。

**示例**：

```cmake
function(COMPLEX)
    cmake_parse_arguments(
        COMPLEX_PREFIX
        "SINGLE;ANOTHER"
        "ONE_VALUE;ALSO_ONE_VALUE"
        "MULTI_VALUES"
        ${ARGN}
    )
    message(STATUS "SINGLE: ${COMPLEX_PREFIX_SINGLE}")
    message(STATUS "ANOTHER: ${COMPLEX_PREFIX_ANOTHER}")
    message(STATUS "ONE_VALUE: ${COMPLEX_PREFIX_ONE_VALUE}")
    message(STATUS "ALSO_ONE_VALUE: ${COMPLEX_PREFIX_ALSO_ONE_VALUE}")
    message(STATUS "MULTI_VALUES: ${COMPLEX_PREFIX_MULTI_VALUES}")
endfunction()

complex(SINGLE ONE_VALUE value MULTI_VALUES some other values)
```

**输出**：

```
SINGLE: TRUE
ANOTHER: FALSE
ONE_VALUE: value
ALSO_ONE_VALUE: 
MULTI_VALUES: some;other;values
```

**说明**：

- `cmake_parse_arguments` 用于解析函数参数，支持具名参数和选项。
- 结果存储在 `COMPLEX_PREFIX_` 前缀的变量中。

---

## 14. CMake 依赖管理

### 14.1 使用 `find_package`

`find_package` 命令用于查找并配置项目所依赖的外部库、工具或模块。

**语法**：

```cmake
find_package(<PackageName> [version] [REQUIRED] [COMPONENTS components...] [QUIET])
```

**参数说明**：

- `<PackageName>`：要查找的包名称，如 Eigen、Boost、OpenCV、Qt 等。
- `[version]`：指定包的最小版本要求。
- `[REQUIRED]`：如果包未找到，则生成错误并停止配置。
- `[COMPONENTS components...]`：指定包的具体组件。
- `[QUIET]`：静默模式，不输出查找失败的信息。

**示例**：

```cmake
find_package(Eigen3 3.4 REQUIRED)
```

**说明**：

- 查找 Eigen3 版本至少为 3.4 的包。
- `REQUIRED` 表示如果未找到 Eigen3，将停止配置并报错。

### 14.2 查找结果变量

如果 `find_package` 成功找到包，CMake 将定义一组变量来帮助使用该包：

- `<PackageName>_FOUND`：布尔值，表示包是否被找到。
- `<PackageName>_INCLUDE_DIRS`：包含目录路径。
- `<PackageName>_LIBRARIES`：需要链接的库文件。

### 14.3 使用目标链接库（Modern CMake）

现代 CMake 推荐使用目标链接库的方式，而不是直接使用变量。这种方法更加模块化和安全。

**示例**：

```cmake
cmake_minimum_required(VERSION 3.21)

# 设置项目名称
project(MonProjet)

# 查找 Eigen3 包
find_package(Eigen3 3.4 REQUIRED)

# 添加可执行文件
add_executable(Hello src/hello.cpp)

# 链接 Eigen3 库到可执行文件
target_link_libraries(Hello Eigen3::Eigen)
```

**说明**：

- `Eigen3::Eigen` 是 Eigen3 包定义的目标库，确保链接正确且包含必要的依赖。
- 使用目标方式避免了直接依赖变量，提高了构建系统的健壮性。

---

## 15. 使用 CTest 进行测试

### 15.1 什么是 CTest？

**CTest** 是 CMake 集成的测试驱动工具，用于自动化执行项目中的测试，并报告测试结果。它支持单元测试、集成测试等多种测试类型。

### 15.2 CTest 的工作流程

CTest 的工作流程包括以下几个步骤：

1. **配置测试**：在 `CMakeLists.txt` 中定义测试。
2. **构建测试**：编译测试代码，生成测试可执行文件。
3. **执行测试**：运行测试可执行文件，验证程序功能。
4. **报告结果**：收集和显示测试结果，包括成功、失败和执行时间等信息。

### 15.3 在 CMake 中启用测试

首先，需要在 `CMakeLists.txt` 中启用测试功能：

```cmake
enable_testing()
```

### 15.4 添加测试

使用 `add_test` 命令定义测试。

**示例**：

```cmake
# 添加测试可执行文件
add_executable(test_hello src/test_hello.cpp)

# 注册测试
add_test(NAME test_hello_a COMMAND test_hello)
add_test(NAME test_hello_b COMMAND test_hello abcdefg)
add_test(NAME test_hello_c COMMAND test_hello "Hello, World!")
```

### 15.5 示例测试代码

#### test_hello.cpp

```cpp
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "no arguments!" << std::endl;
        return 1;
    }

    std::string msg = argv[1];
    std::cout << "msg: " << msg << std::endl;

    if (msg != "Hello, World!") {
        return 2;
    }

    return 0;
}
```

**说明**：

- 程序接受一个参数并输出。
- 如果参数不是 `"Hello, World!"`，则返回错误代码 `2`，否则返回 `0`。

### 15.6 执行测试

构建并运行测试：

```bash
cmake --build build
ctest --test-dir build
```

**输出示例**：

```
Test project build
    Start 1: test_hello_a
1/3 Test #1: test_hello_a .................   Passed    0.00 sec
    Start 2: test_hello_b
2/3 Test #2: test_hello_b .................   Passed    0.00 sec
    Start 3: test_hello_c
3/3 Test #3: test_hello_c .................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 3

Total Test time (real) =   0.01 sec
```

### 15.7 CTest 命令选项

- **并行执行测试**：

  ```bash
  ctest --test-dir build -j4
  ```

  **说明**：使用 `-j4` 并行执行 4 个测试。

- **运行特定测试**：

  ```bash
  ctest --test-dir build -R test_hello_c
  ```

  **说明**：仅运行名称匹配 `test_hello_c` 的测试。

- **查看详细输出**：

  ```bash
  ctest --test-dir build -VV
  ```

  **说明**：显示详细的测试执行过程和输出。

- **重新运行失败的测试**：

  ```bash
  ctest --test-dir build --rerun-failed
  ```

  **说明**：仅重新运行上一次失败的测试。

### 15.8 添加带有超时的测试

可以为测试设置执行时间限制，防止测试陷入无限循环或长时间运行。

**示例**：

```cmake
# 添加可能导致无限循环的测试
add_test(NAME test_hello_d COMMAND test_hello "infinite_loop")

# 设置执行超时为 10 秒
set_tests_properties(test_hello_d PROPERTIES TIMEOUT 10)
```

**解释**：

- `add_test` 定义了一个名为 `test_hello_d` 的测试，运行时传递参数 `"infinite_loop"`。
- `set_tests_properties` 设置该测试的超时时间为 10 秒。如果测试在 10 秒内未完成，将被强制终止并标记为失败。

### 15.9 测试结果示例

```bash
ctest --test-dir build
```

**输出结果**：

```
Test project build
    Start 1: test_hello_a
1/4 Test #1: test_hello_a .................   Passed    0.00 sec
    Start 2: test_hello_b
2/4 Test #2: test_hello_b .................   Passed    0.00 sec
    Start 3: test_hello_c
3/4 Test #3: test_hello_c .................   Passed    0.00 sec
    Start 4: test_hello_d
4/4 Test #4: test_hello_d .................   Failed    10.00 sec

50% tests passed, 1 tests failed out of 4

Total Test time (real) =   10.02 sec
```

**说明**：

- `test_hello_d` 超时未完成，因此被标记为失败。

---

## 16. 进一步学习

### 16.1 高级流操作

- **处理二进制文件**：学习如何使用 CMake 配置和构建处理二进制数据的项目。
- **自定义缓冲区**：了解流缓冲区的工作原理，并学习如何自定义缓冲区以优化性能。

### 16.2 异常处理

- 学习如何在 C++ 中使用异常处理机制（`try-catch`）来处理运行时错误，提升程序的健壮性。

### 16.3 格式化读取

- 学习使用 `std::getline`、`std::ws` 等函数进行复杂的输入操作，如读取整行数据或忽略特定字符。

### 16.4 自定义操纵符

- 学习如何创建自定义的流操纵符（Manipulators），以扩展流的功能，满足特定的格式化需求。

### 16.5 国际化和本地化

- 学习如何使用 C++ 流处理国际化和本地化的数据格式，如不同语言的字符编码和日期格式。

### 16.6 模板和泛型编程

- 掌握如何使用 C++ 模板编程与 CMake 结合，实现更通用和可复用的输入输出功能。

### 16.7 标准模板库（STL）

- 深入学习 C++ 提供的各种容器（如 `vector`、`list`、`map` 等）、算法和迭代器，提升编程效率。

---

## 推荐资源

### 书籍

- **《C++ Primer》**：全面介绍 C++ 基础知识和编程技巧。
- **《Effective C++》**：深入探讨 C++ 编程的最佳实践。
- **《The C++ Programming Language》 by Bjarne Stroustrup**：由 C++ 语言的创建者编写，详尽介绍 C++ 的各个方面。

### 在线教程

- **[cplusplus.com](http://www.cplusplus.com/)**：提供 C++ 标准库和语言特性的详细文档。
- **[Learn C++](https://www.learncpp.com/)**：系统化的 C++ 学习资源，适合初学者和进阶者。

### 视频课程

- **[Coursera - C++ For C Programmers](https://www.coursera.org/learn/c-plus-plus-a)**：适合有 C 语言基础的学习者，深入学习 C++。
- **[edX - Introduction to C++](https://www.edx.org/course/introduction-to-c-plus-plus)**：全面的 C++ 入门课程，涵盖语言基础和面向对象编程。

---

**祝您在 CMake 和 C++ 的学习中取得更大的进步！如果有任何疑问，请随时联系我。**
