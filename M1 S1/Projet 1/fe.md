### 说明

- **Point、Triangle、Segment** 是用于表示几何数据的结构体。
- **Mesh** 保存了所有节点、三角形及边界线段，并提供计算面积与边界长度的成员函数。
- **MshReader** 提供静态函数 `readMsh` 用于从 .msh 文件中读取数据填充 Mesh 实例。
- **VtkExporter** 提供两个静态方法，用于导出仅包含几何信息的 VTK 文件以及带有标量场数据的 VTK 文件。
- **Quadrature** 类中包含静态函数 `integrateOnMesh`，通过不同积分阶数对 Mesh 上的函数值进行积分。
- **PoissonProblem** 用于描述 Poisson 方程的右端项和边界条件。
- **PoissonSolver** 负责组装线性系统、处理 Dirichlet 边界条件，最后调用线性求解器计算数值解。


下面对项目中所有主要函数进行整体介绍，帮助你了解各个函数的作用及相互关系。

---

## 1. **main.cpp 中的全局函数和 main 函数**

- **全局函数**

  - `double myFunction(double x, double y)`  
    返回函数值 *f(x, y) = x² + y²*，用于计算节点处的标量场并导出结果。

  - `double oneFunction(double x, double y)`  
    返回常数 1.0，用于数值积分测试，即对常量 1 在整个区域上积分（积分值应等于区域面积）。

  - `double poissonF(double x, double y)`  
    表示 Poisson 方程的源项，返回常数 1.0，即 -Δu = 1。

  - `double poissonG(double x, double y)`  
    表示 Dirichlet 边界条件，返回 0.0，即 u(x, y) = 0 在边界上。

  - `double computeMaxValue(const std::vector<double>& solution)`  
    接收数值解向量，使用标准库算法 `max_element` 求解向量中最大值，用于监控收敛性测试时数值解的最大值。

  - `void convergenceTest(const std::vector<std::string>& meshFiles, const PoissonProblem& prob)`  
    传入多个 mesh 文件名和 Poisson 问题描述，对不同 mesh 网格求解 Poisson 方程。对每个网格：  
    - 读取网格数据  
    - 使用 `PoissonSolver` 求解方程  
    - 计算解向量中最大值并打印  
    用于测试网格细化下数值解的收敛性。

- **main 函数**

  - 定义了网格数据和结果输出所在的目录，创建结果目录；
  - 对一组已知网格文件进行处理：
    - 从 .msh 文件中读取网格数据（调用 `MshReader::readMsh`）；
    - 计算网格的总面积和边界长度（调用 `Mesh::computeTotalArea` 与 `Mesh::computeBoundaryLength`）；
    - 将面积和边界信息写入文本文件；
    - 将网格几何信息导出为 VTK 格式文件（调用 `VtkExporter::exportMesh`）；
    - 对节点上赋予标量场值（通过函数 `myFunction` 计算）后导出完整 VTK 文件（调用 `VtkExporter::exportCompleteVTK`）；
    - 利用 `Quadrature::integrateOnMesh` 对常量函数进行数值积分测试，验证积分结果；
  - 进行收敛性测试，针对不同密度的网格求解 Poisson 方程（调用 `convergenceTest`）；
  - 分别针对两种网格（square2d_M2.msh 与 square2d_perforated.msh）求解 Poisson 方程，组装矩阵、求解方程，并导出计算结果为 VTK 文件（调用 `PoissonSolver::solve` 及 `VtkExporter::exportCompleteVTK`）；
  - 最后打印提示信息，结束程序运行。

---

## 2. **Mesh.hpp 与 Mesh.cpp：网格数据与几何计算**

- **结构体**

  - `Point`  
    用于存储二维点的坐标（x, y）。

  - `Triangle`  
    存储三角形单元，包含三个节点（以节点数组索引的形式保存）。

  - `Segment`  
    存储边界线段，由两个节点索引和物理标签组成（物理标签可用于区分不同边界条件）。

- **类 Mesh**

  - **成员数据**  
    - `nodes`：存储所有点的信息。  
    - `triangles`：存储网格中所有三角形单元。  
    - `segments`：存储描述网格边界的线段信息。

  - **成员函数**

    - `double computeTotalArea() const`  
      遍历所有三角形，通过公式 *area = 0.5 * |det|*（det 为三角形顶点构成的行列式）计算每个三角形的面积，并将所有面积累加，得到整个网格的总面积。

    - `double computeBoundaryLength() const`  
      遍历所有边界段（segments），计算每个线段的长度（两端节点的欧氏距离），累加后返回总边界长度。

---

## 3. **MshReader.hpp 与 MshReader.cpp：.msh 文件读取**

- **类 MshReader**

  - **静态成员函数**

    - `static void readMsh(const std::string &filename, Mesh &mesh)`  
      负责读取 .msh 格式网格文件，并填充传入的 Mesh 实例。  
      主要步骤：
      - 清空 Mesh 中的 nodes、triangles、segments；
      - 打开文件，依次读取节点段（$Nodes）和单元段（$Elements）；
      - 根据单元类型（elemType），当为 1 时构造 Segment，当为 2 时构造 Triangle；
      - 节点编号在 .msh 文件中通常以 1 为起始，读取时相应减 1 存储。

---

## 4. **VtkExporter.hpp 与 VtkExporter.cpp：VTK 文件导出**

- **类 VtkExporter**

  - **静态成员函数**

    - `static void exportMesh(const Mesh &mesh, const std::string &outFilename)`  
      导出仅包含几何信息的 VTK 文件。步骤包括：
      - 写入 VTK 文件头部、版本信息、数据格式说明；
      - 输出点坐标（所有节点位置）；
      - 输出单元信息（三角形单元的节点索引）及单元类型（VTK_CELL_TYPE 对应数字 5 表示三角形）。

    - `static void exportCompleteVTK(const Mesh &mesh, const std::vector<double> &nodeValues, const std::string &fieldName, const std::string &outFilename)`  
      在导出几何信息的基础上，增加节点数据（标量场），用于后续可视化。步骤：
      - 与 `exportMesh` 相似地输出点和单元信息；
      - 额外写入节点数据部分（使用 POINT_DATA 标签），输出标量数据的名称和对应值。

---

## 5. **Quadrature.hpp 与 Quadrature.cpp：数值积分**

- **枚举类型 QuadratureOrder**
  
  定义数值积分的精度阶数：
  
  - `ORDER1`：一阶（使用三角形重心进行积分）；
  - `ORDER2`：二阶（三点积分公式）；
  - `ORDER3`：三阶（四点积分公式）。

- **类 Quadrature**

  - **静态成员函数**

    - `static double integrateOnMesh(const Mesh &mesh, double (*func)(double, double), QuadratureOrder order)`  
      对传入的函数 `func` 在整个网格上进行数值积分。  
      具体步骤：
      - 遍历网格中所有三角形，调用辅助函数 `integrateOnTriangle` 计算每个三角形的积分值；
      - 累加各三角形积分值得到整个域上的积分。

- **内部辅助函数**

  - `static double triArea(const Mesh &mesh, int triIdx)`  
    计算单个三角形面积（在 `integrateOnTriangle` 中调用）。

  - `static double integrateOnTriangle(const Mesh &mesh, int triIdx, double (*f)(double, double), QuadratureOrder order)`  
    根据积分阶数，采用不同的积分公式：
    - **ORDER1**：使用三角形重心积分；
    - **ORDER2**：使用三个积分点，每个点赋权重 area/3；
    - **ORDER3**：使用四个积分点，权重和位置按照四点公式计算。

---

## 6. **PoissonSolver.hpp 与 PoissonSolver.cpp：泊松方程求解**

- **数据结构 PoissonProblem**

  一个结构体，用于描述 Poisson 方程问题，包括：
  
  - `std::function<double(double, double)> f`：右侧源项函数；
  - `std::function<double(double, double)> g`：Dirichlet 边界条件函数。

- **类 PoissonSolver**

  用于组装全局线性系统并求解离散化后的 Poisson 方程。  
  **主要成员函数和内部流程：**

  - **构造函数**  
    `PoissonSolver(const Mesh &mesh, const PoissonProblem &problem)`  
    保存对网格数据和问题描述的引用，供后续求解使用。

  - **`std::vector<double> solve()`**  
    主求解函数，整体步骤：
    1. **组装系统**  
       - 调用私有函数 `assembleSystem`，构造稀疏矩阵的系数列表（存储为二维 `vector`，每个元素为 `(col, value)` 对）和负载向量 `F`。
       - 使用 Eigen 的 `Triplet` 结构将稀疏矩阵数据进行转换。
    2. **处理 Dirichlet 边界条件**  
       - 对于被标记为边界的节点（通过内部函数 `isBoundaryNode` 判断），将对应方程修改为 u = g，即将该节点所在行清零并将对角线设为 1，同时修改负载向量。
    3. **求解线性系统**  
       - 使用 Eigen 的 `SparseLU` 求解器求解该线性系统，并计时输出装配和求解时间。
    4. **返回解向量**  
       - 将 Eigen 计算的解复制到标准库的 `std::vector<double>` 中返回。

  - **内部函数**
  
    - `bool isBoundaryNode(int nodeIdx) const`  
      判断给定节点是否位于边界上，具体实现为扫描网格中所有边界线段，只要节点编号匹配即返回 `true`。
  
    - `void assembleSystem(std::vector< std::vector< std::pair<int, double> > > &A_coeff, std::vector<double> &F)`  
      组装全局刚度矩阵和负载向量的过程：
      - 遍历所有三角形，计算局部刚度矩阵（基于形函数梯度）以及局部负载向量（使用重心积分计算右侧函数值乘以面积分布）；
      - 将局部矩阵和向量累加到全局系统的对应位置。

---

## 总体说明

项目整体目标为读取网格数据、计算几何量、导出可视化结果，并利用有限元方法求解 Poisson 方程。各模块相互配合：

- **Mesh** 与 **MshReader** 负责网格信息的存储和读取；
- **VtkExporter** 实现了网格几何和数值解的可视化导出；
- **Quadrature** 提供数值积分支持，用于验证网格的积分精度；
- **PoissonSolver** 组装并求解离散化后的 Poisson 方程，结合 **PoissonProblem** 描述的边界和源项条件。

通过 `main` 函数，将上述各个功能模块串联起来，对不同网格进行处理、积分验证与方程求解，并将结果保存为文本和 VTK 文件，方便后续分析和可视化。