以下为题目（法语）翻译成的中文版本：

---

**斯特拉斯堡大学 (Université de Strasbourg)

计算科学2 (Cours Calcul Scientifique 2)

实验课 (TP) 4：有限元方法 (Méthode des éléments finis)**

本次实验课的目标是用一维有限元方法来求解带有狄利克雷边界条件的椭圆方程：

$$
-\,u''(x) + u(x) = f(x),\quad x \in (0,1),
$$

$$
u(0) = 0,\quad u(1) = 0,
$$

其中 $f(x)$ 为已知函数，$u(x)$ 为未知函数。

请先导入以下 Python 模块：

```python

import numpy as np

import scipy as sp

import scipy.sparse as spsp

import scipy.sparse.linalg as spsplin

```

注意在每一步都要测试所编写的代码。

---

### 第1部分. 一次元 (P1) 有限元

1. **定义 `mesh` 类**

   - 包含以下属性：
   - `Nel`：网格的单元（子区间）数量
   - `Ndof`：自由度（degree of freedom）的数量
   - `xmin, xmax`：区间的左右端点
   - `nodes`：储存网格节点坐标的数组
   - `h`：储存各个单元长度（大小）的数组
2. **在该类中定义函数**

   - `init_uniform(self, Nel, xmin, xmax)`：根据给定的单元数 Nel、区间 $[xmin, xmax]$ 构造**均匀**网格
   - `init_random(self, Nel, xmin, xmax)`：根据给定的单元数 Nel、区间 $[xmin, xmax]$ 构造**随机**网格
3. **定义 `fem` 类**

   - 其中包含一个 `mesh` 对象（即将刚才定义好的网格类实例化）。
4. **构造有限元矩阵**

   矩阵 $A = (A_{i,j})_{i,j}$ 在弱形式下可写为

   $$
   A = \bigl(A_{i,j}\bigr)_{i,j} 

   = \left(\int_{x_{\min}}^{x_{\max}} \phi_i'(x)\,\phi_j'(x)\,dx + \int_{x_{\min}}^{x_{\max}} \phi_i(x)\,\phi_j(x)\,dx \right)_{i,j}
   $$

   这里 $\{\phi_i\}$ 是一次元(P1)的基函数族。

   - 问：矩阵 $A$ 的维度多大？
   - 问：如何根据单元大小 $h_j$ 计算出各系数的显式表达式？

   在 `fem` 类中，定义函数 `matrixA_P1(self)` 用于构造一次元的有限元矩阵 $A$，并且建议使用**稀疏**的对角结构（或更通用的稀疏结构）来存储。
5. **右端项向量**

   定义函数 `rhs_P1(self, f)`：

   - 输入：一个函数 $f$。
   - 输出：一个向量 $\mathbf{b}$，其中每个分量近似

   $$
   \int_{x_{\min}}^{x_{\max}} f(x)\,\phi_i(x)\,dx.
   $$

   - 要求：分片使用梯形公式（trapezoid rule）来数值近似该积分。
6. **验证方法**

   为了测试与验证，我们希望使用一个“制造”出来的解（manufactured solution）。

   假设

   $$
   u(x) = \sin(\pi x)
   $$

   请推导相应的 $f(x)$，使得上述 $u(x)$ 满足给定的方程和边界条件。
7. **在 `fem` 类中编写 `solve(self, f, plot=True)` 函数**

   - 输入：一个函数 $f$。
   - 过程：

   1. 组装矩阵 $A$。
   2. 组装右端项 $\mathbf{b}$。
   3. 求解离散方程 $A \cdot \mathbf{u} = \mathbf{b}$，得到有限元解 $\mathbf{u}$。

   - 若参数 `plot=True`，则在同一张图上绘制精确解（例如 $\sin(\pi x)$）和有限元的近似解。
   - 用上面制造的解来测试方法。

---

### 第2部分. (数值研究)

8. **在 `mesh` 类中添加函数 `norm_P1(self, u)`**

   - 输出：
   - $\|u\|_{L^2}$ 范数
   - $\|u\|_{H^1}$ 的半范数（离散意义下），分别使用合适的数值积分来计算。
   - 在 `fem` 的 `solve` 函数中做相应修改，使其返回与精确解之间的 $L^2$ 误差。
9. **收敛阶的数值实验**

   - 令网格步长 $h = \max_j h_j$。
   - 选取单元数量列表 `[20, 40, 80, 160, 320, 640]`，每次生成网格并求解，计算在 $L^2$ 范数中的误差，做对数坐标下的误差-步长图。
   - 问：数值上得到的收敛阶是多少？与理论期望的收敛阶是否一致？
10. **条件数**

    - 使用 `np.linalg.cond` 函数来计算矩阵的条件数。
    - 分别针对**均匀网格**和**随机网格**，随 $N_{el}$ 改变，输出或绘制条件数，并进行分析评论。

---

### 第3部分. 二次元 (P2) 有限元

11. **在 `mesh` 类中添加属性 `deg` 和数组 `dof`**

    - `deg`：元素的多项式次数（在这里为2，即二次元 P2）
    - `dof`：大小为 `deg*Nel + 1` 的数组，它包含了二次元网格所有自由度对应的坐标。
    - 对每个单元 $[x_j, x_{j+1}]$，在其内部根据 `deg+1` 个点的方式均匀分布：

    $$
    x_{j,k}^{(\mathrm{deg})} = x_j + \frac{h_j \cdot k}{\mathrm{deg}},\quad 0 \le k \le \mathrm{deg},
    $$

    其中 $h_j = x_{j+1} - x_j$。

    - 注意处理整体编号以及最后一个点 $x_{\max}$ 的情况（边界）。
    - 更新自由度数量 `Ndof` 的定义（现在每个单元有 `deg+1` 个局部基函数，但相邻单元会有节点重合，需要小心不重复计数）。
12. **定义 `connect(self, el, k)` 函数**

    - 功能：根据单元编号 `el`（从0或1开始，要与自己的约定保持一致）以及局部自由度编号 `k`，返回全局自由度编号 `i`。
    - 即：`i = connect(el, k)`。
13. **在 `init_uniform` 与 `init_rand` 函数中**

    - 完成对 `dof` 的构造（可使用双重循环）。
14. **组装矩阵 $A$**

    - 矩阵形式与一次元类似：

    $$
    A = \bigl(A_{i,j}\bigr)_{i,j}

    = \left(\int_{x_{\min}}^{x_{\max}} \phi_i'(x)\,\phi_j'(x)\,dx + \int_{x_{\min}}^{x_{\max}} \phi_i(x)\,\phi_j(x)\,dx \right)_{i,j}
    $$

    不同之处在于基函数换成了二次元(P2)的基函数。

    - 在参考单元 $[0,1]$ 上，二次元基函数记为

    $$
    \bar{\phi}_0(x) = (2x - 1)(x - 1), \quad

    \bar{\phi}_1(x) = 4x(1 - x), \quad

    \bar{\phi}_2(x) = x(2x - 1).
    $$

    在每个单元 $[x_\ell, x_{\ell+1}]$ 上，做适当的平移与缩放后得到实际基函数。

    - 已知在参考单元 $[0,1]$ 上有如下矩阵（供组装使用）

    $$
    M =

    \begin{pmatrix}

    2 & 1 & -0.5 \\

    1 & 8 & 1 \\

    -0.5 & 1 & 2

    \end{pmatrix}\Big/15

    \quad,\quad

    K =

    \begin{pmatrix}

    7 & -8 & 1 \\

    -8 & 16 & -8 \\

    1 & -8 & 7

    \end{pmatrix}\Big/3
    $$

    其中 $M$ 用于 $\int \bar{\phi}_k \bar{\phi}_{k'}$ 的计算，$K$ 用于 $\int \bar{\phi}_k' \bar{\phi}_{k'}'$ 的计算。

    - 组装思路：遍历所有单元 $\ell$，在参考单元上计算好对应的局部贡献，然后乘以单元大小或缩放系数，再加到全局矩阵正确的行列位置上。

    例如，伪代码：

    ```python

    for el in range(self.mesh.Nel):

    for ni in range(self.mesh.deg+1):

    i = self.mesh.connect(el, ni)

    for nj in range(self.mesh.deg+1):

    j = self.mesh.connect(el, nj)

    if 0 < i <= self.mesh.Ndof and 0 < j <= self.mesh.Ndof:

    A[i-1, j-1] += K[ni, nj] * (...) # 与导数相关的缩放

    A[i-1, j-1] += M[ni, nj] * (...) # 与函数值相关的缩放

    ```

    可使用 DOK（Dictionary Of Keys）形式的稀疏矩阵来存储以方便组装。最后打印或检查该矩阵。
15. **右端项向量 `rhs(self, f)`**

    - 输入：函数 $f$。
    - 输出：对向量

    $$
    b_i \approx \int_{x_{\min}}^{x_{\max}} f(x)\,\phi_i(x)\,dx
    $$

    的数值近似。

    - 要求：在每个单元上使用辛普森（Simpson）积分或类似方法，然后累加到全局 `b` 向量上。
16. **在 `mesh` 类中编写 `norm(self, u)` 函数**

    - 返回：向量 $u$ 对应的函数在 $L^2$ 范数下的数值近似。
    - 依然可在每个单元上分片使用辛普森积分等方法，然后在全局上求和取平方根。
17. **重新进行收敛性研究**

    - 参考第2部分的做法，用二次元来替换一次元，并比较误差与理论收敛阶。
18. **[可选]**

    - 若要让 `matrixA`, `rhs`, `norm` 等函数在 P1 和 P2 的情形下都通用，可在函数内部根据 `mesh.deg` 分支处理
