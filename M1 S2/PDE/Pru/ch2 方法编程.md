# 方法编程（Programmation de la méthode）

作者：Christophe Prud’homme, Laurent Navoret

---

## 1. Gmsh

Gmsh（开源网格生成器）：[https://gmsh.info](https://gmsh.info)

* 用于 2D/3D 有限元网格生成
* 提供图形界面

> **注意**：Gmsh 存在多个主要版本（2.x 与 4.x），语法略有差异

---

## 2. 示例：生成方形网格

### 2.1 创建 `carre.geo`

```geo
// 参数
h = 0.2;
L = 1;
// 定义四个角点（x, y, z, 网格尺寸）
Point(1) = {0, 0, 0, h};
Point(2) = {L, 0, 0, h};
Point(3) = {L, L, 0, h};
Point(4) = {0, L, 0, h};
// 定义边
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
// 定义面
Line Loop(5)    = {1,2,3,4};
Plane Surface(10) = {5};
// 标记物理边界和区域
Physical Line("Dirichlet")    = {1,2,3,4};
Physical Surface("Omega")     = {10};
```

### 2.2 命令行生成网格

```bash
# 进入当前目录后运行
gmsh carre.geo
```

在 GUI 中：

* **Mesh → 2D** 生成网格
* **File → Save Mesh** 保存 `.msh` 文件
* **Tools → Mesh → Visibility** 打开节点/单元编号显示

命令行直接输出 `.msh`：

```bash
gmsh -2 -order 1 carre.geo -o carre.msh
```

---

## 3. `.msh` 文件结构

### 3.1 格式与实体段

```text
$MeshFormat
4 0 8
$EndMeshFormat

$Entities
4 4 1 0
// 4 个点，4 条线，1 个面
1 0 0 0 0 0 0 0   // 点 1
2 1 0 0 1 0 0 0   // 点 2
3 1 1 0 1 1 0 0   // 点 3
4 0 1 0 0 1 0 0   // 点 4
// … (边定义略)
$EndEntities
```

### 3.2 节点段

```text
$Nodes
9 45            // 9 个实体，共 45 个节点条目
1  0 0 1        // 点 1, tag=1, 坐标(0,0,0)
…               // 边上中间节点
$EndNodes
```

### 3.3 单元段

```text
$Elements
1 68            // 1 类实体，共 68 个三角形
1 21 43 29      // 单元 1，节点 21,43,29
2 29 43 33      // 单元 2
…               
$EndElements
```

> **物理分组**：
> 通过 `Physical Surface(1) = {…}` 控制哪些单元包含在该物理域中；未标记的实体仍被输出
> 可用多条 `Physical Line` 来标记不同边界

---

## 4. 其他 Gmsh 指令

* 生成规则三角网格：

  ```geo
  Transfinite Surface {1};
  ```
* 将三角网格重组成四边形：

  ```geo
  Recombine Surface {1};
  ```
* 3D 挤出生成：

  ```geo
  Extrude {0,1.,1.5} {
    Surface {1};
    Layers {3};
    Recombine;
  }
  ```

---

## 5. 刚度矩阵装配

对于全局矩阵

$$
  A_{ij}
  = \int_\Omega \nabla\varphi_i\cdot\nabla\varphi_j
  = \sum_{K}
    \int_{K} \nabla\psi_{K,r_i}\cdot\nabla\psi_{K,r_j}
$$

* 大部分积分为零，只需遍历所有单元 $K$，累加局部贡献

### 5.1 连接矩阵

```math
connect(\mathrm{ind}_K, r) = i
```

* 局部基函数编号 $r$ 映射到全局自由度 $i$
* 装配时，对每个单元 $K$ 的每对本地基函数 $(r_i,r_j)$，计算积分并加到 $A[i,j]$

---

## 6. 数值积分（Quadrature）

### 6.1 定义

对单元 $K$ 及权重-点对 $\{(\xi_\ell,\omega_\ell)\}_{\ell=1}^n$，定义

$$
  Q(\phi) = \sum_{\ell=1}^n \omega_\ell\,\phi(\xi_\ell)
  \approx \int_K \phi(x)\,dx
$$

* $\xi_\ell\in K$：积分点
* $\omega_\ell\in\mathbb{R}$：积分权重

### 6.2 精度阶

最大整数 $m$ 使得对所有多项式 $\phi\in\mathbb P_m$ 都能精确积分。误差估计：

$$
  \bigl|Q(\phi)-\!\int_K\phi\bigr|
  \le C\,\mathrm{mes}(K)\,h_K^{m+1}\,\|\phi\|_{C^{m+1}(K)}.
$$

### 6.3 三角形参考单元示例

* 参考三角形 $\hat K=\{(x,y)\ge0,\;x+y\le1\}$，面积 $S=1/2$
* 常见配点与权重：

| 点数 $n$ |  精度 | 重心坐标                           | 重复次数 |        权重        |
| :----: | :-: | :----------------------------- | :--: | :--------------: |
|    3   |  1  | $(1,0,0)$                      |   3  |   $\tfrac13\,S$  |
|    3   |  2  | $(\tfrac12,\tfrac12,0)$        |   3  |   $\tfrac13\,S$  |
|    7   |  3  | $(\tfrac13,\tfrac13,\tfrac13)$ |   1  | $\tfrac9{20}\,S$ |
|    …   |  …  | …                              |   …  |         …        |

“**重复次数**”指对重心坐标各自置换后得到的点数。

---

## 7. 装配算法示例（Python 伪码）

```python
# 遍历所有单元
for ik in range(Nel):
    compute detTK, inv_dTK
    # 遍历所有积分点
    for l in range(nq):
        w = poids[l]
        # 遍历本地基函数对 (ni, nj)
        for ni in range(nk):
            i = connect(ik, ni)
            derpsii = derpsi[ni,l] @ inv_dTK
            for nj in range(nk):
                j = connect(ik, nj)
                derpsij = derpsi[nj,l] @ inv_dTK
                A[i,j] += detTK * w * (derpsii @ derpsij)
```

同理可组装右端项 `F`。

---

## 8. Dirichlet 边界条件

有多种实现方式：

* **消元法（Elimination）**：将边界自由度对应行置零，对角置 1，并在载荷中写入边界值
* **惩罚法（Penalisation）**：对边界节点添加大系数 $\varepsilon^{-1}$
* **Nitsche 方法**：弱施加边界，参考 Feel++ 文档

### 8.1 消元法示例

```python
for i in dir_nodes:
    A[i, :] = 0       # 对应行清零
    A[i, i] = 1       # 对角置 1
    F[i]    = g_i     # 边界值
```

> 为保持矩阵对称，可同时修改对应列和载荷，但共轭梯度等迭代法不受影响。

---

## 9. 误差与范数计算

### 9.1 $L^2$ 范数

对离散解 $u_h=\sum_i u_i\phi_i$：

$$
  \|u_h\|_{L^2}^2
  \approx \sum_K \sum_{\ell} w_\ell\;
    \bigl(u_h\circ T_K(\xi_\ell)\bigr)^2
    \;\bigl|\det\nabla T_K\bigr|.
$$

> **提示**：若 $u_h\in P^k$，需选用至少 $2k$ 阶配点。

### 9.2 $H^1$ 半范数

$$
  |\!|u_h|\!|_{H^1}^2
  = \int_\Omega \|\nabla u_h\|^2
  \approx \sum_K \sum_{\ell} w_\ell\;
    \bigl(\hat\nabla u_h(\xi_\ell)\,\nabla T_K^{-1}\bigr)^2
    \;\bigl|\det\nabla T_K\bigr|.
$$

* $\hat\nabla u_h$ 在参考单元上由基函数梯度与系数线性组合得到
* 若需对任意 $u$ 计算，可先插值到 $P^1$ 空间，再按上述公式积分

---

