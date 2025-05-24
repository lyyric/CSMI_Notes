# 数值方法与偏微分方程（Méthodes numériques pour les EDP）

作者：Christophe Prud’homme, Laurent Navoret

---

## 1. 微积分回顾

### 1.1 标量场（Champs scalaires）

* 定义：

$$
u: \mathbb{R}^d \to \mathbb{R},\quad x \mapsto u(x)
$$
* 梯度（gradient）：

$$
\nabla u = \bigl(\partial_{x_1}u,\dots,\partial_{x_d}u\bigr)^T \in \mathbb{R}^d
$$

### 1.2 向量场（Champs de vecteur）

* 定义：

$$
U: \mathbb{R}^d \to \mathbb{R}^d,\quad x \mapsto (U_1(x),\dots,U_d(x))
$$
* 散度（divergence）：

$$
\nabla\cdot U = \sum_{i=1}^d \partial_{x_i}U_i
$$
* 梯度（Jacobian 矩阵）：

$$
\nabla U = \bigl(\partial_{x_j}U_i\bigr)_{i,j}
= \begin{pmatrix}
   \nabla U_1^T\\
   \vdots\\
   \nabla U_d^T
  \end{pmatrix}
$$

### 1.3 拉普拉斯算子（Laplacien）

* 对于标量场：

$$
\Delta u = \nabla\cdot\nabla u
= \sum_{i=1}^d \partial_{x_i}^2 u
$$
* 对于向量场：

$$
\Delta U = \bigl(\Delta U_1,\dots,\Delta U_d\bigr),
\quad
\Delta U_i = \sum_{j=1}^d \partial_{x_j}^2 U_i
$$

### 1.4 可视化（Visualisation）

* 常用软件：Paraview、Visit、Mayavi、Gmsh 等
* 标量场

  * 等值面：$\{x\mid u(x)=c\}$
  * 图形：$\{(x,\alpha\,u(x))\}\subset\mathbb{R}^{d+1}$，$\alpha$ 为缩放因子
* 向量场

  * 按分量或大小作等值面 $x\mapsto |U(x)|$
  * 流线：解 $X'(t)=U(X(t)),\,X(0)=x_0$
  * 箭头图（glyphs）：在各点绘制向量

---

## 2. 基本积分公式

### 2.1 Green 公式

* 若 $\Omega$ 为有界 Lipschitz 开集，$u\in C^1(\overline\Omega)$，则

$$
\int_{\Omega} \partial_{x_i}u\,dx
= \int_{\partial\Omega}u\,n_i\,d\sigma
$$
* 分部积分：

$$
\int_{\Omega}\partial_{x_i}u\,v\,dx
= -\int_{\Omega}u\,\partial_{x_i}v\,dx
  +\int_{\partial\Omega}u\,v\,n_i\,d\sigma
$$
* 对标量场的 Green 公式：

$$
\int_{\Omega}\Delta u\,v\,dx
= -\int_{\Omega}\nabla u\cdot\nabla v\,dx
  +\int_{\partial\Omega}(n\cdot\nabla u)\,v\,d\sigma
$$

### 2.2 Gauss 定理

* 若 $U\in C^1(\overline\Omega,\mathbb{R}^d)$，则

$$
\int_{\Omega}\nabla\cdot U\,dx
= \int_{\partial\Omega} U\cdot n\,d\sigma
$$

### 2.3 曲线与曲面积分

* 曲线积分：若 $\Sigma\subset\mathbb{R}^2$ 由 $x(t)$, $t\in[a,b]$ 参数化，则

$$
\int_\Sigma f(x)\,d\sigma
= \int_a^b f\bigl(x(t)\bigr)\,\|x'(t)\|\,dt
$$
* 曲面积分：若 $\Sigma\subset\mathbb{R}^3$ 由 $(t,s)\mapsto x(t,s)$ 参数化，法向量 $n=\partial_t x\times\partial_s x$，则

$$
\int_\Sigma f(x)\,d\sigma
= \int\!\!\!\int f\bigl(x(t,s)\bigr)\,\|\partial_t x\times\partial_s x\|\,dt\,ds
$$
* 向量场通量：

$$
\int_\Sigma U\cdot d\sigma
= \int_\Sigma (U\cdot n)\,d\sigma
$$

---

## 3. Poisson 问题

### 3.1 问题描述

在有界 Lipschitz 区域 $\Omega$ 上求

$$
  \begin{cases}
    -\Delta u(x) = f(x), & x\in\Omega,\\
    u(x)=0,           & x\in\partial\Omega.
  \end{cases}
$$

### 3.2 物理意义

* $u$：弹性膜位移、化学浓度、温度、静电势等
* 平衡状态模型（无时间项）

### 3.3 变分解释

等价于对任意子区域 $\omega\subset\Omega$，

$$
  -\int_{\partial\omega}n\cdot\nabla u\,d\sigma
  = \int_\omega f\,dx,
$$

对应力平衡或物质守恒。

---

## 4. 有限元方法（FEM）步骤

1. **变分问题**
   在 Sobolev 空间

$$
 H^1(\Omega)=\{u\in L^2,\,\nabla u\in (L^2)^d\},\quad
 H^1_0(\Omega)=\{u\in H^1,\,u|_{\partial\Omega}=0\},
$$

   寻求 $u\in H^1_0(\Omega)$，使

$$
 \int_\Omega\nabla u\cdot\nabla v\,dx
 = \int_\Omega f\,v\,dx,\quad\forall v\in H^1_0(\Omega).
$$

   Lax–Milgram 定理保证唯一解。

2. **Galerkin 近似**
   取有限维子空间 $V_h\subset H^1_0(\Omega)$，基 $\{\varphi_j\}_{j=1}^{N_h}$，求

$$
 u_h=\sum_{j=1}^{N_h}u_j\varphi_j,\quad
 a(u_h,v_h)=\ell(v_h)\ \forall v_h\in V_h.
$$

   离散方程 $A\mathbf u=\mathbf b$，其中

$$
 A_{ij}=\int_\Omega\nabla\varphi_i\cdot\nabla\varphi_j,\quad
 b_i=\int_\Omega f\,\varphi_i.
$$

3. **网格划分（Maillage）**
   将 $\Omega$ 划分为三角形/四面体等单元，记作 $\mathcal T_h$。

   * 单元直径 $h_K$，最大网格大小 $h=\max_K h_K$。
   * 要求网格正规（形状比受限）。

4. **几何映射**
   从参考单元 $\hat K$ 通过光滑映射 $T_K$ 构造实际单元。

5. **Lagrange 元**
   在每个单元上选多项式空间 $\mathbb P_k$，并以节点（如顶点、边中点等）为自由度，构造全局连续空间 $P_{c,h}^k$ 及其边界零空间 $P_{c,h,0}^k$。

6. **装配与求解**
   构造稀疏对称正定矩阵 $A$ 和载荷向量 $\mathbf b$，使用 Cholesky 分解、共轭梯度、多重网格等方法求解。

7. **收敛性**
   对正规网格族，当 $h\to0$ 时
$$
 \|u-u_h\|_{H^1}\to0,\quad
 \|u-u_h\|_{L^2}=O(h^2).
$$

---

> **参考资料**
>
> * “Feel++ Laplacian” 文档系列
> * Paraview、Gmsh 等开源工具手册
