# Stokes 方程

作者：Christophe Prud’homme, Laurent Navoret

---

## 1. Stokes 问题

### 非稳态形式

$$
\begin{cases}
\partial_t u - \Delta u + \nabla p = f, & (x,t)\in\Omega\times\mathbb R^+,\\
\nabla\cdot u = 0, & \Omega\times\mathbb R^+,\\
u=0, & \partial\Omega.
\end{cases}
$$

* $u:\Omega\times\mathbb R^+\to\mathbb R^d$：速度场
* $p:\Omega\times\mathbb R^+\to\mathbb R$：压力场

### 稳态形式

$$
\begin{cases}
-\Delta u + \nabla p = f, & x\in\Omega,\\
\nabla\cdot u = 0, & x\in\Omega,\\
u=0, & x\in\partial\Omega.
\end{cases}
$$

---

## 2. 不可压缩性

* **流体不可压缩** 等价于 $\nabla\cdot u=0$。
* 从“输运定理”得：

  $$
  \frac{d}{dt}\!\int_{\Omega(t)}\!1\,dx
  =\int_{\partial\Omega(t)}(u\cdot n)\,ds
  =\int_{\Omega(t)}\nabla\cdot u\,dx.
  $$

  若体积恒定，则 $\nabla\cdot u\equiv0$。

---

## 3. 弱（变分）形式

对动量方程和不可压缩性分别乘以试验函数并分部积分，得到：

**形式 1（Hodge 分解版）**
在

$$
V=\{v\in H_0^1(\Omega)^d,\;\nabla\cdot v=0\}
$$

上求 $u$，使

$$
\int_\Omega\nabla u:\nabla v\,dx
=\int_\Omega f\cdot v\,dx,\quad\forall v\in V.
$$

* 此时 $p$ 作为拉格朗日乘子被省略。

> **注意**：构造离散子空间 $V_h\subset V$ 极其困难。

---

## 4. 经典混合型弱形式

引入压力试验空间 $L^2_*(\Omega)=\{q\in L^2(\Omega):\int_\Omega q=0\}$，令

$$
X=H_0^1(\Omega)^d,\quad M=L^2_*(\Omega).
$$

求 $(u,p)\in X\times M$，使

$$
\begin{cases}
\displaystyle\int_\Omega\nabla u:\nabla v\,dx-\int_\Omega p\,\nabla\cdot v\,dx
=\int_\Omega f\cdot v\,dx,
&\forall v\in X,\\[0.5em]
\displaystyle\int_\Omega q\,\nabla\cdot u\,dx=0,
&\forall q\in M.
\end{cases}
$$

* 取 $L^2_*$ 确保 $p$ 唯一（去掉常数模）。

---

## 5. Banach–Nečas–Babuška 条件

对一般鞍点问题：给定 Hilbert 空间 $X, M$，双线性型

$$
a:X\times X\to\mathbb R,\quad b:X\times M\to\mathbb R,
$$

线性泛函 $\ell$ 和 $g$。弱问题有唯一解 $(u,p)$ 当且仅当满足：

1. $a$ 在 $X$ 上连续且 coercive。
2. $b$ 满足“inf–sup 条件”：

   $$
   \inf_{q\in M}\;\sup_{v\in X}\;\frac{b(v,q)}{\|v\|_X\,\|q\|_M}\;\ge\beta>0.
   $$

---

## 6. Galerkin 离散

选离散空间 $X_h\subset X$、$M_h\subset M$，基 $\{\varphi_i\}_{i=1}^{N_u}$、$\{\psi_j\}_{j=1}^{N_p}$，写

$$
u_h=\sum_j u_j\,\varphi_j,\quad
p_h=\sum_k p_k\,\psi_k,
$$

得到矩阵方程

$$
\begin{bmatrix}
A & B^T\\
B & 0
\end{bmatrix}
\begin{bmatrix}U\\P\end{bmatrix}
=
\begin{bmatrix}F\\G\end{bmatrix},
$$

其中

$$
A_{ij}=a(\varphi_j,\varphi_i),\;
B_{kj}=b(\varphi_j,\psi_k).
$$

* 若离散 inf–sup 条件成立，则矩阵可逆。

---

## 7. 常见速度-压力元

|                    速度 元                    |      压力 元     |            inf–sup 稳定性           | 收敛阶 |
| :----------------------------------------: | :-----------: | :------------------------------: | :-: |
|          $\mathbb P_1/\mathbb P_0$         |  锁死（locking）  |                 ✗                |  —  |
|          $\mathbb P_1/\mathbb P_1$         | 伪模式（spurious） |                 ✗                |  —  |
| Mini 元（$\mathbb P_1$+bubble）/$\mathbb P_1$ |       ✓       |   一阶 $O(h)$（H¹）、二阶 $O(h^2)$（L²）  |     |
|   Taylor–Hood ($\mathbb P_2/\mathbb P_1$)  |       ✓       | 二阶 $O(h^2)$（H¹）, 三阶 $O(h^3)$（L²） |     |

---

## 8. 求解方法

Stokes 矩阵对称但不正定，直接迭代收敛差。两种常用策略：

1. **鞍点分解**：Schur 补 $S = B A^{-1}B^T$ 正定，先解压力再解速度（Uzawa 迭代）。
2. **人工可压缩**：引入小参数 $\varepsilon>0$，替换系统为
   $\begin{pmatrix}A & B^T\\ B & \varepsilon M\end{pmatrix}$，正定。

---

## 9. 实践要点

* 离散压力空间通常取 $\tilde M_h$，不强制零均值；解得 $p_h$ 后再减去平均值。
* 人工可压缩方法直接选出零均值压力。

---

