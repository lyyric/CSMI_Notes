## 🧩 **Chapitre I – Contrôlabilité des systèmes différentiels ordinaires**

### §1 Introduction

* 研究一般控制系统
$$
\dot x = f(x,u), \quad x\in\mathbb R^n, ; u\in \mathbb R^m,
$$
  其中 $f \in C^1, f(0,0)=0$。
* 定义：若存在控制 $u\in U_{ad}$ 使得从初态 $x_0$ 到终态 $x(T)=0$，则 $x_0$ 在时间 (T) 可控。
* 可控集 $C(T)$ 的拓扑性质：连通且当 $0\in \text{int},C$ 时为开集。
* 举例：二阶系统（火车减速模型）$\ddot x = u$，分析 bang-bang 控制下可控域为两段抛物线。

### §2 Critère de Kalman

* 线性系统：
$$
\dot x = Ax + Bu.
$$
* **可控性判据（Kalman条件）**：
$$
\mathrm{rank}[B, AB, \dots, A^{n-1}B] = n.
$$
* 若该秩条件成立且 $\Re\lambda(A)\le0$，系统对任意有界控制 $u\in [-1,1]^m$ 可控。
* 几何意义：可控性 ⇔ 可通过有限控制组合生成整个状态空间。

### §3 Contrôle en temps minimal

* 定义最小时间 $T^*$：使系统能从 $x_0$ 到 (0) 的最短时间。
* 由 **庞特里亚金极大值原理（PMP）** 导出必要条件：
$$
\dot p = -A^Tp, \quad (p(t),Bu(t)) = \max_{v\in U}(p(t),Bv).
$$
* 控制呈现 **bang-bang 形式**（取极值）。
* 实例：二阶火车系统中 $u = \pm 1$ 控制在切换点处改变符号。

---

## ⚙️ **Chapitre II – Contrôlabilité de l’équation des ondes**

### §1 Définition

* 研究波动方程：
$$
u_{tt} - \Delta u = 0, \quad u|_{\Gamma}=v(t,x),
$$
  控制 $v$ 作用于边界。
* **精确可控性（exact controllability）**：存在 $v$ 使 $u(T)=u_t(T)=0$。

### §2-3 Homogène & Observabilité

* 定义伴随系统（adjoint problem）并引入**可观测性（observability）**：
$$
\partial_\nu \varphi = 0 \Rightarrow \varphi \equiv 0.
$$
* 可控性与可观测性互为对偶。
* 一维情形：通过能量恒等式与 Holmgren 定理证明当 (T>2) 时系统可观测。
* 推出**直接与逆向不等式（Direct/Inverse Inequalities）**：
$$
C_1|(\varphi_0,\varphi_1)|^2 \le \int_0^T |\varphi_x(t,1)|^2 dt \le C_2|(\varphi_0,\varphi_1)|^2.
$$
### §4-5 Non-homogène & HUM 方法

* 非齐次边界控制问题写成变分形式；
* 利用 **Hilbert Uniqueness Method (HUM, Lions 1988)**：

  * 定义算子 $\Lambda(\varphi_0,\varphi_1) = (u_t(0),-u(0))$；
  * 通过 Lax-Milgram 引理建立同构性；
  * 得出：系统在 **(T>2)** 时精确可控；
  * 且最小能量控制 (v) 满足
$$
|v|*{L^2(0,T)} \le C|(u_0,u_1)|*{L^2\times H^{-1}}.
$$
### §6 Optimalité & Équivalence

* 最小范数控制 $v_0$ 的存在唯一性；
* 证明了：
$$
\text{可控性} ;\Longleftrightarrow; \text{可观测性}.
$$
* 最优控制由 HUM 给出。

### §7 Stabilisation

* 将反馈控制 $u(t,1)=u_t(t,1)$ 加入波方程；
* 能量函数 $E(t)$ 满足 $E'(t)=-|u_t(1,t)|^2\le0$；
* 利用积分估计与指数衰减引理得出：
$$
E(t)\le Me^{-\omega t}E(0),
$$
  即**稳定性与可控性等价**（Russell原理）。

---

## 💻 **Chapitre III – Approximation numérique**

### §1 Formulation duale

* 将 HUM 问题写成极值问题
$$
J(\varphi_0,\varphi_1)=\tfrac12\langle\Lambda(\varphi_0,\varphi_1),(\varphi_0,\varphi_1)\rangle+\langle(u_1,-u_0),(\varphi_0,\varphi_1)\rangle.
$$
* Euler 方程即为控制方程。

### §2 Méthode du gradient

* 迭代求解极小化问题的 **梯度下降算法**；
* 给出离散步骤与停止准则。

### §3 Semi-discrétisation

* 用有限差分（空间半离散）：
$$
\Phi_{tt} + \tfrac1{h^2}A\Phi = 0,
$$
  并在边界施加控制；
* 得到矩阵系统与离散能量守恒。

### §4 Discrétisation sur 2-grilles

* **双网格方法 $2-grid$**：去除高频数值波动，提高收敛性；
* 控制 $v_h = -\frac{\phi_N(t)}{h}$ 在 $L^2$ 中一致有界；
* 当 $h\to0$ 时，$(u_h,v_h)\to(u,v)$。

### §5 Discrétisation totale

* 时间离散（Euler / Crank–Nicolson / Runge–Kutta），稳定性条件 $k=\Delta t/h\le1$。

### §6 Méthode du gradient conjugué

* 给出共轭梯度法 (CG) 的迭代公式：
$$
f_{n+1}=f_n-\rho_n w_n, \quad w_{n+1}=g_{n+1}+\gamma_n w_n.
$$
### §7 Méthode des éléments finis

* 提及可将空间离散改为有限元法（FEM）以适应复杂几何。

---

## 📘 **Exercices**

* 一系列练习涵盖理论与应用：

  1. 参数化系统的 Kalman 可控性；
  2. 简谐振子系统的 bang-bang 控制；
  3. Hill 系统的空间会合控制；
  4. 二阶系统的最小时间控制；
5-8. 波动方程的唯一性、双边界控制、内部控制及 Neumann 控制的 HUM 分析。

---

## 🧠 **总体总结**

| 主题               | 核心思想                          |
| ---------------- | ----------------------------- |
| **线性系统可控性**      | Kalman 秩条件决定可控性。              |
| **最优控制理论**       | Pontryagin极大值原理给出bang-bang控制。 |
| **偏微分系统（PDE）控制** | HUM将控制问题转化为可观测性与变分问题。         |
| **波动方程可控性**      | 一维情形当 (T>2) 时可控，可观测不等式成立。     |
| **稳定性**          | 指数稳定 ⇔ 可控性。                   |
| **数值近似**         | 差分、双网格与有限元法实现 HUM 控制。         |

---
