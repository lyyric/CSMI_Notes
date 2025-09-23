### 引言

为了更好地理解本课程的内容，我们先从几个具有代表性的最优控制问题入手。

**例1：两个航天器在地球附近的会合**
设想航天器1处于被动状态，沿圆轨道飞行；航天器2通过发动机产生推力

$$
v = (v_1, v_2)
$$

来追赶航天器1。记 $x$ 为航天器2相对于航天器1的线性化位置向量，在以航天器1为原点的移动坐标系中，

$$
x = (x_1, x_2)
$$

满足 **Hill方程**：

$$
\begin{cases}
\ddot{x}_1(t) = 3\omega^2 x_1(t) + 2\omega \dot{x}_2(t) + v_1(t), \\
\ddot{x}_2(t) = -2\omega \dot{x}_1(t) + v_2(t), \\
(x_1(0), \dot{x}_1(0), x_2(0), \dot{x}_2(0)) = X_0 \in \mathbb{R}^4.
\end{cases}
$$

记

$$
X_v = (x_1, \dot{x}_1, x_2, \dot{x}_2),
$$

它描述了在移动参考系下两航天器的相对位置和速度。这里 $\omega$ 是航天器1的角频率，满足周期

$$
T = \frac{2\pi}{\omega},
$$

在国际空间站的情况下 $\omega$ 对应的周期约为 5480 秒。

该问题可通过不同的建模方式来描述：

1. **最短时间问题**
   考虑发动机功率的限制，控制集合为

   $$
   U_{ad} = \{ v \in C^0_M([0,T]) \mid v(s) \in U \},
   $$

   其中 $U \subset \mathbb{R}^2$ 为一个紧集。例如假设两个发动机相互独立，并满足约束 $|v_i(s)| \leq 1$, $i=1,2$。
   目标是找到一个控制 $v$，使得从初态 $X_0$ 出发，系统在最短时间 $T$ 到达

   $$
   X_v(T) = 0,
   $$

   即两航天器实现会合。

2. **能量与精度的折中问题**
   固定一个时间区间 $T > 0$，希望在此时刻两航天器相遇。可引入如下泛函：

   $$
   J(v) = \frac{\varepsilon}{2} \int_0^T (v_1(t)^2 + v_2(t)^2)\,dt  
   + (1-\varepsilon)\left(x_1(T)^2 + \dot{x}_1(T)^2 + x_2(T)^2 + \dot{x}_2(T)^2\right),
   $$

   其中 $\varepsilon \in (0,1)$ 为权重。第一项表示**控制能量的代价**，第二项惩罚终态与目标状态的偏差。
   问题转化为：

   $$
   \min_{v \in U_{ad}} J(v).
   $$

   在这种建模下，系统的终态 $X_v(T)$ 未必严格等于零，但通过代价泛函中的惩罚项体现对目标的追求。

---

**例2：最小能量消耗**

设 $\Omega \subset \mathbb{R}^n$ 是一个有界区域，表示一个导热体。系统状态是温度场

$$
y : \mathbb{R}^+ \times \Omega \to \mathbb{R},
$$

满足 **热方程**：

$$
\begin{cases}
\frac{\partial y}{\partial t}(t,x) - \Delta y(t,x) = f(x) + v(t,x), & (t,x) \in [0,T]\times\Omega, \\
y(t,x) = 0, & (t,x) \in [0,T]\times \partial\Omega, \\
y(0,\cdot) = y_0(\cdot).
\end{cases}
$$

* 边界条件 $y=0$ 表示边界温度保持恒定（作为温度刻度的零点）。
* 函数 $f(x)$ 描述不可控的热源；控制函数 $v(t,x)$ 则对应可操控的热源。

设有期望温度分布 $z_d(x)$，引入代价泛函：

$$
J(v) = \frac{1}{2} \int_0^T \int_\Omega |y(t,x) - z_d(x)|^2 dx\,dt  
+ \frac{\alpha}{2} \int_0^T \int_\Omega |v(t,x)|^2 dx\,dt,
$$

其中：

* 第一项度量系统温度与期望分布的差异；
* 第二项度量控制能量消耗；
* $\alpha > 0$ 为权衡系数。

控制集合典型为：

$$
U_{ad} = \{ v \in L^\infty(]0,T[) \mid a \leq v(s) \leq b \ \text{a.e. 在 } ]0,T[ \times \omega \},
$$

其中 $\omega \subset \Omega$ 为受控区域。

最终的最优控制问题是：

$$
\text{寻找 } v \in U_{ad} \text{ 使得 } J(v) \text{ 最小}.
$$

换言之，目标是 **以最小能耗使温度尽可能接近期望分布**。

---

### 总结

无论在有限维还是无限维情况下，最优控制问题实质上都是一个优化问题。但其特殊性在于：

* 代价泛函 $J$ 并不是直接依赖于控制 $v$，
* 而是通过系统的**状态变量**（满足状态方程）间接依赖于 $v$。

这给 **梯度的计算** 带来困难，而梯度在最优化方法中起着至关重要的作用。
为解决此问题，引入 **伴随状态（adjoint state）** 的方法，使得能够有效计算 $J$ 的梯度，并用更易操作的形式表达最优性条件。

---

## 常微分系统的可控性

本部分研究的一般问题如下。设 $(n,m) \in (\mathbb{N}^\ast)^2$，$I$ 是 $\mathbb{R}$ 的一个区间，
$f : \mathbb{R}^+ \times \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^n$ 是一个足够正则的函数，使得下文的一切推导都有意义。

设 $U \subset \mathbb{R}^m$，初始条件 $x_0 \in \mathbb{R}^n$。考虑如下控制系统：

$$
\dot{x}(t) = f(t, x(t), u(t)), \quad t \in I, \tag{I.1}
$$

$$
x(0) = x_0,
$$

其中控制集合由所有在 $I$ 上有界且可测的函数 $u(\cdot)$ 构成，它们的值落在子集 $U \subset \mathbb{R}^m$ 内。

常微分方程解的经典存在唯一性定理保证：对任意控制 $u$，系统 $(I.1)$ 存在唯一解 $x(\cdot): I \to \mathbb{R}^n$，并且是绝对连续的。

因此，系统的解 $x_u$ 依赖于控制 $u$。当改变 $u$ 时，得到的轨迹 $t \mapsto x(t)$ 在 $\mathbb{R}^n$ 中也随之改变。

自然会产生两个基本问题：

* **精确可控性问题**：给定 $x_1 \in \mathbb{R}^n$，是否存在某个控制 $u$，使得系统轨迹从 $x_0$ 出发在有限时间 $T$ 内到达 $x_1$？
* **最优控制问题**：若前一个条件满足，是否存在一个控制，不仅能使轨迹从 $x_0$ 到达 $x_1$，而且还能最小化某个代价泛函 $u \mapsto C(u)$？

这里，泛函 $C(u)$ 是一个优化准则，被称为**代价**。例如，当代价是飞行时间时，问题就变成了**最短时间控制问题**。

---

### I.1 线性系统的可控性

#### I.1.1 Kalman 判别准则

设 $T > 0$ 为固定的时间区间。考虑如下线性系统：

$$
\dot{x}(t) = A x(t) + B u(t), \quad t \in [0,T], \tag{I.2}
$$

$$
x(0) = x_0,
$$

其中 $A \in M_n(\mathbb{R}), \ B \in M_{n,m}(\mathbb{R})$，控制函数 $u(\cdot): [0,T] \to \mathbb{R}^m$。系统解 $x(t)$ 依赖于控制 $u$，我们记为 $x_u$。

合适的函数空间框架是**绝对连续函数**空间 AC$[0,T]$，而不是要求轨迹为 $C^1$。

**命题 I.1.1 （Duhamel 公式）**
对任意 $u \in L^1(]0,T[;\mathbb{R}^m)$，系统 $(I.2)$ 存在唯一解 $x_u \in AC([0,T],\mathbb{R}^n)$，并满足：

$$
x_u(t) = e^{tA} x_0 + \int_0^t e^{(t-s)A} B u(s)\,ds, \quad t \in [0,T].
$$

**定义 I.1.2 （精确可控性）**
若对所有 $x_1 \in \mathbb{R}^n$，存在某个 $u \in L^1(0,T;\mathbb{R}^m)$，使得 $x_u(T) = x_1$，则称系统 $(I.2)$ 在时间 $T$ 内从 $x_0$ 可控。

**定理 I.1.3 （Kalman 判别准则）**
系统 $(I.2)$ 在任意 $T > 0$ 和任意初始状态 $x_0$ 下都是精确可控的，当且仅当 **Kalman 可控性矩阵**

$$
\text{Kal} = [B, AB, \dots, A^{n-1}B] \in M_{n, nm}(\mathbb{R})
$$

的秩为满秩，即 $\operatorname{rg}(\text{Kal}) = n$。

**备注 I.1.4**
Kalman 条件与时间区间 $T$ 和初值 $x_0$ 无关。这意味着一旦系统可控，就能在任意短时间内到达任意目标点。但需要注意的是，代价是控制可能会变得无限大。

**例 I.1.5**
考虑系统

$$
\begin{cases}
\dot{x} = 2x + (\alpha - 3) y + u, \\
\dot{y} = 2y + \alpha^2 x - \alpha y + v,
\end{cases}
$$

问：对哪些 $\alpha$ 值，该系统是可控的？

**备注 I.1.6 （非自治系统的情况）**
若 $A(t), B(t), r(t)$ 是 $L^\infty$ 函数，考虑系统

$$
\dot{x}(t) = A(t)x(t) + B(t)u(t), \quad x(0) = x_0, \tag{I.3}
$$

此时 Duhamel 公式不再成立，需要引入**解算算子（resolvante）** $R(t)$。系统的解为

$$
x_u(t) = R(t)x_0 + R(t)\int_0^t R(s)^{-1} B(s) u(s)\, ds.
$$

可证明系统 $(I.3)$ 在时间 $T$ 可控，当且仅当矩阵

$$
G_T = \int_0^T R(s)^{-1} B(s) B(s)^\top (R(s)^{-1})^\top ds
$$

是可逆的。

---

#### I.1.2 可达集

考虑系统：

$$
\dot{x}(t) = A(t)x(t) + B(t)u(t), \quad x(0)=x_0. \tag{I.4}
$$

**定义 I.1.7 （可达集）**
在时间 $T > 0$ 内从 $x_0$ 出发的可达集为

$$
A(x_0, T) = \{ x_u(T) \mid u \in L^\infty([0,T],U)\}.
$$

**定理 I.1.8**
若 $U \subset \mathbb{R}^m$ 是非空、紧、凸集，则对系统 $(I.4)$，在任意 $t \in [0,T]$，集合 $A(x_0,t)$ 是紧的、凸的，并且随时间 $t$ 连续变化。

**备注 I.1.9**
令人惊讶的是，即使只要求 $U$ 是紧集（而不一定凸），结论依然成立。并且可证明：对应于 $U$、其边界及其凸包的可达集是相同的。

为了形式化地描述集合的连续变化，引入**Hausdorff 距离**：

$$
d_H(A,B) = \max\left\{ \sup_{x \in A} d_B(x), \ \sup_{x \in B} d_A(x) \right\},
$$

其中 $d_E(x) = \inf_{y \in E} |x-y|$。

**例 I.1.11**
考虑一维点质量，控制其速度 $u(t) \in U = [-1,1]$，系统为

$$
\dot{x}(t) = u(t), \quad x(0) = 0.
$$

此时在时间 $t$ 的可达集为

$$
A(0,t) = [-t, t],
$$

它确实是紧的、凸的，并且随时间连续变化。

---

## I.2 非线性系统的可控性

考虑系统

$$
\dot x(t)=f(t,x(t),u(t)),\quad t\in[0,T],\qquad x(0)=x_0, \tag{I.5}
$$

其中 $u\in L^1(]0,T[,\mathbb{R}^m)$，且 $f:[0,T]\times\mathbb{R}^n\times\mathbb{R}^m\to\mathbb{R}^n$。即便 $f$ 关于 $u$ 是正则的，由于控制 $u$ 对时间不必连续，因而 $f$ 不一定关于 $t$ 连续。

### I.2.1 Cauchy–Lipschitz 定理回顾（可测情形与全局Lipschitz）

**定理 I.2.1（Cauchy–Lipschitz，$t$ 可测且 $x$ 全局Lipschitz）**
设 $F:[0,T]\times\mathbb{R}^n\to\mathbb{R}^n$ 满足：

1. $F$ 对 $t$ 可测、对 $x$ 连续：对每个 $x$，$t\mapsto F(t,x)$ 可测；对几乎处处的 $t$，$x\mapsto F(t,x)$ 连续。
2. $F$ 关于 $t$ 可积：$\forall x\in\mathbb{R}^n$，存在 $\beta\in L^1(]0,T[,\mathbb{R}_+)$ 使得 $|F(t,x)|\le \beta(t)$。
3. $F$ 关于 $x$ 全局Lipschitz：$\exists C_0\in L^1(]0,T[,\mathbb{R}_+)$ 使得对几乎处处的 $t$ 与任意 $x_1,x_2$,

$$
|F(t,x_1)-F(t,x_2)|\le C_0(t)\,|x_1-x_2|.
$$

则 Cauchy 问题

$$
\dot x(t)=F(t,x(t)),\ t\in[0,T],\qquad x(0)=x_0
$$

存在唯一解 $x\in AC([0,T],\mathbb{R}^n)$，并满足

$$
x(t)=x(0)+\int_0^t F(s,x(s))\,ds,\quad \forall t\in[0,T].
$$

该解定义在区间 $J\subset[0,T]$ 上，并且要么 $J=[0,T]$，要么 $J=[0,T^\ast[$ 且 $T^\ast<T$ 并满足 $\lim_{t\to T^\ast}|x(t)|=+\infty$。

**备注 I.2.2（推广）**
若将(iii) 替换为：$\forall x\in\mathbb{R}^n$，$\exists r>0$、$\exists C_0\in L^1(]0,T[)$，使得对几乎处处的 $t$ 与任意 $x_1,x_2\in B(x,r)$，

$$
|F(t,x_1)-F(t,x_2)|\le C_0(t)\,|x_1-x_2|,
$$

则上述 Cauchy 问题仍存在唯一的极大解。

**例 I.2.3（控制系统的爆破）**
在 $\mathbb{R}$ 上（$n=1$）考虑

$$
\dot x(t)=f(t,x(t),u(t)),\quad x(0)=0,\quad u(t)\in U,
$$

其中 $m=1$，并取 $f(t,x,u)=x^2+u$（不显含 $t$）。Cauchy 问题为

$$
\dot x(t)=x(t)^2+u(t).
$$

取初值 $x_0=0$，且令控制为常数 $u(t)\equiv u_0\in\mathbb{R}_+$。则解为

$$
x(t)=u_0\,\tan(u_0 t),
$$

在有限时间 $T^\ast=\pi/(2u_0)$ 发生爆破，该时间取决于控制常值 $u_0$。

---

### I.2.2 自主非线性系统的局部可控性

固定时间域 $T>0$ 与初值 $x_0\in\mathbb{R}^n$，考虑非线性控制系统

$$
\dot x(t)=f(x(t),u(t)),\quad t\in[0,T],\qquad x(0)=x_0, \tag{I.6}
$$

其中 $u\in L^1(]0,T[,\mathbb{R}^m)$，且 $f:\mathbb{R}^n\times\mathbb{R}^m\to\mathbb{R}^n$。本节假设 $f$ 关于 $(x,u)$ 为 $C^1$。

若 $y\in A(T,x_0)$，按定义存在某个控制 $u_y$ 使得系统在时间 $T$ 将状态由 $x_0$ 驱动至 $y$。局部可控性问题关心：在 $y$ 的邻域内，此性质是否仍成立。

考虑自主系统的平衡点 $(\bar x,\bar u)$，满足 $f(\bar x,\bar u)=0$。注意到取常控制 $\bar u$ 可使 $\bar x\in A(t,\bar x)$。

**定义 I.2.4（局部可控性）**
若存在 $\varepsilon>0$，对每个满足 $\|x_1-\bar x\|<\varepsilon$ 的点 $x_1$，都存在某个控制 $u\in L^\infty(0,T;\mathbb{R}^m)$ 使得由初值 $x_0$ 出发的解 $x_u(T)=x_1$，则称非线性系统 (I.6) 在平衡点 $(\bar x,\bar u)$ **局部可控**（时间 $T$ 下）。

为陈述局部可控性结果，引入在平衡点处的线性化系统。

**定义 I.2.5（线性化系统）**
设

$$
A=\frac{\partial f}{\partial x}(x_0,u_0)\in M_n(\mathbb{R}),\qquad 
B=\frac{\partial f}{\partial u}(x_0,u_0)\in M_{n,m}(\mathbb{R}),
$$

分别为 $f$ 关于 $x$ 与 $u$ 的雅可比矩阵在 $(x_0,u_0)$ 处的取值。则围绕平衡点 $(x_0,u_0)$ 的线性化系统为

$$
\frac{d(\delta x)}{dt}=A\,\delta x(t)+B\,\delta u(t),\quad t\in[0,T],\qquad \delta x(0)=0, \tag{I.7}
$$

其中 $\delta u\in L^\infty(0,T;\mathbb{R}^m)$ 给定。

**定理 I.2.6（局部可控性）**
在对控制不施加额外约束的情形下，若线性化系统在时间 $T$ 可控（即满足 Kalman 条件），则非线性系统在时间 $T$ 从 $x_0$ 出发**局部可控**。

**例 I.2.7（倒立摆）**
考虑倒立摆（上端为质点、下为杆，质量与长度取1），摆在平面内运动。以上端与竖直方向的夹角为 $\theta$（顺时针为正）描述其姿态，控制量为杆底端的水平加速度。动力学为

$$
\ddot{\theta}(t)=\sin\theta(t)-u(t)\cos\theta(t).
$$

令 $x=(x_1,x_2)=(\theta,\dot\theta)$，则一阶形式 (I.6) 写为

$$
f(x,u)=
\begin{pmatrix}
x_2\\
\sin x_1 - u\cos x_1
\end{pmatrix}.
$$

问题：系统在不稳定平衡点 $(x_0,u_0)=((0,0)^\top,\,0)$ 的邻域内是否**局部可控**？

---

