以下是一份对题目内容的中文翻译（保留原题的数学符号与编号，以便对应）：

---

## 科学计算考试准备

在下文中，$E$ 表示在 $\mathbb{R}_x \times \mathbb{R}_t$ 上的函数集合 $u(x, t)$，满足以下条件：

1. 对任意 $t$，映射 $x \mapsto u(x, t)$ 属于 $L^2(\mathbb{R}) \cap L^1(\mathbb{R})$。
2. 对任意 $t$，映射 $x \mapsto u(x, t)$ 是 $C^2$ 级（即二阶连续可导）。
3. 对任意 $t$，当 $x \to \pm \infty$ 时，$u(x, t) \to 0$。
4. 存在一个实数 $T$ 使得
   $$
   \forall\, t \leq T,\quad u(x, t) = 0.
   $$

在下文中，属于集合 $E$ 的函数 $u$ 若满足上述条件，则称为**因果函数（causale）**。

---

### 1）  
证明：若一个因果函数同时满足热方程（方程 de la chaleur）或输运方程（方程 de transport），则该函数必定恒为零。

---

### 2）  
设初值条件 $u_0(x)$ 是一个具有紧支撑且 $C^2$ 级的函数。考虑下面定义的函数 $u$：

$$
\begin{cases}
\partial_t u - \partial_{xx} u = 0,\quad & t \geq 0,\\
u(x, t = 0) = u_0(x), \\
u(x, t) = 0,\quad & t < 0.
\end{cases}
$$

- 证明：此函数 $u$ 定义良好，并且它是一个因果函数。

---

### 3）  
证明：在分布（distribution）的意义下，对于所有 $(x, t) \in \mathbb{R}^2$，函数 $u$ 满足

$$
\partial_t u - \partial_{xx} u = \delta\,u_0,
$$

其中 $\delta$ 是**时间变量**上的狄拉克分布，形式定义为

$$
\int \delta(t)\,\varphi(t)\,\mathrm{d}t := \varphi(0).
$$

---

### 4）  
因此，可以将问题 2）中的“随时间给定初值条件”改写为“在所有时间上给出一个带有分布项（狄拉克）作为源项的方程”：

$$
\partial_t u \;-\; \partial_{xx} u \;=\; \delta \,u_0.
$$

要求：寻找一个因果函数，满足上式。证明：在因果函数的集合中，该问题的解是唯一的。

---

### 5）  
为了数值计算，需要会近似狄拉克分布。  
首先考虑一个函数 $\rho : \mathbb{R} \to \mathbb{R}^+$，其支撑在区间 $[-1, 1]$，并且满足

$$
\int \rho = 1.
$$

令 $\tau > 0$，定义函数（称为“近似单位”）$\delta_\tau$ 为

$$
\delta_\tau(t) = \frac{1}{\tau}\,\rho\!\bigl(\tfrac{t}{\tau}\bigr).
$$

1. 证明：当 $\tau \to 0$ 时，$\delta_\tau$ 在分布意义下趋于 $\delta$（即对所有测试函数 $\varphi$，都有 $\lim_{\tau \to 0} \langle \delta_\tau, \varphi \rangle = \langle \delta, \varphi \rangle$）。  
2. 若无额外说明，之后我们取
   $$
   \rho(t) =
   \begin{cases}
   1, & t \in [0, 1],\\
   0, & \text{否则}.
   \end{cases}
   $$

---

### 6）  
对于 $\tau > 0$ 和 $h > 0$，在集合 $E$ 上定义时间与空间的平移算子：

$$
(T_\tau u)(x, t) \;=\; u(x,\,t-\tau),
\quad
(S_h u)(x, t) \;=\; u(x-h,\,t).
$$

证明：这两个算子都能将 $E$ 映射到自身（即 $T_\tau, S_h : E \to E$）。

---

### 7）  
回顾对变量 $x$ 的傅里叶变换定义（可参考 [Fourier变换 - 维基百科](https://fr.wikipedia.org/wiki/Transformation_de_Fourier)）：

$$
\widehat{u}(\xi, t)
\;=\;
\int_{-\infty}^{\infty} u(x, t)\,e^{-\,\mathrm{i}\,\xi\,x}\,\mathrm{d}x.
$$

1. 计算 $\widehat{(S_h u)}(\xi, t)$。  
2. 证明：若 $u$ 是因果函数，则 $(\xi, t) \mapsto \widehat{u}(\xi, t)$ 也是因果函数。

---

### 8）  
为逼近热方程解，考虑下面的函数型差分格式（“函数性”是指直接在函数空间中写出的格式），

$$
\frac{v - T_\tau v}{\tau}
\;+\;
\frac{-\,S_h v \;+\; 2\,v \;-\; S_{-h}v}{h^2}
\;=\;
\delta_\tau \,v_0.
\tag{1}
$$

1. 证明：若设 $v_i^n = v(i\,h,\,n\,\tau)$，则此函数型格式与常见的显式差分格式等价。  
2. 证明：此函数型解 $v$ 依旧是因果的。  
3. 如果更改选取的 $\rho$（从而改变 $\delta_\tau$ 的形状），是否会带来实质影响（是否“有趣”）？

---

### 9）  
在离散差分格式中，我们讲过对解的 $\ell^1$ 稳定性与 $\ell^2$ 稳定性。如何在**函数型差分格式**层面上翻译（对应）这两种稳定性的概念？

---

### 10）  
将 von Neumann 的**放大因子（coefficient d’amplification）** 的概念推广到此函数型差分格式的情形。应如何定义与计算？

---

### 11）  
针对格式 (1)，分别求其在 $\mathrm{L}^2$ 和 $\mathrm{L}^\infty$ 范数意义下的稳定性条件。

---

### 12）  
就输运方程  
$$
\partial_t u \;+\; c\,\partial_x u \;=\; 0,\quad c > 0,
$$  
以及一阶偏风（迎风）差分格式（schéma décentré d’ordre 1）：

- 讨论与前面类似的问题：稳定性、因果性、唯一性等。

---

### 13）  
如何验证格式 (1) 的一致性（consistance）？在傅里叶空间中，这个一致性又怎样体现？

---

### 14）  
为求解输运方程
$$
\partial_t u + c\,\partial_x u = 0,\quad c > 0,
$$
我们考虑如下差分格式：
$$
\frac{u_i^{n+1} - u_i^n}{\tau}
\;+\;
c \,\frac{3\,u_i^n \;-\;4\,u_{i-1}^n \;+\; u_{i-2}^n}{h}
\;=\;
d_n\,u_0(i\,h),
$$
其中
$$
d_n \;=\;
\begin{cases}
1/\tau, & n = 0,\\
0, & \text{否则}.
\end{cases}
$$

用“函数型”方法（同前面对热方程的思路）计算此格式的稳定性条件。

---
 