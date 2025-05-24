# 扩散方程（Equation de diffusion）

作者：Christophe Prud’homme, Laurent Navoret

---

## 1. 扩散问题

### 1.1 方程形式

$$
\begin{cases}
\displaystyle \partial_t u - \Delta u = f, & (x,t)\in \Omega\times\mathbb{R}^+,\\
u(x,0) = u_0(x), & x\in\Omega,\\
u(x,t) = g(x,t), & (x,t)\in\partial\Omega_D\times\mathbb{R}^+.
\end{cases}
$$

* $u(x,t)$：待求的密度场
* $u_0(x)$：初始分布
* $f(x,t)$：源项
* $g(x,t)$：Dirichlet 边界值

---

## 2. 物理与变分描述

* **守恒定律**：

$$
\frac{d}{dt}\int_\Omega u
= \int_{\partial\Omega}n\cdot\nabla u
  + \int_\Omega f.
$$
* **能量耗散**（$L^2$ 范数随时间衰减）：

$$
\frac{d}{dt}\int_\Omega u^2
= -\int_\Omega|\nabla u|^2
  + \int_{\partial\Omega}u\,\partial_nu
  + \int_\Omega f\,u.
$$

---

## 3. 数值求解思路

对于时间相关 PDE，一种常见做法是**先离散时间**、再做空间有限元。

1. **时离散**（Method of Lines）：

   * 取时间步长 $\Delta t$，$t_n = n\,\Delta t$，定义 $u^n(x)=u(x,t_n)$。
   * 向后 Euler（隐式）格式：

 $$
   \frac{u^n - u^{n-1}}{\Delta t}
   = \Delta u^n + f^n.
$$
   * 得到每一步的**椭圆问题**：

 $$
   u^n - \Delta t\,\Delta u^n
   = u^{n-1} + \Delta t\,f^n.
$$

2. **空间变分**：
   令

$$
 V = \{v\in H^1(\Omega)\mid v=g^n\text{ on }\partial\Omega\},\quad
 V_0=\{v\in H^1(\Omega)\mid v=0\text{ on }\partial\Omega\}.
$$

   对每步求解变分问题：

$$
\begin{cases}
 \text{找 }u^n\in V,\quad
 a(u^n,v)=\ell(v),\quad\forall v\in V_0,\\[6pt]
 a(u,v)=\displaystyle\int_\Omega u\,v
		 +\Delta t\int_\Omega\nabla u\cdot\nabla v,\\[4pt]
 \ell(v)=\displaystyle\int_\Omega\bigl(u^{n-1}+\Delta t\,f^n\bigr)\,v.
\end{cases}
$$

   然后在 $V_h\subset V$ 上用有限元离散。

---

## 4. 初始值处理与高阶时序

* **初始解 $u^0$ 的构造**：

  * **插值**：$u^0(x_i)=u_0(x_i)$。
  * **$L^2$ 投影**：满足
    $\displaystyle\int_\Omega u^0\,v = \int_\Omega u_0\,v$, $\forall v\in V_h$。
* **更高阶时序格式**：

  * Crank–Nicolson（二阶精度）等。

---

## 5. 参考示例

* 带有代码示例或视频演示（如 YouTube 片段）展示具体实现过程。

---

