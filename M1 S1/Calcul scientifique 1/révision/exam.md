以下是这份法语考试复习题的中文翻译：

---

### 科学计算考试准备

设 $E$ 是定义在 $\mathbb{R}_x \times \mathbb{R}_t$ 上的函数 $u(x, t)$ 的集合，满足：

1. 对任意 $t$，$x \mapsto u(x, t) \in L^2(\mathbb{R}) \cap L^1(\mathbb{R})$；
2. 对任意 $t$，$x \mapsto u(x, t)$ 是 $C^2$ 类函数；
3. 对任意 $t$，$u(x, t) \to 0$ 当 $x \to \pm\infty$；
4. 存在一个实数 $T$ 使得：对任意 $t \leq T$，有 $u(x, t) = 0$。

在下文中，称属于 $E$ 的函数为“因果函数”（causale）。

---

1）证明：满足热方程或输运方程的因果函数必为零函数。

2）设初始条件 $u_0(x)$ 是具有紧支集的 $C^2$ 函数，考虑下列问题定义的函数 $u$：

$$
\begin{cases}
\partial_t u - \partial_{xx} u = 0, & t \geq 0, \\
u(x, 0) = u_0(x), \\
u(x, t) = 0, & t < 0.
\end{cases}
$$

证明：$u$ 被良好定义，并且是因果函数。

3）证明：在分布意义下，$u$ 满足如下方程：

$$
\partial_t u - \partial_{xx} u = \delta u_0,
$$

其中 $\delta$ 是时间上的狄拉克分布，其形式定义为：

$$
\int \delta(t) \phi(t) dt := \phi(0)。
$$

4）因此，可以将第2问中的初始条件问题转换为一个定义在全时间轴上的分布右端项问题：寻找一个因果函数 $u$，使得：

$$
\partial_t u - \partial_{xx} u = \delta u_0.
$$

证明：该问题在因果函数集合中有唯一解。

5）为了进行数值求解，需要近似狄拉克分布。我们考虑函数 $\rho : \mathbb{R} \to \mathbb{R}_+$，其支集在区间 \[−1, 1] 上，且满足：

$$
\int \rho = 1。
$$

设 $\tau > 0$，定义逼近单位（unité approchée）：

$$
\delta_\tau(t) = \frac{1}{\tau} \rho\left(\frac{t}{\tau}\right)。
$$

证明：当 $\tau \to 0$ 时，$\delta_\tau$ 以分布意义收敛于 $\delta$（即对所有测试函数 $\phi$，有 $\langle \delta_\tau, \phi \rangle \to \langle \delta, \phi \rangle$）。除非另有说明，后续中取：

$$
\rho(t) = 
\begin{cases}
1, & t \in [0, 1], \\
0, & \text{其他情况}.
\end{cases}
$$

6）设 $\tau > 0$, $h > 0$，定义时间和平移算子：

$$
(T_\tau u)(x, t) = u(x, t - \tau), \quad (S_h u)(x, t) = u(x - h, t)。
$$

证明：这些算子将 $E$ 映射到自身。

7）回顾关于 $x$ 变量在 $\mathbb{R}$ 上的傅里叶变换定义：

$$
\hat{u}(\xi, t) = \int_{-\infty}^{\infty} u(x, t) e^{-i\xi x} dx。
$$

计算 $S_h u$ 的傅里叶变换。证明：如果 $u$ 是因果函数，则 $(\xi, t) \mapsto \hat{u}(\xi, t)$ 也是因果函数。

8）为近似热方程的解，考虑如下差分函数格式（函数型）：

$$
\frac{v - T_\tau v}{\tau} + \frac{-S_h v + 2v - S_{-h}v}{h^2} = \delta_\tau v_0. \quad (1)
$$

证明：该格式等价于显式差分格式（若设 $v^n_i = v(ih, n\tau)$）。证明：函数型解是因果的。是否有必要改变函数 $\rho$？

9）如何将离散格式中学过的 $\ell^1$ 和 $\ell^2$ 稳定性翻译到函数型格式中？

10）将冯·诺依曼放大系数（coefficient d'amplification）的一般概念推广到函数型差分格式中。

11）计算格式 (1) 的 $L^2$ 与 $L^\infty$ 稳定性条件。

12）将上述问题应用于输运方程与一阶偏移格式（schéma décentré d’ordre 1）。

13）如何验证格式 (1) 的一致性？这种一致性如何在傅里叶空间中表现出来？

14）为求解输运方程：

$$
\partial_t u + c \partial_x u = 0, \quad c > 0,
$$

考虑如下格式：

$$
\frac{u^{n+1}_i - u^n_i}{\tau} + c \cdot \frac{3u^n_i - 4u^n_{i-1} + u^n_{i-2}}{h} = d_n u_0(ih),
$$

其中：

$$
d_n = 
\begin{cases}
1/\tau, & n = 0, \\
0, & \text{否则}.
\end{cases}
$$

使用函数型方法计算该格式的稳定性。

---

需要我为你重点讲解哪一部分内容？
	