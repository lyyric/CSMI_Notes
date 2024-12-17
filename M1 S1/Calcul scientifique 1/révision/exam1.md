**题目：计算科学考试准备**

设 $E$ 为在 $\mathbb{R}_x \times \mathbb{R}_t$ 上定义的函数类 $u(x,t)$，满足以下条件：

1. 对任意固定 $t$，映射 $x \mapsto u(x,t)$ 属于 $L^2(\mathbb{R}) \cap L^1(\mathbb{R})$。
2. 对任意固定 $t$，映射 $x \mapsto u(x,t)$ 是 $C^2$ 类函数。
3. 对任意固定 $t$，有 $u(x,t) \to 0$ 当 $x \to \pm \infty$。
4. 存在一个实数 $T$ 使得对所有 $t \leq T$，有 $u(x,t) = 0$。

在以下内容中，我们称满足上述条件的函数 $u \in E$ 为**因果函数（causale）**。

**问题：**

1. 证明：若 $u$ 是因果函数并满足热方程或输运方程（运输方程），则 $u$ 必为零函数。

2. 对于一个满足 $C^2$ 且有紧支集的初值 $u_0(x)$，考虑如下定义的函数 $u$：
   $$
   \partial_t u - \partial_{xx}u = 0, \quad t \geq 0,
   $$
   $$
   u(x,t=0) = u_0(x),
   $$
   $$
   u(x,t) = 0, \quad t < 0.
   $$
   证明 $u$ 是良好定义的，并且是一个因果函数。

3. 证明在分布意义下，$u$ 对任意 $(x,t)\in \mathbb{R}^2$ 满足
   $$
   \partial_t u - \partial_{xx}u = \delta u_0,
   $$
   其中 $\delta$ 是时间变量上的 Dirac 分布（形式定义如下：
   $$
   \int \delta(t)\varphi(t)\,dt = \varphi(0)
   $$）。

4. 因此，可以将第2问中有时间初值条件的问题，用有分布型右端项的问题替代，即在所有时间上求解下式：
   $$
   \partial_t u - \partial_{xx}u = \delta u_0,
   $$
   且 $u$ 为因果函数。证明在因果函数集合中，此问题的解是唯一的。

5. 为了进行数值计算，我们需要逼近 Dirac 分布。考虑一个函数 $\rho: \mathbb{R} \to \mathbb{R}_+$，其支集在 $[-1,1]$ 上，并满足
   $$
   \int \rho = 1.
   $$
   对于 $\tau > 0$，定义函数（称为近似单位函数）
   $$
   \delta_\tau(t) = \frac{1}{\tau}\rho\left(\frac{t}{\tau}\right).
   $$
   证明当 $\tau \to 0$ 时，$\delta_\tau$ 在分布意义下收敛于 $\delta$，即对所有测试函数 $\varphi$ 有
   $$
   \lim_{\tau \to 0}\langle \delta_\tau,\varphi \rangle = \langle \delta,\varphi \rangle.
   $$
   若无特别说明，以下均取
   $$
   \rho(t) = \begin{cases}
   1 & \text{若 } t \in [0,1],\\
   0 & \text{否则}.
   \end{cases}
   $$

6. 对于 $\tau > 0$ 和 $h > 0$，在 $E$ 上定义时间与空间的平移算子：
   $$
   (T_\tau u)(x,t) = u(x,t-\tau), \quad (S_h u)(x,t) = u(x - h, t).
   $$
   证明这些算子都是从 $E$ 到 $E$ 的映射。

7. 回顾在 $R$ 上对 $x$ 变量的 Fourier 变换定义为
   $$
   \hat{u}(\xi,t) = \int_{-\infty}^\infty u(x,t)e^{-i\xi x}\,dx.
   $$
   计算 $S_h u$ 的 Fourier 变换，并证明如果 $u$ 是因果函数，则 $(\xi,t) \mapsto \hat{u}(\xi,t)$ 也是因果函数。

8. 为了逼近热方程的解，我们考虑如下的函数型差分格式：
   $$
   \frac{v - T_\tau v}{\tau} + \frac{-S_h v + 2v - S_{-h}v}{h^2} = \delta_\tau v_0. \quad (1)
   $$
   若令 $v_i^n = v(ih, n\tau)$，证明此函数型格式等价于经典的显式差分格式。证明该函数型解是因果的。改变函数 $\rho$ 是否有意义？

9. 如何在函数型差分格式的层面表达在离散格式中所学过的 $\ell^1$ 和 $\ell^2$ 稳定性？

10. 将 von Neumann 增幅因子（amplification factor）的概念推广到函数型差分格式的情况。

11. 计算格式（1）的 $L^2$ 和 $L^\infty$ 稳定性条件。

12. 用同样的问题和方法研究输运方程以及一阶偏风（上风）差分格式的情况。

13. 如何验证格式（1）的相容性（consistency）？在 Fourier 空间中该相容性性质如何体现？

14. 为了求解输运方程
   $$
   \partial_t u + c\partial_x u = 0, \quad c>0,
   $$
   考虑如下格式：
   $$
   \frac{u^{n+1}_i - u^n_i}{\tau} + c \frac{3u^n_i - 4u^n_{i-1} + u^n_{i-2}}{h} = d_n u_0(ih),
   $$
   其中
   $$
   d_n = \begin{cases}
   \frac{1}{\tau} & \text{若 } n=0,\\
   0 & \text{若 } n > 0.
   \end{cases}
   $$

   用函数型方法计算该格式的稳定性。

---
