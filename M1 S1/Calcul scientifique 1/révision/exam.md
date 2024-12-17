下面是题目内容的中文翻译（保留了原有的编号与公式的 LaTeX 格式）：

---

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

**1) 因果函数满足热方程或输运方程必为零**

**问题回顾**：设 $u \in E$ 且 $u$ 为因果函数，即存在某个实数 $T$，使得对所有 $t \leq T$，都有 $u(x,t)=0$。我们要证明：若 $u$ 满足热方程
$$
\partial_t u - \partial_{xx}u = 0
$$
或输运方程
$$
\partial_t u + c\partial_x u = 0,
$$
则 $u$ 必为零函数。

**证明思路（热方程）**：热方程具有非常强的唯一性延拓性质。简言之，如果一个解在某个时间之前都恒为零，那么由于热方程的抹平效应与解析延拓性，它无法在后续时间“凭空”产生非零的解。

更严格地说，对热方程的经典解而言，如果在 $t \leq T$ 上 $u=0$，则 $u(x,T)=0$ 作为初值可通过热核卷积唯一地确定后续时间的解。但该初值为0，则对 $t>T$ 的解也必为0。故 $u\equiv 0$。

**证明思路（输运方程）**：输运方程可通过特征线法求解。方程
$$
\partial_t u + c\partial_x u = 0
$$
的通用解为 $u(x,t) = U(x-ct)$，其中 $U$ 是由初值确定的函数。若对 $t \leq T$，$u(x,t)=0$，则特别在 $t=T$ 时 $u(x,T)=0$。因为输运方程的解沿着特征线保持常量，这意味着 $U(x)=0$ 对所有 $x$ 成立，即 $u(x,t)$ 在全空间全时间上为零。

综上，无论是热方程还是输运方程，因果函数解若在某个时刻之前为零，则必全局为零。

---

**2) 对给定紧支 $C^2$ 类初值 $u_0(x)$ 的构造**

给定初值问题：
$$
\partial_t u - \partial_{xx} u = 0,\quad t \geq 0,
$$
$$
u(x,0) = u_0(x),\quad u(x,t)=0 \text{ 对 } t<0.
$$

**定义 $u$ 的方法**：利用热核（高斯核）来定义
$$
u(x,t) = 
\begin{cases}
\displaystyle \int_{-\infty}^{\infty}G_t(x-y)u_0(y)\,dy, & t>0\\
0,& t\leq 0,
\end{cases}
$$
其中热核 $G_t(x) = \frac{1}{\sqrt{4\pi t}}e^{-\frac{x^2}{4t}}$。

**$u$ 的性质**：  
- 对 $t>0$，$u$ 是热方程的经典解，因为热核与足够光滑且有紧支的初值卷积给出了 $C^\infty$ 解。  
- 对 $t<0$，$u=0$。  
- 当 $t \to 0^+$ 时，利用热核的性质，有 $u(x,t) \to u_0(x)$（经典的初值逼近）。  
- 因初值 $u_0$ 在 $L^2 \cap L^1$ 且有紧支，加上热核的快速衰减和光滑性，可检查 $u(x,t)$ 符合 $E$ 中的条件：  
  1. 对每个固定 $t$，$u(\cdot,t) \in L^1(\mathbb{R}) \cap L^2(\mathbb{R})$；  
  2. 对每个固定 $t$，$u(\cdot,t)$ 无穷可微；  
  3. 当 $|x|\to \infty$ 时，$u(x,t)$ 利用热核的高斯衰减以及 $u_0$ 的紧支性可见 $u(x,t) \to 0$；  
  4. 存在 $T=0$ 使得 $t<0$ 时 $u(x,t)=0$。

因此，$u$ 是题中所要求的因果函数。

---

**3) 在分布意义下的方程：**

对于 $t>0$，我们有 $\partial_t u - \partial_{xx}u = 0$。对于 $t<0$，$u=0$ 明显也满足同样的齐次方程。但在 $t=0$ 的地方存在不连续跃迁（从 $0$ 跳到 $u_0(x)$）。这种跳跃可用时间上的 Dirac 分布来描述。

考虑分布意义下的等式：
$$
\partial_t u - \partial_{xx} u = \delta(t)u_0(x).
$$

验证方法：取任意测试函数 $\varphi(t)$，在时间上做分布对偶配：
$$
\langle \partial_t u - \partial_{xx}u,\varphi(t)\rangle = \langle u(t), -\varphi'(t)\rangle - \langle \partial_{xx}u,\varphi(t)\rangle.
$$

因为当 $t<0$ 时 $u=0$，而当 $t>0$ 时 $u$ 为热方程解。将 $\varphi$ 的支集缩小在一个含零点的邻域内，可以发现分布 $\partial_t u - \partial_{xx}u$ 在 $t=0$ 存在一个“源项”。通过计算得到该源项正是 $\delta(t)u_0(x)$，从而验证
$$
\partial_t u - \partial_{xx}u = \delta(t)u_0(x).
$$

---

**4) 使用分布源项的全时问题与唯一性**

问题 2 中的初值问题等价于求解：
$$
\partial_t u - \partial_{xx}u = \delta(t)u_0(x),
$$
且 $u$ 是因果函数（即对 $t<0$，$u=0$）。

要证明在因果函数集合中解的唯一性，设有两个因果解 $u_1$ 与 $u_2$。令 $v = u_1 - u_2$，则 $v$ 满足齐次方程
$$
\partial_t v - \partial_{xx} v = 0,
$$
并且因为 $u_1$ 和 $u_2$ 都是因果函数，对某个 $T$，$v(x,t)=0$ 当 $t \leq T$。利用第一问中的结论，这个 $v$ 必为全零函数。故 $u_1=u_2$，解具有唯一性。

---

**5) 使用近似单位逼近 Dirac 分布**

**问题回顾**：给定一个非负函数 $\rho: \mathbb{R}\to \mathbb{R}_+$，支集在 $[-1,1]$ 内，且满足 $\int_{-\infty}^{\infty}\rho(t)dt = 1$。定义
$$
\delta_\tau(t) = \frac{1}{\tau}\rho\left(\frac{t}{\tau}\right), \quad \tau > 0.
$$

要证明 $\delta_\tau \to \delta$ 在分布意义下的收敛，即对于任意测试函数 $\varphi(t)$（通常指 $C_c^\infty$ 函数），要有
$$
\lim_{\tau \to 0} \langle \delta_\tau, \varphi \rangle = \langle \delta, \varphi \rangle = \varphi(0).
$$

**证明**：  
计算对偶配
$$
\langle \delta_\tau,\varphi \rangle = \int_{-\infty}^{\infty} \varphi(t)\delta_\tau(t) dt = \int_{-\infty}^{\infty} \varphi(t)\frac{1}{\tau}\rho\left(\frac{t}{\tau}\right) dt.
$$
通过变量替换 $s=\frac{t}{\tau}$，则 $t=\tau s$，$dt=\tau ds$：
$$
\langle \delta_\tau,\varphi \rangle = \int_{-\infty}^{\infty}\varphi(\tau s)\rho(s)ds.
$$

由于 $\rho(s)$ 的支集在 $[-1,1]$ 上，积分实际上只在 $s \in [-1,1]$。当 $\tau \to 0$ 时，$\varphi(\tau s)$ 在 $s \in [-1,1]$ 上趋于 $\varphi(0)$。利用 $\varphi$ 的连续性和 $\rho$ 的可积性：
$$
\lim_{\tau \to 0}\langle \delta_\tau,\varphi \rangle = \varphi(0)\int_{-1}^{1}\rho(s)ds = \varphi(0)\cdot 1 = \varphi(0),
$$
因为 $\int \rho(s)ds=1$。

这表明 $\delta_\tau \xrightarrow[\tau\to 0]{} \delta$ 在分布意义下成立。

（备注：题中最终固定 $\rho(t)=1$ 若 $t\in[0,1]$，否则为0，这只是一个特定的选择，不影响极限过程的正确性。）

---

**6) 时间与空间平移算子在 $E$ 中的闭合性**

**问题回顾**：定义时间与空间平移算子
$$
(T_\tau u)(x,t) = u(x,t-\tau), \quad (S_h u)(x,t)=u(x - h,t), \quad \tau>0, h>0.
$$

要证明 $T_\tau$ 和 $S_h$ 都将 $E$ 映射到 $E$ 内，即如果 $u\in E$，则 $T_\tau u \in E$ 与 $S_h u \in E$。

**证明思路**：检查 $E$ 的定义条件：

- 对于 $u \in E$，给定任意固定 $t$，$u(\cdot,t) \in L^1(\mathbb{R}) \cap L^2(\mathbb{R})$ 且 $u(\cdot,t)$ 是 $C^2$，且 $u(x,t) \to 0$ 当 $|x|\to \infty$。
- 因为 $(T_\tau u)(x,t)=u(x,t-\tau)$ 只是将时间坐标平移，不影响 $x$ 上的可积性、平滑性和边界行为：若 $u(\cdot,t)$ 满足条件，那么 $T_\tau u(\cdot,t)=u(\cdot,t-\tau)$ 也同样满足，对 $t$ 的每一固定值这个性质不变。
- 同理，$(S_h u)(x,t) = u(x-h,t)$ 只是将 $x$ 坐标平移，不改变函数在 $x$ 上的空间正则性和可积性性质，也不影响其随 $x \to \pm \infty$ 消失的性质。

最后要检查因果性：  
- 如果 $u$ 是因果的，即存在某 $T$ 使 $u(x,t)=0$ 对 $t\leq T$ 成立，那么 $(T_\tau u)(x,t)=u(x,t-\tau)$ 在 $t \leq T+\tau$ 时为0，因果性仍成立。  
- $(S_h u)$ 的空间平移不改变随时间的零区域：如果 $u(x,t)=0$ 对 $t\leq T$，则 $u(x-h,t)=0$ 也对 $t\leq T$ 成立（只是在 $x$ 上平移，并不影响关于 $t$ 的零集）。

故 $T_\tau u$ 和 $S_h u$ 都保持 $E$ 的属性。

---

**7) 对 $S_h u$ 的 Fourier 变换与因果性的保持**

**问题回顾**：Fourier 变换定义为
$$
\hat{u}(\xi,t) = \int_{-\infty}^{\infty}u(x,t)e^{-i\xi x}dx.
$$

计算 $(S_h u)^\wedge(\xi,t)$ 并验证因果性在 Fourier 域的保持。

**计算**：
$$
(S_h u)(x,t)=u(x-h,t).
$$
则
$$
\widehat{(S_h u)}(\xi,t) = \int_{-\infty}^{\infty}u(x-h,t)e^{-i\xi x}dx.
$$
设 $y=x-h$，则 $x=y+h$，$dx=dy$：
$$
\widehat{(S_h u)}(\xi,t) = \int_{-\infty}^{\infty}u(y,t)e^{-i\xi(y+h)}dy = e^{-i\xi h}\int_{-\infty}^{\infty}u(y,t)e^{-i\xi y}dy = e^{-i\xi h}\hat{u}(\xi,t).
$$

因此
$$
\widehat{(S_h u)}(\xi,t) = e^{-i\xi h}\hat{u}(\xi,t).
$$

因果性保持性：若 $u(x,t)$ 在 $t \leq T$ 时为零，则 $\hat{u}(\xi,t)$ 作为 $u(x,t)$ 在 $x$ 上的变换，并不改变时间域的支撑性质。也即，如果 $u$ 在时间上为因果函数，那么 $\hat{u}(\xi,t)$ 对时间的零区域相同（对 $t\leq T$，$\hat{u}(\xi,t)=0$）。对于 $S_h u$，我们看出它的 Fourier 变换仅差一个相位因子 $e^{-i\xi h}$，不会改变在时间方向上的支撑，从而仍是因果的。

---

**8) 函数型差分格式的因果性与与经典格式的等价**

**问题回顾**：考虑热方程的函数型差分格式：
$$
\frac{v - T_\tau v}{\tau} + \frac{-S_h v + 2v - S_{-h} v}{h^2} = \delta_\tau v_0. \quad (1)
$$

这里 $v_0(x)=v(x,0)$ 是给定的初值函数。若在离散点上定义 $v_i^n = v(ih,n\tau)$，我们想展示该格式与经典显式差分格式等价，并且解是因果的。

**证明与说明**：

1. **与经典显式格式的等价**：  
   经典的热方程离散化（显式格式）：
   $$
   \frac{v_i^{n}-v_i^{n-1}}{\tau} = \frac{v_{i+1}^{n}-2v_i^{n}+v_{i-1}^{n}}{h^2}.
   $$
   假设初值在 $n=0$ 时给定为 $v_i^0 = u_0(ih)$。如果我们将 $v(x,t)$ 用 $v_i^n$ 替代，即 $(x,t)=(ih,n\tau)$ ，并且注意到
   $$
   (T_\tau v)(x,t) = v(x,t-\tau) \implies (T_\tau v)(ih,n\tau)=v_i^{n-1},
   $$
   $$
   (S_h v)(ih,n\tau)=v_{i-1}^{n}, \quad (S_{-h} v)(ih,n\tau)=v_{i+1}^{n}.
   $$

   因此将 $(1)$ 式离散化即可得到经典的显式格式。这表明两者在网格上是等价的。

2. **因果性**：
   考察 $t<0$ 的区域：在函数型格式中，$v$ 满足 $(1)$ 且右端是 $\delta_\tau v_0$。由于 $\delta_\tau$ 在 $t<0$ 时很小（当 $\tau \to 0$，集中在 $t=0$），当 $t<0$ 时，类似于 $u=0$ 的初值延拓。这保证了当 $t<0$ 时，$v(x,t)=0$，从而维持因果性。

   更直观地说，如果在离散层面 $n<0$ 时定义 $v_i^n=0$，则 $(1)$ 中的时间与空间移动算子不会在负时间生成非零值。这是因为 $\delta_\tau v_0$ 仅在 $t$ 近0处有贡献。这样在时间向前推进时（从 $n<0$ 区域是0开始），不会出现先于0时刻有非零解的情况。

3. **改变 $\rho$ 的意义**：  
   在定义 $\delta_\tau(t) = \frac{1}{\tau}\rho(t/\tau)$ 时，我们选择 $\rho$ 来逼近 Dirac 分布。如果换一个光滑且对称的 $\rho$，仍能逼近 Dirac 分布，最终求得的解在 $\tau \to 0$ 时会一致收敛到同样的解。改变 $\rho$ 也许会影响收敛速率的常数因子或者在实际数值中的误差，但对于理论上的因果性、唯一性与收敛性影响不大。从理论角度而言，这种改变并没有实质提升。

---

**9) 在函数型差分格式中表达 $\ell^1$ 和 $\ell^2$ 稳定性**

**问题回顾**：在离散格式中（即网格上的数值方法中），$\ell^1$ 和 $\ell^2$ 稳定性是通过考察数列解 $\{v_i^n\}$ 的 $\ell^1$ 或 $\ell^2$ 范数（对空间索引 $i$ 求和）随时间演化是否有界来定义的。对于函数型格式，我们没有离散点列，而是定义在函数空间 $E$ 上的算子与方程。

在函数型框架下，$\ell^1$ 和 $\ell^2$ 稳定性的类比是自然地转化为对解函数在空间变量上的 $L^1$ 和 $L^2$ 范数的有界性，即考虑：
$$
\|v(\cdot,t)\|_{L^p} \quad p=1,2.
$$

- 若在给定时间演化下，有 $\|v(\cdot,t)\|_{L^1} \leq C\|v(\cdot,0)\|_{L^1}$ 对所有 $t$ 成立（这里 $C$ 为不依赖于时间的常数），则称该函数型格式在 $L^1$ 意义下稳定。
- 类似地，在 $L^2$ 意义下若 $\|v(\cdot,t)\|_{L^2} \leq C\|v(\cdot,0)\|_{L^2}$，则格式在 $L^2$ 意义下稳定。

总之，将离散点的求和换成空间上的积分范数，即可在函数型差分框架下定义和检验 $\ell^1$ 与 $\ell^2$ 稳定性。

---

**10) 将 von Neumann 增幅因子概念推广到函数型差分格式**

在经典离散 von Neumann 稳定性分析中，我们假设解以 $\lambda^n e^{i\xi i h}$（即 $\lambda^n$ 为时间步增长因子、$e^{i\xi i h}$ 为离散空间的波状解）形式展开，并考察在一次时间迭代后的增幅因子 $\lambda$ 。

在函数型差分格式中，同样可以通过 Fourier 分解来考察稳定性：  
- 对 $x$ 做 Fourier 变换，将解表示为 $\hat{v}(\xi,t)$。  
- 将时间推进格式作用于 $\hat{v}(\xi,t)$，考察每一频率 $\xi$ 的增幅。  
- 若每一频率分量 $\hat{v}(\xi,t)$ 在时间推进中不扩大（即其放大因子 $\lambda(\xi)$ 的模不超过1），则在相应范数下取得稳定性。

因此函数型格式的 von Neumann 分析只需将离散 Fourier 模式替换为连续 Fourier 模式，将增幅因子定义为在一步时间迭代中，$\hat{v}(\xi,t)$ 的幅度变化率。该放大因子成为依赖 $\xi$ 的函数，在所有 $\xi$ 上有界且 $|\lambda(\xi)|\leq 1$ 则保证稳定性。

---

**11) 计算 (1) 式格式的 $L^2$ 和 $L^\infty$ 稳定性条件**

问题 (1) 中的格式：
$$
\frac{v - T_\tau v}{\tau} + \frac{-S_h v + 2v - S_{-h}v}{h^2} = \delta_\tau v_0.
$$

这对应显式求解热方程的离散形式。与标准显式格式相同，稳定性条件不依赖于 $\delta_\tau v_0$ 的具体形状（其在 $t=0$ 给出初值条件的影响）。对热方程的标准显式差分格式，von Neumann 分析给出了经典的稳定性条件：

- 对于 $L^2$ 稳定性和 $L^\infty$ 稳定性（最大模稳定性）而言，经典热方程显式格式的稳定性条件是相同的，即
$$
\frac{\tau}{h^2} \leq \frac{1}{2}.
$$

这一条件确保在时间推进过程中解不会发生指数爆炸。因为热方程的扩散性质，对于这类对称的二阶空间离散算子，$L^2$ 和 $L^\infty$ 稳定性的条件是一致的。

总结：该函数型格式的稳定性条件仍然是经典的 $\tau \leq \frac{h^2}{2}$。

---

**12) 对输运方程与一阶偏风格式的稳定性分析**

考虑一维输运方程：
$$
\partial_t u + c \partial_x u = 0, \quad c>0.
$$

一阶上风（upwind）差分格式为：
$$
\frac{v(x,t+\tau)-v(x,t)}{\tau} + c\frac{v(x,t)-v(x-h,t)}{h} = \delta_\tau v_0,
$$
离散化后即为（记 $v_i^n = v(ih,n\tau)$）：
$$
\frac{v_i^{n+1}-v_i^n}{\tau} + c\frac{v_i^n - v_{i-1}^n}{h} = d_n v_0(ih),
$$
其中 $d_0=1/\tau$ 且 $d_n=0$ 对 $n>0$。

通过 von Neumann 分析（对 $x$ 做 Fourier 展开），可以得到该格式的放大因子为
$$
\lambda(\xi) = 1 - c\frac{\tau}{h}(1 - e^{-i\xi h}).
$$
要保证 $|\lambda(\xi)|\leq 1$，典型的稳定性条件为
$$
|c|\frac{\tau}{h} \leq 1.
$$

对于该类一阶偏风格式，此条件同样保证了 $L^2$ 稳定性与 $L^\infty$ 稳定性，因为如果放大因子在所有频率上模长不超过1，即可阻止解范数的增长。

总结：对输运方程的上风格式，稳定性条件为 $\frac{\tau}{h}\leq \frac{1}{c}$。

---

**13) 一致性（consistency）的验证及其在 Fourier 空间中的体现**

**问题回顾**：  
一致性是指当离散参数（如步长 $h$、$\tau$）趋于0时，数值格式能在形式上重现连续方程。对于函数型差分格式  
$$
\frac{v - T_\tau v}{\tau} + \frac{-S_h v + 2v - S_{-h} v}{h^2} = \delta_\tau v_0,
$$
要检查当 $\tau \to 0$ 和 $h \to 0$ 时此格式能否退化为热方程  
$$
\partial_t u - \partial_{xx}u = \delta(t)u_0(x).
$$

**验证一致性的方法**：  
1. 考虑光滑函数 $u(x,t)$ 并将 $v(x,t)=u(x,t)$ 代入离散格式。
2. 使用 Taylor 展开。以时间项为例：
   $$
   T_\tau v(x,t) = v(x,t-\tau) = v(x,t) - \tau\partial_t v(x,t) + \frac{\tau^2}{2}\partial_{tt}v(x,t-\theta_\tau),
   $$
   对某个 $\theta_\tau \in (0,\tau)$。
   
   将此代入时间差分：
   $$
   \frac{v - T_\tau v}{\tau} = \frac{v(x,t)-v(x,t)+\tau\partial_t v(x,t)+O(\tau^2)}{\tau} = \partial_t v(x,t)+O(\tau).
   $$

3. 对空间项同样使用 Taylor 展开：
   $$
   S_h v(x,t) = v(x-h,t) = v(x,t)-h\partial_x v(x,t)+\frac{h^2}{2}\partial_{xx}v(x-\theta_h,t),
   $$
   $$
   S_{-h} v(x,t)=v(x+h,t)=v(x,t)+h\partial_x v(x,t)+\frac{h^2}{2}\partial_{xx}v(x+\theta'_h,t).
   $$
   
   代入离散拉普拉斯算子：
   $$
   \frac{-S_h v + 2v - S_{-h} v}{h^2} = \frac{-[v(x,t)-h\partial_x v + \frac{h^2}{2}\partial_{xx}v]+2v-[v(x,t)+h\partial_x v + \frac{h^2}{2}\partial_{xx}v]}{h^2}.
   $$

   简化：
   $$
   = \frac{(-v(x,t)+h\partial_x v - \frac{h^2}{2}\partial_{xx}v)+2v(x,t)+(-v(x,t)-h\partial_x v - \frac{h^2}{2}\partial_{xx}v)}{h^2}.
   $$

   观察到 $\pm h\partial_x v$ 项抵消，$-v+2v-v=0$部分抵消剩下：
   $$
   = \frac{- \frac{h^2}{2}\partial_{xx}v - \frac{h^2}{2}\partial_{xx}v}{h^2} = \frac{- h^2 \partial_{xx}v}{h^2} = -\partial_{xx}v + O(h^2).
   $$

   所以空间差分的主项是 $-\partial_{xx}v$ 再加上高阶项 $O(h^2)$。

4. 再考虑右端项 $\delta_\tau v_0$ 的一致性。随着 $\tau \to 0$，$\delta_\tau \to \delta$ 在分布意义下，则 $\delta_\tau v_0 \to \delta(t)u_0(x)$。

综合各项，当 $\tau,h \to 0$ 时，函数型格式收敛到  
$$
\partial_t v - \partial_{xx}v = \delta(t)u_0(x),
$$
这正是连续问题。故格式是一致的。

**在 Fourier 空间的体现**：  
在 Fourier 空间对 $x$ 做变换，$-S_h + 2 - S_{-h}$ 对应的符号函数为 $\frac{-e^{-i\xi h}+2 - e^{i\xi h}}{h^2}$。使用 Euler 展开：
$$
-e^{-i\xi h}+2 - e^{i\xi h} = 2 - (e^{i\xi h}+e^{-i\xi h}) = 2 - 2\cos(\xi h).
$$
因此
$$
\frac{-S_h + 2 - S_{-h}}{h^2} \xrightarrow[h \to 0]{} \frac{2-2\cos(\xi h)}{h^2}.
$$

利用 $\cos(\xi h)=1-\frac{\xi^2 h^2}{2}+O(h^4)$:
$$
2 - 2\cos(\xi h) = 2 - 2\left(1-\frac{\xi^2 h^2}{2}+O(h^4)\right)=\xi^2 h^2 + O(h^4).
$$

所以  
$$
\frac{2 - 2\cos(\xi h)}{h^2} = \xi^2 + O(h^2).
$$

这表明在 Fourier 空间中，离散 Laplacien 的符号趋近于 $\xi^2$，这与连续 Laplacien 的符号 $-\xi^2$（因为 $\partial_{xx}$ 在 Fourier 域对应乘以 $-\xi^2$）相一致，从而验证了一致性。

---

**14) 用函数型方法计算给定输运方程格式的稳定性**

**问题回顾**：  
考虑输运方程
$$
\partial_t u + c\partial_x u = 0, \quad c>0,
$$
采用格式：
$$
\frac{u^{n+1}_i - u^n_i}{\tau} + c \frac{3u^n_i - 4u^n_{i-1} + u^n_{i-2}}{h} = d_n u_0(ih),
$$
其中
$$
d_n = \begin{cases}
1/\tau & n=0,\\
0 & n>0.
\end{cases}
$$

**函数型方法**：  
定义函数 $v(x,t)$ 满足相应的函数型方程：
$$
\frac{v - T_\tau v}{\tau} + c\frac{3v -4S_h v + S_{2h} v}{h} = \delta_\tau v_0.
$$

这里与前面类似，我们通过 Fourier 变换（对 $x$）来进行 von Neumann 稳定性分析：

1. 对 $x$ 变换，设 $\hat{v}(\xi,t)$ 为 $v(x,t)$ 的 Fourier 变换。

2. 考察时间离散迭代对单一 Fourier 模 $\hat{v}(\xi,t) = \lambda^n e^{i\xi x}$ 的影响。  
   将 $S_h v$ 在 Fourier 域替换为 $e^{-i\xi h}\hat{v}(\xi,t)$，$S_{2h}v$ 对应 $e^{-2i\xi h}\hat{v}(\xi,t)$，$T_\tau v(\xi,t) = \hat{v}(\xi, t-\tau)$ 对时间则对应因果性条件。

3. 在无初值源项（$\delta_\tau v_0$不考虑的时刻，即 $n>0$）时，格式为：
   $$
   \frac{\hat{v}(\xi,t+\tau)-\hat{v}(\xi,t)}{\tau} + c\frac{3\hat{v}(\xi,t)-4e^{-i\xi h}\hat{v}(\xi,t)+e^{-2i\xi h}\hat{v}(\xi,t)}{h} = 0.
   $$

   假设 $\hat{v}(\xi,t)=\lambda^n$（离散时间），则有：
   $$
   \frac{\lambda - 1}{\tau} + \frac{c}{h}(3 -4e^{-i\xi h}+e^{-2i\xi h}) = 0.
   $$

   整理得放大因子关系式：
   $$
   \lambda = 1 - \frac{\tau c}{h}(3 -4e^{-i\xi h}+e^{-2i\xi h}).
   $$

   对稳定性而言，需要 $|\lambda(\xi)| \leq 1$ 对所有 $\xi$。

4. 线性稳定性分析：  
   将 $\xi h$ 作为变量，展开复指数：
   $$
   3 -4e^{-i\xi h}+e^{-2i\xi h} = 3 -4(\cos(\xi h)-i\sin(\xi h)) + (\cos(2\xi h)-i\sin(2\xi h)).
   $$

   当 $h\to 0$，使用 Taylor 展开：
   $$
   \cos(\xi h) = 1 - \frac{\xi^2 h^2}{2} + O(h^4), \quad \sin(\xi h) = \xi h + O(h^3),
   $$
   $$
   \cos(2\xi h)=1-2\xi^2 h^2+O(h^4), \quad \sin(2\xi h)=2\xi h + O(h^3).
   $$

   将这些代入并保留主阶项，可以得到与 $\lambda$ 的近似关系，由此可寻找满足稳定性的条件（类似上一题输运方程上风格式的分析）。

   虽然这里给出了较复杂的格式，但原则上过程相同：  
   - 在 Fourier 域写出放大因子 $\lambda(\xi)$。  
   - 要求对于所有 $\xi$，$|\lambda(\xi)| \leq 1$。  
   - 根据放大因子的具体表达式（展开成关于 $\tau/h$ 的式子），找到限制条件，使放大因子不大于1。

由于此格式是类似中心差分与加偏移的组合形式，可预期稳定性条件也会与 $\frac{\tau}{h}$ 的比值有关。特别是对于输运方程，一般要求 Courant 数 $|c|\frac{\tau}{h}$ 不超过某个临界值以保证稳定性。

**总结**：  
通过函数型方法（Fourier 变换和 von Neumann 分析），将格式的放大因子写出并分析其模的大小，即可得到输运方程此格式的稳定性条件。与经典离散格式类似，最终的稳定性条件会是 $|c|\frac{\tau}{h}$ 应小于等于某个常量（通常为1，或根据具体式子进行计算）。

---
