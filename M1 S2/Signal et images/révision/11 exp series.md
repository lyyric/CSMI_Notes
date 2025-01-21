## 指数展开

在前面的正弦余弦展开中，由于基函数取实值，概念较易理解；但在数值计算中，更常使用的是复数指数基（exponential family）。该基底在实际应用中往往更为实用与高效。

### 理论总结

我们现在考虑复值信号，其定义域仍是区间 $[0,T]$。此时我们需要引入**厄米内积**（hermitian product），定义为  
$$
\forall\, f,g,\qquad \mathtt{her}(f,g)=\frac{1}{T} \int_0^T f(t)\,\overline{g(t)}\, dt,
$$
其中 $\overline{g(t)}$ 表示 $g(t)$ 的共轭复数。

> [!attention]
> **注意：**  
> 如果你之前没有接触过厄米内积，不必担心，它只是标量积的一种变体。下面就来讨论其中的几个要点：
>
> 1. **$(1\heartsuit)$** 对称性被替换成了“共轭对称性”：也就是说，一般有  
>    $$
>    \mathtt{her}(f,g)=\overline{\mathtt{her}(g,f)},
>    $$
>    而不再是严格的对称（即 $=$ 的关系）。
>
> 2. **$(1\heartsuit)$** 关于 $g\to \mathtt{her}(f,g)$ 的映射，其不是严格线性的，而是**半线性的**（或称为共轭线性），这意味着对于复标量 $a$ 和 $b$，我们有  
>    $$
>    \mathtt{her}\bigl(f,\, a\, g_1+b\, g_2\bigr)= \overline{a}\,\mathtt{her}(f, g_1)+\overline{b}\,\mathtt{her}(f, g_2).
>    $$
>
> 3. **$(1\heartsuit)$** 第三个点（正定性）依然成立：即  
>    $$
>    \mathtt{her}(f,f)\ge 0,\quad \text{且只有当 } f\equiv 0 \text{时},\quad \mathtt{her}(f,f)=0.
>    $$
>    证明可以参照标量积的证明，加上复数情况的额外处理。

我们定义指数基底族 $(e_n)_{n\in \mathbb Z}$ 为  
$$
e_n(t)=\exp\Bigl(2i\pi \frac{n}{T}t\Bigr),
$$
其中 $i=\sqrt{-1}$。

> [!theorem]
> **指数版本的 3 点定理：**
> 
> 1. 指数基底族 $(e_n)_{n\in \mathbb Z}$ 在上述厄米内积下正交归一。
> 2. 复值信号 $f$ 可写为  
>    $$
>    f(t)=\sum_{n\in \mathbb Z} \alpha_n\, e_n(t)
>    $$
>    - 当 $f$ 在 $t$ 点连续时，该级数对每个 $t$ 收敛；
>    - 当 $f$ 连续且满足 $f(0)=f(T)$ 时，级数一致收敛。
> 3. 傅里叶指数系数的计算公式为  
>    $$
>    \alpha_n=\mathtt{her}(f,e_n).
>    $$
> 
> **补充说明：**
> 
> - 第一条结论是通过基本积分计算（利用指数函数的原函数）可直接验证的（下面有相关练习）。
> - 第二条是较难的部分，说明 $(e_n)$ 实际上构成了复值信号空间的一个基底。
> - 注意：不要写错顺序，使用 $\mathtt{her}(e_n,f)$ 得到的结果并非 $\alpha_n$，而是其共轭（这是因为内积的共轭对称性）。

---

### 练习与讨论

#### 练习（$2\heartsuit$）：验证正交归一性

**任务：** 请利用指数函数积分的基本公式验证  
$$
\frac{1}{T}\int_0^T e_n(t)\overline{e_m(t)}\, dt=\delta_{n,m}.
$$
  
**证明提示：**

- 当 $n=m$ 时，积分为  
  $$
  \frac{1}{T}\int_0^T 1\, dt=1.
  $$
- 当 $n\ne m$ 时，有  
  $$
  \frac{1}{T}\int_0^T \exp\Bigl(2i\pi \frac{n-m}{T}t\Bigr)dt = \frac{1}{T} \cdot \frac{\exp\Bigl(2i\pi (n-m)\Bigr)-1}{2i\pi (n-m)/T} = 0,
  $$
  因为 $\exp(2i\pi (n-m))=1$。

#### 练习（$2\heartsuit$）：由指数基正交归一性推出正弦余弦基底的正交归一性

**提示：**

- 对于实值函数 $f$ 与 $g$，有  
  $$
  \mathtt{dot}(f,g)=\frac{2}{T}\int_0^T f(t)g(t)\, dt = 2\,\mathtt{her}(f,g),
  $$
  因为 $f$ 和 $g$ 取实值，所以复共轭没有影响。
  
- 注意正弦余弦函数与指数函数之间的关系  
  $$
  \cos_n(t)=\frac{e_n(t)+e_{-n}(t)}{2},\quad
  \sin_n(t)=\frac{e_n(t)-e_{-n}(t)}{2i}.
  $$
  
由这些公式可把正弦余弦内积转化为指数内积，从而利用指数基的正交归一性证明正弦余弦基底的性质。

#### 复数在 Python 中

例如下面这行代码：
```python
(2+1j*4)*1j  # 这里 1j 表示 sqrt(-1)
np.exp(1j*np.pi)
```
  
**练习（$1\heartsuit$）：**  
为什么上面第二行代码输出的复数含有非零虚部？  
**解释：**  
$\exp(i\pi)=\cos\pi + i\sin\pi = -1 + i\cdot 0$ 理论上应该是 -1；但在计算机中，由于数值计算误差，很可能得到诸如 $-1+1.2246468e-16j$ 的结果，即虚部非常小但非零。这说明计算机中复数计算的舍入误差导致了非零虚部。

---

### Python 实战

下面给出构造指数基底的代码示例。

```python
T = 2
t = np.linspace(T*0, T, 200, endpoint=False)

def compute_basis_exp(t, T, M):
    nb_points = len(t)
    # 构造大小为 (2*M+1) x nb_points 的矩阵，数据类型为复数
    basis = np.empty([2*M+1, nb_points], dtype=np.complex128)
    for n in range(-M, M+1):
        # 将 n 从 -M 到 M 映射到行索引：行号 = n+M
        basis[n+M, :] = np.exp(+2*1j*np.pi*n*t/T)
    return basis

M = 5
N = 2*M+1
basis_expo = compute_basis_exp(t, T, M)
print(basis_expo.shape, basis_expo.dtype)
```

以下代码展示了如何将基底的实部和虚部分别绘制出来：

```python
fig, axs = plt.subplots(N, 2, figsize=(8, N), sharex=True, sharey=True)
for n in range(-M, M+1):
    i = n + M  # 行号
    axs[i, 0].plot(t, np.real(basis_expo[i, :]))
    axs[i, 1].plot(t, np.imag(basis_expo[i, :]))
    axs[i, 0].set_title("实部, n=%d" % n)
    axs[i, 1].set_title("虚部, n=%d" % n)
fig.tight_layout()
```

接下来，定义厄米内积函数，与之前定义的 `Decomposer` 类配合使用：

```python
def her(f, g):
    return np.sum(f * np.conj(g)) / len(f)

# 用复数类型构造 decomposer_expo 对象
decomposer_expo = Decomposer(t, basis_expo, her, dtype=np.complex128)

# 示例：对信号 f(t)= t*(2-t)**2+1 进行展开与逼近
decomposer_expo.compute_approximations(t*(2-t)**2+1, [2,5,7,9,11], True);
```

**练习题讨论：**

1. **练习（$2\heartsuit$）：**  
   为什么在使用较少基底项进行逼近时，橙色（逼近曲线）的效果与蓝色（原信号）相差较远？  
   **解释：**  
   初始级数逼近中如果选取的基底顺序按照频率排序不当，可能会先引入高频分量，从而产生严重的震荡现象（Gibbs 现象）。因此，低频分量较为重要，而高频分量对于信号的局部平滑性起到微调作用。当低频部分尚未完全逼近原信号时，混入高频成分可能导致整体形状偏离。

2. **练习（$1\heartsuit$）：**  
   设想一种更好的基底排序方法，使得一开始使用的基底项主要为低频部分，避免上述问题。  
   **思路：**  
   可以重新组织基底的顺序，将零频项、正低频和负低频项放在前面，而较高频率的项放在后面。具体来说，可以构造一个新的排序：例如  
   $$
   e_0,\, e_{1},\, e_{-1},\, e_{2},\, e_{-2},\, \dots
   $$
   这样排序之后，低频项先出现在展开中，有助于改善初期逼近效果。若实现时可写代码对基底矩阵的行重新排序，再将排序后的基底传递给 `Decomposer` 类。

---

### 坐标变换

每个实信号 $f$ 都可以用两种方式展开：  
- 一种是用正弦余弦基底 $\left(\frac{1}{\sqrt2},\cos_n, \sin_n\right)$ 展开；
- 另一种是用指数基底 $(e_n)$ 展开。

要在这两种展开之间建立转换关系，可利用傅里叶变换公式进行坐标变换。

对于 $n>0$，有：
$$
\alpha_n=\frac{1}{T}\int_0^T f(t)\,\overline{e_n(t)}\, dt = \frac{1}{T}\int_0^T f(t)(\cos_n(t) - i\sin_n(t))\, dt.
$$
**练习（$1\heartsuit$）：**  
请补全上述公式的计算，讨论 $n>0$、$n<0$ 和 $n=0$ 时的区别。

**解题提示：**

- 当 $n>0$ 时，上式可得到
  $$
  \alpha_n = \frac{1}{T}\int_0^T f(t)\cos_n(t)\, dt - i\,\frac{1}{T}\int_0^T f(t)\sin_n(t)\, dt.
  $$
- 当 $n<0$ 时，因为 $e_{-n}(t)=\overline{e_n(t)}$，可以推得  
  $$
  \alpha_n = \overline{\alpha_{-n}}.
  $$
- 当 $n=0$ 时，  
  $$
  \alpha_0=\frac{1}{T}\int_0^T f(t)\, dt.
  $$

**练习（$2\heartsuit$）：**  
试找出反向变换的公式，即如何由正弦余弦系数 $(a_n)$ 表达指数系数 $(\alpha_n)$。

答案可通过逆变换关系推导得到，通常有：
$$
a_0 = \sqrt{2}\,\alpha_0,\quad a_{2n-1} = 2\,\Re(\alpha_n),\quad a_{2n} = -2\,\Im(\alpha_n).
$$

**练习（$1\heartsuit$）：**  
在什么条件下，指数傅里叶系数 $\alpha_n$ 是实数？  
**提示：**  
令 $f(t)=g(t)+ih(t)$ 且 $g,h$ 均为实函数，然后注意  
$$
\Im(\alpha_n) = \frac{1}{T}\int_0^T \Im\Bigl(f(t)\,\overline{e_n(t)}\Bigr) dt.
$$
展开后可发现，当 $h(t)=0$（即 $f$ 为实信号），并且信号满足对称性条件时，虚部会消失；或者更精确地说，当 $f$ 具备**厄米对称性**时（见下文），则有 $\alpha_{-n}=\overline{\alpha_n}$，使得系数对应的实部与虚部分布对称，从而若系数自身为共轭对称且某些特殊条件满足时，系数可能为实。

---

### 厄米对称性

**证明练习：**  
证明当 $f$ 为实信号时，其复数傅里叶系数满足  
$$
\alpha_{-n}=\overline{\alpha_n}.
$$

**证明提示：**  
利用定义
$$
\alpha_n=\frac{1}{T}\int_0^T f(t)e^{-2i\pi n t/T}\, dt,
$$
并注意 $f(t)$ 为实数，取共轭便得  
$$
\overline{\alpha_n}=\frac{1}{T}\int_0^T f(t)e^{2i\pi n t/T}\, dt=\alpha_{-n}.
$$

下面代码展示了系数计算及绘图：

```python
alpha = decomposer_expo.compute_coordinates(t*(2-t)**2+1)
print(alpha)

fig, axs = plt.subplots(2, 1)
axs[0].plot(np.real(alpha), '.')
axs[1].plot(np.imag(alpha), '.')
axs[1].set_xticks(range(0, N))
axs[1].set_xticklabels(range(-M, M+1))
fig, ax = plt.subplots()
ax.plot(np.abs(alpha), '.')
```

---

### 频谱

下面给出一些关键的词汇：

- **频谱（spectrum）：** 就是傅里叶系数的集合 $(\alpha_n)_{n\in \mathbb Z}$；
- **振幅频谱（amplitude spectrum）：** 即频谱中每个系数的模 $(|\alpha_n|)_{n\in \mathbb Z}$；
- 当信号为实值时，由于厄米对称性，我们通常只考察半频谱  
  $$
  (\alpha_n)_{n\in \mathbb N}\quad \text{或}\quad (|\alpha_n|)_{n\in \mathbb N}.
  $$
- 由于计算机只能处理有限数据，我们通常只画截断频谱，即 $|n|\le M$ 部分，称之为截断（truncated）频谱。

**练习（$2\heartsuit$）：**  
请绘制我们信号的截断半振幅频谱。  
**提示：**  
- 利用前面计算得到的傅里叶系数，取 $n\ge 0$ 部分，并计算模值，再绘制出条形图或散点图。你可以参考前面绘制正余弦系数的代码进行修改。

---

## 信号滤波练习

信号分解的两个主要应用是**信号滤波**和**信号压缩**。

- **信号滤波：**  
  由于任一信号都能分解为若干基础波（如正弦或余弦），滤波就是保留部分频率成分：
  - **低通滤波器（Low-pass）：** 去掉高频成分；
  - **高通滤波器（High-pass）：** 去掉低频成分；
  - **带通滤波器（Band-pass）：** 只保留指定频带内的分量。
  
- **信号压缩：**  
  本质上与滤波类似，只保留主要的、可见的频率成分，其它细节舍去，从而实现数据压缩。

下面给出一段包含噪音的信号示例：

```python
f_noisy = np.loadtxt("assets_signal/signalToFilter.txt")
T = 2  # 信号时长2秒
t = np.linspace(0, T, 2000)
plt.plot(t, f_noisy)
```

**练习（$3\heartsuit$）：**  
**要求：** 不使用 `Decomposer` 类，而是手动写出所有公式，对噪声信号在正弦余弦基下进行展开。  
注意：该信号包含较高频噪声。

**练习（$2\heartsuit$）：**  
绘制该信号的振幅频谱图（截断版本）。

**练习（$4\heartsuit$）：**  
对信号进行滤波——只保留特定频率的成分（例如仅保留一个低频分量），并绘制滤波后的信号，同时将原始频谱与滤波后的频谱绘制在同一图中以便比较。

**解题提示：**

1. 根据正弦余弦展开公式  
   $$
   a_n = \mathtt{dot}(f, sc_n),
   $$
   逐项计算傅里叶系数。
2. 绘制频谱时，取振幅 $|a_n|$ （或者用指数系数 $|\alpha_n|$）。
3. 进行滤波时，只保留低频项，其余系数置零，再利用基底重构信号：
   $$
   f_{\text{filtered}}(t)= \sum_{\text{低频项}} a_n\, sc_n(t).
   $$
4. 用 matplotlib 将原始信号与滤波后信号以及对应频谱进行比较绘图。

---

## 完美采样

在前面所有代码中，我们看到积分均采用了矩形法近似（比较简单，但效果意外“完美”）：  
在采用“完美离散”的情况下，基底函数之间的内积正好等于 0 或 1。  
  
**注意关键：**  
- 离散采样时必须在 $[0,T]$ 内均匀采样，但**不包括端点 $T$**。  
- 原因在于：周期延拓时，$T$ 与 $0$ 是同一个点，若包含 $T$ 则重复计算，会破坏正交性。

代码示例：

```python
T = 2
t_perfect = np.linspace(0, T, 100, endpoint=False)
basis_sc_perfect = compute_sinCos_basis(t_perfect, T, 5)
decomposer_perfect = Decomposer(t_perfect, basis_sc_perfect, dot)
decomposer_perfect.check_ortho()
decomposer_perfect.compute_approximations((t_perfect-1)**2, [5, 11], True);
```

而如果采用了包含端点的采样：

```python
t_bad = np.linspace(0, T, 100)  # 默认 endpoint=True
f_bad = (t_bad-1)**2
print(t_bad)
basis_sc_bad = compute_sinCos_basis(t_bad, T, 5)
decomposer_bad = Decomposer(t_bad, basis_sc_bad, dot)
decomposer_bad.check_ortho()
decomposer_bad.compute_approximations(f_bad, [5, 11], True);
```

结果说明：  
- 当采用完美采样时，正弦余弦基函数在离散点下严格正交（依据离散内积  
  $$
  \mathtt{dot_{dis}}(f, g)=\frac{1}{N}\sum_n f(x_n)g(x_n)
  $$
  ，其中 $x_n$ 为采样点）。
- 若采样不完美，则正交性只有近似成立，逼近结果会明显偏差。

这种思想正是离散傅里叶变换（DFT）的理论基础，后续章节将详细讨论。

---

## 编程挑战：正交多项式

最后给出一个例子，证明除了正弦余弦之外，还有其他正交基。例如，Legendre 多项式常用于数值积分（高斯积分）中。  

**定义：**  
Legendre 多项式在区间 $[-1, 1]$ 上递归定义为：
$$
\begin{aligned}
nP_n(x) &= (2n-1)x\,P_{n-1}(x) - (n-1)\,P_{n-2}(x),\$$1mm]
P_0(x)&=1,\$$1mm]
P_1(x)&=x.
\end{aligned}
$$
它们对于积分内积  
$$
\mathtt{dot}(f,g)=\int_{-1}^{1} f(x)g(x)\, dx
$$
正交。

要构造完美离散的 Legendre 基，需要选取**高斯节点**，即 $P_{n+1}(x)$ 的根。令 $(x_0,\dots,x_n)$ 为这些节点，在节点上定义的权重 $w_i$ 使得  
$$
\mathtt{dot_{dis}}(f,g) = \sum_{i=0}^n w_i\, f(x_i)g(x_i)
$$
上，多项式集合正交。

下面代码利用 `scipy` 得到 Gauss-Legendre 节点及权重：

```python
import scipy.special as spe

degreeMax = 19
nbStep = 50  # 将区间 [-1,1] 划分成 nbStep 个点
# 获得 Gauss 点与对应的权重
x, w, _ = spe.roots_legendre(nbStep, True)
print("离散点：", x)
plt.plot(x, w, '.')
```

**练习：**

1. 定义在 Gauss 节点 $x$ 上采样的 Legendre 基，将其按行叠加到矩阵 `basis` 中，并绘制各基函数的图像：
   ```python
   for i in range(len(basis)):
       plt.plot(x, basis[i, :])
   ```
2. 计算所有基函数之间的内积。**Bonus：** 尝试不使用循环，用矩阵乘法 `@` 实现。  
   **提示：** 构造一个对角矩阵  
   ```python
   W = np.diag(w)
   ```
   则离散内积可写为：  
   $$
   \text{inner\_prod} = \text{basis} \; @\; W \; @\; \text{basis}^T.
   $$
3. 注意：Legendre 基函数固然正交，但通常不是归一化的。请构造一个新矩阵 `basisNor` 使得各行向量均归一化（即每个基函数的离散范数为 1）。
4. 利用该正交归一基，计算向量  
   $$
   y=\sqrt{|x|}
   $$
   关于 Legendre 多项式的逼近，并绘制近似结果。
5. 寻找另一个导致 Gipps 现象（高频震荡）明显的向量逼近案例。
6. 注意观察当基函数个数正好等于离散节点数（例如：`degreeMax = 19` 与 `nbStep = 20`）时会出现怎样的现象。

---

## 希尔伯特空间理论

下面介绍一些理论，但十分简单。

### 希尔伯特基

考虑 $L_2([0,T]\to\mathbb{C})$ 空间（平方可积的复值函数），则正弦余弦或指数傅里叶基构成了一个希尔伯特正交基。这意味着对任一 $f\in L_2$ 有
$$
\Bigl\|\sum_{n=-M}^M \mathtt{her}(f,e_n)e_n - f\Bigr\| \to 0,
$$
其中  
$$
\|g\|^2=\mathtt{her}(g,g).
$$

类似地，对于 $f\in L_2([0,T]\to\mathbb{R})$，正弦余弦展开也满足
$$
\Bigl\|\sum_{n=0}^N \mathtt{dot}(f,sc_n)\,sc_n - f\Bigr\| \to 0,
$$
其中  
$$
\|g\|^2=\mathtt{dot}(g,g).
$$

实际上，对于复值函数 $f$，只需在内积中加共轭即可使上述成立。

### 傅里叶变换与等距映射

我们引入新的记号：
- $\mathbb{F}_{sc}[f]_n = \mathtt{dot}(f,sc_n)$
- $\mathbb{F}_{e}[f]_n = \mathtt{her}(f,e_n)$

**性质：**
- 映射 $f\to \mathbb{F}_{sc}[f]$ 是从 $L_2([0,T]\to\mathbb{R})$ 到 $\ell_2(\mathbb{N}\to\mathbb{R})$ 的双射。
- 映射 $f\to \mathbb{F}_{e}[f]$ 是从 $L_2([0,T]\to\mathbb{C})$ 到 $\ell_2(\mathbb{Z}\to\mathbb{C})$ 的双射。
- 当 $f$ 为实信号时，$f\to \mathbb{F}_{e}[f]$ 的像是 $\ell_2(\mathbb{Z}\to\mathbb{C})$ 中具厄米对称性的元素。

并有 **Plancherel–Parseval 等式：**
$$
\mathtt{dot}(f, g) = \sum_{n\in\mathbb{N}} \mathbb{F}_{sc}[f]_n\,\mathbb{F}_{sc}[g]_n
$$
$$
\mathtt{her}(f, g) = \sum_{n\in\mathbb{Z}} \mathbb{F}_{e}[f]_n\,\mathbb{F}_{e}[g]_n
$$
  
**练习：**

1. **练习（$1\heartsuit$）：**  
   请更正上述公式中存在的小错误。（提示：检查常数因子是否正确。通常，正弦余弦版本的等式因子可能应为 2/T 或类似形式。）
2. **练习（$2\heartsuit$）：**  
   利用正交归一性证明以上等式。证明时需要说明标量积（或厄米内积）具有连续性。

3. **讨论投影：**  
   设 $f_{filtered}$ 为对信号 $f$ 的某种逼近：例如  
   - 仅保留余弦项；
   - 仅保留正弦项；
   - 只保留低频成分；
   - 只保留高频成分；
   - 去除常数项；
   等等。  
   **问题：** 如何用投影的角度定义 $f_{filtered}$？

   **答案思路：**  
   这实际上相当于将 $f$ 在希尔伯特空间中投影到某个子空间上。设 $P$ 为该子空间的正交投影，则  
   $$
   f_{filtered}= P(f)=\sum_{n\in I} \text{(对应的傅里叶系数)}\times \text{基底函数},
   $$
   其中 $I$ 是保留的频率指标集合。

### 信号能量

信号的平均能量定义为  
$$
E(f)=\frac{1}{T}\int_0^T |f'(t)|^2\, dt.
$$
  
**练习：**

1. 计算单个基函数 $e_n(t)$ 的能量 $E(e_n)$。  
2. 对于任意信号 $f$，试图用其指数傅里叶系数表达能量。  
3. 比较低音（低频）与高音（高频）的能量。问：低音是否比高音更“有能量”？  
4. 尝试直接给出用指数傅里叶系数表示能量的公式（类似地，正弦余弦系数也有对应表达）。

**提示：**

- 对于基函数，有  
  $$
  e_n(t)=\exp(2i\pi n t/T) \quad \Rightarrow \quad e_n'(t)=\frac{2i\pi n}{T}\exp(2i\pi n t/T),
  $$
  于是  
  $$
  |e_n'(t)|^2=\Bigl(\frac{2\pi n}{T}\Bigr)^2.
  $$
  因此，
  $$
  E(e_n)=\frac{1}{T}\int_0^T \Bigl|\frac{2i\pi n}{T}\Bigr|^2 dt = \Bigl(\frac{2\pi n}{T}\Bigr)^2.
  $$
- 对于任意信号，如果  
  $$
  f(t)=\sum_{n\in\mathbb{Z}} \alpha_n\, e_n(t),
  $$
  则利用傅里叶变换的等距性，有  
  $$
  \frac{1}{T}\int_0^T |f'(t)|^2\, dt=\sum_{n\in\mathbb{Z}} \Bigl|\frac{2i\pi n}{T}\alpha_n\Bigr|^2.
  $$

**讨论：**  
- 一个低频信号（低调）并不必然能量更高，因为能量与导数有关；  
- 高频成分的导数模长更大，因此若信号中高频成分占较大比例，则能量也会较高。  
- 但在真实的物理中，通常低音（低频）振动周期较长、变化较慢，其瞬时能量不一定高于高频成分。需要根据具体物理意义（例如压力变化）选择合适定义。

---

## 总结

本文讲义详细介绍了指数傅里叶展开的理论和数值实现，涵盖了以下主要内容：

1. **厄米内积的定义**以及与实值标量积的差别（共轭对称、半线性性质等）；
2. **指数基底的构造**及其正交归一性证明；
3. **傅里叶系数的计算**及正弦余弦展开与指数展开的互相转换；
4. **如何利用 Python 实现指数展开**、绘制基函数图形、计算傅里叶系数及逼近重构；
5. **频谱概念的阐述与绘制**（包括振幅谱、半频谱等）；
6. **信号滤波**与**完美采样**的讨论，强调离散化时不包含端点 $T$ 的必要性；
7. **正交多项式（Legendre 多项式）的使用**，以及如何利用 Gauss 节点与权重实现完美离散内积；
8. 简单讨论了**希尔伯特空间理论**，包括等距映射与 Parseval 等式；
9. 最后探讨了**信号能量**的定义及其与傅里叶系数之间的关系。

通过本讲义，希望能帮助你更好地掌握傅里叶展开及其在信号处理中的广泛应用。如有疑问，可以参考相关教材或文献进一步探讨每一部分的细节。