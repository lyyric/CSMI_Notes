# Box-Muller 方法与球面采样

## Box-Muller 方法

### 原理

接下来，我们将介绍 Box-Muller 技术，用于模拟高斯随机变量（为什么不使用之前介绍的其他方法呢？）。我们将通过“分析”和“合成”两个步骤来进行。

**符号说明：** 如果 $(x,y)$ 是 $\mathbb{R}^2$ 中的一个点，我们记 $r(x,y) = \sqrt{x^2 + y^2}$，$\theta(x,y)$ 为其极坐标。

**分析部分：** 假设 $(X,Y)$ 是一对独立的高斯随机变量。我们来计算 $R = r(X,Y)$ 和 $\Theta = \theta(X,Y)$ 的分布：设 $\phi$ 是一个测试函数，有：
$$
\mathbb{E}[\phi(R, \Theta)] = \text{常数} \int \int \phi\big(r(x,y), \theta(x,y)\big) e^{-\frac{1}{2}(x^2 + y^2)} \, dx \, dy
$$

我们将坐标转换为极坐标：
$(r(x,y), \theta(x,y)) \to (r, \theta)$，因此 $(dx\,dy) \to \color{red}{r \, dr \, d\theta}$

（需要牢记这一点，否则可能需要计算雅可比行列式）。积分区域也相应变化：
$\mathbb{R} \times \mathbb{R} \to \mathbb{R}_+ \times [0, 2\pi]$。
因此：
$$
\begin{align*}
\mathbb{E}[\phi(R, \Theta)]
&= \frac{1}{2\pi} \iint \phi(r(x,y), \theta(x,y)) e^{-\frac{1}{2}(x^2 + y^2)} \, dx \, dy \\
&= \frac{1}{2\pi} \int_{\theta=0}^{2\pi} \int_{r=0}^{+\infty} \phi(r, \theta) e^{-\frac{r^2}{2}} \, \color{red}{r \, dr \, d\theta}
\end{align*}
$$
$$
= \int \int \phi(r,\theta) \Big(\color{red}{\frac{1}{2\pi}}\Big) \Big(\color{red}{r e^{-\frac{r^2}{2}}}\Big)
$$

* 可以看出，$R$ 和 $\Theta$ 是独立的，因为 $\dfrac{1}{2\pi}$ 和 $r e^{-\frac{r^2}{2}}$ 是分离的。

* $\Theta$ 服从 $[0, 2\pi]$ 上的均匀分布。

* 通过一些经验可以发现，$R$ 服从参数为形状参数 $\alpha=2$ 和尺度参数 $\sqrt{2}$ 的韦布尔分布，这意味着 $R^2$ 服从参数为尺度 $2$ 的指数分布（参见关于反函数法的实验课）。

**合成部分：** 为了模拟一对独立的高斯随机变量，我们可以模拟 $R \sim \text{Weibull}(scale=\sqrt{2}, \alpha=2)$ 和 $\Theta \sim \text{Unif}[0, 2\pi]$，然后考虑极坐标点 $(R, \Theta)$。

实际上，我们取 $U$ 和 $V$ 两个服从 $\text{Unif}[0,1]$ 的随机变量，设 $R = \sqrt{2 |\ln(U)|}$，然后
$$
X = \sqrt{2 |\ln(U)|} \cos(2\pi V)
$$
$$
Y = \sqrt{2 |\ln(U)|} \sin(2\pi V)
$$

### 模拟 ▹

**任务：** 实现一个函数，一次性生成 $n$ 个高斯随机变量。尽量避免浪费随机数生成器的调用，利用正弦和余弦的对称性。通过将模拟结果的直方图与高斯密度函数叠加来验证。

#### ♡♡♡

```python
# 使用 Box-Muller 方法模拟高斯随机变量
import numpy as np
import matplotlib.pyplot as plt

def simul_normal(n):
    U = np.random.uniform(0, 1, n//2)
    V = np.random.uniform(0, 1, n//2)
    R = np.sqrt(-2 * np.log(U))
    Theta = 2 * np.pi * V
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    return np.concatenate([X, Y])

n = 100000
X = simul_normal(n)
plt.hist(X, bins=20, density=True, edgecolor='k')

x = np.linspace(X.min(), X.max(), 100)
plt.plot(x, 1/(np.sqrt(2*np.pi)) * np.exp(-x**2 / 2))
plt.show()
```

### 附加内容：计算机中的极坐标

在 `numpy` 中，一个点 `(x,y)` 的角度可以用 `np.arctan2(y, x)` 计算，而在其他编程语言中通常也有 `arctan2` 函数；但它与普通的反正切函数有什么联系呢？

#### ♡

点 $(x,y) \neq (0,0)$ 的有向角度在 $(-\pi, \pi]$ 内由以下情况给出：
$$
\left\{
\begin{array}{lcl}
\arctan\left(\frac{y}{x}\right) & & \text{如果 } x > 0, y \geq 0 \\
\frac{\pi}{2} & & \text{如果 } x = 0, y > 0 \\
\pi + \arctan\left(\frac{y}{x}\right) & & \text{如果 } x < 0, y \geq 0 \\
\arctan\left(\frac{y}{x}\right) - \pi & & \text{如果 } x < 0, y < 0 \\
-\frac{\pi}{2} & & \text{如果 } x = 0, y < 0 \\
\arctan\left(\frac{y}{x}\right) & & \text{如果 } x > 0, y \leq 0 \\
\end{array}
\right.
$$

## 直观理解如何在球面上模拟点

在继续阅读之前：

**问题：** 你会如何模拟均匀分布在球面上的点？

### 错误的“天真”方法

需要注意，使用“天真”方法，即将一个在 $[-1,1]^d$ 上均匀分布的随机变量进行归一化，并不能得到均匀分布的球面点。例如，在二维情况下，北、南、西、东四个区域的点比东北、东南、西南、西北四个区域的点要少。

```python
n = 10000

X = np.random.uniform(-1, +1, n)
Y = np.random.uniform(-1, +1, n)
N = np.sqrt(X**2 + Y**2)

plt.plot(X/N, Y/N, "o", alpha=0.01)
plt.show()
```

### 正确的“天真”方法

如果我们先在球内均匀采样（使用拒绝采样），然后进行归一化呢？

但是：

球体的体积
$$
\frac{\pi^{n/2} R^n}{\Gamma(n/2 + 1)}
$$

立方体的体积：$2^n$

```python
import scipy

def ratio_sphere_cube(n):
    return np.pi**(n/2) / scipy.special.gamma(n/2 + 1) / 2**n

n = np.arange(1, 16, 1)
ratio = ratio_sphere_cube(n)
fig, ax = plt.subplots()
ax.plot(ratio)
ax.set_yscale("log")
plt.show()
```

**结论：** 随着维度的增加，球体在立方体中的体积比急剧减小，因此拒绝采样在高维空间中效率极低。

## 正确的方法

### 旋转不变性

这一部分的目标是理解如何有效地在任意维度上模拟均匀分布在球面上的点。

**词汇：** 记 $\|x\|$ 为向量 $x$ 的范数。当一个函数 $f(x)$ 可以表示为 $f(x) = g(\|x\|)$ 时，我们称其具有旋转不变性。

考虑：

* 一个在 $\mathbb{R}^d$ 上旋转不变的密度函数 $f(x)$。
* 一个服从 $f(x)\,dx$ 的随机变量 $X$。
* 一个以原点为中心的旋转变换 $T$。

#### ♡♡♡

验证 $T(X)$ 也服从 $f(x)dx$。因为旋转是可逆的，且其行列式（决定因子）为 1，所以对测试函数 $\phi$ 有：
\begin{align*}
\mathbb{E}[\phi(T(X))] &= \int_{\mathbb{R}^n} \phi(T(x)) f(x) dx \\
&= \int_{\mathbb{R}^n} \phi(y) f(T^{-1}(y)) dy \\
&= \int_{\mathbb{R}^n} \phi(y) f(y) dy = \mathbb{E}[\phi(X)]
\end{align*}

#### ♡♡

现在寻找一个既旋转不变又是 $d$ 个独立变量的密度函数。

设 $f(x) = f(x_0, \dots, x_{d-1})$ 满足旋转不变且是 $d$ 个独立变量的密度函数。那么存在 $f_0, \dots, f_{d-1}$ 使得
$$
f(x_0, \dots, x_{d-1}) = \color{red}{f_0(x_0) \cdots f_{d-1}(x_{d-1})}
$$
又因为 $f$ 是旋转不变的，存在函数 $g$ 使得
$$
f(x_0, \dots, x_{d-1}) = g(\|x\|) = g(x_0^2 + \cdots + x_{d-1}^2)
$$
因此，
$$
f_0(x_0) \cdots f_{d-1}(x_{d-1}) = g(x_0^2 + \cdots + x_{d-1}^2)
$$
注意到对于任意 $i$，$f_i(0) \neq 0$。否则，$f$ 在超平面 $\{x_i = 0\}$ 上为零，由于 $f$ 旋转不变，这意味着 $f$ 在整个空间上都为零，这与密度函数的定义矛盾。通过将每个 $f_i$ 除以 $f_i(0)$，可以假设 $f_i(0) = 1$。将所有 $x_j$ 除 $x_i$ 以外设为零，可以得出：
$$
f_i(x_i) = g(x_i^2)
$$
设 $X_i = x_i^2$，则有：
$$
g(X_0) \cdots g(X_{d-1}) = g(X_0 + \dots + X_{d-1}) \quad \text{且} \quad g(0) = 1
$$
非平凡解为：
$$
g(x) = \color{red}{e^{c x}} \quad \text{其中} \quad c < 0
$$
因此，$f$ 可以表示为：
$$
f(x_0, \dots, x_{d-1}) = \text{常数} \cdot e^{-\lambda (x_0^2 + \dots + x_{d-1}^2)}
$$
这就是 $d$ 个独立标准正态分布的联合密度，可以用之前的方法进行模拟。

### 实现

通过归一化结果，我们就得到了均匀分布在 $S^{d-1}$ 上的模拟。下面实现一个函数：

```python
def simul_unif_sphere(n, d):
    X = np.random.normal(size=(n, d))
    norm = np.linalg.norm(X, axis=1)
    return X / norm[:, None]
```

我们可以在二维或三维中可视化模拟结果：

```python
# 二维可视化
n = 1000
d = 2

X = simul_unif_sphere(n, d)
plt.plot(X[:,0], X[:,1], 'o', alpha=0.1)
plt.axis('equal')
plt.show()

# 三维可视化
n = 1000
d = 3

X = simul_unif_sphere(n, d)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], alpha=0.1)
plt.show()
```

### 在球面上计算积分

这种在任意维度上均匀模拟球面点的方法，可以用来近似球面上的积分。具体来说，如果 $h$ 是一个从 $S^{d-1}$ 到 $\mathbb{R}$ 的函数，且 $(X_k)_k$ 是一组独立同分布的均匀分布在 $S^{d-1}$ 上的随机变量，根据大数定律：
$$
\lim_{n \to \infty} \frac{1}{n} \sum_{k=1}^n h(X_k) = \mathbb{E}[h(X_1)]
$$
而
$$
\mathbb{E}[h(X_1)] = \frac{1}{\text{surface}(S^{d-1})} \int_{S^{d-1}} h(x) \, dx
$$
因此，通过进行 $N$ 次独立模拟 $x_1, \dots, x_N$，我们有：
$$
\int_{S^{d-1}} h(x) \, dx \approx \frac{\text{surface}(S^{d-1})}{N} \sum_{k=1}^N h(x_k)
$$

我想，没有概率论基础的同学可能会在 $d=1$ 和 $d=2$ 时，利用欧拉角等方法估计这类积分，但高维情况下这会变得困难。蒙特卡洛方法正是为此设计的：用于计算非矩形、非平面（例如球面）、带孔的复杂区域或高维空间上的积分。

### 应用：涂漆问题

**任务：** 计算涂漆一个去除北极和南极的 $S^2$ 球面所需的油漆量，即去除满足 $x^2 + y^2 < 0.1$ 的点 $(x,y,z)$。

#### ♡♡

```python
n = 100000
X = simul_unif_sphere(n, 3)

# 创建掩码：保留 x^2 + y^2 >= 0.1 的点
mask = (X[:,0]**2 + X[:,1]**2) >= 0.1

X_in = X[mask]
X_out = X[~mask]

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(X_in[:,0], X_in[:,1], X_in[:,2], c='g', marker='.', alpha=0.1, label='涂漆区域')
ax.scatter(X_out[:,0], X_out[:,1], X_out[:,2], c='r', marker='.', alpha=0.1, label='未涂漆区域')
ax.legend()
plt.show()

# 估计涂漆的表面积
surface_area = 4 * np.pi  # S^2 的表面积
painted_area = surface_area * (np.sum(mask) / n)
print(f"估计需要的油漆量: {painted_area}")
```

**解释：** 通过模拟大量均匀分布在 $S^2$ 上的点，并筛选出满足 $x^2 + y^2 \geq 0.1$ 的点，我们可以估计需要涂漆的表面积占总表面积的比例，从而计算出所需的油漆量。

# 结语

通过 Box-Muller 方法，我们可以高效地模拟高斯随机变量；通过旋转不变性的方法，我们能够在任意维度上均匀地模拟球面上的点。这些技术在概率模拟和蒙特卡洛积分中有广泛的应用，帮助我们解决复杂的积分和概率分布问题。