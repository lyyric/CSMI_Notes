# Python 常用语法整理：概率分布与 `scipy.stats` 使用

本文整理了在使用 `scipy.stats` 和相关库进行概率分布模拟与可视化过程中涉及的所有Python语法和函数。内容涵盖了库的导入、分布的定义与模拟、绘图方法及参数设置等。以下内容按功能模块分类，详细介绍每个函数的用法、参数及示例。

## 1. 导入必要的库

在进行概率分布模拟与可视化时，常用以下库：

```python
%reset -f
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
plt.style.use("default")

# 科学计算库
import scipy.stats as stats
from scipy.special import factorial as fac
from scipy.stats import kde
```

- **numpy (`np`)**：用于数值计算和数组操作。
- **matplotlib.pyplot (`plt`)**：用于绘制各种图表。
- **scipy.stats (`stats`)**：包含各种概率分布和统计函数。
- **scipy.special.factorial (`fac`)**：计算阶乘。
- **scipy.stats.kde**：用于核密度估计。

## 2. 关键术语

理解以下关键术语对于使用 `scipy.stats` 十分重要：

- **pdf (Probability Density Function)**：概率密度函数，适用于连续分布，以实数为自变量。
- **pmf (Probability Mass Function)**：概率质量函数，适用于离散分布，以整数为自变量。
- **cdf (Cumulative Density Function)**：累积分布函数，表示 $P(X \leq x)$。
- **ppf (Percent Point Function)**：分位函数，累积分布函数的逆函数，用于计算分位点。
- **rvs (Random Variables Simulation)**：随机变量模拟，用于生成具有指定分布的样本。

## 3. 连续分布

### 3.1 正态分布（高斯分布）

定义在实数轴上的分布，常用于描述自然现象。

#### 模拟与绘图

```python
# 模拟正态分布随机变量
simus = stats.norm.rvs(loc=0, scale=1, size=1000)

# 使用numpy模拟相同分布
simus_np = np.random.normal(loc=0, scale=1, size=1000)

# 计算密度、累积分布、分位数
x = np.linspace(-3, 3, 100)
pdf = stats.norm.pdf(x, loc=0, scale=1)
cdf = stats.norm.cdf(x, loc=0, scale=1)
ppf = stats.norm.ppf(x, loc=0, scale=1)

# 绘制直方图与密度曲线
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 左图：直方图与密度、累积分布
axs[0].hist(simus, bins=20, density=True, label="模拟数据", edgecolor="k")
axs[0].plot(x, pdf, label="PDF")
axs[0].plot(x, cdf, label="CDF")
axs[0].legend()

# 左图：累积分布与分位数
axs[1].plot(x, cdf, label="CDF")
axs[1].plot(x, ppf, label="PPF")
axs[1].plot(x, x, label="y=x")
axs[1].legend()

plt.show()
```

#### 使用现代Matplotlib语法

```python
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 左图
axs[0].hist(simus, bins=20, density=True, label="模拟数据", edgecolor="k")
axs[0].plot(x, pdf, label="PDF")
axs[0].plot(x, cdf, label="CDF")
axs[0].legend()
axs[0].set_title("正态分布：PDF与CDF")

# 右图
axs[1].plot(x, cdf, label="CDF")
axs[1].plot(x, ppf, label="PPF")
axs[1].plot(x, x, label="y=x")
axs[1].legend()
axs[1].set_title("正态分布：CDF与PPF")

plt.tight_layout()
plt.show()
```

#### 改进PPF曲线的绘制

```python
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 左图
axs[0].hist(simus, bins=20, density=True, label="模拟数据", edgecolor="k")
axs[0].plot(x, pdf, label="PDF")
axs[0].plot(x, cdf, label="CDF")
axs[0].legend()
axs[0].set_title("正态分布：PDF与CDF")

# 右图：仅绘制有效范围内的PPF
valid = (x >= stats.norm.ppf(0.001)) & (x <= stats.norm.ppf(0.999))
axs[1].plot(x[valid], ppf[valid], label="PPF")
axs[1].plot(x[valid], x[valid], label="y=x")
axs[1].legend()
axs[1].set_title("正态分布：PPF与y=x")

plt.tight_layout()
plt.show()
```

### 3.2 Gamma分布

定义在非负实数上的分布，形状参数为 `a`，无位置和尺度参数时，默认 `loc=0`, `scale=1`。

#### 密度函数

$$
\text{gamma.pdf}(x, a) = \frac{x^{a-1} e^{-x}}{\Gamma(a)} \quad \text{for } x > 0
$$

#### 模拟与绘图

```python
a = 2  # 形状参数
plt.hist(stats.gamma.rvs(a=a, size=400), bins=30, density=True, edgecolor="k", label="模拟数据")

x = np.linspace(0, 8, 100)
plt.plot(x, stats.gamma.pdf(x, a=a), label="Gamma PDF")
plt.legend()
plt.title("Gamma分布 (a=2)")
plt.show()
```

#### 调整位置和尺度参数

```python
a = 2
loc = 1
scale = 2
size = 400

# 模拟Gamma分布随机变量
simus = stats.gamma.rvs(a=a, loc=loc, scale=scale, size=size)

# 计算密度
x = np.linspace(stats.gamma.ppf(0.01, a, loc=loc, scale=scale),
                stats.gamma.ppf(0.99, a, loc=loc, scale=scale), 100)

# 绘制直方图与密度曲线
plt.hist(simus, bins=30, density=True, alpha=0.6, edgecolor="k", label=f'a={a}, loc={loc}, scale={scale}')
plt.plot(x, stats.gamma.pdf(x, a=a, loc=loc, scale=scale), 'r-', lw=2)
plt.legend()
plt.title("Gamma分布调整参数后的效果")
plt.show()

# 叠加不同参数的Gamma分布
params = [
    {'a': 1, 'loc': 0, 'scale': 1},
    {'a': 2, 'loc': 0, 'scale': 1},
    {'a': 2, 'loc': 2, 'scale': 1},
    {'a': 2, 'loc': 0, 'scale': 2},
]

for param in params:
    simus = stats.gamma.rvs(a=param['a'], loc=param['loc'], scale=param['scale'], size=size)
    x = np.linspace(stats.gamma.ppf(0.01, param['a'], loc=param['loc'], scale=param['scale']),
                    stats.gamma.ppf(0.99, param['a'], loc=param['loc'], scale=param['scale']), 100)
    plt.hist(simus, bins=30, density=True, alpha=0.3, edgecolor="k")
    plt.plot(x, stats.gamma.pdf(x, a=param['a'], loc=param['loc'], scale=param['scale']), label=f'a={param["a"]}, loc={param["loc"]}, scale={param["scale"]}')

plt.legend()
plt.title("不同参数下的Gamma分布")
plt.show()
```

#### Gamma分布的单调性

- **单调递减**：当形状参数 `a <= 1` 时，密度函数单调递减。
- **先增后减**：当形状参数 `a > 1` 时，密度函数先增加至峰值后单调递减。

### 3.3 Beta分布

定义在区间 $[0, 1]$ 上的分布，形状参数为 `a` 和 `b`。

#### 密度函数

$$
\text{beta.pdf}(x, a, b) = \frac{\Gamma(a + b)}{\Gamma(a) \Gamma(b)} x^{a-1} (1 - x)^{b-1} \quad \text{for } x \in [0, 1], a > 0, b > 0
$$

#### 模拟与绘图

```python
a = 2
b = 2
plt.hist(stats.beta.rvs(a=a, b=b, size=400), bins=30, density=True, edgecolor="k", label="模拟数据")

x = np.linspace(0, 1, 100)
plt.plot(x, stats.beta.pdf(x, a=a, b=b), label="Beta PDF")
plt.legend()
plt.title("Beta分布 (a=2, b=2)")
plt.show()
```

#### 调整形状参数

```python
a_values = [0.5, 2]
b_values = [0.5, 2]
x = np.linspace(0, 1, 100)

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        a = a_values[i]
        b = b_values[j]
        ax[i, j].plot(x, stats.beta.pdf(x, a=a, b=b), label=f'a={a}, b={b}')
        ax[i, j].legend()
        ax[i, j].set_title(f'Beta(a={a}, b={b})')

plt.tight_layout()
plt.show()
```

#### 分布形态

通过调整 `a` 和 `b` 参数，可以得到不同形态的Beta分布：

- **钟形**：`a > 1` 且 `b > 1`
- **递增**：`a < 1` 且 `b > 1`
- **递减**：`a > 1` 且 `b < 1`
- **U型**：`a < 1` 且 `b < 1`

## 4. 离散分布

### 4.1 二项分布

定义为在 `n` 次独立试验中成功的次数，成功概率为 `p`。

#### 密度函数

$$
\text{binom.pmf}(k, n, p) = \binom{n}{k} p^k (1 - p)^{n - k}
$$

#### 模拟与绘图

```python
n = 7
p = 0.2
simus = stats.binom.rvs(n, p, size=1000)

x = np.arange(0, n + 1)
bins = np.arange(0, n + 2) - 0.5

pdf = stats.binom.pmf(x, n, p)
cdf = stats.binom.cdf(x, n, p)

plt.hist(simus, bins=bins, density=True, edgecolor="k", label="模拟数据")
plt.plot(x, pdf, 'o', label="PMF")
plt.plot(x, cdf, '+', label="CDF")
plt.title("二项分布 (n=7, p=0.2)")
plt.legend()
plt.show()

# 阶梯式分布函数
fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(13, 5))

for xc in x:
    ax0.axvline(x=xc, color='0.9')
    ax1.axvline(x=xc, color='0.9')

ax0.plot(x, cdf, drawstyle='steps-post', label="CDF (steps-post)")
ax0.plot(x, cdf, "o")
ax0.legend()
ax0.set_title("阶梯式CDF (steps-post)")

ax1.plot(x, cdf, drawstyle='steps-pre', label="CDF (steps-pre)")
ax1.plot(x, cdf, "o")
ax1.legend()
ax1.set_title("阶梯式CDF (steps-pre)")

plt.tight_layout()
plt.show()
```

#### 分布函数的阶梯表示

- **steps-post**：对应累积分布函数 $F(x) = P(X \leq x)$。
- **steps-pre**：不对应累积分布函数的标准定义。

### 4.2 几何分布

定义为首次成功所需的试验次数，成功概率为 `p`。

#### 密度函数

$$
\text{geom.pmf}(k, p) = (1 - p)^{k-1} p \quad \text{for } k \geq 1
$$

#### 模拟与绘图

```python
p = 0.3
simus = stats.geom.rvs(p, size=1000)

x = np.arange(1, np.max(simus)+1)
bins = np.arange(0.5, np.max(simus)+1.5)  # 对齐整数

pmf = stats.geom.pmf(x, p)
cdf = stats.geom.cdf(x, p)

plt.hist(simus, bins=bins, density=True, edgecolor="k", label="模拟数据")
plt.plot(x, pmf, 'o', label="PMF")
plt.plot(x, cdf, '+', label="CDF")
plt.title("几何分布 (p=0.3)")
plt.legend()
plt.show()
```

### 4.3 泊松分布

定义为在固定时间间隔内事件发生的次数，事件平均发生率为 `mu`。

#### 密度函数

$$
\text{poisson.pmf}(k, \mu) = e^{-\mu} \frac{\mu^k}{k!} \quad \text{for } k \geq 0
$$

#### 模拟与绘图

```python
mu = 4
simus = stats.poisson.rvs(mu=mu, size=1000)

x = np.arange(0, np.max(simus)+1)
bins = np.arange(-0.5, np.max(simus)+1.5)

pmf = stats.poisson.pmf(x, mu)

plt.hist(simus, bins=bins, density=True, edgecolor="k", label="模拟数据")
plt.plot(x, pmf, 'o', label="PMF")
plt.title("泊松分布 (mu=4)")
plt.legend()
plt.show()
```

### 4.4 超几何分布

定义为从有限总体中不放回地抽取样本中成功次数的分布。

#### 密度函数

$$
\text{hypergeom.pmf}(k, M, n, N) = \frac{\binom{n}{k} \binom{M-n}{N-k}}{\binom{M}{N}} \quad \text{for } k = \max(0, N-M+n) \text{ to } \min(n, N)
$$

#### 模拟与绘图

```python
M = 20  # 总体大小
n = 7   # 成功个数
N = 5   # 抽样次数
simus = stats.hypergeom.rvs(M=M, n=n, N=N, size=1000)

x = np.arange(max(0, N - (M - n)), min(n, N)+1)
bins = np.arange(-0.5, min(n, N)+1.5)

pmf = stats.hypergeom.pmf(x, M, n, N)

plt.hist(simus, bins=bins, density=True, edgecolor="k", label="模拟数据")
plt.plot(x, pmf, 'o', label="PMF")
plt.title("超几何分布 (M=20, n=7, N=5)")
plt.legend()
plt.show()
```

### 4.5 整数均匀分布

定义为在区间 $[a, b)$ 上均匀分布的离散分布。

#### 密度函数

$$
\text{randint.pmf}(k, a, b) = \frac{1}{b - a} \cdot \mathbb{1}_{\{a \leq k < b\}}
$$

#### 模拟与绘图

```python
low = 1
high = 5
simus = stats.randint.rvs(low=low, high=high, size=1000)

x = np.arange(low, high)
bins = np.arange(low - 0.5, high + 0.5)

pmf = stats.randint.pmf(x, low, high)

plt.hist(simus, bins=bins, density=True, edgecolor="k", label="模拟数据")
plt.plot(x, pmf, 'o', label="PMF")
plt.title("整数均匀分布 (low=1, high=5)")
plt.legend()
plt.show()
```

#### 注意事项

- Python中的区间是左闭右开的，即包含 `a` 但不包含 `b`。

### 4.6 自定义概率选择

使用 `numpy.random.choice` 根据指定概率选择元素。

#### 示例

```python
elements = [2, 4, 6]
probabilities = [0.2, 0.2, 0.6]
simus = np.random.choice(a=elements, p=probabilities, size=1000)

plt.hist(simus, bins=np.arange(1.5, 7.5, 2), density=True, edgecolor="k", label="模拟数据")
plt.plot(elements, probabilities, 'o', label="指定概率")
plt.legend()
plt.title("自定义概率选择")
plt.show()
```

## 5. 多变量分布

### 5.1 多元正态分布

定义在多维空间上的正态分布，参数包括均值向量和协方差矩阵。

#### 密度函数

$$
\text{multivariate_normal.pdf}(x, \mu, \Sigma) = \frac{1}{\sqrt{\det(2\pi \Sigma)}} \exp\left(-\frac{1}{2} (\mathbf{x} - \mu)^\top \Sigma^{-1} (\mathbf{x} - \mu)\right)
$$

#### 模拟与绘图

```python
# 生成网格数据
n = 5
a = np.linspace(-2, 2, n)
XX, YY = np.meshgrid(a, a)
XY = np.stack([XX.flatten(), YY.flatten()], axis=1)

# 计算多元正态分布的密度
Z = stats.multivariate_normal.pdf(XY, mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
Z = Z.reshape([n, n])

# 绘制密度图
fig, ax = plt.subplots()
cax = ax.imshow(Z, origin="lower", extent=[-2, 2, -2, 2], cmap="viridis")
fig.colorbar(cax)
plt.title("多元正态分布密度图")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

### 5.2 多项式分布

定义为在固定试验次数下，每个类别的成功次数分布。

#### 密度函数

$$
\text{multinomial.pmf}(x_1, \dots, x_k; n, p_1, \dots, p_k) = \frac{n!}{x_1! \cdots x_k!} p_1^{x_1} \cdots p_k^{x_k} \quad \text{where } \sum x_i = n
$$

#### 模拟与绘图

```python
# 模拟多项式分布随机变量
n = 10
p = [0.2, 0.5, 0.3]
simus = stats.multinomial.rvs(n=n, p=p, size=1000)

# 统计每个类别的出现次数
counts = simus.sum(axis=0)

# 绘制每个类别的分布
labels = ['类别1', '类别2', '类别3']
for i in range(len(p)):
    plt.hist(simus[:, i], bins=n+1, density=True, alpha=0.5, label=labels[i], edgecolor="k")

plt.legend()
plt.title("多项式分布模拟")
plt.show()
```

## 6. 核密度估计（KDE）

用于估计数据的概率密度函数，提供平滑的密度曲线。

### 6.1 一维KDE示例

```python
X = np.random.beta(a=0.5, b=2, size=15)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(X, np.zeros_like(X), '.', label="数据点")

x = np.linspace(np.min(X), np.max(X), 200)

# 带宽为0.1的KDE
kernel = stats.gaussian_kde(X, bw_method=0.1)
y = kernel(x)
ax.plot(x, y, label="KDE 带宽=0.1")

# 带宽为0.2的KDE
kernel2 = stats.gaussian_kde(X, bw_method=0.2)
y2 = kernel2(x)
ax.plot(x, y2, label="KDE 带宽=0.2")

ax.legend()
plt.title("一维核密度估计")
plt.show()
```

### 6.2 二维KDE示例

```python
from scipy.stats import kde

# 模拟二维数据
def makeTrajectories(T, nbSimu):
    pre_pas = np.random.randint(0, 4, size=[nbSimu, T])
    pas = np.zeros(shape=[nbSimu, T, 2])
    pas[pre_pas == 0] = [2, 1]
    pas[pre_pas == 1] = [-1, 0]
    pas[pre_pas == 2] = [0, 1]
    pas[pre_pas == 3] = [0, -1]
    return np.cumsum(pas, axis=1)

trajectories = makeTrajectories(T=200, nbSimu=50000)
arrivals = trajectories[:, -1, :]

# 核密度估计
kernel = kde.gaussian_kde(arrivals.T)

x = np.linspace(-10, 100, 50)
XX, YY = np.meshgrid(x, x)
z = np.vstack([XX.flatten(), YY.flatten()])
ZZ = kernel(z).reshape(XX.shape)

# 绘制二维密度图
fig, ax = plt.subplots()
cax = ax.imshow(ZZ, cmap="jet", interpolation="bilinear", origin="lower",
                extent=[-10, 100, -10, 100])
fig.colorbar(cax)
plt.title("二维核密度估计")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

## 7. 分布参数概述

### 7.1 位置参数 (`loc`) 和尺度参数 (`scale`)

所有连续分布在 `scipy.stats` 中都有位置参数 `loc` 和尺度参数 `scale`：

- **位置参数 (`loc`)**：控制分布的平移。
- **尺度参数 (`scale`)**：控制分布的缩放。

**性质：**
如果 $X$ 的密度函数为 $f(x)$，则 $Y = \sigma X + \mu$ 的密度函数为：

$$
f_Y(y) = \frac{1}{\sigma} f\left(\frac{y - \mu}{\sigma}\right)
$$

### 7.2 形状参数

不同的分布有不同的形状参数，控制分布的具体形态。例如：

- **Gamma分布**：形状参数 `a`
- **Beta分布**：形状参数 `a` 和 `b`
- **t分布**：自由度参数 `df`

## 8. 常见概率分布总结

### 8.1 连续分布

| 分布名称         | 简化形式                  | 完整形式                                                   | Numpy 对应 |
|------------------|---------------------------|------------------------------------------------------------|------------|
| **Beta分布**     | $x^{a-1}(1-x)^{b-1}$      | $\frac{1}{B(a,b)} x^{a-1}(1-x)^{b-1} \quad x \in [0,1]$    | `np.random.beta(a, b)` |
| **柯西分布**     | $\frac{1}{\pi(1+x^2)}$    | $\frac{1}{\pi(1+x^2)}$                                     | `np.random.standard_cauchy()` |
| **卡方分布**     | $x^{k/2-1}e^{-x/2}$        | $\frac{1}{2^{k/2}\Gamma(k/2)} x^{k/2-1} e^{-x/2} \quad x > 0$ | -          |
| **Weibull分布**   | $k x^{k-1} e^{-x^k}$      | $\frac{k}{\sigma} \left(\frac{x - \mu}{\sigma}\right)^{k-1} e^{-\left(\frac{x - \mu}{\sigma}\right)^k} \quad x > 0$ | `np.random.weibull(a)` |
| **指数分布**     | $e^{-x} \cdot \mathbb{1}_{\{x>0\}}$ | $\frac{1}{\sigma} e^{-\frac{x - \mu}{\sigma}} \cdot \mathbb{1}_{\{x > \mu\}}$ | `np.random.exponential()` |
| **Gamma分布**    | $x^{a-1} e^{-x}$           | $\frac{x^{a-1} e^{-x}}{\Gamma(a)} \quad x > 0$             | `np.random.gamma(shape=a)` |
| **正态分布**     | $\frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ | $\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$ | `np.random.normal()` |
| **学生t分布**    | $\frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi} \Gamma(\nu/2)} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}$ | - | - |

### 8.2 离散分布

| 分布名称           | 密度函数                               | Numpy 对应                     |
|--------------------|----------------------------------------|--------------------------------|
| **二项分布**       | $\binom{n}{k} p^k (1-p)^{n-k}$          | `np.random.binomial(n, p)`     |
| **几何分布**       | $(1-p)^{k-1} p \quad k \geq 1$          | `np.random.geometric(p)`       |
| **超几何分布**     | $\frac{\binom{n}{k} \binom{M-n}{N-k}}{\binom{M}{N}}$ | `np.random.hypergeometric(ngood=n, nbad=M-n, nsample=N)` |
| **泊松分布**       | $e^{-\mu} \frac{\mu^k}{k!} \quad k \geq 0$ | `np.random.poisson(lam=mu)`     |
| **整数均匀分布**   | $\frac{1}{b - a} \cdot \mathbb{1}_{\{a \leq k < b\}}$ | `np.random.randint(a, b)`       |
| **自定义概率选择** | 根据指定概率选择元素                    | `np.random.choice(a=[2,4,6], p=[0.2,0.2,0.6])` |

### 8.3 多变量分布

| 分布名称                 | 密度函数                                                                        | Numpy 对应                 |
|--------------------------|---------------------------------------------------------------------------------|----------------------------|
| **多元正态分布**         | $\frac{1}{\sqrt{\det(2\pi \Sigma)}} \exp\left(-\frac{1}{2} (\mathbf{x} - \mu)^\top \Sigma^{-1} (\mathbf{x} - \mu)\right)$ | `np.random.multivariate_normal(mean, cov)` |
| **多项式分布**           | $\frac{n!}{x_1! \cdots x_k!} p_1^{x_1} \cdots p_k^{x_k} \quad \sum x_i = n$        | `np.random.multinomial(n, p)` |

## 9. 绘图相关语法

### 9.1 直方图 (`plt.hist`)

用于展示数值数据的分布情况，通过分箱将数据分组。

#### 基本用法

```python
plt.hist(data, bins=number_of_bins, density=True, edgecolor="k", label="标签")
plt.legend()
plt.show()
```

#### 参数解释

- `data`：数据数组。
- `bins`：分箱数量或分箱边界（列表或数组）。
- `density=True`：将直方图归一化，使其表示概率密度。
- `edgecolor`：条形边缘颜色，如 `"k"` 表示黑色。
- `label`：图例标签。

#### 示例：正态分布

```python
X = np.random.normal(size=100)
plt.hist(X, bins=10, color="red", density=True, rwidth=0.8, orientation="horizontal", edgecolor="k")
plt.show()
```

### 9.2 条形图 (`plt.bar`)

用于展示分类数据的分布情况。

#### 基本用法

```python
fig, ax = plt.subplots()
ax.bar(x, height, width=width, edgecolor="k", color="颜色", label="标签")
ax.legend()
plt.show()
```

#### 参数解释

- `x`：条形图的x轴位置。
- `height`：条形的高度。
- `width`：条形的宽度。
- `edgecolor`：条形边缘颜色。
- `color`：条形填充颜色。
- `label`：图例标签。

#### 示例：人口年龄金字塔

```python
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(ages, nb_hommes, width=1, label="男性")
ax.bar(ages, nb_femmes, width=1, label="女性")
xticks = np.arange(0, 101, 10)
ax.set_xticks(xticks)
for x in xticks:
    ax.axvline(x, color="0.9", linewidth=0.3)
ax.legend()
plt.show()
```

### 9.3 二维直方图 (`plt.hist2d`)

用于展示二维数据的分布情况。

#### 基本用法

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist2d(x, y, bins=[bins_x, bins_y], cmap="jet")
plt.colorbar()
plt.show()
```

#### 参数解释

- `x`, `y`：二维数据的x和y坐标。
- `bins`：分箱数量或分箱边界（列表或数组）。
- `cmap`：颜色映射，如 `"jet"`。

#### 示例

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist2d(arrivals[:, 0], arrivals[:, 1], bins=[bins, bins], cmap="jet")
plt.colorbar()
plt.title("二维直方图示例")
plt.show()
```

### 9.4 六边形分箱 (`plt.hexbin`)

用于展示二维数据的密度，替代矩形分箱。

#### 基本用法

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.hexbin(x, y, gridsize=40, cmap="jet")
plt.colorbar()
plt.show()
```

#### 参数解释

- `x`, `y`：二维数据的x和y坐标。
- `gridsize`：六边形网格的数量，控制分辨率。
- `cmap`：颜色映射，如 `"jet"`。

#### 示例

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.hexbin(arrivals[:, 0], arrivals[:, 1], gridsize=40, cmap="jet")
plt.colorbar()
plt.title("六边形分箱的二维直方图")
plt.show()
```

### 9.5 核密度估计（KDE）

用于估计数据的概率密度函数，生成平滑的密度曲线。

#### 一维KDE示例

```python
X = np.random.beta(a=0.5, b=2, size=15)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(X, np.zeros_like(X), '.', label="数据点")

x = np.linspace(np.min(X), np.max(X), 200)

# 带宽为0.1的KDE
kernel = stats.gaussian_kde(X, bw_method=0.1)
y = kernel(x)
ax.plot(x, y, label="KDE 带宽=0.1")

# 带宽为0.2的KDE
kernel2 = stats.gaussian_kde(X, bw_method=0.2)
y2 = kernel2(x)
ax.plot(x, y2, label="KDE 带宽=0.2")

ax.legend()
plt.title("一维核密度估计")
plt.show()
```

#### 二维KDE示例

```python
from scipy.stats import kde

# 模拟二维数据
def makeTrajectories(T, nbSimu):
    pre_pas = np.random.randint(0, 4, size=[nbSimu, T])
    pas = np.zeros(shape=[nbSimu, T, 2])
    pas[pre_pas == 0] = [2, 1]
    pas[pre_pas == 1] = [-1, 0]
    pas[pre_pas == 2] = [0, 1]
    pas[pre_pas == 3] = [0, -1]
    return np.cumsum(pas, axis=1)

trajectories = makeTrajectories(T=200, nbSimu=50000)
arrivals = trajectories[:, -1, :]

# 核密度估计
kernel = kde.gaussian_kde(arrivals.T)

x = np.linspace(-10, 100, 50)
XX, YY = np.meshgrid(x, x)
z = np.vstack([XX.flatten(), YY.flatten()])
ZZ = kernel(z).reshape(XX.shape)

# 绘制二维密度图
fig, ax = plt.subplots()
cax = ax.imshow(ZZ, cmap="jet", interpolation="bilinear", origin="lower",
                extent=[-10, 100, -10, 100])
fig.colorbar(cax)
plt.title("二维核密度估计")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

### 9.6 等高线绘制 (`plt.contour`)

用于在图中绘制等高线，常与二维直方图或KDE结合使用。

#### 示例

```python
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(ZZ, interpolation="bilinear",
          origin="lower", extent=[gauche, droite, gauche, droite],
          cmap="jet", vmin=0, vmax=0.002, alpha=0.8)

# 计算多元正态分布的密度
mean_theoretical = np.array([1/4 * T, 1/4 * T])
cov_theoretical = np.array([[19/16 * T, 1/2 * T], [1/2 * T, 11/16 * T]])
Z_density = stats.multivariate_normal.pdf(XY, mean=mean_theoretical, cov=cov_theoretical)
Z_density = Z_density.reshape(XX.shape)
ax.contour(XX, YY, Z_density, cmap="jet", vmin=0, vmax=0.002)
plt.show()
```

## 10. 常用函数与参数

### 10.1 `numpy` 函数

- **生成正态分布数据**

  ```python
  X = np.random.normal(loc=0, scale=1, size=1000)
  ```

- **生成二项分布数据**

  ```python
  X = np.random.binomial(n, p, size=1000)
  ```

- **生成几何分布数据**

  ```python
  X = np.random.geometric(p, size=1000)
  ```

- **生成对数正态分布数据**

  ```python
  Y = np.random.lognormal(mean=0, sigma=1, size=1000)
  ```

- **生成标准柯西分布数据**

  ```python
  X = np.random.standard_cauchy(size=200)
  ```

### 10.2 `scipy.stats` 分布方法

每个分布对象（如 `stats.norm`）通常包含以下方法：

- **模拟随机变量 (`rvs`)**

  ```python
  simus = stats.norm.rvs(loc=0, scale=1, size=1000)
  ```

  参数：
  - `loc`：位置参数，$\mu$
  - `scale`：尺度参数，$\sigma$
  - `size`：生成的随机变量数量，可以是整数或元组

- **计算概率密度函数 (`pdf`)**

  ```python
  pdf_values = stats.norm.pdf(x, loc=0, scale=1)
  ```

  参数：
  - `x`：计算密度的点
  - `loc`：位置参数，$\mu$
  - `scale`：尺度参数，$\sigma$

- **计算累积分布函数 (`cdf`)**

  ```python
  cdf_values = stats.norm.cdf(x, loc=0, scale=1)
  ```

  参数同 `pdf`

- **计算分位函数 (`ppf`)**

  ```python
  ppf_values = stats.norm.ppf(q, loc=0, scale=1)
  ```

  参数：
  - `q`：分位点（概率）

- **计算矩 (`moment`)**

  ```python
  moments = stats.norm.moment(n, loc=0, scale=1)
  ```

  参数：
  - `n`：矩的阶数

### 10.3 特殊函数

- **阶乘函数 (`scipy.special.factorial`)**

  ```python
  from scipy.special import factorial as fac
  k_factorial = fac(k)
  ```

- **多元正态分布的概率密度函数**

  ```python
  Z_density = stats.multivariate_normal.pdf(xy, mean=mean, cov=cov)
  ```

  参数：
  - `xy`：二维点坐标数组
  - `mean`：均值向量
  - `cov`：协方差矩阵

## 11. 示例代码汇总

### 11.1 绘制基本条形图

```python
import matplotlib.pyplot as plt

x = [1, 1.5, 2, 2.5, 3]
y = [1, 4, 3, 4, 1]

fig, ax = plt.subplots()
ax.bar(x, y, edgecolor="k", width=0.5)
plt.title("基本条形图示例")
plt.xlabel("类别")
plt.ylabel("值")
plt.show()
```

### 11.2 绘制直方图并获取数值信息

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.random.normal(0, 1, size=1000)
a = plt.hist(X, bins=5, edgecolor="k")
print("条形高度\n", a[0])
print("分箱边界\n", a[1])

b = np.histogram(X, bins=5)
print("条形高度\n", b[0])
print("分箱边界\n", b[1])
plt.show()
```

### 11.3 绘制离散分布的条形图

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial as fac

n = 12
p = 0.2
X = np.random.binomial(n, p, size=2000)

x = np.arange(0, n + 1)
bins = np.arange(0, n + 2) - 0.5
pdf = stats.binom.pmf(x, n, p)
cdf = stats.binom.cdf(x, n, p)

plt.hist(X, bins=bins, density=True, edgecolor="k", label="模拟数据")
plt.plot(x, pdf, 'o', label="理论 PMF")
plt.plot(x, cdf, '+', label="理论 CDF")
plt.title("二项分布 (n=12, p=0.2)")
plt.legend()
plt.show()
```

### 11.4 二维直方图与等高线叠加

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gaussian_kde

# 生成示例数据
def makeTrajectories(T, nbSimu):
    pre_pas = np.random.randint(0, 4, size=[nbSimu, T])
    pas = np.zeros(shape=[nbSimu, T, 2])
    pas[pre_pas == 0] = [2, 1]
    pas[pre_pas == 1] = [-1, 0]
    pas[pre_pas == 2] = [0, 1]
    pas[pre_pas == 3] = [0, -1]
    return np.cumsum(pas, axis=1)

trajectories = makeTrajectories(T=200, nbSimu=50000)
arrivals = trajectories[:, -1, :]

gauche = -10
droite = 100
bins = np.arange(gauche, droite + 1, 1, dtype=np.float32)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect("equal")
output = ax.hist2d(arrivals[:, 0], arrivals[:, 1], bins=[bins, bins], cmap="jet")
plt.colorbar(output[3], ax=ax)
plt.title("二维直方图示例")
plt.show()

# 核密度估计与等高线
kernel = gaussian_kde(arrivals.T)
x = np.linspace(gauche, droite, 50)
XX, YY = np.meshgrid(x, x)
z = np.vstack([XX.flatten(), YY.flatten()])
ZZ = kernel(z).reshape(XX.shape)

mean_theoretical = np.array([1/4 * 200, 1/4 * 200])
cov_theoretical = np.array([[19/16 * 200, 1/2 * 200], [1/2 * 200, 11/16 * 200]])

fig = plt.figure(figsize=(6, 6))
ax_xy = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax_xy.set_aspect("equal")
ax_xy.imshow(ZZ, interpolation="bilinear",
             origin="lower", extent=[gauche, droite, gauche, droite],
             cmap="jet", vmin=0, vmax=0.002, alpha=0.8)

# 计算多元正态分布的密度
xy = np.stack([XX, YY], axis=2)
Z_density = multivariate_normal.pdf(xy, mean=mean_theoretical, cov=cov_theoretical)
Z_density = Z_density.reshape(XX.shape)
ax_xy.contour(XX, YY, Z_density, cmap="jet", vmin=0, vmax=0.002)

# x轴边缘直方图
ax_x = plt.subplot2grid((3, 3), (0, 0), colspan=2)
ax_x.hist(arrivals[:, 0], bins=bins, density=True, edgecolor="k")
ax_x.plot(bins, stats.norm.pdf(bins, mean_theoretical[0], np.sqrt(cov_theoretical[0, 0])), 'r-', lw=2)

ax_x.set_xticks([])
ax_x.set_yticks([])
ax_x.set_title("X方向分布")

# y轴边缘直方图
ax_y = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax_y.hist(arrivals[:, 1], bins=bins, density=True,
          orientation='horizontal', edgecolor="k")
ax_y.plot(stats.norm.pdf(bins, mean_theoretical[1], np.sqrt(cov_theoretical[1, 1])), bins, 'r-', lw=2)

ax_y.set_xticks([])
ax_y.set_yticks([])
ax_y.set_title("Y方向分布")
plt.show()
```

### 11.5 绘制叠加直方图与密度曲线

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

nbSimu = 1000
Simu = np.random.normal(size=nbSimu)

def gaussian_density(x):
    return 1 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

plt.hist(Simu, bins=30, density=True, edgecolor="k", alpha=0.6, label="模拟数据")
x = np.linspace(-3, 3, 200)
plt.plot(x, gaussian_density(x), 'r-', lw=2, label="正态分布密度")
plt.legend()
plt.title("叠加直方图与密度曲线")
plt.show()
```

### 11.6 对数正态分布与正态分布叠加

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

size = 10000
Y = np.random.lognormal(size=size)
bins = np.linspace(0, 10, 80)
plt.hist(Y, bins=bins, edgecolor="k", label=["Y"], density=True)

# 计算并绘制对数正态分布的密度
y = np.linspace(0.001, 10, 1000)
plt.plot(y, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.log(y)**2) / y, 'r-', lw=2, label="理论密度")
plt.legend()
plt.title("对数正态分布与正态分布叠加")
plt.show()
```

## 12. 分布之间的关系

### 12.1 Gamma分布与Beta分布

**命题：**

设 $X_1 \sim \text{Gamma}(\alpha)$，$X_2 \sim \text{Gamma}(\beta)$，且 $X_1$ 与 $X_2$ 独立，则：

$$
\frac{X_1}{X_1 + X_2} \sim \text{Beta}(\alpha, \beta)
$$

#### 模拟验证

```python
alpha = 2
beta_param = 3
size = 10000

# 模拟Gamma分布随机变量
X1 = stats.gamma.rvs(a=alpha, size=size)
X2 = stats.gamma.rvs(a=beta_param, size=size)

# 计算比例
ratio = X1 / (X1 + X2)

# 绘制比例的直方图与Beta分布密度曲线
plt.hist(ratio, bins=50, density=True, alpha=0.6, edgecolor="k", label="模拟比例")

# 绘制Beta分布的密度函数
x = np.linspace(0, 1, 100)
plt.plot(x, stats.beta.pdf(x, a=alpha, b=beta_param), 'r-', lw=2, label="Beta 分布密度")

plt.legend()
plt.title("Gamma分布随机变量比例的分布")
plt.xlabel("比例")
plt.ylabel("密度")
plt.show()
```

**解释：**

通过模拟大量的 $X_1$ 和 $X_2$，计算它们的比例，并将其与理论上的Beta分布密度函数进行比较，可以验证比例确实服从Beta分布。

## 13. `plt.hist()` 常用参数总结

**主要参数：**

- **必需参数：**
  - `x`：数据数组。

- **数据处理相关参数：**
  - `bins`：分箱数目或具体的分箱边界（列表或数组）。
  - `density=True`：将直方图归一化，使其表示概率密度。
  - `range=[左, 右]`：限制数据的范围。
  - `weights`：为每个数据点分配权重，常用于自定义归一化。

- **显示相关参数：**
  - `orientation`：条形图方向，`"horizontal"` 或 `"vertical"`。
  - `rwidth`：条形宽度比例，如 `rwidth=0.5` 表示条形宽度占分箱宽度的一半。
  - `edgecolor`：条形边缘颜色，如 `"k"` 表示黑色。
  - `color`：条形填充颜色。

**示例：**

```python
X = np.random.normal(size=100)
plt.hist(X, bins=10, color="red", density=True, rwidth=0.8, orientation="horizontal", edgecolor="k")
plt.title("示例直方图")
plt.xlabel("密度")
plt.ylabel("值")
plt.show()
```

## 14. 总结

通过本整理，涵盖了使用 `scipy.stats` 和 `matplotlib` 进行概率分布模拟与可视化的常用语法和函数。掌握这些语法和函数能够帮助您高效地进行数据分析和统计建模。

### 关键要点：

- **分布参数：** 理解各分布的形状参数、位置参数和尺度参数，并掌握它们对分布形态的影响。
- **模拟与密度计算：** 使用 `rvs` 方法模拟随机变量，使用 `pdf` 和 `cdf` 方法计算概率密度函数和累积分布函数。
- **绘图技巧：** 熟悉直方图 (`hist`)、条形图 (`bar`)、二维直方图 (`hist2d`)、六边形分箱 (`hexbin`) 等方法，用于可视化数据分布。
- **核密度估计（KDE）：** 使用 `gaussian_kde` 进行平滑的密度估计，提供更直观的数据分布展示。
- **分布关系：** 理解不同分布之间的关系，如Gamma分布与Beta分布、学生t分布与正态分布等，有助于选择合适的模型进行分析。

建议结合实际数据进行练习，熟悉各个函数的参数设置和使用方法，以便在数据分析过程中灵活应用。