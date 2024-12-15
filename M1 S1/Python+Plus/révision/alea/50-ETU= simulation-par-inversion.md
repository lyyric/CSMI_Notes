# Python 常用语法整理：模拟技术与蒙特卡洛方法

本文整理了在进行模拟技术与蒙特卡洛方法时常用的Python语法和函数。内容涵盖了库的导入、随机变量的生成、累积分布函数的逆变换、绘图方法及其参数设置等。以下内容按功能模块分类，详细介绍每个函数的用法、参数及示例。

## 1. 导入必要的库

在进行模拟和蒙特卡洛方法时，常用以下库：

```python
import numpy as np
np.set_printoptions(precision=2, suppress=True)  # 设置Numpy打印选项
import matplotlib.pyplot as plt
plt.style.use("default")  # 设置Matplotlib样式
import scipy.stats as stats
```

- **numpy (`np`)**：用于数值计算和数组操作。
- **matplotlib.pyplot (`plt`)**：用于绘制各种图表。
- **scipy.stats (`stats`)**：包含各种概率分布和统计函数。

## 2. 随机数生成

### 2.1 生成均匀分布随机变量

计算机通常使用伪随机数生成器生成 $[0,1)$ 上的均匀随机变量。

```python
# 使用 np.random.rand 生成 2x3 的均匀分布随机数
A = np.random.rand(2, 3)
print(A)

# 使用 np.random.uniform 生成相同的随机数
A = np.random.uniform(0, 1, size=[2, 3])
print(A)
```

- `np.random.rand(d0, d1, ...)`：生成 $[0,1)$ 上均匀分布的随机数，形状为 `(d0, d1, ...)` 的数组。
- `np.random.uniform(low, high, size)`：生成 $[low, high)$ 上均匀分布的随机数，形状由 `size` 指定。

### 2.2 生成离散分布随机变量

#### 2.2.1 模拟有限离散分布

例如，生成取值 `0, 1, 2, 3`，概率分别为 `0.2, 0.1, 0.5, 0.2` 的随机整数。

```python
# 定义概率
p = [0.2, 0.1, 0.5, 0.2]

# 定义模拟函数
def simul_disc(p):
    cdf = np.cumsum(p)  # 计算累积分布函数
    U = np.random.random()  # 生成一个均匀随机数
    i = 0
    while U > cdf[i]:
        i += 1
    return i

# 模拟样本
simus = []
for _ in range(500):
    simus.append(simul_disc(p))

# 绘制直方图
bins = [-0.5, 0.5, 1.5, 2.5, 3.5]
plt.hist(simus, bins=bins, edgecolor="k")
plt.title("离散分布模拟")
plt.xlabel("取值")
plt.ylabel("频率")
plt.show()

# 使用 np.random.choice 生成同样的分布
simus = np.random.choice(a=[0, 1, 2, 3], p=[0.2, 0.1, 0.5, 0.2], size=100)
plt.hist(simus, bins=bins, edgecolor="k")
plt.title("使用 np.random.choice 模拟离散分布")
plt.xlabel("取值")
plt.ylabel("频率")
plt.show()
```

- `np.cumsum(p)`：计算数组 `p` 的累加和，得到累积分布函数（CDF）。
- `np.random.random()`：生成 $[0,1)$ 上的均匀随机数。
- `np.random.choice(a, size, p)`：从数组 `a` 中按概率 `p` 选择元素，生成 `size` 个样本。

#### 2.2.2 模拟无限离散分布（几何分布）

几何分布的模拟较为复杂，因为其取值无限。以下是模拟几何分布的示例：

```python
# 定义几何分布的概率 p
p = 0.5

# 模拟几何分布函数
def simul_geom(p):
    i = 1
    cdfi = p
    U = np.random.random()
    while U > cdfi:
        i += 1
        cdfi += (1 - p) ** (i - 1) * p
    return i

# 模拟样本
simus = []
for _ in range(500):
    simus.append(simul_geom(p))

# 绘制直方图
x = np.arange(1, max(simus)+1)
pmf = stats.geom.pmf(x, p=0.5)
plt.hist(simus, bins=np.arange(0.5, max(simus)+1.5), density=True, edgecolor="k", alpha=0.6, label="模拟数据")
plt.plot(x, pmf, 'o', label="理论 PMF")
plt.title("几何分布模拟 (p=0.5)")
plt.xlabel("k")
plt.ylabel("概率")
plt.legend()
plt.show()
```

- `stats.geom.pmf(k, p)`：几何分布的概率质量函数（PMF）。
- `stats.geom.cdf(k, p)`：几何分布的累积分布函数（CDF）。
- `stats.geom.rvs(p, size)`：生成几何分布随机变量。

## 3. 逆变换法模拟连续分布

逆变换法基于累积分布函数（CDF）的逆函数来模拟随机变量。

### 3.1 理论基础

对于连续且严格单调递增的CDF $F$，其逆函数 $F^{-1}$ 满足：
$$
F(F^{-1}(u)) = u \quad \text{且} \quad F^{-1}(F(x)) = x
$$
其中 $u$ 服从 $[0,1]$ 上的均匀分布。

设 $X = F^{-1}(U)$，则 $X$ 服从分布 $\mu$。

### 3.2 指数分布模拟

指数分布的逆CDF方法示例如下：

```python
# 生成均匀分布随机数
U = np.random.random(size=1000)
mu = 0.5

# 使用逆变换法模拟指数分布随机变量
X = -mu * np.log(U)

# 绘制直方图与理论密度
plt.hist(X, bins=50, density=True, edgecolor="k", alpha=0.6, label="模拟数据")
x = np.linspace(0, 4, 200)
plt.plot(x, np.exp(-x/mu)/mu, 'r-', lw=2, label="理论密度")
plt.title("指数分布模拟")
plt.xlabel("X")
plt.ylabel("密度")
plt.legend()
plt.show()
```

- `np.log(U)`：计算自然对数。
- 指数分布的CDF为 $F(x) = 1 - e^{-x/\mu}$，其逆函数为 $F^{-1}(u) = -\mu \ln(1 - u)$。由于 $1 - u$ 仍服从均匀分布，故可简化为 $F^{-1}(u) = -\mu \ln(u)$。

### 3.3 Weibull分布模拟

Weibull分布的累积分布函数为：
$$
F(x) = 1 - e^{-x^\alpha}
$$
其逆函数为：
$$
F^{-1}(u) = (-\ln(1 - u))^{1/\alpha}
$$
同样可简化为：
$$
F^{-1}(u) = (-\ln(u))^{1/\alpha}
$$

#### 模拟示例

```python
# 定义逆CDF函数
def simul_weibull(size, alpha):
    U = np.random.random(size=size)
    return (-np.log(1 - U))**(1/alpha)

# 参数设置
alphas = [0.7, 2]
m = 1000
n = len(alphas)
simul = [simul_weibull(m, a) for a in alphas]

# 绘制直方图与理论密度
for i, a in enumerate(alphas):
    plt.subplot(n, 1, i+1)
    plt.hist(simul[i], bins=20, density=True, edgecolor="k", alpha=0.6, label=f"模拟数据 α={a}")
    x = np.linspace(0.01, simul[i].max(), 200)
    plt.plot(x, a * x**(a-1) * np.exp(-x**a), 'r-', lw=2, label="理论密度")
    plt.legend()
    plt.title(f"Weibull分布模拟 α={a}")

plt.tight_layout()
plt.show()

# 与 numpy 的 Weibull 分布进行比较
X1 = np.random.weibull(0.7, size=2000)
X2 = np.random.weibull(2, size=2000)
bins = np.linspace(0, 10, 50)
plt.hist(X1, bins=50, density=True, alpha=0.6, edgecolor="k", label="np.random.weibull(0.7)")
plt.hist(X2, bins=50, density=True, alpha=0.5, edgecolor="k", label="np.random.weibull(2)")
plt.legend()
plt.title("Weibull分布与 numpy 比较")
plt.xlabel("X")
plt.ylabel("密度")
plt.show()
```

- `np.random.weibull(a, size)`：生成Weibull分布随机变量，参数为形状参数 `a`。

#### 模拟逆变换函数的完善

```python
# 定义广义逆函数
def f_inv(u):
    if 0 <= u < 1/6:
        return 6 * u
    elif 1/6 <= u < 3/6:
        return 1
    elif 3/6 <= u < 5/6:
        return (6 * u) / 2
    else:
        return 3

# 向量化函数
f_inv_vectorized = np.vectorize(f_inv)

# 绘制 F 和 F^{-1} 的对称性
x = np.linspace(0, 3, 200)
y = f_np(x)  # 定义在[0,3]上的累积分布函数
plt.plot(x, y, label=r"$F(x)$")

u = np.linspace(0, 1, 200)
x_inv = f_inv_vectorized(u)
plt.plot(u, x_inv, label=r"$F^{-1}(u)$")

plt.plot(np.linspace(0,3,20), np.linspace(0,3,20), ".", label="y = x")
plt.axis("equal")
plt.legend()
plt.title("累积分布函数 F(x) 与广义逆 F⁻¹(u) 的对称性")
plt.xlabel("x 或 u")
plt.ylabel("F(x) 或 F⁻¹(u)")
plt.show()

# 使用逆变换方法模拟随机变量
def simul_custom_distribution(size):
    U = np.random.random(size=size)
    return f_inv_vectorized(U)

X = simul_custom_distribution(1000)

bins = np.arange(0, 3.2, 0.2)
plt.hist(X, bins=bins, edgecolor="k", density=True, alpha=0.6, label="模拟数据")

# 绘制理论密度
x = np.linspace(0, 4, 200)
plt.plot(x, [f(x) for x in x], 'r-', lw=2, label="理论密度 F(x)")
plt.title("自定义分布模拟")
plt.xlabel("X")
plt.ylabel("密度")
plt.legend()
plt.show()
```

- `np.vectorize(function)`：将标量函数向量化以处理数组输入。
- 自定义逆函数 `f_inv` 用于处理不连续的累积分布函数。

## 4. 生成和绘制核密度估计（KDE）

核密度估计用于估计数据的概率密度函数，提供平滑的密度曲线。

### 4.1 一维KDE示例

```python
# 生成数据
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
plt.xlabel("X")
plt.ylabel("密度")
plt.show()
```

- `stats.gaussian_kde(dataset, bw_method)`：进行高斯核密度估计。
  - `dataset`：数据数组，形状为 `(n_samples, )` 或 `(n_dimensions, n_samples)`。
  - `bw_method`：带宽方法，控制平滑程度，可以是浮点数或字符串。

### 4.2 二维KDE示例

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

- `kde.gaussian_kde(arrivals.T)`：对二维数据进行高斯核密度估计。
- `np.meshgrid(x, y)`：生成网格数据用于绘图。
- `ax.imshow()`：显示二维密度图。

## 5. 绘图相关语法

### 5.1 直方图 (`plt.hist`)

用于展示数值数据的分布情况，通过分箱将数据分组。

```python
plt.hist(data, bins=number_of_bins, density=True, edgecolor="k", alpha=0.6, label="标签")
plt.legend()
plt.title("直方图示例")
plt.xlabel("值")
plt.ylabel("密度")
plt.show()
```

- `data`：数据数组。
- `bins`：分箱数量或分箱边界（列表或数组）。
- `density=True`：将直方图归一化，使其表示概率密度。
- `edgecolor`：条形边缘颜色，如 `"k"` 表示黑色。
- `alpha`：透明度。
- `label`：图例标签。

#### 示例：正态分布

```python
X = np.random.normal(size=1000)
plt.hist(X, bins=50, density=True, edgecolor="k", alpha=0.6, label="模拟数据")
plt.plot(x, stats.norm.pdf(x, loc=0, scale=1), 'r-', lw=2, label="正态分布密度")
plt.legend()
plt.title("正态分布直方图与密度曲线")
plt.xlabel("X")
plt.ylabel("密度")
plt.show()
```

### 5.2 条形图 (`plt.bar`)

用于展示分类数据的分布情况。

```python
fig, ax = plt.subplots()
ax.bar(x, height, width=width, edgecolor="k", color="颜色", label="标签")
ax.legend()
plt.title("条形图示例")
plt.xlabel("类别")
plt.ylabel("值")
plt.show()
```

- `x`：条形图的x轴位置。
- `height`：条形的高度。
- `width`：条形的宽度。
- `edgecolor`：条形边缘颜色。
- `color`：条形填充颜色。
- `label`：图例标签。

#### 示例：基本条形图

```python
x = [1, 1.5, 2, 2.5, 3]
y = [1, 4, 3, 4, 1]

fig, ax = plt.subplots()
ax.bar(x, y, edgecolor="k", width=0.5)
plt.title("基本条形图示例")
plt.xlabel("类别")
plt.ylabel("值")
plt.show()
```

### 5.3 二维直方图 (`plt.hist2d`)

用于展示二维数据的分布情况。

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist2d(x, y, bins=[bins_x, bins_y], cmap="jet")
plt.colorbar()
plt.title("二维直方图示例")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

- `x`, `y`：二维数据的x和y坐标。
- `bins`：分箱数量或分箱边界（列表或数组）。
- `cmap`：颜色映射，如 `"jet"`。

#### 示例

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist2d(arrivals[:, 0], arrivals[:, 1], bins=[bins, bins], cmap="jet")
plt.colorbar()
plt.title("二维直方图示例")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

### 5.4 六边形分箱 (`plt.hexbin`)

用于展示二维数据的密度，替代矩形分箱。

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.hexbin(x, y, gridsize=40, cmap="jet")
plt.colorbar()
plt.title("六边形分箱的二维直方图")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

- `x`, `y`：二维数据的x和y坐标。
- `gridsize`：六边形网格的数量，控制分辨率。
- `cmap`：颜色映射，如 `"jet"`。

#### 示例

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.hexbin(arrivals[:, 0], arrivals[:, 1], gridsize=40, cmap="jet")
plt.colorbar()
plt.title("六边形分箱的二维直方图")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

### 5.5 等高线绘制 (`plt.contour`)

用于在图中绘制等高线，常与二维直方图或KDE结合使用。

```python
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(ZZ, cmap="jet", interpolation="bilinear", origin="lower",
          extent=[gauche, droite, gauche, droite], vmin=0, vmax=0.002, alpha=0.8)
ax.contour(XX, YY, Z_density, cmap="jet", vmin=0, vmax=0.002)
plt.colorbar()
plt.title("二维核密度估计与等高线")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

- `ax.contour(X, Y, Z, cmap)`：绘制等高线。

#### 示例

```python
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(ZZ, cmap="jet", interpolation="bilinear", origin="lower",
          extent=[gauche, droite, gauche, droite], vmin=0, vmax=0.002, alpha=0.8)
ax.contour(XX, YY, Z_density, cmap="jet", vmin=0, vmax=0.002)
plt.colorbar()
plt.title("二维核密度估计与等高线")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

## 6. 核密度估计（KDE）

核密度估计用于估计数据的概率密度函数，提供平滑的密度曲线。

### 6.1 一维KDE示例

```python
# 生成数据
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
plt.xlabel("X")
plt.ylabel("密度")
plt.show()
```

- `stats.gaussian_kde(dataset, bw_method)`：进行高斯核密度估计。
  - `dataset`：数据数组，形状为 `(n_samples, )` 或 `(n_dimensions, n_samples)`。
  - `bw_method`：带宽方法，控制平滑程度，可以是浮点数或字符串。

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

- `kde.gaussian_kde(arrivals.T)`：对二维数据进行高斯核密度估计。
- `np.meshgrid(x, y)`：生成网格数据用于绘图。
- `ax.imshow(Z, ...)`：显示二维密度图。

## 7. 模拟分布函数的广义逆

当累积分布函数（CDF）不是严格单调时，可以使用广义逆函数进行模拟。

### 7.1 定义累积分布函数及其逆

```python
# 定义分段连续的累积分布函数 F(x)
def f(x):
    if 0 <= x < 1:
        return x / 6
    elif 1 <= x < 2:
        return 1 / 6
    elif 2 <= x < 3:
        return 2 * x / 6
    else:
        return 1

# 使用Numpy向量化函数
def f_np(x):
    y = np.concatenate([
        x[(0 <= x) & (x < 1)],
        np.ones_like(x[(1 <= x) & (x < 2)]),
        2 * x[(2 <= x) & (x < 3)],
        6 * np.ones_like(x[x >= 3])
    ])
    return y / 6

# 定义广义逆函数
def f_inv(u):
    if 0 <= u < 1/6:
        return 6 * u
    elif 1/6 <= u < 3/6:
        return 1
    elif 3/6 <= u < 5/6:
        return (6 * u) / 2
    else:
        return 3

# 向量化逆函数
f_inv_vectorized = np.vectorize(f_inv)
```

- `np.concatenate([...])`：连接多个数组。
- `np.ones_like(array)`：生成与给定数组形状相同的全1数组。
- `np.vectorize(function)`：将标量函数向量化以处理数组输入。

### 7.2 绘制累积分布函数及其广义逆

```python
# 绘制 F(x) 和 F^{-1}(u) 的对称性
x = np.linspace(0, 3, 200)
y = f_np(x)
plt.plot(x, y, label=r"$F(x)$")

u = np.linspace(0, 1, 200)
x_inv = f_inv_vectorized(u)
plt.plot(u, x_inv, label=r"$F^{-1}(u)$")

plt.plot(np.linspace(0, 3, 20), np.linspace(0, 3, 20), ".", label="y = x")
plt.axis("equal")
plt.legend()
plt.title("累积分布函数 F(x) 与广义逆 F⁻¹(u) 的对称性")
plt.xlabel("x 或 u")
plt.ylabel("F(x) 或 F⁻¹(u)")
plt.show()
```

- `plt.plot()`：绘制折线图或散点图。
- `plt.axis("equal")`：设置坐标轴比例相同。

### 7.3 使用逆变换法模拟自定义分布

```python
# 使用逆变换方法模拟随机变量
def simul_custom_distribution(size):
    U = np.random.random(size=size)
    return f_inv_vectorized(U)

X = simul_custom_distribution(1000)

# 绘制直方图与理论密度
bins = np.arange(0, 3.2, 0.2)
plt.hist(X, bins=bins, edgecolor="k", density=True, alpha=0.6, label="模拟数据")

# 绘制理论密度
x = np.linspace(0, 4, 200)
plt.plot(x, [f(xi) for xi in x], 'r-', lw=2, label="理论密度 F(x)")
plt.title("自定义分布模拟")
plt.xlabel("X")
plt.ylabel("密度")
plt.legend()
plt.show()
```

- `np.arange(start, stop, step)`：生成从 `start` 到 `stop`（不包括 `stop`）的数组，步长为 `step`。
- `plt.hist()`：绘制直方图。
- `plt.plot()`：绘制折线图。

## 8. 生成和绘制学生t分布

### 8.1 学生t分布模拟与绘图

```python
df = 3  # 自由度
X = stats.t.rvs(df=df, size=1000)

plt.hist(X, bins=50, density=True, alpha=0.6, edgecolor="k", label="模拟数据")
x = np.linspace(-10, 10, 200)
plt.plot(x, stats.t.pdf(x, df=df), 'r-', lw=2, label="学生t分布密度")
plt.title("学生t分布模拟 (df=3)")
plt.xlabel("X")
plt.ylabel("密度")
plt.legend()
plt.show()
```

- `stats.t.rvs(df, size)`：生成学生t分布随机变量。
- `stats.t.pdf(x, df)`：学生t分布的概率密度函数。

### 8.2 不同自由度下的学生t分布

```python
# 自由度较小，重尾
df_small = 2
X_small = stats.t.rvs(df=df_small, size=1000)

# 自由度较大，接近正态
df_large = 30
X_large = stats.t.rvs(df=df_large, size=1000)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(X_small, bins=50, density=True, alpha=0.6, edgecolor="k", label=f'df={df_small}')
plt.plot(x, stats.t.pdf(x, df=df_small), 'r-', lw=2)
plt.title(f'Student t Distribution (df={df_small})')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(X_large, bins=50, density=True, alpha=0.6, edgecolor="k", label=f'df={df_large}')
plt.plot(x, stats.t.pdf(x, df=df_large), 'r-', lw=2)
plt.title(f'Student t Distribution (df={df_large})')
plt.legend()

plt.show()
```

- `stats.t.pdf(x, df)`：学生t分布的概率密度函数。
- `plt.subplot(nrows, ncols, index)`：在一个图中创建子图。

## 9. 常用概率分布总结

### 9.1 连续分布

| 分布名称         | 定义域         | 累积分布函数 (CDF) | 概率密度函数 (PDF)                                     | Numpy 对应函数                    |
|------------------|----------------|---------------------|--------------------------------------------------------|-----------------------------------|
| **正态分布**     | $(-\infty, \infty)$ | $F(x) = \frac{1}{2} [1 + \text{erf}(\frac{x-\mu}{\sigma\sqrt{2}})]$ | $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | `np.random.normal(loc, scale, size)` |
| **指数分布**     | $[0, \infty)$   | $F(x) = 1 - e^{-\lambda x}$ | $f(x) = \lambda e^{-\lambda x}$                            | `np.random.exponential(scale=1/lambda, size)` |
| **Gamma分布**    | $[0, \infty)$   | $F(x) = \int_0^x \frac{t^{k-1} e^{-t/\theta}}{\Gamma(k)\theta^k} dt$ | $f(x) = \frac{x^{k-1} e^{-x/\theta}}{\Gamma(k)\theta^k}$      | `np.random.gamma(shape=k, scale=theta, size)` |
| **Beta分布**     | $[0, 1]$        | $F(x) = \int_0^x \frac{t^{\alpha-1} (1-t)^{\beta-1}}{B(\alpha, \beta)} dt$ | $f(x) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha, \beta)}$ | `np.random.beta(a, b, size)` |
| **Weibull分布**   | $[0, \infty)$   | $F(x) = 1 - e^{-x^\alpha}$  | $f(x) = \alpha x^{\alpha-1} e^{-x^\alpha}$                   | `np.random.weibull(a, size)`     |
| **多元正态分布** | $\mathbb{R}^n$   | -                   | $f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n \det(\Sigma)}} e^{-\frac{1}{2} (\mathbf{x}-\mu)^\top \Sigma^{-1} (\mathbf{x}-\mu)}$ | `np.random.multivariate_normal(mean, cov, size)` |

### 9.2 离散分布

| 分布名称           | 定义域       | 概率质量函数 (PMF)                          | 累积分布函数 (CDF)                          | Numpy 对应函数                                |
|--------------------|--------------|---------------------------------------------|---------------------------------------------|-----------------------------------------------|
| **二项分布**       | $\{0, 1, \dots, n\}$ | $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$       | $F(k) = \sum_{i=0}^k \binom{n}{i} p^i (1-p)^{n-i}$ | `np.random.binomial(n, p, size)`             |
| **几何分布**       | $\{1, 2, \dots\}$ | $P(X=k) = (1-p)^{k-1} p$                      | $F(k) = 1 - (1-p)^k$                         | `np.random.geometric(p, size)`               |
| **泊松分布**       | $\{0, 1, 2, \dots\}$ | $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$   | $F(k) = \sum_{i=0}^k \frac{\lambda^i e^{-\lambda}}{i!}$ | `np.random.poisson(lam, size)`                |
| **超几何分布**     | $\{0, 1, \dots\}$ | $P(X=k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}$ | $F(k) = \sum_{i=0}^k \frac{\binom{K}{i} \binom{N-K}{n-i}}{\binom{N}{n}}$ | `np.random.hypergeometric(ngood, nbad, nsample)` |
| **整数均匀分布**   | $\{a, a+1, \dots, b-1\}$ | $P(X=k) = \frac{1}{b - a}$                       | $F(k) = \frac{k - a + 1}{b - a}$             | `np.random.randint(low=a, high=b, size)`      |
| **多项式分布**     | $\{0, 1, \dots, n\}^k$ | $P(X_1=k_1, \dots, X_k=k_k) = \frac{n!}{k_1! \dots k_k!} p_1^{k_1} \dots p_k^{k_k}$ | -                                             | `np.random.multinomial(n, p, size)`          |

### 9.3 多变量分布

| 分布名称             | 定义域           | 概率密度函数 (PDF)                                                                                                     | Numpy 对应函数                                 |
|----------------------|------------------|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| **多元正态分布**     | $\mathbb{R}^n$     | $f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n \det(\Sigma)}} e^{-\frac{1}{2} (\mathbf{x}-\mu)^\top \Sigma^{-1} (\mathbf{x}-\mu)}$ | `np.random.multivariate_normal(mean, cov, size)` |
| **多项式分布**       | $\mathbb{N}^k$     | $P(X_1=k_1, \dots, X_k=k_k) = \frac{n!}{k_1! \dots k_k!} p_1^{k_1} \dots p_k^{k_k}$                                       | `np.random.multinomial(n, p, size)`            |

## 10. 模拟与绘图示例

### 10.1 模拟离散分布

#### 10.1.1 二项分布

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

n = 12
p = 0.2
X = np.random.binomial(n, p, size=2000)

x = np.arange(0, n + 1)
bins = np.arange(-0.5, n + 1.5)
pmf = binom.pmf(x, n, p)

plt.hist(X, bins=bins, density=True, edgecolor="k", label="模拟数据")
plt.plot(x, pmf, 'o', label="理论 PMF")
plt.title("二项分布模拟 (n=12, p=0.2)")
plt.xlabel("k")
plt.ylabel("概率")
plt.legend()
plt.show()
```

- `binom.pmf(k, n, p)`：计算二项分布的概率质量函数。
- `binom.rvs(n, p, size)`：生成二项分布随机变量。

### 10.2 模拟连续分布

#### 10.2.1 指数分布

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# 使用逆变换法模拟指数分布
U = np.random.random(size=1000)
mu = 0.5
X = -mu * np.log(U)

# 绘制直方图与理论密度
plt.hist(X, bins=50, density=True, edgecolor="k", alpha=0.6, label="模拟数据")
x = np.linspace(0, 4, 200)
plt.plot(x, np.exp(-x/mu)/mu, 'r-', lw=2, label="理论密度")
plt.title("指数分布模拟")
plt.xlabel("X")
plt.ylabel("密度")
plt.legend()
plt.show()
```

- `expon.pdf(x, scale=mu)`：指数分布的概率密度函数。
- `expon.cdf(x, scale=mu)`：指数分布的累积分布函数。
- `expon.rvs(scale=mu, size)`：生成指数分布随机变量。

#### 10.2.2 Weibull分布

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# 定义逆CDF函数
def simul_weibull(size, alpha):
    U = np.random.random(size=size)
    return (-np.log(1 - U))**(1/alpha)

# 参数设置
alphas = [0.7, 2]
m = 1000
simul = [simul_weibull(m, a) for a in alphas]

# 绘制直方图与理论密度
for i, a in enumerate(alphas):
    plt.subplot(len(alphas), 1, i+1)
    plt.hist(simul[i], bins=20, density=True, edgecolor="k", alpha=0.6, label=f"模拟数据 α={a}")
    x = np.linspace(0.01, simul[i].max(), 200)
    plt.plot(x, a * x**(a-1) * np.exp(-x**a), 'r-', lw=2, label="理论密度")
    plt.legend()
    plt.title(f"Weibull分布模拟 α={a}")

plt.tight_layout()
plt.show()

# 与 numpy 的 Weibull 分布进行比较
X1 = np.random.weibull(0.7, size=2000)
X2 = np.random.weibull(2, size=2000)
bins = np.linspace(0, 10, 50)
plt.hist(X1, bins=50, density=True, alpha=0.6, edgecolor="k", label="np.random.weibull(0.7)")
plt.hist(X2, bins=50, density=True, alpha=0.5, edgecolor="k", label="np.random.weibull(2)")
plt.legend()
plt.title("Weibull分布与 numpy 比较")
plt.xlabel("X")
plt.ylabel("密度")
plt.show()
```

- `weibull_min.pdf(x, c)`：Weibull分布的概率密度函数。
- `weibull_min.cdf(x, c)`：Weibull分布的累积分布函数。
- `weibull_min.rvs(c, size)`：生成Weibull分布随机变量。

### 10.3 模拟多元分布

#### 10.3.1 多元正态分布

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 生成网格数据
n = 5
a = np.linspace(-2, 2, n)
XX, YY = np.meshgrid(a, a)
XY = np.stack([XX.flatten(), YY.flatten()], axis=1)

# 计算多元正态分布的密度
Z = multivariate_normal.pdf(XY, mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
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

- `multivariate_normal.pdf(x, mean, cov)`：多元正态分布的概率密度函数。
- `np.meshgrid(x, y)`：生成网格数据用于绘图。
- `ax.imshow(Z, ...)`：显示二维密度图。

#### 10.3.2 多项式分布

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial

# 模拟多项式分布随机变量
n = 10
p = [0.2, 0.5, 0.3]
simus = multinomial.rvs(n=n, p=p, size=1000)

# 统计每个类别的出现次数
counts = simus.sum(axis=0)

# 绘制每个类别的分布
labels = ['类别1', '类别2', '类别3']
for i in range(len(p)):
    plt.hist(simus[:, i], bins=n+1, density=True, alpha=0.5, label=labels[i], edgecolor="k")

plt.legend()
plt.title("多项式分布模拟")
plt.xlabel("类别")
plt.ylabel("频率")
plt.show()
```

- `multinomial.pmf(x, n, p)`：多项式分布的概率质量函数。
- `multinomial.rvs(n, p, size)`：生成多项式分布随机变量。

## 11. 常用函数与参数

### 11.1 `numpy` 函数

- **生成均匀分布随机数**

  ```python
  A = np.random.rand(2, 3)  # 生成2x3的均匀分布随机数
  A = np.random.uniform(0, 1, size=[2, 3])  # 相同效果
  ```

- **生成其他分布随机数**

  ```python
  X = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布
  X = np.random.gamma(shape=2, scale=1, size=1000)  # Gamma分布
  X = np.random.binomial(n=10, p=0.5, size=1000)  # 二项分布
  X = np.random.geometric(p=0.3, size=1000)  # 几何分布
  X = np.random.poisson(lam=4, size=1000)  # 泊松分布
  X = np.random.randint(low=1, high=5, size=1000)  # 整数均匀分布
  X = np.random.weibull(a=2, size=1000)  # Weibull分布
  X = np.random.lognormal(mean=0, sigma=1, size=1000)  # 对数正态分布
  ```

### 11.2 `scipy.stats` 分布方法

每个分布对象（如 `stats.norm`）通常包含以下方法：

- **模拟随机变量 (`rvs`)**

  ```python
  simus = stats.norm.rvs(loc=0, scale=1, size=1000)
  simus = stats.gamma.rvs(a=2, scale=1, size=1000)
  ```

  - `loc`：位置参数，$\mu$
  - `scale`：尺度参数，$\sigma$
  - `size`：生成的随机变量数量，可以是整数或元组

- **计算概率密度函数 (`pdf`)**

  ```python
  pdf_values = stats.norm.pdf(x, loc=0, scale=1)
  pdf_values = stats.gamma.pdf(x, a=2, scale=1)
  ```

  - `x`：计算密度的点
  - `loc`：位置参数，$\mu$
  - `scale`：尺度参数，$\sigma$

- **计算累积分布函数 (`cdf`)**

  ```python
  cdf_values = stats.norm.cdf(x, loc=0, scale=1)
  cdf_values = stats.gamma.cdf(x, a=2, scale=1)
  ```

  - 参数同 `pdf`

- **计算分位函数 (`ppf`)**

  ```python
  ppf_values = stats.norm.ppf(q, loc=0, scale=1)
  ppf_values = stats.gamma.ppf(q, a=2, scale=1)
  ```

  - `q`：分位点（概率）

- **计算矩 (`moment`)**

  ```python
  moments = stats.norm.moment(n=1, loc=0, scale=1)
  moments = stats.gamma.moment(n=2, a=2, scale=1)
  ```

  - `n`：矩的阶数

### 11.3 特殊函数

- **阶乘函数 (`scipy.special.factorial`)**

  ```python
  from scipy.special import factorial as fac
  k_factorial = fac(5)  # 计算5的阶乘
  ```

- **多元正态分布的概率密度函数**

  ```python
  Z_density = stats.multivariate_normal.pdf(xy, mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
  ```

## 12. 示例代码汇总

### 12.1 绘制基本条形图

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

### 12.2 绘制直方图并获取数值信息

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

- `plt.hist()`：绘制直方图，并返回条形高度和分箱边界。
- `np.histogram()`：计算直方图数据，但不绘图。

### 12.3 绘制离散分布的条形图

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial as fac
from scipy.stats import binom

n = 12
p = 0.2
X = np.random.binomial(n, p, size=2000)

x = np.arange(0, n + 1)
bins = np.arange(0, n + 2) - 0.5
pmf = binom.pmf(x, n, p)
cdf = binom.cdf(x, n, p)

plt.hist(X, bins=bins, density=True, edgecolor="k", label="模拟数据")
plt.plot(x, pmf, 'o', label="理论 PMF")
plt.plot(x, cdf, '+', label="理论 CDF")
plt.title("二项分布 (n=12, p=0.2)")
plt.xlabel("k")
plt.ylabel("概率")
plt.legend()
plt.show()
```

- `binom.pmf(k, n, p)`：二项分布的概率质量函数。
- `binom.cdf(k, n, p)`：二项分布的累积分布函数。

### 12.4 二维直方图与等高线叠加

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
plt.xlabel("X")
plt.ylabel("Y")
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
ax_y.hist(arrivals[:, 1], bins=bins, density=True, orientation='horizontal', edgecolor="k")
ax_y.plot(stats.norm.pdf(bins, mean_theoretical[1], np.sqrt(cov_theoretical[1, 1])), bins, 'r-', lw=2)
ax_y.set_xticks([])
ax_y.set_yticks([])
ax_y.set_title("Y方向分布")

plt.tight_layout()
plt.show()
```

- `plt.subplot2grid(shape, loc, colspan, rowspan)`：在指定的网格中创建子图。
- `ax.imshow()`：显示二维密度图。
- `ax.contour()`：绘制等高线。
- `ax.hist()`：绘制直方图。
- `stats.multivariate_normal.pdf(x, mean, cov)`：多元正态分布的概率密度函数。

### 12.5 绘制叠加直方图与密度曲线

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
plt.xlabel("X")
plt.ylabel("密度")
plt.show()
```

- `def function(x):`：定义函数。
- `plt.hist()`：绘制直方图。
- `plt.plot()`：绘制折线图。

### 12.6 对数正态分布与正态分布叠加

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
plt.xlabel("Y")
plt.ylabel("密度")
plt.show()
```

- `np.random.lognormal(mean, sigma, size)`：生成对数正态分布随机变量。
- `lognorm.pdf(x, s, scale)`：对数正态分布的概率密度函数。

## 13. 模拟分布之间的关系

### 13.1 Gamma分布与Beta分布的关系

**命题：**

设 $X_1 \sim \text{Gamma}(\alpha)$，$X_2 \sim \text{Gamma}(\beta)$，且 $X_1$ 与 $X_2$ 独立，则：
$$
\frac{X_1}{X_1 + X_2} \sim \text{Beta}(\alpha, \beta)
$$

#### 模拟验证

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, beta

alpha = 2
beta_param = 3
size = 10000

# 模拟 Gamma 分布随机变量
X1 = gamma.rvs(a=alpha, size=size)
X2 = gamma.rvs(a=beta_param, size=size)

# 计算比例
ratio = X1 / (X1 + X2)

# 绘制比例的直方图与Beta分布密度曲线
plt.hist(ratio, bins=50, density=True, alpha=0.6, edgecolor="k", label="模拟比例")

# 绘制 Beta 分布的密度函数
x = np.linspace(0, 1, 100)
plt.plot(x, beta.pdf(x, a=alpha, b=beta_param), 'r-', lw=2, label="Beta 分布密度")

plt.legend()
plt.title("Gamma分布随机变量比例的分布")
plt.xlabel("比例")
plt.ylabel("密度")
plt.show()
```

- `gamma.rvs(a, size)`：生成Gamma分布随机变量。
- `beta.pdf(x, a, b)`：Beta分布的概率密度函数。

**解释：**

通过模拟大量的 $X_1$ 和 $X_2$，计算它们的比例，并将其与理论上的Beta分布密度函数进行比较，可以验证比例确实服从Beta分布。

## 14. 总结

通过以上整理，涵盖了使用Python进行概率分布模拟与可视化的常用语法和函数。内容包括连续分布、离散分布、多变量分布的模拟方法，以及直方图、条形图、二维直方图、六边形分箱图、核密度估计等绘图技巧。掌握这些语法和函数能够帮助您高效地进行数据分析和统计建模。

### 关键要点：

- **分布参数：** 理解各分布的形状参数、位置参数和尺度参数，并掌握它们对分布形态的影响。
- **模拟与密度计算：** 使用 `rvs` 方法模拟随机变量，使用 `pdf` 和 `cdf` 方法计算概率密度函数和累积分布函数。
- **绘图技巧：** 熟悉直方图 (`hist`)、条形图 (`bar`)、二维直方图 (`hist2d`)、六边形分箱图 (`hexbin`) 等方法，用于可视化数据分布。
- **核密度估计（KDE）：** 使用 `gaussian_kde` 进行平滑的密度估计，有助于更直观地展示数据分布。
- **分布关系：** 理解不同分布之间的关系，如Gamma分布与Beta分布等，有助于选择合适的模型进行分析。

### 建议：

结合实际数据进行练习，熟悉各个函数的参数设置和使用方法，以便在数据分析过程中灵活应用。