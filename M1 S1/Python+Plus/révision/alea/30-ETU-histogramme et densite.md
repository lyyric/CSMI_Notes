# Python 语法整理：直方图与相关图表

本文整理了在直方图及相关图表绘制过程中使用的所有Python语法和函数。内容涵盖了导入必要的库、绘制条形图、直方图、二维直方图、核密度估计等。以下内容将按功能模块分类，详细介绍每个函数的用法、参数及示例。

## 1. 导入必要的库

在进行数据可视化和统计分析时，常用以下库：

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import factorial as fac
from scipy.stats import kde
```

- **numpy (`np`)**：用于数值计算和数组操作。
- **matplotlib.pyplot (`plt`)**：用于绘制各种图表。
- **scipy.stats (`stats`)**：包含统计分布和统计函数。
- **scipy.special.factorial (`fac`)**：计算阶乘。
- **scipy.stats.kde**：用于核密度估计。

## 2. 条形图 (`bar`)

用于绘制条形图，展示分类数据的分布情况。

### 基本用法

```python
fig, ax = plt.subplots()
ax.bar(x, y, edgecolor="k", width=0.5)
plt.show()
```

- **参数解释**：
  - `x`：条形图的x轴位置，可以是类别或数值。
  - `y`：每个条形的高度。
  - `edgecolor`：条形边缘颜色，如 `"k"` 表示黑色。
  - `width`：条形的宽度，默认为0.8。

### 示例：人口年龄金字塔

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

- **功能**：绘制按年龄和性别分布的人口金字塔。
- **说明**：男性数据为正，女性数据为负，以实现对称显示。

## 3. 直方图 (`hist`)

用于展示数值数据的分布情况，通过分箱（bins）将数据分组。

### 自动分箱

```python
X = np.random.normal(0, 1, size=1000)
plt.hist(X, bins=15, edgecolor="k")
plt.show()
```

- **参数解释**：
  - `bins`：分箱的数量或具体的分箱边界。
  - `edgecolor`：条形边缘颜色。

### 获取数值信息

```python
a = plt.hist(X, bins=5, edgecolor="k")
print("条形高度\n", a[0])
print("分箱边界\n", a[1])

b = np.histogram(X, bins=5)
print("条形高度\n", b[0])
print("分箱边界\n", b[1])
```

- **功能**：`plt.hist` 和 `np.histogram` 可以返回直方图的数据，如条形高度和分箱边界。

### 手动分箱

```python
plt.hist(X, bins=[-2, 0.5, 1, 5], edgecolor="k")
plt.show()
```

- **说明**：指定具体的分箱区间，灵活控制数据分组。

### 离散分布的直方图

对于离散型数据（如二项分布），需明确设置分箱，以确保每个类别独立。

```python
n = 12
X = np.random.binomial(n, 0.8, size=2000)
bins = np.arange(0, n + 2, 1) - 0.5
plt.hist(X, bins=bins, edgecolor="k")
plt.xticks(np.arange(0, n + 1, 1))
plt.show()
```

- **关键点**：通过设置`bins`为`np.arange(0, n + 2, 1) - 0.5`，确保每个整数值在独立的分箱内。

### 使用 `np.unique` 绘制离散分布

```python
value, count = np.unique(X, return_counts=True)
plt.bar(value, count)
plt.xticks(value)
plt.show()
```

- **功能**：统计每个唯一值的出现次数，并用条形图展示。

## 4. 二维直方图 (`hist2d` 和 `hexbin`)

用于展示二维数据的分布情况。

### 使用 `hist2d`

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect("equal")
output = ax.hist2d(arrivals[:, 0], arrivals[:, 1], bins=[bins, bins], cmap="jet")
plt.show()
```

- **参数解释**：
  - `arrivals[:, 0]`, `arrivals[:, 1]`：二维数据的x和y坐标。
  - `bins`：分箱数量或边界。
  - `cmap`：颜色映射，如 `"jet"`。

### 使用 `hexbin`

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.hexbin(arrivals[:, 0], arrivals[:, 1], gridsize=40, cmap="jet")
plt.show()
```

- **参数解释**：
  - `gridsize`：六边形网格的数量，控制分辨率。
  - `cmap`：颜色映射。

## 5. 核密度估计 (`gaussian_kde`)

用于估计数据的概率密度函数，生成平滑的密度曲线。

### 一维核密度估计

```python
kernel = stats.gaussian_kde(X, bw_method=0.1)
y = kernel(x)
ax.plot(x, y)
```

- **参数解释**：
  - `bw_method`：带宽方法，控制平滑程度。数值越大，平滑越强。

### 二维核密度估计

```python
kernel = kde.gaussian_kde(arrivals.T)
ZZ = kernel(z).reshape(XX.shape)
ax.imshow(ZZ, cmap="jet", interpolation="bilinear", origin="lower", extent=[gauche, droite, gauche, droite])
```

- **步骤**：
  1. 创建核密度估计对象。
  2. 生成网格点。
  3. 计算密度值。
  4. 使用 `imshow` 展示密度图。

## 6. 绘制密度与直方图叠加

将直方图与理论密度曲线叠加，便于比较。

### 一维叠加

```python
plt.hist(Simu, bins=30, density=True, edgecolor="k")
plt.plot(x, gaussian_density(x))
plt.show()
```

- **关键点**：使用`density=True`将直方图归一化，使其与密度曲线对齐。

### 二维叠加

```python
ax_xy.imshow(ZZ, cmap="jet", interpolation="bilinear", origin="lower", extent=[gauche, droite, gauche, droite], vmin=0, vmax=0.002, alpha=0.8)
ax_xy.contour(XX, YY, Z_density, cmap="jet", vmin=0, vmax=0.002)
```

- **功能**：在二维直方图上叠加理论密度的等高线。

## 7. 对数转换处理偏斜数据

对数转换可以将偏斜分布的数据转化为更对称的分布，便于分析。

```python
Y = np.log(X)
plt.hist(Y, bins=30, edgecolor="k")
plt.show()
```

- **说明**：将比值`X`取对数，得到`Y`，通常能使数据分布更接近正态分布。

## 8. 其他常用函数与参数

### `np.linspace`

生成线性间隔的数值序列。

```python
bins = np.linspace(gauche, droite, nb_batons + 1)
```

- **参数**：
  - `start`：起始值。
  - `stop`：结束值。
  - `num`：生成的样本数量。

### `np.arange`

生成指定范围内的数值序列。

```python
bins = np.arange(0, n + 2, 1) - 0.5
```

- **参数**：
  - `start`：起始值。
  - `stop`：结束值（不包含）。
  - `step`：步长。

### `plt.plot`

绘制线条图，用于叠加密度曲线。

```python
plt.plot(x, gaussian_density(x))
plt.plot(k, density, 'o')
```

- **参数**：
  - `'o'`：标记样式，表示点标记。

### `plt.axvline` 和 `plt.axhline`

在图表中绘制垂直或水平参考线。

```python
plt.axvline(1, color="r")
plt.axvline(x, color="0.9", linewidth=0.3)
```

- **参数**：
  - `x`：垂直线的x坐标。
  - `color`：颜色。
  - `linewidth`：线宽。

### `plt.imshow`

显示图像数据，常用于展示二维密度图。

```python
ax.imshow(ZZ, cmap="jet", interpolation="bilinear", origin="lower", extent=[gauche, droite, gauche, droite])
```

- **参数解释**：
  - `cmap`：颜色映射。
  - `interpolation`：插值方法。
  - `origin`：图像的原点位置。
  - `extent`：图像的坐标范围。

### `plt.contour`

绘制等高线，用于叠加理论密度曲线。

```python
ax.contour(XX, YY, Z_density, cmap="jet", vmin=0, vmax=0.002)
```

- **参数解释**：
  - `levels`：等高线的级数。
  - `cmap`：颜色映射。

### `plt.hexbin`

绘制六边形直方图，用于展示二维数据的密度。

```python
ax.hexbin(arrivals[:, 0], arrivals[:, 1], gridsize=40, cmap="jet")
```

- **参数解释**：
  - `gridsize`：六边形网格的数量，控制分辨率。
  - `cmap`：颜色映射。

### `plt.subplot2grid`

创建网格布局中的子图，用于复杂图表的布局。

```python
ax_xy = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax_x = plt.subplot2grid((3, 3), (0, 0), colspan=2)
ax_y = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
```

- **参数解释**：
  - `(3, 3)`：网格的形状。
  - `(1, 0)`：子图起始位置。
  - `colspan`、`rowspan`：子图跨越的列数和行数。

### `plt.subplots`

创建图表和子图对象。

```python
fig, ax = plt.subplots(figsize=(10, 5))
```

- **参数解释**：
  - `figsize`：图表尺寸，单位为英寸。

### `plt.legend`

添加图例，标注不同数据系列。

```python
ax.legend()
plt.legend()
```

- **参数解释**：
  - 可以通过`label`参数在绘图函数中指定图例标签。

### `plt.show`

显示图表。

```python
plt.show()
```

## 9. 其他函数与方法

### `np.random` 模块

用于生成各种随机数和随机分布数据。

- **生成正态分布数据**

  ```python
  X = np.random.normal(0, 1, size=1000)
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
  Y = np.random.lognormal(size=1000)
  ```

- **生成标准柯西分布数据**

  ```python
  X = np.random.standard_cauchy(size=200)
  ```

### `np.unique`

用于找到数组中的唯一值，并可返回其他相关信息。

```python
val, count = np.unique(X, return_counts=True)
```

- **参数解释**：
  - `return_counts=True`：返回每个唯一值的出现次数。

### `np.cumsum`

计算数组的累积和，用于生成轨迹数据。

```python
trajectories = np.cumsum(pas, axis=1)
```

- **参数解释**：
  - `axis=1`：沿着指定轴计算累积和。

### `stats.multivariate_normal.pdf`

计算多变量正态分布的概率密度函数。

```python
Z_density = stats.multivariate_normal.pdf(xy, mean=mean_theoretical, cov=cov_theoretical)
```

- **参数解释**：
  - `xy`：网格点坐标。
  - `mean`：均值向量。
  - `cov`：协方差矩阵。

## 10. 参数详解总结

### `plt.hist` 常用参数

- **必需参数**：
  - `x`：数据数组。

- **数据处理相关参数**：
  - `bins`：分箱数目或具体的分箱边界（列表或数组）。
  - `density`：布尔值，是否归一化直方图，使其表示概率密度。
  - `range`：限制数据的范围，如`range=[左, 右]`。
  - `weights`：为每个数据点分配权重，常用于自定义归一化。

- **显示相关参数**：
  - `orientation`：条形图方向，`"horizontal"` 或 `"vertical"`。
  - `rwidth`：条形宽度比例，如`rwidth=0.5`表示条形宽度占分箱宽度的一半。
  - `edgecolor`：条形边缘颜色，如`"k"`表示黑色。
  - `color`：条形填充颜色。

### `plt.bar` 常用参数

- `x`：条形图的x轴位置。
- `height`：条形的高度。
- `width`：条形的宽度。
- `edgecolor`：条形边缘颜色。
- `color`：条形填充颜色。
- `label`：图例标签。

### `plt.hist2d` 常用参数

- `x`，`y`：二维数据的x和y坐标。
- `bins`：分箱数目或具体的分箱边界（列表或数组）。
- `cmap`：颜色映射，如`"jet"`。

### `plt.hexbin` 常用参数

- `x`，`y`：二维数据的x和y坐标。
- `gridsize`：六边形网格的数量，控制分辨率。
- `cmap`：颜色映射，如`"jet"`。

### `stats.gaussian_kde` 常用参数

- `dataset`：数据数组，形状为（维度，样本数）。
- `bw_method`：带宽方法，控制平滑程度，可以是浮点数或字符串。

## 11. 示例代码汇总

### 绘制基本条形图

```python
import matplotlib.pyplot as plt

x = [1, 1.5, 2, 2.5, 3]
y = [1, 4, 3, 4, 1]

fig, ax = plt.subplots()
ax.bar(x, y, edgecolor="k", width=0.5)
plt.show()
```

### 绘制直方图并获取数值信息

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

### 绘制离散分布的条形图

```python
import numpy as np
import matplotlib.pyplot as plt

n = 12
X = np.random.binomial(n, 0.8, size=2000)
bins = np.arange(0, n + 2, 1) - 0.5
plt.hist(X, bins=bins, edgecolor="k")
plt.xticks(np.arange(0, n + 1, 1))
plt.show()
```

### 二维直方图与等高线叠加

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

trajectories = makeTrajectories(T=100, nbSimu=50000)
arrivals = trajectories[:, -1, :]

gauche = -10
droite = 100
bins = np.arange(gauche, droite + 1, 1, dtype=np.float32)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect("equal")
output = ax.hist2d(arrivals[:, 0], arrivals[:, 1], bins=[bins, bins], cmap="jet")
plt.show()

# 核密度估计与等高线
kernel = gaussian_kde(arrivals.T)
x = np.linspace(gauche, droite, 50)
XX, YY = np.meshgrid(x, x)
z = np.vstack([XX.reshape(-1), YY.reshape(-1)])
ZZ = kernel(z).reshape(XX.shape)

mean_theoretical = np.array([1/4 * 100, 1/4 * 100])
cov_theoretical = np.array([[19/16 * 100, 1/2 * 100], [1/2 * 100, 11/16 * 100]])

fig = plt.figure(figsize=(6, 6))
ax_xy = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax_xy.set_aspect("equal")
ax_xy.imshow(ZZ, interpolation="bilinear", origin="lower",
             extent=[gauche, droite, gauche, droite],
             cmap="jet", vmin=0, vmax=0.002, alpha=0.8)

xy = np.stack([XX, YY], axis=2)
Z_density = multivariate_normal.pdf(xy, mean=mean_theoretical, cov=cov_theoretical)
ax_xy.contour(XX, YY, Z_density, cmap="jet", vmin=0, vmax=0.002)

ax_x = plt.subplot2grid((3, 3), (0, 0), colspan=2)
ax_x.hist(arrivals[:, 0], bins=bins, density=True, edgecolor="k")
ax_x.plot(bins, stats.norm.pdf(bins, mean_theoretical[0], np.sqrt(cov_theoretical[0, 0])))

ax_x.set_xticks([])
ax_x.set_yticks([])

ax_y = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax_y.hist(arrivals[:, 1], bins=bins, density=True, orientation='horizontal', edgecolor="k")
ax_y.plot(stats.norm.pdf(bins, mean_theoretical[1], np.sqrt(cov_theoretical[1, 1])), bins)

ax_y.set_xticks([])
ax_y.set_yticks([])
plt.show()
```

### 绘制叠加直方图与密度曲线

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

nbSimu = 1000
Simu = np.random.normal(size=nbSimu)

def gaussian_density(x):
    return 1 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

plt.hist(Simu, bins=30, density=True, edgecolor="k")
x = np.linspace(-3, 3, 200)
plt.plot(x, gaussian_density(x))
plt.show()
```

### 对数正态分布与正态分布叠加

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

size = 10000
Y = np.random.lognormal(size=size)
bins = np.linspace(0, 10, 80)
plt.hist(Y, bins=bins, edgecolor="k", label=["Y"], density=True)

y = np.linspace(0.001, 10, 1000)
plt.plot(y, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.log(y)**2) / y, label="理论密度")
plt.legend()
plt.show()
```

## 12. 总结

通过上述整理，涵盖了绘制条形图、直方图、二维直方图、核密度估计等常用的Python语法和函数。掌握这些语法和函数能够帮助你高效地进行数据可视化和分布分析。以下是一些关键要点：

- **分箱（Bins）**：在绘制直方图时，分箱的数量和区间对图形效果影响显著。对于离散数据，需明确设置分箱边界。
- **归一化（Density）**：使用`density=True`可以将直方图归一化，便于与理论密度曲线叠加。
- **颜色映射（Colormap）**：选择合适的颜色映射，如`"jet"`，提升图表的可读性。
- **核密度估计（KDE）**：提供平滑的密度估计，比直方图更直观地展示数据分布。

建议结合实际数据进行练习，熟悉各个函数的参数设置和使用方法，以便在数据分析过程中灵活应用。