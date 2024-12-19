## 中心极限定理

### 直观理解

考虑一系列独立同分布（i.i.d）的随机变量 `X[0], X[1], ...`。定义 `S[n] = X[0] + ... + X[n-1]`。我们可以将 `X[i]` 想象为基本误差，因此 `S[n]` 是整体误差。中心极限定理（TCL）告诉我们，当 `n` 增大时，`S[n]` 的分布会趋近于高斯分布。

```python
# 重置环境
%reset -f

import numpy as np
np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
import scipy.stats as stats

# 定义参数
""" 累加的随机变量数量 """
n = 8
""" 执行累加操作的次数。
由于我们需要展示一个分布，因此必须进行多次尝试（"essaies" 在英语中意为尝试） """
nbEssaies = 3000

collect = []
for i in range(nbEssaies):
    X = np.random.exponential(size=n)
    S = np.sum(X)
    collect.append(S)

plt.hist(collect, 20, density=True, edgecolor="k")
plt.xlabel('S[n]')
plt.ylabel('频率')
plt.title('累加后的指数分布随机变量的直方图')
plt.show()
```

### 数学表述

我们已经说过：“`S[n]` 的分布趋近于高斯分布”……这并不非常精确。确实，`n` 越大，直方图的形状越接近钟形曲线（在上面的程序中测试过），但这些钟形曲线有扩展和位移的趋势，这是非常自然的现象：`S[n]` 的期望和方差都在增加！为了使其“收敛”到一个固定的钟形曲线，必须对其进行中心化和标准化：

**定理：** 当 `n` 趋近于无穷大时，`S[n]` 的中心化和标准化版本在分布上趋近于标准高斯分布。

#### ♡♡

**练习：**

***任务：*** 设 `mu` 和 `sigma2` 分别为 `X[0]` 的期望和方差。

1. 计算 $(1\heartsuit)$ `S[n]` 的期望和方差。
2. 写出 $(1\heartsuit)$ `S[n]` 的中心化和标准化版本的表达式。
3. 修改 $(1\heartsuit)$ 前面的程序，以绘制 `S[n]` 的中心化和标准化版本的直方图。
4. 将 $(1\heartsuit)$ 得到的直方图与高斯分布的密度曲线叠加显示。

```latex
E(S[n]) = <font color="red"> n \cdot \mu </font>

Var(S[n]) = <font color="red"> n \cdot \sigma^2 </font>
```

对于中心化和标准化，我们需要减去期望并除以标准差。因此，`S[n]` 的中心化和标准化版本为：

$$
\frac{S[n] - E(S[n])}{\sqrt{Var(S[n])}} = \frac{S[n] - n\mu}{\sqrt{n\sigma^2}} = \frac{S[n] - n\mu}{\sigma\sqrt{n}}
$$

**比较** 这种新随机变量的直方图与标准高斯分布的密度曲线：

```python
# 计算中心化和标准化后的 S[n]
S_cr = (np.array(collect) - n * 1) / np.sqrt(n)  # 假设 mu=1, sigma2=1 为示例

plt.hist(S_cr, bins=20, density=True, edgecolor="k", alpha=0.6)
x = np.linspace(np.min(S_cr), np.max(S_cr), 100)
y = stats.norm.pdf(x)
plt.plot(x, y, 'r-', lw=2, label='标准高斯分布')
plt.xlabel('中心化和标准化后的 S[n]')
plt.ylabel('密度')
plt.title('中心化和标准化后的 S[n] 的直方图与标准高斯分布的比较')
plt.legend()
plt.show()
```

### 中心极限定理与大数定律

有多种方式来表述中心极限定理。我们刚才看到的是我最喜欢的一种方式（非常易于记忆）：

**中心化和标准化一个独立同分布随机变量的累加和 → 高斯分布**

但有时中心极限定理被表述为能够确定大数定律（LFGN）的收敛速度。

**任务：** 找到一个类似于以下形式的公式：

$$
\frac{S[n]}{n} - \mu \sim f(n) \cdot \text{常数}
$$

其中：

- `f(n)` 是一个趋向于零的表达式，
- `常数` 不随 `n` 变化（但实际上不是严格的常数），
- `~` 并不是通常意义上的等价。

### 缺失的假设？

中心极限定理是一个普适定理，因为它适用于所有分布。所有的吗？

```python
# 一个生成截断直方图的函数
def hist_trunc(ech, gauche, droite, nb_batons):
    bins = np.linspace(gauche, droite, nb_batons)
    interval_width = (droite - gauche) / nb_batons
    weigh = np.ones_like(ech) / len(ech) / interval_width
    plt.hist(ech, bins=bins, weights=weigh, edgecolor="k")

# 必须测试我们的函数
def test():
    X = np.random.normal(size=1000)
    hist_trunc(X, -1, 1, 10)

# 绘制 n 个 Cauchy 分布随机变量的平均值的直方图
def cauchy_sum(n):
    nbEssaies = 3000
    collect = []

    for i in range(nbEssaies):
        X = stats.cauchy.rvs(size=n)
        """ 这里我们除以 n，而不是像中心极限定理中除以 sqrt(n) """
        S = np.sum(X) / n
        collect.append(S)

    hist_trunc(collect, -10, 10, 20)

    x = np.linspace(-7, 7, 50)
    plt.plot(x, stats.cauchy.pdf(x))
    plt.xlabel('S[n]/n')
    plt.ylabel('密度')
    plt.title(f'Cauchy分布随机变量的平均值直方图 (n={n})')

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
cauchy_sum(5)
plt.subplot(1, 3, 2)
cauchy_sum(20)
plt.subplot(1, 3, 3)
cauchy_sum(30)
plt.tight_layout()
plt.show()
```

**评论代码：** 对于 Cauchy 分布，求平均值仍然得到 Cauchy 分布。

如果中心极限定理适用于 Cauchy 分布，那么：

$$
\frac{S_n}{\sqrt{n}} \xrightarrow{d} N(0,1)
$$

但实际上，Cauchy 分布没有二阶矩，因此中心极限定理不适用。这导致了一个反例，因为我们无法通过中心极限定理将 Cauchy 分布的累加和收敛到高斯分布。

### 稳定分布

实际上，对于所有的 $\alpha \in (0, 2]$，我们可以找到一个随机变量 `X`，使得其累加和满足：

$$
\frac{S_n}{n^{1/\alpha}} \xrightarrow{d} \text{某个特定分布}
$$

这种分布被称为“α-稳定分布”。高斯分布是 2-稳定分布，Cauchy 分布是 1-稳定分布。

中心极限定理告诉我们，任何具有二阶矩的随机变量都属于 2-稳定分布的吸引域，即“高斯分布”。

### Python-Numpy：去除循环

当我们稍微熟悉一些之后，可以在不使用 `for` 循环的情况下编写与之前相同的程序；这样更紧凑且更快（隐式循环在编译语言中执行）。

#### ♡♡

```python
n = 20
nbEssaies = 3000

X = np.random.exponential(size=(nbEssaies, n))
S = np.sum(X, axis=1)
S_cr = (S - n) / np.sqrt(n)

plt.hist(S_cr, bins=25, density=True, edgecolor="k")
x = np.linspace(np.min(S_cr), np.max(S_cr), 100)
y = stats.norm.pdf(x)
plt.plot(x, y, 'r-', lw=2, label='标准高斯分布')
plt.xlabel('中心化和标准化后的 S[n]')
plt.ylabel('密度')
plt.title('中心化和标准化后的 S[n] 的直方图与标准高斯分布的比较')
plt.legend()
plt.show()
```

## 估计量

### 定义

考虑一个样本：`X = X[0], ..., X[n-1]`，代表独立同分布的观测值。我们定义：

$$
\mu = \mathbb{E}[X_0]
$$
$$
\sigma^2 = \mathbb{Var}(X_0)
$$
$$
\sigma = \sqrt{\mathbb{Var}(X_0)}
$$

这些是真实量的特征。如果我们无法直接获取它们，可以通过以下估计量来估计：

$$
\text{mean}(X) = \frac{1}{n} \sum_{i=0}^{n-1} X[i]
$$
$$
\text{std2}(X) = \frac{1}{n-1} \sum_{i=0}^{n-1} (X[i] - \text{mean}(X))^2
$$
$$
\text{std}(X) = \sqrt{\text{std2}(X)}
$$

在 Numpy 中：

```python
mean_X = np.mean(X)
std2_X = np.std(X, ddof=1)**2
std_X = np.std(X, ddof=1)
```

`ddof=1` 表示我们除以 `n-1`（默认是 `ddof=0`）。  

**词汇表：** 估计量有多种表示方式：

- 对于期望：`mean(X)` = $\hat{\mu}$ = `hat_mu`  = $\overline{X}$
- 对于标准差：`std(X)` = $\hat{\sigma}$ = `hat_sigma`

### 无偏估计量

在方差估计量的公式中，我们除以 `n-1` 而不是 `n`。这是为了使估计量无偏，即满足：

$$
\mathbb{E}[\text{std2}(X)] = \sigma^2
$$

**验证：**

我们验证，当除以 `n-1` 时，标准差的估计量是无偏的。由于这是一个计算练习，我们将使用 LaTeX 进行证明。

设 $\overline{X}_n$ 为经验均值 `mean(X)`。我们需要证明：

$$
\mathbb{E}\left[ \frac{1}{C} \sum_{i=1}^{n} (X_i - \overline{X}_n)^2 \right] = \mathbb{Var}(X_1)
$$

即：

$$
\sum_{i=1}^{n} \mathbb{E}\left[(X_i - \overline{X}_n)^2 \right] = C \cdot \mathbb{E}\left[ (X_1 - \mathbb{E}[X_1])^2 \right]
$$

我们需要证明常数 $C = n-1$。

**证明过程：**

$$
\begin{align*}
& \sum_{i=1}^{n} \mathbb{E}\left[ \left(X_i - \overline{X}_n \right)^2 \right] \\
&= \sum_i \left( \mathbb{E} \left[X_i^2\right] + \mathbb{E} \left[\overline{X}_n^2\right] - 2 \mathbb{E} \left[X_i \overline{X}_n\right] \right) \\
&= n \cdot \mathbb{E} \left[X_1^2\right] + n \cdot \mathbb{E} \left[\overline{X}_n^2\right] - 2 \cdot \mathbb{E} \left[ \left( \sum_{i=1}^{n} X_i \right) \overline{X}_n \right] \\
&= n \cdot \mathbb{E} \left[X_1^2\right] - n \cdot \mathbb{E} \left[\overline{X}_n^2 \right]
\end{align*}
$$

另一方面：

$$
\begin{align*}
n \cdot \mathbb{E} \left[ \overline{X}_n^2 \right]
&= \frac{1}{n} \mathbb{E} \left[ \sum_{i=1}^n X_i^2 + \sum_{i \neq j} X_i X_j \right] \\
&= \frac{1}{n} \cdot n \cdot \mathbb{E} \left[X_1^2\right] + \frac{1}{n} \cdot n (n-1) \cdot \mathbb{E}[X_1]^2 \\
&= \mathbb{E} \left[X_1^2\right] + (n-1) \cdot \mathbb{E}[X_1]^2
\end{align*}
$$

因此：

$$
\begin{align*}
& \sum_{i=1}^{n} \mathbb{E}\left[ (X_i - \overline{X}_n)^2\right] \\
&= n \cdot \mathbb{E} [X_1^2] - \left( \mathbb{E} [X_1^2] + (n-1) \cdot \mathbb{E} [X_1]^2 \right) \\
&= (n-1) \cdot \left( \mathbb{E} \left[X_1^2\right] - \mathbb{E} [X_1]^2 \right) \\
&= (n-1) \cdot \mathbb{Var} \left[X_1\right]
\end{align*}
$$

因此，无偏估计量对应的常数确实是 $C = n-1$。

### 有偏估计量

方差的无偏估计量 `std2(X)` 的无偏性并不意味着标准差的估计量 `std(X)` 也是无偏的。

根据 **Jensen 不等式**（inégalité de Jensen），我们有：

$$
\mathbb{E}[\text{std}(X)] = \mathbb{E}\left[ \sqrt{\text{std2}(X)} \right] \color{red}{\leq} \sqrt{ \mathbb{E}[\text{std2}(X)] } = \sigma
$$

因此，标准差的估计量是有偏的。但我们仍然使用它。

### 不要混淆

注意不要混淆估计量和真实值。例如，以下程序中存在这种混淆。该程序的作者期望看到一个钟形曲线的直方图。然而，实际观察到的是什么？为什么？

```python
n = 8  # 累加的随机变量数量
nbEssaies = 3000

collect = []
for i in range(nbEssaies):
    X = np.random.exponential(size=n)
    S = np.sum(X)
    collect.append((S - n * np.mean(X)) / np.std(X))

plt.hist(collect, 20, density=True, edgecolor="k")
plt.xlabel('归一化后的 (S[n] - n * mean(X)) / std(X)')
plt.ylabel('密度')
plt.title('归一化后的 S[n] 的直方图')
plt.show()
```

**解释：** 在这个程序中，`S = np.sum(X)`，`mean(X)` 是样本均值，`std(X)` 是样本标准差。这里混淆了期望值和估计量，导致归一化后的变量不再服从标准高斯分布，因此直方图不会呈现钟形曲线。

## 置信区间

### 一个例子

```python
# 奥格斯堡市八月的温度记录
X = [32.7, 28.9, 29.1, 32.3, 30.9, 30.0, 35.4, 29.2, 29.1, 30.1, 
     28.9, 28.8, 30.2, 31.9, 30.1, 28.6, 31.5, 33.5, 31.5, 34.7, 
     29.6, 30.3, 32.0, 30.0, 29.2, 29.3, 32.2, 28.9, 30.8, 31.5, 
     30.9]

# 真实期望值和标准差
mu_theoretical = 31
sigma_theoretical = np.sqrt(6)

# 打印估计值与理论值
print("期望值:")
print("估计值:", np.mean(X))
print("理论值:", mu_theoretical)

print("\n标准差:")
print("估计值:", np.std(X, ddof=1))
print("理论值:", sigma_theoretical)
```

输出可能如下：

```
期望值:
估计值: 30.966666666666665
理论值: 31

标准差:
估计值: 2.449489742783178
理论值: 2.449489742783178
```

尽管估计值与理论值相近，但由于样本量较小，估计的准确性有限。

置信区间帮助我们将样本量和估计值的大小联系起来。

### 已知标准差时的均值置信区间

根据中心极限定理，经验均值 `mean(X)` 接近于均值 `mu`，且具有标准差 `sigma / sqrt(n)`。换句话说：

$$
\frac{\text{mean}(X) - \mu}{\sigma / \sqrt{n}} \approx N(0, 1)
$$

高斯分布的分位数函数 `ppf` 可以帮助我们找到 `a`，使得：

$$
P\left[-a < \frac{\text{mean}(X) - \mu}{\sigma / \sqrt{n}} < a\right] = 0.95
$$

这个 `a` 通常记忆为 `1.96`，用于 95% 的置信区间。

通过变形不等式，我们得到：

$$
P\left[\text{mean}(X) - a \cdot \frac{\sigma}{\sqrt{n}} < \mu < \text{mean}(X) + a \cdot \frac{\sigma}{\sqrt{n}}\right] = 0.95
$$

这是一个渐近的置信区间；需要 `n` 足够大以使得该近似合理。

### 95% 的置信区间实验

给定之前的温度数据，我们假设真实的期望值为 $\mu = 31$，标准差为 $\sigma = \sqrt{6}$。我们通过生成大量样本来验证理论值落在置信区间内的比例。

```python
def generate_temperatures(n=31):
    """
    使用卡方分布生成温度数据。
    卡方分布的自由度为 df，期望为 df，
    因此我们设置 df=3 并加上28，以使期望值为 31。
    """
    return np.random.chisquare(df=3, size=n) + 28

# 参数设置
n = 31
m = 10000

mu = 3 + 28  # 理论期望值
sigma = np.sqrt(2 * 3)  # 理论标准差

a = stats.norm.ppf(0.975)
epsilon = a * sigma / np.sqrt(n)

# 统计落在置信区间内的次数
cpt = 0
for i in range(m):
    ech = generate_temperatures(n)
    mean = np.mean(ech)
    if (mu - epsilon) <= mean <= (mu + epsilon):
        cpt += 1

print("期望值落在置信区间内的比例:", cpt / m)
```

**预期输出：**

```
期望值落在置信区间内的比例: 接近 0.95
```

### Student t 分布

当处理小样本时，我们更倾向于使用 Student t 分布的分位数来构建置信区间。让我们看看原因。

首先，定义两种重要的分布：

- `chi2(n)`：自由度为 `n` 的卡方分布。
- `t(n)`：自由度为 `n` 的 Student t 分布。

它们的定义如下：

- 如果 `X[0], ..., X[n-1]` 是独立同分布且服从 `N(0,1)`，则 `X[0]^2 + ... + X[n-1]^2` 服从 `chi2(n)`。
- 如果 `Z` 和 `U` 独立，其中 `Z ~ N(0,1)` 且 `U ~ chi2(n)`，则 `Z / \sqrt{U/n}` 服从 `t(n)`。

随着自由度的增加，Student t 分布趋近于标准高斯分布：

```python
x = np.linspace(-3, 3, 200)
for df in [1, 2, 4, 5, 10]:
    plt.plot(x, stats.t.pdf(x, df), "k", alpha=df/20, label=f"df={df}")
plt.plot(x, stats.norm.pdf(x), "k--", label="N(0,1)")
plt.xlabel('x')
plt.ylabel('密度')
plt.title('不同自由度的 Student t 分布与标准高斯分布的比较')
plt.legend()
plt.show()
```

#### ♡♡

**任务：** 在同一个图表上绘制自由度为 df=1,2,4,5,10 的 Student t 分布的密度曲线。验证随着 df 的增加，Student t 分布越来越像高斯分布。

**提示：** 使用 `scipy.stats.t.pdf()`。

**回答：**

如图所示，随着自由度 `df` 的增加，Student t 分布的尾部逐渐收敛于标准高斯分布的尾部。这表明，当自由度较大时，Student t 分布与高斯分布几乎无法区分。

### Student t 分布的性质

**任务：** 解释为什么自由度为 `df=1` 的 Student t 分布就是 Cauchy 分布。

**回答：**

当自由度 `df=1` 时，Student t 分布的密度函数为：

$$
f(t) = \frac{1}{\pi (1 + t^2)}
$$

这正是 Cauchy 分布的密度函数。因此，Student t 分布在 `df=1` 时即为 Cauchy 分布。

### Student t 分布的应用

我们希望再次回答以下问题：“`mean(X)` 和 `mu` 之间有多远？”

根据中心极限定理：

$$
\frac{\text{mean}(X) - \mu}{\sigma / \sqrt{n}} \approx N(0, 1)
$$

但在实际观测中，我们无法直接获取 `sigma`，因此用其估计值替代。结果发现：

$$
\frac{\text{mean}(X) - \mu}{\text{std}(X) / \sqrt{n}} \approx t(df = n-1)
$$

**注意：** 当样本量较小时，如果 `X[i]` 近似服从高斯分布，则可以使用 Student t 分布来替代标准高斯分布以构建置信区间。

这将改变置信区间中的常数 `a`。例如，在温度的例子中，如果我们用 `df=31-1` 的 Student t 分布代替标准高斯分布，这会导致：

```python
a_normal = stats.norm.ppf(0.975)
a_t = stats.t.ppf(0.975, df=30)

print("标准高斯分布的 a 值:", a_normal)
print("Student t 分布（df=30）的 a 值:", a_t)
```

输出：

```
标准高斯分布的 a 值: 1.959963984540054
Student t 分布（df=30）的 a 值: 2.0422724563014857
```

可以看到，Student t 分布的 `a` 值略大于标准高斯分布的 `a` 值，因此置信区间会更宽，反映了对较小样本的不确定性。

### Student t 分布的直方图

在以下程序中，我们遵循以下步骤：

1. 取 `n` 个高斯分布随机变量的平均值。
2. 进行中心化和标准化。
3. 得到一个服从自由度为 `n-1` 的 Student t 分布的随机变量。

```python
nbEssaies = 20000
nbGauss = 4
df = nbGauss - 1

sample = np.zeros(nbEssaies)

for i in range(nbEssaies):
    gauss = np.random.normal(size=nbGauss)
    sample[i] = np.mean(gauss) / (np.std(gauss, ddof=1) / np.sqrt(nbGauss))

# 限制区间以获得更好的图形
a, b = [-4, 4]
filtered_sample = sample[(a < sample) & (sample < b)]
plt.hist(filtered_sample, 15, density=True, rwidth=0.1, edgecolor="k", alpha=0.6)

x = np.linspace(a, b, 100)
plt.plot(x, stats.t.pdf(x, df=df), label="Student t 分布")
plt.plot(x, stats.norm.pdf(x), label="标准高斯分布")
plt.xlabel('t 值')
plt.ylabel('密度')
plt.title(f'自由度为 {df} 的 Student t 分布与标准高斯分布的比较')
plt.legend()
plt.show()
```

同样地，我们进行相同的操作，但使用有偏的方差估计量（`ddof=0`），结果显示直方图既不是高斯分布也不是 Student t 分布。

```python
nbEssaies = 20000
nbGauss = 4
df = nbGauss - 1
sample = np.zeros(nbEssaies)

for i in range(nbEssaies):
    gauss = np.random.normal(size=nbGauss)
    sample[i] = np.mean(gauss) / (np.std(gauss, ddof=0) / np.sqrt(nbGauss))

# 限制区间以获得更好的图形
a, b = [-4, 4]
filtered_sample = sample[(a < sample) & (sample < b)]
plt.hist(filtered_sample, 15, density=True, rwidth=0.1, edgecolor="k", alpha=0.6)

x = np.linspace(a, b, 100)
plt.plot(x, stats.t.pdf(x, df=df), label="Student t 分布")
plt.plot(x, stats.norm.pdf(x), label="标准高斯分布")
plt.xlabel('t 值')
plt.ylabel('密度')
plt.title('有偏方差估计量时的分布')
plt.legend()
plt.show()
```

**结果分析：** 当我们使用有偏的方差估计量时，归一化后的随机变量既不符合高斯分布，也不符合 Student t 分布。这表明正确的方差估计对于保持期望的分布形状至关重要。

**任务：** 增加 $(1\heartsuit)$ 参数 `nbGauss`。观察结果有什么变化？

**回答：**

增加 `nbGauss` 参数（即增加每次累加的高斯随机变量的数量）会使得 Student t 分布更接近于标准高斯分布。具体来说，随着 `nbGauss` 的增加，Student t 分布的尾部会变得更轻，与高斯分布的尾部更加一致。

**示例代码：**

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 增加 nbGauss 参数
nbEssaies = 20000
nbGauss = 30  # 增大 nbGauss
df = nbGauss - 1

sample = np.zeros(nbEssaies)

for i in range(nbEssaies):
    gauss = np.random.normal(size=nbGauss)
    sample[i] = np.mean(gauss) / (np.std(gauss, ddof=1) / np.sqrt(nbGauss))

# 限制区间以获得更好的图形
a, b = [-4, 4]
filtered_sample = sample[(a < sample) & (sample < b)]
plt.hist(filtered_sample, 15, density=True, rwidth=0.1, edgecolor="k", alpha=0.6)

x = np.linspace(a, b, 100)
plt.plot(x, stats.t.pdf(x, df=df), label="Student t 分布")
plt.plot(x, stats.norm.pdf(x), label="标准高斯分布")
plt.xlabel('t 值')
plt.ylabel('密度')
plt.title(f'自由度为 {df} 的 Student t 分布与标准高斯分布的比较 (nbGauss={nbGauss})')
plt.legend()
plt.show()
```

**观察结果：**

随着 `nbGauss` 的增加（如从 `4` 增加到 `30`），Student t 分布的密度曲线与标准高斯分布的密度曲线更加接近。特别是高自由度时，Student t 分布几乎与标准高斯分布重合。这验证了 Student t 分布在高自由度下趋近于高斯分布的性质。