# 随机变量对另一个的影响

```python
%reset -f

import numpy as np
import matplotlib.pyplot as plt
import scipy
np.set_printoptions(precision=2, suppress=True, linewidth=100)
```

## 引言

### 随机变量与数据

本课程的主角是两个随机变量 $X$ 和 $Y$。它们取值于实数集 $\mathbb{R}$。然而，许多概念可以推广到 $X$ 和 $Y$ 取值于 $\mathbb{R}^n$ 的情况。

通常，取值于 $\mathbb{R}^n$ 的随机变量被称为向量随机变量。我们经常使用的向量随机变量是 $Z = (X, Y)$，它取值于 $\mathbb{R}^2$。

在数据处理中，我们只需要数据 $\mathtt{X} = (\mathtt{X}_i)$ 和 $\mathtt{Y} = (\mathtt{Y}_i)$，它们是 $X$ 和 $Y$ 的独立抽样。我们用类似于编程语言的字体来表示它们，因为在实际操作中，我们是用数据进行计算的。

**备注**：统计学家将“样本”称为随机变量 $X, Y$ 的一系列独立拷贝。数据科学家（像我们）主要需要的是“数据”：即样本的“实现”。

### 具体示例 1

```python
def simulate_concrete_1(size):
    X = np.random.normal(size=size)
    Y = X * np.random.exponential(size=size)
    return X, Y

X, Y = simulate_concrete_1(5000)
# 这是我们的实现:
X, Y
```

使用 `pairplot` 或者 $\mathtt{X}, \mathtt{Y}$ 的直方图可以对 $X, Y$ 的分布有一个大致的了解。

```python
def plot_simu(X, Y):
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey="all", figsize=(10,5))
    ax0.set_xlabel("X")
    ax1.set_xlabel("X")
    ax0.set_ylabel("Y")
    ax0.scatter(X, Y, s=3, alpha=0.2, linewidths=0)
    ax1.hist2d(X, Y, bins=30)
plot_simu(X, Y)
```

在这个例子中，我们可以看到 $X, Y$ 之间有相互影响，但并没有因果关系：不存在函数 $f$ 使得 $Y = f(X)$ 或者 $X = f(Y)$。

本课程的目标是量化这种影响，特别是通过相关性的概念。

### 日常语言中的相关性

在日常语言中，两个现象“相关”是指它们有同时发生的趋势。例如，雨水和河流泛滥是相关的。

当我们可以用两个分数量化这些现象时，相关性意味着这两个分数有同时变大，或同时变小的趋势。例如：降雨量与河流水位相关。

有时也称为反相关：干旱天数与河流水位反相关。

数学家更倾向于称之为正相关和负相关，因为他们发明了一个系数，当现象相关时该系数为正，反相关时为负。

### 不要混淆

注意不要将“相关性”和“因果性”混淆。这里有一个小故事，你可以在周围讲述：

“我们对一所小学所有孩子从一年级到五年级进行了数学测试。通过观察测试结果，校长得出结论：脚越大，数学成绩越好。”

#### ♡

***任务：*** 校长并不是笨蛋，他只是将因果关系和相关性混淆了。解释原因。

**回答：**

校长观察到脚的大小与数学成绩之间存在相关性，但这并不意味着脚的大小导致了数学成绩的提高。实际上，这两者可能都受到另一个变量（如年龄或身体发育）的影响。因此，相关性不等于因果关系。

## 联合分布与条件分布

### 联合分布的定义

$X$ 和 $Y$ 的分布是定义在实数集上的测度：

$$
\begin{align}
\text{Loi}_X[A] &= \mathbf{P}[X \in A] \\
\text{Loi}_Y[B] &= \mathbf{P}[Y \in B]
\end{align}
$$

但这两个分布没有包含 $X$ 和 $Y$ 之间相互作用的信息。联合分布包含了这些信息：

$$
\forall C \subset \mathbb{R}^2 \quad \text{Loi}_{X,Y}[C] = \mathbf{P}[(X,Y) \in C]
$$

特别是当 $C = A \times B$ 时：

$$
\text{Loi}_{X,Y}[A \times B] = \mathbf{P}[X \in A, Y \in B]
$$

（概率中的逗号表示“与”）。注意，联合分布可以恢复边缘分布（也称为边际分布）：

$$
\begin{align}
\text{Loi}_X[A] &= \text{Loi}_{X,Y}[A \times \mathbb{R}] \\
\text{Loi}_Y[B] &= \text{Loi}_{X,Y}[\mathbb{R} \times B]
\end{align}
$$

### 无穷小记法

我们可以考虑 $\mathbb{R}^2$ 中的一个无穷小元素 $dx \times dy$，也记为 $dx\, dy$，并写为：

$$
\text{Loi}_{X,Y}(dx\, dy) = \mathbf{P}[X \in dx, Y \in dy]
$$

联合分布允许我们计算期望：对于任意函数 $\phi : \mathbb{R}^2 \to \mathbb{R}$：

$$
\mathbf{E}[\phi(X,Y)] = \int_{\mathbb{R}^2} \phi(x, y) \text{Loi}_{X,Y}(dx\, dy)
$$

当联合分布有密度函数 $f_{X,Y}$ 时，我们可以写成：

$$
\text{Loi}_{X,Y}(dx\, dy) = f_{X,Y}(x, y) \, dx \, dy
$$

因此：

$$
\mathbf{E}[\phi(X,Y)] = \int_{\mathbb{R}^2} \phi(x, y) f_{X,Y}(x, y) \, dx \, dy
$$

### 条件分布的定义

为了更好地描述 $X$ 对 $Y$ 的影响，我们考虑在给定 $X=x$ 时，$Y$ 的分布：

$$
\text{Loi}_{Y|X=x}(dy)
$$

如符号所示，这是关于 $dy$ 的测度，是关于 $x$ 的函数。它允许我们通过“重建方程”恢复联合分布：

$$
\text{Loi}_{X,Y}(dx\, dy) = \text{Loi}_{Y|X=x}(dy) \cdot \text{Loi}_X(dx)
$$

另一种记法：

$$
\mathbf{P}[X \in dx, Y \in dy] = \mathbf{P}[Y \in dy \mid X=x] \cdot \mathbf{P}[X \in dx]
$$

但要注意，不能简单地写成：

$$
\mathbf{P}[Y \in dy \mid X=x] = \frac{\mathbf{P}[X \in dx, Y \in dy]}{\mathbf{P}[X \in dx]}
$$

因为测度不能直接相除（或者需要明确除法的意义）。

但在联合分布有密度函数 $f_{X,Y}$ 的特殊情况下，边缘分布也有密度函数，特别是 $X$ 的密度函数：

$$
f_X(x) = \int f_{X,Y}(x, y) \, dy
$$

条件分布的密度函数为：

$$
\text{Loi}_{Y|X=x}(dy) = \frac{f_{X,Y}(x, y)}{f_X(x)} \, dy
$$

（分函数是有意义的，并且在这个等式中，约定 $\frac{0}{0} = 0$ 合理地解决了除以零的问题）。

**技术点**：条件分布不完全唯一。因为如果我们考虑一个变体 $\text{Loi2}_{Y|X=x}$，它与 $\text{Loi}_{Y|X=x}$ 仅在 $x$ 的一个 $Loi_X$ 忽略的集合上不同，那么重建方程仍然有效，因此 $\text{Loi2}_{Y|X=x}$ 也与 $\text{Loi}_{Y|X=x}$ 一样合法。

### 具体示例 2

我们模拟一对随机变量 $X,Y$ 如下：

```python
def simulate_concrete_2(size):
    X = np.random.uniform(0, 1, size=size)
    Y = np.random.normal(loc=X, scale=np.sqrt(X))
    return X, Y

X, Y = simulate_concrete_2(50_000)
plot_simu(X, Y)
```

* $X$ 服从 $[0,1]$ 上的均匀分布
* 给定 $X=x$ 时，随机变量 $Y$ 服从均值为 $x$，标准差为 $\sqrt{x}$ 的正态分布。

用数学符号重新表示：

$$
\text{Loi}_X(dx) = 1_{[0,1]}(x) \, dx
$$

$$
\text{Loi}_{Y|X=x}(dy) = \frac{1}{\sqrt{2\pi x}} \exp\left(-\frac{1}{2} \frac{(y - x)^2}{x}\right) dy
$$

因此：

$$
\text{Loi}_{X,Y}(dx\, dy) = \frac{1}{\sqrt{2\pi x}} \exp\left(-\frac{1}{2} \frac{(y - x)^2}{x}\right) 1_{[0,1]}(x) \, dx \, dy
$$

#### ♡♡

对于具体示例 1，我们有：

```python
X = np.random.normal(size=size)
Y = X * np.random.exponential(size=size)
```

所以用数学符号表示：

$$
\text{Loi}_X(dx) = \frac{1}{\sqrt{2\pi}} \color{red}{\text{N(0,1)}}
$$

如果 $x > 0$ 时：

$$
\text{Loi}_{Y|X=x}(dy) = \frac{1}{x} \color{red}{\text{Exponential}(\frac{1}{x})} \cdot 1_{y > 0}
$$

如果 $x < 0$ 时：

$$
\text{Loi}_{Y|X=x}(dy) = \frac{1}{|x|} \color{red}{\text{Exponential}(\frac{1}{|x|})} \cdot 1_{y < 0}
$$

不需要处理 $x=0$ 的情况，因为它在 $Loi_X$ 下是可以忽略的。

### 层叠条件

注意，在具体示例 2 中，我们是通过条件分布构建 $(X,Y)$ 的联合分布。这种情况在建模中非常常见，甚至经常有多层条件。

* 这里是一个两层条件的例子：我们想模拟一种鱼类的大小：设 $X$ 为水温。假设水温服从参数为 1 的指数分布。然后给定 $X=x$，鱼的大小服从均值为 $(x+10)^2$，标准差为 $\sqrt{x}$ 的正态分布。

* 这里是一个三层条件的例子：仍然是鱼的大小。设 $P$ 为鱼生活的水深的随机变量。$P$ 服从参数为 1 的伽玛分布。然后给定 $P=p$，水温服从参数为 $p$ 的指数分布。然后给定 $X=x$，鱼的大小服从均值为 $(x+10)^2$，标准差为 $\sqrt{x}$ 的正态分布。

* 这里是一个四层条件的例子：设 $O$ 为描述鱼所在海洋的离散随机变量：如果鱼在大西洋生活，$O=1$；如果在太平洋，$O=2$ 等等。假设 $O=k$ 的概率与海洋的大小成正比。然后给定 $O=k$，水深 $P$ 服从一个依赖于 $k$ 的伽玛分布。接下来，描述水温的条件分布，最后描述鱼的大小的条件分布。

## 独立性

### 定义与初步特征

> **定义**：当且仅当

$$
\forall A, B \quad \mathbf{P}[X \in A, Y \in B] = \mathbf{P}[X \in A] \cdot \mathbf{P}[Y \in B]
$$

时，我们说 $X$ 和 $Y$ 是独立的。

**备注**：与非概率学家交流时，最好说“统计独立”，因为“独立”这个词有很多含义，包括在数学中，例如“线性独立”。

上面的等式也可以写成：

$$
\text{Loi}_{X,Y}(dx\, dy) = \text{Loi}_X(dx) \cdot \text{Loi}_Y(dy)
$$

当联合分布有密度函数 $f_{X,Y}$ 时，这等价于函数可以写成：

$$
f_{X,Y}(x,y) = g(x) \cdot h(y)
$$

并且进行归一化：

* $\frac{g}{\int g}$ 必然是 $X$ 的密度函数
* $\frac{h}{\int h}$ 必然是 $Y$ 的密度函数

下面是独立性的最经典表征：能够“分解”期望值。

> **命题**：$X$ 和 $Y$ 是独立的，当且仅当

$$
\forall \phi, \psi \quad \mathbf{E}[\phi(X)\psi(Y)] = \mathbf{E}[\phi(X)] \cdot \mathbf{E}[\psi(Y)]
$$

### 通过条件分布表征

我觉得你们对上一节的独立性表征已经很熟悉了。但接下来的内容较少人知道：

> **命题**：以下三个点是等价的：
> * $X$ 和 $Y$ 是独立的
> * $\text{Loi}_{Y|X=x}(dy)$ 不依赖于 $x$（总是相同的测度）
> * $\text{Loi}_{X|Y=y}(dx)$ 不依赖于 $y$（总是相同的测度）

在我看来，这是理解概率独立性的好方法：$X$ 取某个值 $x$ 或 $x'$ 不影响 $Y$ 落在 $dy$ 或 $dy'$ 的概率。同样地，反过来也如此。

#### ♡♡

***任务***：在 $X, Y$ 有密度函数 $f_{X,Y}$ 的特例中证明这个命题。

**答案**：

假设 $X$ 和 $Y$ 有联合密度函数 $f_{X,Y}(x, y)$，以及边缘密度函数 $f_X(x)$ 和 $f_Y(y)$。

1. **如果 $X$ 和 $Y$ 独立**，则：

$$
f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y)
$$

因此，条件密度为：

$$
f_{Y|X=x}(y) = \frac{f_{X,Y}(x,y)}{f_X(x)} = f_Y(y)
$$

这说明 $\text{Loi}_{Y|X=x}(dy)$ 不依赖于 $x$，即独立。

2. **反之，如果 $\text{Loi}_{Y|X=x}(dy)$ 不依赖于 $x$**，即 $f_{Y|X=x}(y) = f_Y(y)$，那么：

$$
f_{X,Y}(x,y) = f_{Y|X=x}(y) \cdot f_X(x) = f_Y(y) \cdot f_X(x)
$$

这意味着 $X$ 和 $Y$ 独立。

因此，这三个条件是等价的。

#### ♡

***任务***：在具体示例 1 和 2 中，如何立即看出 $X$ 和 $Y$ 之间没有独立性？

**答案**：

在具体示例 1 和 2 中，$Y$ 是通过 $X$ 计算得到的。例如，在示例 1 中，$Y = X \cdot \text{Exponential}$，因此 $Y$ 明显依赖于 $X$。同样，在示例 2 中，$Y$ 的分布依赖于 $X$ 的值。因此，$X$ 和 $Y$ 之间不独立。

### 与因果性的关系

我们说有 $X$ 对 $Y$ 的因果关系，当且仅当存在函数 $f$ 使得 $Y = f(X)$。直观上，这会阻止 $X$ 和 $Y$ 独立。

> **命题**：设 $X, Y$ 是两个非恒定的随机变量。如果存在 $X$ 对 $Y$ 或者 $Y$ 对 $X$ 的因果关系，那么它们不独立。

***证明***：例如，假设 $Y = f(X)$，则：

$$
\text{Loi}_{Y|X=x}(dy) = \delta_{f(x)}(dy)
$$

其中 $\delta$ 是狄拉克测度。这个量随 $x$ 变化，因此不独立。相同的证明方式适用于 $X = f(Y)$。

**备注**：一个常数随机变量与任何其他随机变量都是独立的。在上面的证明中，我们确实使用了 $X$ 和 $Y$ 不是常数的事实。

### 非因果且不独立

但要注意：不是因为没有因果关系，就意味着独立。

一个反例很容易构造，通过观察联合分布的支撑集：$X$ 对 $Y$ 的因果关系等价于联合分布的支撑集是某个函数的图像。反之：如果 $X$ 和 $Y$ 独立，那么联合分布的支撑集是一个矩形（可能是无界的）。因此，在支撑集既不是图像，也不是矩形的所有情况下，都存在非因果且不独立的情况。

## 协方差

### 定义

> **定义**：设 $X$ 和 $Y$ 是两个随机变量。它们的协方差定义为：

$$
\text{cov}(X,Y) = \mathbf{E}[(X - \mathbf{E}(X))(Y - \mathbf{E}(Y))]
$$

***任务***：通过展开上面期望中的乘积，有一个非常实用的第二种计算协方差的公式。

#### ♡

$$
\text{cov}(X,Y) = \mathbf{E}[XY] - \mathbf{E}[X] \cdot \mathbf{E}[Y]
$$

### 独立 ⇒ 不相关

#### ♡

> **命题**：如果 $X$ 和 $Y$ 独立，则

$$
\text{cov}(X,Y) = 0
$$

### 不相关不意味着独立

反例：考虑 $\mathbb{R}^2$ 中任何关于 x 轴对称的图形 $F$。设 $X, Y$ 是均匀在 $F$ 上取点的随机变量。则 $\text{cov}(X,Y) = 0$。

这是因为对称性意味着 $(X, Y)$ 和 $(-X, Y)$ 的分布相同。利用协方差的双线性性质：

$$
\text{cov}(X,Y) = \text{cov}(-X,Y) = -\text{cov}(X,Y) \implies \text{cov}(X,Y) = 0
$$

然而，如果 $F$ 不是矩形，则 $X$ 和 $Y$ 不独立。

```python
def simulate_one_unif_in_F():
    go_on = True
    while go_on:
        X = np.random.uniform(-1.5, 1.5)
        Y = np.random.uniform(-2.5, 2.5)
        go_on = -1 < X < 1 and -1 < Y < 1
    return X, Y

def simulate_several_unif_in_F(size):
    Xs, Ys = [], []
    for _ in range(size):
        X, Y = simulate_one_unif_in_F()
        Xs.append(X)
        Ys.append(Y)
    return np.array(Xs), np.array(Ys)

X, Y = simulate_several_unif_in_F(10)
X, Y

# 向量化版本，更快，但样本大小是随机的
def simulate_unif_in_F(n):
    X = np.random.uniform(-1.5, 1.5, size=n)
    Y = np.random.uniform(-2.5, 2.5, size=n)
    in_center = (-1 < X) & (X < 1) & (-1 < Y) & (Y < 1)
    not_in_center = ~in_center

    return X[not_in_center], Y[not_in_center]

X, Y = simulate_unif_in_F(20_000)
plot_simu(X, Y)
```

#### ♡♡♡

***任务***：如果图形 $F$ 关于 y 轴对称呢？或者关于中心对称呢？

**回答：**

1. **关于 y 轴对称**：

   如果图形 $F$ 关于 y 轴对称，那么对于每一个 $(x, y) \in F$，都有 $(-x, y) \in F$。这意味着 $X$ 和 $-X$ 在分布上是对称的，而 $Y$ 的分布独立于 $X$。由于对称性，$\text{cov}(X,Y)$ 仍然为零，因为 $E[XY] = E[-X Y]$，所以 $E[XY] = 0$。但是，$X$ 和 $Y$ 不独立，除非 $F$ 是矩形。

2. **中心对称**：

   如果图形 $F$ 关于原点中心对称，即对于每一个 $(x, y) \in F$，都有 $(-x, -y) \in F$，那么同样地，$\text{cov}(X,Y) = 0$。这是因为 $E[XY] = E[(-X)(-Y)] = E[XY]$，所以 $E[XY] = 0$。但同样，$X$ 和 $Y$ 不独立，除非 $F$ 是矩形。

### 协方差矩阵

**定义**：设 $Z$ 是一个取值于 $\mathbb{R}^n$ 的向量随机变量。$Z = (Z_i)$ 的协方差矩阵是矩阵：

$$
\text{cov}(Z)_{i,j} = \text{cov}(Z_i, Z_j)
$$

特别地，对角线上的元素：

$$
\text{cov}(Z)_{i,i} = \text{var}(Z_i)
$$

## 协方差的估计

### 手动编程

* $\mathbf{E}[X]$ 可以用 `mean(X)`（数据 $X$ 的平均值）来估计。
* $\mathbf{E}[Y]$ 可以用 `mean(Y)` 来估计。

协方差 $\text{cov}(X,Y)$ 也可以用数据来估计：

$$
\text{cov}(X,Y) = \frac{1}{n-1} \sum_i (X_i - \text{mean}(X)) (Y_i - \text{mean}(Y))
$$

特别地，$\text{cov}(X,X) = \text{std}^2(X)$ 是方差的经典估计量。

请始终区分从观测数据估计的量和理论量。如果在书写时没有使用不同的字体，可以用波浪线或帽子符号来表示估计量：$\tilde{\text{cov}}$, $\hat{\text{cov}}$ 以区分真实的协方差 $\text{cov}$。

我们实现一个函数来返回协方差的估计：

```python
def cov(X, Y):
    n = len(X)
    assert n == len(Y)
    X = X - np.mean(X)
    Y = Y - np.mean(Y)
    return X @ Y / (n - 1)

def test():
    np.random.seed(1234)  # 为了每次结果相同
    X, Y = simulate_unif_in_F(100_000)
    print(cov(X, Y))

test()
```

在测试中，我们得到一个接近零的值……但不完全为零：这只是一个估计值！

#### ♡

***任务***：为什么下面这个变体是个很糟糕的主意：

```python
def cov(X, Y):
    n = len(X)
    assert n == len(Y)
    X -= np.mean(X)
    Y -= np.mean(Y)
    return X @ Y / (n - 1)
```

**回答**：

因为 `-=` 是一种就地操作，会修改输入的 `X` 和 `Y` 数据。这意味着用户传入的数据会被改变，这是非常不好的，因为函数的行为不应有副作用。正确的做法是创建新的变量，而不是直接修改输入数据。

### 在具体示例 2 上

```python
X, Y = simulate_concrete_2(10_000)
plot_simu(X, Y)
```

看起来 $X$ 和 $Y$ 有轻微的正相关。我们通过估计验证一下：

```python
cov(X, Y)
```

另一种计算方式：考虑向量随机变量 $Z = (X, Y)$，用 numpy 计算它的协方差矩阵，然后取其中一个非对角元素。

```python
def cov_np(X, Y):
    XY = np.stack([X, Y], axis=0)
    return np.cov(XY, ddof=1)[0,1]
cov_np(X, Y)
```

## 精确计算协方差矩阵

当我们处理现实生活中的数据时，通常无法获得它们的分布，因此只能有协方差、方差、期望的估计值。

但在我们的两个具体示例中，$(X, Y)$ 有一个简单明确的联合分布。我们可以进行理论计算。这里我们关注具体示例 2：

* $X$ 服从 $[0,1]$ 上的均匀分布
* 给定 $X = x$ 时，随机变量 $Y$ 服从均值为 $x$，标准差为 $\sqrt{x}$ 的正态分布。

### 协方差的计算

#### ♡♡♡

* 计算 $\mathbf{E}[Y]$。给定 $X = x$，$Y$ 的期望是 $x$，所以：

$$
\mathbf{E}[Y \mid X = x] = x
$$

---

通过积分，我们有：

$$
\mathbf{E}[Y] = \int \mathbf{E}[Y \mid X = x] \cdot \mathbf{P}[X \in dx] = \int_0^1 x \cdot 1 \, dx = \frac{1}{2}
$$

---

* 计算 $\text{cov}(X,Y)$。

$$
\mathbf{E}[XY \mid X = x] = \mathbf{E}[xY \mid X = x] = x \cdot \mathbf{E}[Y \mid X = x] = x \cdot x = x^2
$$

---

通过积分：

$$
\mathbf{E}[XY] = \int \mathbf{E}[XY \mid X = x] \cdot \mathbf{P}[X \in dx] = \int_0^1 x^2 \cdot 1 \, dx = \frac{1}{3}
$$

---

最后，

$$
\text{cov}(X,Y) = \mathbf{E}[XY] - \mathbf{E}[X] \cdot \mathbf{E}[Y] = \frac{1}{3} - \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}
$$

最终结果以最简分数形式表示：

$$
\text{cov}(X,Y) = \frac{1}{12}
$$

### 一个经典的错误

注意经典错误：随机变量的方差不能通过积分条件方差来获得。特别是对于我们的例子：

$$
\text{var}(Y) \neq \int \mathbf{E}[(Y - \mathbf{E}[Y \mid X = x])^2 \mid X = x] \cdot \mathbf{P}[X \in dx] = \int_0^1 x \, dx
$$

为了理解为什么这不起作用，我们展开右边的项，得到：

$$
\int (\mathbf{E}[Y^2 \mid X = x] - \mathbf{E}[Y \mid X = x]^2) \cdot \mathbf{P}[X \in dx] = \mathbf{E}[Y^2] - \int \mathbf{E}[Y \mid X = x]^2 \cdot \mathbf{P}[X \in dx]
$$

重建方程无法简化最后一项。

### 方差的计算

#### ♡♡♡

* $X$ 的方差是 $\frac{1}{12}$，因为 $[a, b]$ 上均匀分布的随机变量的方差是：

$$
\text{var}(X) = \frac{(b - a)^2}{12}
$$

在这里 $a = 0$, $b = 1$，所以：

$$
\text{var}(X) = \frac{1}{12}
$$

---

* 计算 $Y$ 的方差：对于一个服从 $N(a, \sigma^2)$ 的随机变量 $K$，我们有：

$$
\mathbf{E}[K^2] = \sigma^2 + a^2
$$

在我们的例子中，给定 $X = x$，$Y$ 服从均值为 $x$，标准差为 $\sqrt{x}$ 的正态分布，因此：

$$
\mathbf{E}[Y^2 \mid X = x] = x + x^2
$$

通过积分：

$$
\mathbf{E}[Y^2] = \int_0^1 (x + x^2) \cdot 1 \, dx = \frac{1}{2} + \frac{1}{3} = \frac{5}{6}
$$

因此，

$$
\text{var}(Y) = \mathbf{E}[Y^2] - (\mathbf{E}[Y])^2 = \frac{5}{6} - \left(\frac{1}{2}\right)^2 = \frac{5}{6} - \frac{1}{4} = \frac{10}{12} - \frac{3}{12} = \frac{7}{12}
$$

...

比较真实的协方差矩阵和它的估计值：

```python
true_cov = np.array([[1/12, 1/12],
                     [1/12, 7/12]])
true_cov
```

```python
#--- To keep following outputs, do not run this cell! ---

# 比较估计值：

np.random.seed(134)
# 我们抽取大量数据以获得良好的估计
X, Y = simulate_concrete_2(100_000)
estimate_cov = np.cov(np.stack([X, Y], axis=0), ddof=1)
estimate_cov
```

## 皮尔逊相关系数和斯皮尔曼相关系数

### 皮尔逊相关系数

> **定义**：皮尔逊相关系数定义为：

$$
\text{cor}(X, Y) = \frac{\text{cov}(X,Y)}{\sqrt{\text{var}(X)} \cdot \sqrt{\text{var}(Y)}}
$$

注意这是协方差的简单归一化。这个相关系数用于表示 $X$ 和 $Y$ 之间的关系，而不受它们的量级影响。特别地：

> **命题**：对于任意非零常数 $a, b$，有：

$$
\text{cor}(aX, bY) = \text{sign}(a) \cdot \text{sign}(b) \cdot \text{cor}(X, Y)
$$

#### ♡♡♡

***任务***：证明这个命题，并在证明过程中，你会发现这个命题成立需要一个非常重要的假设。修正这个命题。

**回答：**

证明：

$$
\text{cor}(aX, bY) = \frac{\text{cov}(aX, bY)}{\sqrt{\text{var}(aX)} \cdot \sqrt{\text{var}(bY)}}} = \frac{a b \cdot \text{cov}(X,Y)}{\sqrt{a^2 \cdot \text{var}(X)} \cdot \sqrt{b^2 \cdot \text{var}(Y)}}} = \frac{a b \cdot \text{cov}(X,Y)}{|a| \cdot |b| \cdot \sqrt{\text{var}(X)} \cdot \sqrt{\text{var}(Y)}}} = \frac{a}{|a|} \cdot \frac{b}{|b|} \cdot \text{cor}(X, Y) = \text{sign}(a) \cdot \text{sign}(b) \cdot \text{cor}(X, Y)
$$

**缺少的假设**：$a$ 和 $b$ 必须是非零的。

因此，修正后的命题为：

**命题**：对于任意非零常数 $a, b$，有：

$$
\text{cor}(aX, bY) = \text{sign}(a) \cdot \text{sign}(b) \cdot \text{cor}(X, Y)
$$

### 皮尔逊相关系数总在 $[-1, 1]$ 之间

> **命题**：相关系数总是位于区间 $[-1, 1]$ 内。

#### ♡♡

***证明**：* 这是柯西-施瓦茨不等式的直接结果：

$$
|\text{cov}(X, Y)| \leq \sqrt{\text{var}(X)} \cdot \sqrt{\text{var}(Y)}
$$

因此，

$$
|\text{cor}(X,Y)| = \left| \frac{\text{cov}(X,Y)}{\sqrt{\text{var}(X)} \cdot \sqrt{\text{var}(Y)}} \right| \leq 1
$$

所以，$\text{cor}(X,Y)$ 总是在 $[-1,1]$ 之间。

### 回到具体示例，计算相关系数

```python
# 重新模拟示例 2
X, Y = simulate_concrete_2(50000)
Z = np.stack([X, Y], axis=0)

np.corrcoef(Z)  # 在这个函数中，没有 ddof 参数。算了。
```

`scipy` 也可以用来计算这个系数：

```python
scipy.stats.pearsonr(X, Y)
```

`scipy` 还给出了一个 p-value：这是关于该系数为零的 t 检验的 p 值。当 p-value 很小（小于 `1e-3`）时，这表明该系数显著地不同于零。

#### ♡

通过使用我们之前的理论计算，计算具体示例 2 的真实相关系数。

做一个小计算，给出相关系数的数值真实值。

```python
#--- To keep following outputs, do not run this cell! ---

# 真实的协方差是 1/12，方差是 1/12 和 7/12
true_cov = np.array([[1/12, 1/12],
                     [1/12, 7/12]])
print(true_cov)

# 相关系数 cor(X,Y) = cov(X,Y) / (sqrt(var(X)) * sqrt(var(Y))) = (1/12) / (sqrt(1/12) * sqrt(7/12)) = (1/12) / (sqrt(7)/12) = 1 / sqrt(7) ≈ 0.377964473
correlation = (1/12) / (np.sqrt(1/12) * np.sqrt(7/12))
print(correlation)  # 输出大约 0.378
```

一个约 37.8% 的相关性：这相当不错。$X,Y$ 强相关；这从协方差的 0.08 量级看不太出来。

### 极端相关系数

皮尔逊相关系数衡量了 $Y$ 与 $X$ 之间接近于线性关系的程度。其极端值 $+1$ 和 $-1$ 表示纯线性关系：

> **命题**：
> * 当 $\text{cor}(X,Y) = 1$ 时，$Y = aX + b$ 且 $a > 0$。
> * 当 $\text{cor}(X,Y) = -1$ 时，$Y = aX + b$ 且 $a < 0$。

***证明方向**：* 只证明一个方向：假设 $Y = aX + b$ 且 $a > 0$，则：

$$
\text{cov}(X,Y) = \text{cov}(X, aX + b) = a \cdot \text{cov}(X, X) = a \cdot \text{var}(X)
$$

而：

$$
\begin{align*}
\text{cor}(X, Y) 
&= \frac{\text{cov}(X, Y)}{\sqrt{\text{var}(X)} \cdot \sqrt{\text{var}(Y)}} \\
&= \frac{a \cdot \text{var}(X)}{\sqrt{\text{var}(X)} \cdot \sqrt{a^2 \cdot \text{var}(X)}} \\
&= \frac{a \cdot \text{var}(X)}{|a| \cdot \text{var}(X)} \\
&= 1
\end{align*}
$$

这个命题在另一个方向也成立。

这个命题对估计量也有类似的结果：设 $\mathtt{X}, \mathtt{Y}$ 是样本数据：

* 当 $\mathtt{\text{cov}(X,Y)} = \frac{1}{12}$ 时，$\forall i: \mathtt{Y_i} = a \mathtt{X_i} + b$ 且 $a > 0$。
* 当 $\mathtt{\text{cov}(X,Y)} = -\frac{1}{12}$ 时，$\forall i: \mathtt{Y_i} = a \mathtt{X_i} + b$ 且 $a < 0$。

### 斯皮尔曼相关系数

现在介绍斯皮尔曼相关系数 $\text{spear}(X,Y)$ 及其估计量 $\mathtt{\text{spear}(X,Y)}$。它们满足：

* 当 $\text{spear}(X,Y) = 1$ 时，$Y = f(X)$ 且 $f$ 是单调递增函数。
* 当 $\text{spear}(X,Y) = -1$ 时，$Y = f(X)$ 且 $f$ 是单调递减函数。

* $\mathtt{\text{spear}(X,Y)} = 1$ 当且仅当 $\forall i: \mathtt{Y_i} = f(\mathtt{X_i})$ 且 $f$ 是单调递增函数。
* $\mathtt{\text{spear}(X,Y)} = -1$ 当且仅当 $\forall i: \mathtt{Y_i} = f(\mathtt{X_i})$ 且 $f$ 是单调递减函数。

这怎么可能呢？很简单：斯皮尔曼相关系数是对排名计算的皮尔逊相关系数。

观察这个计算过程。

```python
size = 30

X = np.random.uniform(0, 10, size=100)
Y = np.random.normal(loc=np.exp(X), scale=X)

plt.plot(X, Y, ".")
```

计算皮尔逊相关系数：

```python
# 方法 1
XY = np.stack([X, Y], axis=0)
print(np.corrcoef(XY)[0,1])

# 方法 2：
scipy.stats.pearsonr(X, Y)[0]
```

我们看到这两个变量在法语意义上的相关性非常高。皮尔逊系数相当大（大约 70%），但从观察到的依赖性来看，我们希望相关系数更接近其最大值：1。斯皮尔曼相关系数就是这种情况。

观察观察值的排名：

```python
# X 的排名
X_ranks = scipy.stats.rankdata(X)
# Y 的排名
Y_ranks = scipy.stats.rankdata(Y)

plt.plot(X_ranks, Y_ranks, ".")
```

我们看到它们的排名高度线性相关。排名的皮尔逊系数（即斯皮尔曼相关系数）将返回一个非常接近 1 的数值。

我们实现斯皮尔曼相关系数：

```python
def compute_spearman(X, Y):
    X_ranks = scipy.stats.rankdata(X)
    Y_ranks = scipy.stats.rankdata(Y)
    XY_ranks = np.stack([X_ranks, Y_ranks], axis=0)
    return np.corrcoef(XY_ranks)[0,1]

my_speareman = compute_spearman(X, Y)
my_speareman

# 比较 scipy 的直接计算结果：
scipy.stats.spearmanr(X, Y)[0]
```

如果将 `[0]` 替换为 `[1]`，则返回斯皮尔曼相关系数的 p-value。

#### ♡♡♡

***任务***：做一些小测试，解释 `np.argsort` 和 `scipy.stats.rankdata` 之间的两个区别。提示：特别是当有相等值时。

**回答：**

1. **处理相等值的方式**：
   - `np.argsort` 返回的是排序后元素的索引，不处理相等值的排名问题。
   - `scipy.stats.rankdata` 在遇到相等值时，会为这些值分配平均排名（或其他方法，具体取决于 `method` 参数）。

2. **输出的类型**：
   - `np.argsort` 返回的是整数索引数组。
   - `scipy.stats.rankdata` 返回的是浮点数排名数组。

```python
# 示例代码
import numpy as np
import scipy.stats

data = [1, 2, 2, 3]
argsort = np.argsort(data)
rankdata = scipy.stats.rankdata(data)

print("np.argsort:", argsort)
print("scipy.stats.rankdata:", rankdata)
```

输出：

```
np.argsort: [0 1 2 3]
scipy.stats.rankdata: [1. 2.5 2.5 4.]
```

如上所示，`scipy.stats.rankdata` 为相等的值分配了平均排名，而 `np.argsort` 只是返回了排序后的索引。

```python
plt.plot(X_ranks, Y_ranks, ".")
```

我们看到它们的排名高度线性相关。排名的皮尔逊系数（即斯皮尔曼相关系数）将返回一个非常接近 1 的数值。

```python
def compute_spearman(X, Y):
    X_ranks = scipy.stats.rankdata(X)
    Y_ranks = scipy.stats.rankdata(Y)
    XY_ranks = np.stack([X_ranks, Y_ranks], axis=0)
    return np.corrcoef(XY_ranks)[0,1]

my_speareman = compute_spearman(X, Y)
my_speareman

# 比较 scipy 的直接计算结果：
scipy.stats.spearmanr(X, Y)[0]
```

如果将 `[0]` 替换为 `[1]`，则返回斯皮尔曼相关系数的 p-value。

## 回归直线和主成分分析（PCA）

### 线性回归

我们寻找最优拟合数据的线性回归直线，因此要最小化：

$$
\hat{a}, \hat{b} = \text{argmin}_{a,b} \sum_i (\mathtt{Y_i} - a \mathtt{X_i} - b)^2
$$

> **命题**：有：

$$
\hat{a} = \mathtt{\frac{\text{cov}(X,Y)}{\text{var}(X)}}
$$
$$
\hat{b} = \mathtt{\text{mean}(\mathtt{Y}) - \hat{a} \cdot \text{mean}(\mathtt{X})}
$$

***证明***：如果我们平移数据，就会平移回归直线（只是原点的变化）。设：

$$
\text{loss}(\mathtt{X}, \mathtt{Y}) = \sum_i (\mathtt{Y_i} - a \mathtt{X_i} - b)^2
$$

如果 $\hat{a}, \hat{b}$ 是最小化这个损失的参数，那么对于一个常数 $k$，最小化 $\text{loss}(\mathtt{X} - k, \mathtt{Y})$ 的参数是 $\hat{a}$ 和 $\hat{b} - \hat{a} k$。

取 $k = \text{mean}(\mathtt{X})$，并从 $\mathtt{X}$ 中减去这个量，我们可以假设 $\sum_i \mathtt{X_i} = 0$，这简化了计算。

设 $n$ 为数据大小。设 $\mathbf{X}$ 为矩阵，第一列是 1，第二列是 $\mathtt{X_i}$。设 $\mathbf{Y}$ 为列向量，包含 $\mathtt{Y_i}$。正规方程给出：

$$
\begin{pmatrix}
\hat{b} \\
\hat{a}
\end{pmatrix}
= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
$$

矩阵 $\mathbf{X}^T \mathbf{X}$ 为：

$$
\begin{pmatrix}
n & \sum_i \mathtt{X_i} \\
\sum_i \mathtt{X_i} & \sum_i \mathtt{X_i}^2
\end{pmatrix}
=
\begin{pmatrix}
n & 0 \\
0 & \sum_i \mathtt{X_i}^2
\end{pmatrix}
$$

其逆矩阵易于计算。因此，我们有：

$$
\begin{pmatrix}
\hat{b} \\
\hat{a}
\end{pmatrix}
=
\begin{pmatrix}
\frac{1}{n} \sum_i \mathtt{Y_i} \\
\frac{\sum_i \mathtt{X_i} \mathtt{Y_i}}{\sum_i \mathtt{X_i}^2}
\end{pmatrix}
=
\begin{pmatrix}
\mathtt{\text{mean}(\mathtt{Y})} \\
\frac{\mathtt{\text{cov}(X,Y)}}{\mathtt{\text{var}(X)}}
\end{pmatrix}
$$

证明在 $\sum_i \mathtt{X_i} = 0$ 的情况下完成。一般情况则通过讨论平移推导。

$\square$

### 向量回归直线

通常，我们希望通过最简单的方式解释 $X$ 对 $Y$ 的影响：例如对于公寓，我们谈论“每平方米价格”（$X$=面积，$Y$=价格）。因此，我们寻找参数：

$$
\hat{c} = \text{argmin}_c \sum_i (\mathtt{Y_i} - c \mathtt{X_i})^2
$$

> **命题**：

$$
\hat{c} = \mathtt{\frac{\sum_i \mathtt{X_i} \mathtt{Y_i}}{\sum_i \mathtt{X_i}^2}}
$$

**备注**：当数据居中时，回归直线和向量回归直线重合。

#### ♡♡♡

***任务***：使用正规方程证明这个命题。但这次，矩阵 $\mathbf{X}^T \mathbf{X}$ 是 $1 \times 1$ 的，非常简单！

**回答：**

设 $c$ 为参数。目标函数为：

$$
\text{loss}(c) = \sum_i (\mathtt{Y_i} - c \mathtt{X_i})^2
$$

取导数并令其等于零：

$$
\frac{d}{dc} \text{loss}(c) = -2 \sum_i \mathtt{X_i} (\mathtt{Y_i} - c \mathtt{X_i}) = 0
$$

解得：

$$
c = \frac{\sum_i \mathtt{X_i} \mathtt{Y_i}}{\sum_i \mathtt{X_i}^2}
$$

这就是所求的 $\hat{c}$。

### PCA 的回归直线

这是关于主成分分析（PCA）的所有必要知识。我们将在二维情况下使用这个结果来处理 $\mathtt{Z} = (X, Y)$，但以下内容在任何维度都成立。

* 协方差矩阵 $\mathtt{\text{cov}(Z)}$ 是对称的半正定的，因此可以在正交基底下对角化。
* 它的特征值是正的。我们按降序排列特征值：$\lambda_0$ 是最大的。设 $U_0$ 是对应的特征向量。
* 通过数据质心并由 $U_0$ 生成的直线，给出了数据云的“主要方向”：在所有直线中，它是最小化与点距离的直线。
* 逻辑结果：当我们将点投影到这条直线上时，比起投影到其他直线，投影的方差最大。
* 点投影的方差是 $\lambda_0$。

技术上，`svd` 函数可以通过有序的变化矩阵进行对角化。

```python
X, Y = simulate_concrete_1(5000)
cov_mat = np.cov(np.stack([X, Y], axis=0), ddof=1)
cov_mat

def compute_m_with_svd(X, Y):
    cov_mat = np.cov(np.stack([X, Y], axis=0), ddof=1)

    # 向量 S 包含有序的特征值，从大到小
    # U 是变化矩阵。U 的列按照与最大特征值对应的顺序排列
    U, S, _ = np.linalg.svd(cov_mat)
    U0 = U[:,0]
    m = U0[1] / U0[0]
    return m

compute_m_with_svd(X, Y)
```

```python
def plot_both_line(X, Y):
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)

    a = cov(X, Y) / np.std(X)**2
    b = np.mean(Y) - a * np.mean(X)
    c = sum(X * Y) / sum(X**2)

    m = compute_m_with_svd(X, Y)

    fig, ax = plt.subplots()
    ax.scatter(X, Y, s=3, label="数据点")
    left, right = min(X), max(X)
    xx = np.linspace(left, right, 2)

    ax.plot(xx, a * xx + b, "r", label="线性回归")
    ax.plot(xx, c * xx, "b", label="向量回归")
    ax.legend()
```

```python
# 修改数据，使回归直线与原点不重合
X, Y = simulate_concrete_2(500)
X += 2
Y -= -3
plot_both_line(X, Y)

X, Y = simulate_concrete_2(500)
X += 4
Y -= 6
plot_both_line(X, Y)
```

对于这些数据，向量回归直线看起来不太理想。

### 主成分分析（PCA）的回归直线

我们如何选择回归直线呢？

主成分分析（PCA）给出的回归直线是数据点“最接近”的直线：如果你将所有点垂直投影到这条直线上，得到的距离要比投影到其他直线更小。

然而，线性回归直线允许我们近似一个因果关系的噪声 $Y = f(X) + \text{噪声}$。在最后一个绘制的例子中，我们更倾向于使用线性回归直线：从图上看，我们可以直观地想象一个因果关系 $Y = f(X) + \text{噪声}$，且噪声随着 $X$ 增大而增大。

### 一个错误的解释示例

```python
loi = scipy.stats.multivariate_normal([0., 0.], [[2.0, 0.3], [0.3, 1.5]])

Z = abs(loi.rvs(size=1000) + 3)
Z.shape

# Z 是一个矩阵，包含代表一个人的数学水平的分数。
# 每一行对应一个夫妇的成绩。第一列代表男性的分数，第二列代表女性的分数。

plot_simu(Z[:,0], Z[:,1])
```

记 $\Delta$ 为直线 $y = x$（第一对角线）。

* 在 $\Delta$ 上的点对应于夫妇中两人的数学水平相同。
* 在 $\Delta$ 上方的点对应于女性更强。
* 在 $\Delta$ 下方的点对应于男性更强。

看起来绘制向量回归直线并将其与 $\Delta$ 比较是有意义的。

```python
def plot_delta_and_regression(X, Y, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.scatter(X, Y, s=3)
    c = np.sum(X * Y) / np.sum(X**2)
    xx = np.linspace(min(X), max(X), 10)
    ax.plot(xx, c * xx, "r", label="向量回归")
    ax.plot(xx, xx, "k:", label="Δ")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_aspect("equal")

plot_delta_and_regression(Z[:,0], Z[:,1], "男性", "女性")
```

图表说明了一切：红色直线在 $\Delta$ 下方，因此男性在数学上大多数情况下更强。

但如果我们将女性作为横轴，男性作为纵轴：

```python
plot_delta_and_regression(Z[:,1], Z[:,0], "女性", "男性")
```

结果令人意外，直线仍然在 $\Delta$ 下方，表明女性在多数情况下更强。

这种表面上的悖论来源于向量回归直线仅最小化垂直距离，而不是对称的。这种方法对 $X$ 和 $Y$ 位置不对称。

### 正确的直线

我们使用一种技巧，强制 PCA 直线通过原点。

```python
def plot_delta_and_acp_from_0(X, Y, xlabel, ylabel):
    fig, ax = plt.subplots()
    XX = np.append(X, -X)
    YY = np.append(Y, -Y)
    mm = compute_m_with_svd(XX, YY)

    ax.scatter(X, Y, s=3)

    xx = np.linspace(min(X), max(X), 10)
    ax.plot(xx, mm * xx, "r", label="PCA 直线")
    ax.plot(xx, xx, "k:", label="Δ")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_aspect("equal")

plot_delta_and_acp_from_0(Z[:,0], Z[:,1], "男性", "女性")
plot_delta_and_acp_from_0(Z[:,1], Z[:,0], "女性", "男性")
```

PCA 直线（红色）接近 $\Delta$。没有明显差异。

**注意**：如果数据分布不接近直线，使用直线来总结会有风险。

## 一些有趣的数学计算

### 两个二元变量

假设 $X$ 和 $Y$ 取值于 $\{0,1\}$，因此 $X = 1_{X=1}$，所以：

$$
\mathbf{E}[X] = \mathbf{P}[X=1] := p_X
$$

$$
\text{var}(X) = p_X(1 - p_X)
$$

$Y$ 同理。

为了理解 $X$ 对 $Y$ 的影响，自然比较 $\mathbf{P}[Y=1 \mid X=1]$ 和 $\mathbf{P}[Y=1]$。因此，自然计算比率：

$$
\frac{\mathbf{P}[Y=1 \mid X=1]}{\mathbf{P}[Y=1]}
$$

> **命题**：

$$
\frac{\mathbf{P}[Y=1 \mid X=1]}{\mathbf{P}[Y=1]} = \frac{\text{cov}(X,Y)}{p_X p_Y} + 1 = \text{cor}(X,Y) \cdot \frac{\sqrt{(1 - p_X)(1 - p_Y)}}{\sqrt{p_X p_Y}} + 1
$$

这个公式是很自然的。

#### ♡♡

***任务***：证明这个命题。

**证明**：

我们有：

$$
\text{cov}(X,Y) = \mathbf{E}[XY] - \mathbf{E}[X] \cdot \mathbf{E}[Y] = \mathbf{P}[X=1, Y=1] - p_X p_Y
$$

因为 $X$ 和 $Y$ 只取 0 和 1：

$$
\mathbf{P}[X=1, Y=1] = \mathbf{P}[Y=1 \mid X=1] \cdot p_X
$$

因此，

$$
\text{cov}(X,Y) = \mathbf{P}[Y=1 \mid X=1] \cdot p_X - p_X p_Y = p_X \left( \mathbf{P}[Y=1 \mid X=1] - p_Y \right)
$$

将其代入比率：

$$
\frac{\mathbf{P}[Y=1 \mid X=1]}{\mathbf{P}[Y=1]} = \frac{\mathbf{P}[Y=1 \mid X=1]}{p_Y} = \frac{\text{cov}(X,Y) + p_X p_Y}{p_X p_Y} = \frac{\text{cov}(X,Y)}{p_X p_Y} + 1
$$

另一方面，使用相关性定义：

$$
\text{cor}(X,Y) = \frac{\text{cov}(X,Y)}{\sqrt{\text{var}(X)} \cdot \sqrt{\text{var}(Y)}}} = \frac{\text{cov}(X,Y)}{\sqrt{p_X (1 - p_X)} \cdot \sqrt{p_Y (1 - p_Y)}}}
$$

因此，

$$
\frac{\text{cov}(X,Y)}{p_X p_Y} = \text{cor}(X,Y) \cdot \frac{\sqrt{p_X (1 - p_X)} \cdot \sqrt{p_Y (1 - p_Y)}}}{p_X p_Y} = \text{cor}(X,Y) \cdot \frac{\sqrt{(1 - p_X)(1 - p_Y)}}{\sqrt{p_X p_Y}}
$$

因此，

$$
\frac{\mathbf{P}[Y=1 \mid X=1]}{\mathbf{P}[Y=1]} = \text{cor}(X,Y) \cdot \frac{\sqrt{(1 - p_X)(1 - p_Y)}}{\sqrt{p_X p_Y}} + 1
$$

这就证明了命题。

### 关于脚和成绩的数学回归

我们用一个模型来描述小学孩子的年龄和数学成绩之间的关系。

设 $A$ 是表示小学孩子年龄的随机变量，取值于 $6, 7, 8, 9, 10, 11$。

设数学成绩表示为：

$$
X = f(A) + \epsilon
$$

其中 $f$ 是一个单调递增函数，$\epsilon$ 是一个中心化的噪声。

设脚的大小表示为：

$$
Y = g(A) + \epsilon'
$$

其中 $g$ 是一个单调递增函数，$\epsilon'$ 是一个中心化的噪声。

我们确实有因果关系（带噪声）：从 $A$ 到 $X$ 和从 $A$ 到 $Y$。但没有从 $Y$ 到 $X$ 的因果关系，如校长所暗示。但我们确实有相关性，甚至：

> **命题**：数学成绩 $X$ 和脚的大小 $Y$ 是正相关的。

**证明**：我们只需要证明协方差是正的。由于噪声是中心化的且独立，我们有：

$$
\text{cov}(X,Y) = \text{cov}(f(A), g(A))
$$

考虑独立的随机变量 $A'$，且 $A'$ 与 $A$ 独立。考虑量：

$$
e = \mathbf{E}[(f(A) - f(A'))(g(A) - g(A'))]
$$

* 一方面，我们有 $e \geq 0$。因为可以将期望分成两部分：

$$
e = \mathbf{E}[1_{A > A'} \cdot (f(A) - f(A'))(g(A) - g(A'))] + \mathbf{E}[1_{A \leq A'} \cdot (f(A) - f(A'))(g(A) - g(A'))]
$$

由于 $f$ 和 $g$ 是单调递增的，$f(A) - f(A')$ 和 $g(A) - g(A')$ 在 $A > A'$ 时都是正的，在 $A \leq A'$ 时都是负的，因此两部分都是正的。

* 另一方面，通过展开乘积，可以看到：

$$
e = \mathbf{E}[f(A)g(A)] - \mathbf{E}[f(A)g(A')] - \mathbf{E}[f(A')g(A)] + \mathbf{E}[f(A')g(A')]
$$

由于 $A$ 和 $A'$ 独立且具有相同分布，$\mathbf{E}[f(A)g(A')] = \mathbf{E}[f(A)] \cdot \mathbf{E}[g(A)]$，同理 $\mathbf{E}[f(A')g(A)] = \mathbf{E}[f(A)] \cdot \mathbf{E}[g(A)]$，而 $\mathbf{E}[f(A')g(A')] = \mathbf{E}[f(A)g(A)]$。

因此，

$$
e = \mathbf{E}[f(A)g(A)] - 2 \mathbf{E}[f(A)] \cdot \mathbf{E}[g(A)] + \mathbf{E}[f(A)g(A)] = 2 (\mathbf{E}[f(A)g(A)] - \mathbf{E}[f(A)] \cdot \mathbf{E}[g(A)])
$$

即，

$$
e = 2 \cdot \text{cov}(f(A), g(A))
$$

因此，$e \geq 0$ 意味着 $\text{cov}(f(A), g(A)) \geq 0$，即 $X$ 和 $Y$ 正相关。

#### ♡♡♡

***任务***：为这部分证明添加细节，使其更加清晰，特别是详细展开 $e$ 的计算以理解第二点。

**回答：**

我们有：

$$
e = \mathbf{E}[(f(A) - f(A'))(g(A) - g(A'))]
$$

展开乘积：

$$
e = \mathbf{E}[f(A)g(A)] - \mathbf{E}[f(A)g(A')] - \mathbf{E}[f(A')g(A)] + \mathbf{E}[f(A')g(A')]
$$

由于 $A$ 和 $A'$ 独立且具有相同分布：

$$
\mathbf{E}[f(A)g(A')] = \mathbf{E}[f(A)] \cdot \mathbf{E}[g(A')]
$$
$$
\mathbf{E}[f(A')g(A)] = \mathbf{E}[f(A')] \cdot \mathbf{E}[g(A)]
$$
$$
\mathbf{E}[f(A')g(A')] = \mathbf{E}[f(A)g(A)]
$$

因此，

$$
e = \mathbf{E}[f(A)g(A)] - 2 \mathbf{E}[f(A)] \cdot \mathbf{E}[g(A)] + \mathbf{E}[f(A)g(A)] = 2 (\mathbf{E}[f(A)g(A)] - \mathbf{E}[f(A)] \cdot \mathbf{E}[g(A)])
$$

即，

$$
e = 2 \cdot \text{cov}(f(A), g(A))
$$

由于我们已经证明 $e \geq 0$，因此：

$$
\text{cov}(f(A), g(A)) \geq 0
$$

这意味着数学成绩 $X$ 和脚的大小 $Y$ 是正相关的。

## 总结

通过上述内容，我们探讨了随机变量之间的相关性、独立性及其量化方法，包括协方差和相关系数的计算与解释。我们还通过具体示例展示了如何计算和理解这些统计量，以及如何避免常见的误区。

# 结语

希望这些笔记能够帮助你更好地理解随机变量之间的关系及其量化方法。记住，相关性不等于因果关系，理解两者的区别对于正确解读数据至关重要。

如果你有任何疑问或需要进一步的解释，欢迎随时提问！