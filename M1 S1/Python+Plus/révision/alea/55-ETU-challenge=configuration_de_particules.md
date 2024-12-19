## 粒子放置

### 无约束

首先，导入必要的库：

```python
import numpy as np
import matplotlib.pyplot as plt
```

我们称配置（或简称 config）为一个大小为 `size × size` 的网格，其中放置了若干粒子。数学上，配置可以表示为
$$\{0,1\}^{size\times size}$$
其中：
* `0` 表示该位置没有粒子。
* `1` 表示该位置有粒子。

### 手动创建一个配置

设定网格大小为 4：

```python
size = 4
a_config = np.zeros([size, size], dtype=bool)
a_config[0, 0] = 1
a_config[2, 2] = 1

fig, ax = plt.subplots()
ax.imshow(a_config)
ax.axis("off");
```

这段代码手动创建了一个 4×4 的网格，其中 `(0,0)` 和 `(2,2)` 位置各放置了一个粒子，并将配置可视化。

### 生成随机配置的函数

定义一个函数用于生成随机配置：

```python
def random_config(size):
    return np.random.randint(2, size=(size, size))
```

### 绘制多个配置的函数

定义一个函数用于绘制多个配置：

```python
def plot_several_configs(nb, generation_fn):
    ncols = int(np.ceil(np.sqrt(nb)))  # 计算需要的列数
    nrows = int(np.ceil(nb / ncols))   # 计算需要的行数
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    axs = axs.flatten()
    for idx in range(nb):
        config = generation_fn()
        axs[idx].imshow(config, cmap='Greys')
        axs[idx].axis("off")
    # 关闭多余的子图
    for idx in range(nb, len(axs)):
        axs[idx].axis("off")
    plt.tight_layout()
    plt.show()
```

调用该函数绘制 7 个 4×4 的随机配置：

```python
plot_several_configs(7, lambda: random_config(4))
```

#### ♡♡

平均观察到的粒子数量为：
$$
\color{red}{\frac{size^2}{2}}
$$

粒子数量的分布是一种 **二项分布**（binomiale） 的参数为 $n=\color{red}{size^2}$ 和 $p=\color{red}{0.5}$。

当 `size` 很大时，尽管粒子分布是随机的，其占据比例似乎相当接近一半。而且，`size` 越大，这种现象越明显。

### 数学化这一观察

**命题：** 设 $Y \in [0,1]$ 为占据位置的比例。则有
$$
\mathbf{P}[|Y - 1/2| > \beta] \leq \frac{1}{4 \, size^2 \, \beta^2}
$$
请使用切比雪夫不等式证明此命题。

**提示：** 切比雪夫不等式表明，对于任意随机变量 $X$，其方差为 $\sigma^2$，有：
$$
\mathbf{P}[|X - \mathbf{E}[X]| > \alpha] \leq \frac{\sigma^2}{\alpha^2}
$$
请选择合适的 $X$ 来应用此不等式。

#### 解答提示

设随机变量 $X$ 为占据位置的数量。记 $n = size \times size$ 为位置总数。则占据比例 $Y = X / n$。将其代入不等式，可以得到所需结果。

### 装置配置的等概率性

尽管我们手动创建的第一个配置（仅有 2 个粒子）看起来并不“典型”，但需要注意，“典型”在这里没有严格的数学定义，只是指一眼看上去不令人惊讶的配置。

然而，所有配置都是等概率的。例如，观察到没有粒子的配置的概率与观察到上述图像中某一配置的概率相同。

**任务：** 解释为什么 `random_config` 函数生成的配置是等概率的。

#### ♡

### 有约束的情况

现在假设粒子太大，无法在同一列或同一行上相邻放置（但可以通过对角线接触）。

定义测试配置是否可接受的函数：

```python
def test_acceptable_config(config):
    if np.max(config[1:, :] + config[:-1, :]) > 1:
        return False
    if np.max(config[:, 1:] + config[:, :-1]) > 1:
        return False
    return True
```

定义生成可接受配置的函数：

```python
def make_acceptable_config(size):
    acceptable = False
    while not acceptable:
        config = random_config(size)
        acceptable = test_acceptable_config(config)
    return config
```

绘制 7 个 4×4 的可接受配置：

```python
plot_several_configs(7, lambda: make_acceptable_config(4))
```

#### ♡

注意到函数 `make_acceptable_config` 使用了 **拒绝采样算法**（algorithme du rejet）。因此，它返回的可接受配置是等概率的。

### 计算拒绝次数

实现一个函数 `count_rejection`，用于估计在找到一个可接受配置之前的平均拒绝次数。

#### ♡♡♡

```python
def count_rejection(size):
    nb_rejection = 0
    acceptable = False
    while not acceptable:
        config = random_config(size)
        acceptable = test_acceptable_config(config)
        if not acceptable:
            nb_rejection += 1
    return nb_rejection
```

```python
nb_rejections = []
for _ in range(1000):
    nb_rejection = count_rejection(4)
    nb_rejections.append(nb_rejection)

plt.hist(nb_rejections, bins=range(max(nb_rejections)+2), edgecolor='black', align='left')
plt.xlabel('拒绝次数')
plt.ylabel('频数')
plt.title('拒绝次数分布直方图')
plt.show()
```

#### ♡

此拒绝次数遵循**几何分布**（géométrique）。

### 利用模拟计算占据率

知道如何模拟粒子配置的意义在于，可以通过蒙特卡洛方法估计配置的平均量，这是一种统计物理学的方法。

首先需要计算“占据率” $c$，即粒子的平均数量期望。

在下面的代码中，`np.mean` 分别用于计算期望和平均数量：

#### ♡

```python
def one_size(size):
    nb_particules = []
    for _ in range(200):
        config = make_acceptable_config(size)
        # 这一步的平均值用于计算单个配置的占据率
        nb_particules.append(np.mean(config))
    # 最终的平均值用于计算所有配置的平均占据率
    return np.mean(nb_particules)

res = []
sizes = [2, 3, 4, 5]  # size >= 6 时由于拒绝次数过多，变得非常耗时
for size in sizes:
    res.append(one_size(size))

fig, ax = plt.subplots()
ax.set_xlabel("size")
ax.set_ylabel("估计的占据率")
ax.plot(sizes, res, "o")
plt.show()
```

#### ♡♡♡

这个占据率随着 `size` 增大而减小。您能给出一个直观的解释吗？

<font color='red'>**解释：** 当网格尺寸增大时，粒子之间的空间限制增加，使得在网格上放置更多粒子变得更加困难。因此，整体的占据率随着网格大小的增加而减少。</font>

### 对角线的约束

现在假设粒子太大，无法通过对角线接触。以下是 `size=4` 时可接受配置的模拟示例：

#### ♡

该情况下生成配置的拒绝次数比之前更 **高**。

#### ♡♡♡♡

模拟满足这种新约束的配置，并绘制类似的图像：

```python
def test_acceptable_config_diagonal(config):
    # 检查水平和垂直相邻
    if np.max(config[1:, :] + config[:-1, :]) > 1:
        return False
    if np.max(config[:, 1:] + config[:, :-1]) > 1:
        return False
    # 检查对角线相邻
    for i in range(config.shape[0] - 1):
        for j in range(config.shape[1] - 1):
            if config[i, j] == 1 and (config[i+1, j+1] == 1 or config[i+1, j-1] == 1):
                return False
    return True

def make_acceptable_config_diagonal(size):
    acceptable = False
    while not acceptable:
        config = random_config(size)
        acceptable = test_acceptable_config_diagonal(config)
    return config

plot_several_configs(7, lambda: make_acceptable_config_diagonal(4))
```

### 更复杂的领域

假设除了之前的对角线约束外，所有粒子的位置 $(i,j)$ 还需满足 $i \leq j$。

使用自然的编程方法，这种情况下的拒绝次数将 **更高**，并且模拟算法将 **更加低效**。

以下是 `size=7` 的示例：

```python
def test_complex_constraints(config):
    # 检查水平和垂直相邻
    if np.max(config[1:, :] + config[:-1, :]) > 1:
        return False
    if np.max(config[:, 1:] + config[:, :-1]) > 1:
        return False
    # 检查对角线相邻
    for i in range(config.shape[0] - 1):
        for j in range(config.shape[1] - 1):
            if config[i, j] == 1 and (config[i+1, j+1] == 1 or config[i+1, j-1] == 1):
                return False
    # 检查 i <= j
    for i in range(config.shape[0]):
        for j in range(config.shape[1]):
            if config[i, j] == 1 and i > j:
                return False
    return True

def make_complex_acceptable_config(size):
    acceptable = False
    while not acceptable:
        config = random_config(size)
        acceptable = test_complex_constraints(config)
    return config

plot_several_configs(1, lambda: make_complex_acceptable_config(7))
```

**任务：**

模拟满足这些新约束的配置，并估计 `size=6` 时的占据率。

## 高效但渐近的算法

当粒子数量很大时，拒绝采样算法变得过于低效。这里介绍 Metropolis-Hastings 技术，它是 MCMC（马尔可夫链蒙特卡洛）方法的一部分。这种方法可以模拟满足约束的大规模配置。

然而，这是一种渐近迭代技术：通过构建一系列配置，最终的配置将遵循均匀分布。

```python
def droite_gauche(i, size):
    v_i = [i]
    if i > 0:
        v_i.append(i - 1)
    if i < size - 1:
        v_i.append(i + 1)
    return v_i

def un_voisin(config, i, j, size):
    for di in droite_gauche(i, size):
        for dj in droite_gauche(j, size):
            if config[di, dj] == 1:
                return True
    return False

def metro(size, n_ite):
    config = np.zeros([size, size], dtype=int)
    for _ in range(n_ite):
        rand_i = np.random.randint(size)
        rand_j = np.random.randint(size)
        if config[rand_i, rand_j] == 1:
            config[rand_i, rand_j] = 0
        else:
            if not un_voisin(config, rand_i, rand_j, size):
                config[rand_i, rand_j] = 1
    return config

config = metro(10, 1000)
fig, ax = plt.subplots()
ax.axis("off")
ax.imshow(config, cmap='Greys');
plt.show()
```

***任务：*** 用您自己的语言描述这一技术。该算法考虑了哪些类型的约束？请根据其他类型的约束重新编写该算法。

#### ♡♡♡♡

**回答：**

Metropolis-Hastings 算法是一种马尔可夫链蒙特卡洛（MCMC）方法，用于从复杂的概率分布中采样。在上述实现中，算法通过在配置中随机选择一个位置，并决定是否在该位置放置或移除一个粒子，以满足相邻粒子不重叠的约束。具体来说：

- **约束类型：** 粒子不能在同一行或同一列上相邻放置，但可以通过对角线接触。
- **工作原理：**
    1. 随机选择一个位置 `(rand_i, rand_j)`。
    2. 如果当前位置已有粒子，则移除它。
    3. 如果当前位置没有粒子，检查其四邻域（上下左右）是否有粒子。如果没有，则在此位置放置一个粒子。

要根据其他类型的约束（例如，不允许对角线接触）重新编写该算法，只需修改 `un_voisin` 函数，使其同时检查对角线相邻的位置。

例如，添加对角线邻居的检查：

```python
def un_voisin_diagonal(config, i, j, size):
    for di in droite_gauche(i, size):
        for dj in droite_gauche(j, size):
            if config[di, dj] == 1:
                return True
    # 检查对角线
    diagonals = [ (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1) ]
    for (di, dj) in diagonals:
        if 0 <= di < size and 0 <= dj < size:
            if config[di, dj] == 1:
                return True
    return False

def metro_diagonal(size, n_ite):
    config = np.zeros([size, size], dtype=int)
    for _ in range(n_ite):
        rand_i = np.random.randint(size)
        rand_j = np.random.randint(size)
        if config[rand_i, rand_j] == 1:
            config[rand_i, rand_j] = 0
        else:
            if not un_voisin_diagonal(config, rand_i, rand_j, size):
                config[rand_i, rand_j] = 1
    return config
```

## 有效但有偏的算法

其他技术（针对更复杂的情况）不会生成“等概率”的配置。这些算法存在“偏差”（类似于作弊的骰子）。但这些算法仍然可以用于模拟诸如“占据率”这样的量。

### 理想情况

设 $\mathcal{C}$ 为所有可接受配置的集合。

假设我们能够模拟一个随机变量 $C$，其均匀分布在 $\mathcal{C}$ 上。那么，占据率可以表示为
$$
\frac{\sum_{c \in \mathcal{C}} \mathtt{np.mean}(c)}{\text{cardinal } \mathcal{C}} = \mathbf{E}[\mathtt{np.mean}(C)]
$$
其中，$\mathtt{np.mean}$ 表示对粒子数量的平均。

通过取一系列 $C_i$ 的实现，可以估计占据率为
$$
\frac{1}{I} \sum_{i=0}^{I-1} \mathtt{np.mean}(C_i)
$$

### 有偏情况

假设我们有一个算法生成一个可接受配置 $C$，其概率分布为
$$
p(c) = \mathbf{P}[C = c]
$$
并且存在一个已知的函数 $q$，使得
$$
\exists k > 0 \quad \forall c \in \mathcal{C}, \quad p(c) = k \cdot q(c)
$$
此时，$q$ 被称为 $C$ 的伪密度。

**命题：** 占据率的期望为
$$
\frac{\mathbf{E}\left[\frac{\mathtt{np.mean}(C)}{q(C)}\right]}{\mathbf{E}\left[\frac{1}{q(C)}\right]}
$$

因此，可以通过以下步骤估计占据率：
1. 对每个 $C_i$ 计算 $\frac{1}{q(C_i)} \mathtt{np.mean}(C_i)$ 并求和。
2. 对每个 $C_i$ 计算 $\frac{1}{q(C_i)}$ 并求和。
3. 取上述两者的比值作为占据率的估计值。

**证明：**

分子为：
$$
\mathbf{E}\left[\frac{\mathtt{np.mean}(C)}{q(C)}\right] = \sum_{c \in \mathcal{C}} p(c) \cdot \frac{\mathtt{np.mean}(c)}{q(c)} = k \sum_{c \in \mathcal{C}} \mathtt{np.mean}(c)
$$

分母为：
$$
\mathbf{E}\left[\frac{1}{q(C)}\right] = \sum_{c \in \mathcal{C}} p(c) \cdot \frac{1}{q(c)} = k \cdot \text{cardinal} \, \mathcal{C}
$$

因此，比例为：
$$
\frac{\mathbf{E}\left[\frac{\mathtt{np.mean}(C)}{q(C)}\right]}{\mathbf{E}\left[\frac{1}{q(C)}\right]} = \frac{\sum_{c \in \mathcal{C}} \mathtt{np.mean}(c)}{\text{cardinal} \, \mathcal{C}}
$$
即占据率的期望。

#### ♡

我们忽略了一个小假设：$p$ 必须 **不为零**。

### 最终备注

能够模拟具有已知伪密度（仅相差一个常数倍）的随机变量在实际中非常常见。例如，当我们模拟满足某种可接受性标准的配置时，我们知道其伪密度是指示函数 $1_{\mathcal{C}}$，并且我们已经在 Python 中编写了该函数。

然而，真正的密度是
$$
\frac{1_{\mathcal{C}}}{\text{cardinal } \mathcal{C}}
$$
一旦可接受性的定义变得复杂，我们便无法计算 $\mathcal{C}$ 中元素的数量。