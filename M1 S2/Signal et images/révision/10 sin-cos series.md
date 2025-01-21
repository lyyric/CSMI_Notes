下面给出一份用中文、通俗语言描述的讲义，内容保留了法语笔记中的细节和逻辑，同时补全了笔记中遗留的习题内容。你可以将下面的内容看作是一份关于傅里叶正弦余弦级数的详细中文笔记。

---

# 傅里叶正弦余弦级数

## 简介

傅里叶级数是一种将定义在时间区间 $[0,T]$ 上的光滑函数（信号）分解为无限多种简单波形（例如正弦波和余弦波）的和的方法。常见的正弦余弦级数分解有三种表达方法：
  
- 用正弦和余弦函数分解
- 仅用余弦函数分解
- 用复指数函数分解

需要区分的情况有：

- **傅里叶级数**：定义在有限时间区间 $[0,T]$ 上。
- **傅里叶变换**：定义在整个实数轴上，一般理论性较强。
- **离散傅里叶变换**：对于有限离散时间序列 $\{0,1,\dots,N-1\}$ 的情况（下个章节会有应用）。

### 术语解释

阅读本章节后，你应该对下列术语有所了解（如果不熟悉，请重读本章节）：
  
- 傅里叶级数  
- 连续时间与离散时间  
- 正弦余弦基底与复指数基底  
- 周期与频率  
- 分段连续函数  
- 标量积、厄米内积、正交性、正交归一性  
- Gipps现象、绝对收敛  
- Hermitian对称性  
- 频谱、半频谱、振幅谱、傅里叶系数、坐标  
- 信号逼近、滤波、信号压缩  
- 低通滤波、高通滤波、带通滤波器  
- 完美采样  
- 正交多项式、Legendre多项式、高斯点  
- 希尔伯特空间、Parseval等式、等距映射  
- 信号能量  

---

## 信号（函数）

### 定义

连续时间信号定义在区间 $[0,T]$ 上，比如常见的声音信号就是由空气压力的快速变化产生的。下面给出了一个简单示例，用于生成一个信号并绘制出来：

```python
# 时间轴
t = np.linspace(0, 2, 2000)
# 信号：由两个正弦波叠加，并乘上 $t^2$
signal0 = (np.sin(4 * 2 * np.pi * t) + 0.5 * np.sin(7 * 2 * np.pi * t)) * t**2

fig, ax = plt.subplots()
ax.plot(t, signal0)
ax.set_xlabel("time")
ax.set_ylabel("pressure")
```

### 周期性信号

周期函数是指函数在经过一定时间后重复自身。

**定义：**

- 如果对于所有 $t$，函数满足 $f(t + T) = f(t)$，我们就说函数 $f$ 是周期 $T>0$ 的。
- **周期**：$T$ 是满足上述条件的最小正数。
- **频率**：每秒钟出现完整周期的个数，是周期 $T$ 的倒数，单位是赫兹（Hz）。

**注意：**

- 如果 $f$ 是周期 $T$ 的，那么它也自动满足周期为 $nT$（$n=2,3,\dots$）。
- 对于非周期信号，不宜说它“有”周期 $0$ 或频率 $\infty$，应称之为非周期信号。
- 常数函数可以看作频率为0。
- 频率单位 Hz 表示每秒钟的周期数。

下面是一个周期信号的例子（虽然图只显示在有限区间内）：

```python
t = np.linspace(0, 2, 2000)
signal1 = np.sin(4 * 2 * np.pi * t) + 0.5 * np.sin(8 * 2 * np.pi * t)
fig, ax = plt.subplots()
ax.plot(t, signal1)
```

#### 练习 1：给出下面信号的周期和频率

1. 信号 $t \mapsto \sin(4 \cdot 2\pi t)$

2. 信号 $t \mapsto \sin(4t)$

3. 信号 $t \mapsto \sin(3\cdot2\pi t)+\sin(4\cdot2\pi t)$

4. 信号 $t \mapsto \sin(3\cdot2\pi t)\cdot\sin(4\cdot2\pi t)$

5. 信号 $t \mapsto \cos\big(\frac{\sin(\pi t/20)}{\sin(\pi t/30)}\big)$

6. 信号 $t \mapsto \sin(4\cdot2\pi t)+\sin(7t)$

**解答提示与答案：**

- **（1）**  
  $\sin(4\cdot2\pi t)=\sin(8\pi t)$  
  周期 $T=\frac{1}{4}$ 秒，因为 $8\pi \cdot (t+T)=8\pi t+8\pi T$ 且要求 $8\pi T=2\pi$ ⇒ $T=\frac{1}{4}$。  
  频率 $f=4$ Hz.

- **（2）**  
  $\sin(4t)$ 中的角频率为4，即 $\omega=4$。  
  周期 $T=\frac{2\pi}{4}=\frac{\pi}{2}$ 秒。  
  频率 $f=\frac{2}{\pi}$ Hz.

- **（3）**  
  分量分别是 $\sin(3\cdot 2\pi t)=\sin(6\pi t)$ 和 $\sin(4\cdot 2\pi t)=\sin(8\pi t)$，其周期分别为 $T_1=\frac{1}{3}$ 秒和 $T_2=\frac{1}{4}$ 秒。  
  要构造一个同时为两者周期的周期，取二者周期的**最小公倍数**。  
  注意：$T_1=\frac{1}{3}$ 与 $T_2=\frac{1}{4}$ 的最小公倍数为 $T = 1$ 秒（因为 $1$ 秒是 $\frac{1}{3}$ 和 $\frac{1}{4}$ 的倍数：1 为 3 和 4 的公倍数的倒数）。  
  因此，合成信号周期 $T=1$ 秒，频率 $f=1$ Hz.

- **（4）**  
  $t \mapsto \sin(3\cdot2\pi t)\sin(4\cdot2\pi t)$  
  两个因子分别周期为 $\frac{1}{3}$ 秒和 $\frac{1}{4}$ 秒，乘积函数的周期也是它们周期的最小公倍数，即 $T=1$ 秒，频率 $f=1$ Hz.  
  （注意：对于乘积函数，若各因子周期为 $T_1$ 和 $T_2$，只要它们是周期函数，乘积函数的周期为二者最小公倍数。）

- **（5）**  
  此信号较复杂，一般分析其周期需要先分析内部参数函数的周期。  
  - $\sin(\pi t/20)$ 的周期为 $T_1=\frac{40}{1}=40$ 秒；  
  - $\sin(\pi t/30)$ 的周期为 $T_2=60$ 秒。  
  因此分式内的函数整体周期为两者最小公倍数（如果分母不为0且函数存在）一般取 $T = \mathrm{lcm}(40,60) = 120$ 秒。  
  然而，由于外层是取余弦，再加上分母可能出现符号变化，总体周期可以推断为 120 秒。  
  频率 $f=\frac{1}{120}$ Hz.  
  （注：这题可能需更详细讨论，但基本思想是：当 $f_1(t)$ 与 $f_2(t)$ 分别周期为 $T_1$ 与 $T_2$ 时，如果令 $T$ 同时是 $T_1$ 与 $T_2$ 的倍数，则整个复合函数必定周期性。**答案空缺处应填写：**周期为二者最小公倍数，频率为该最小公倍数的倒数。）

- **（6）**  
  $\sin(4\cdot2\pi t)=\sin(8\pi t)$ 的周期为 $\frac{1}{4}$秒，  
  $\sin(7t)$ 的周期为 $T=\frac{2\pi}{7}$秒。  
  整体函数的周期是这两个周期的**最小公倍数**。  
  一般来说，若两个周期$T_1=\frac{1}{4}$ 与 $T_2=\frac{2\pi}{7}$不成有理数比，则整体函数就不严格周期。如果比值为有理数，则整体周期为二者最小公倍数；否则信号非周期。  
  在本习题中，由于 $\frac{T_2}{T_1} = \frac{2\pi/7}{1/4} = \frac{8\pi}{7}$ 不是有理数，因而整体信号不是严格周期信号。

**进一步说明：**

> [!hint]
> **提示：**  
> 假设 $f_1$ 是周期 $T_1$ 的函数，$f_2$ 是周期 $T_2$ 的函数，那么当你构造函数  
> $$
> f(t) = \text{fonction}(f_1(t), f_2(t))
> $$
> 如果存在 $T$ 同时满足 $T$ 是 $T_1$ 与 $T_2$ 的倍数，则对于所有 $t$，有  
> $$
> f(t + T) = f(t)
> $$
> 因此，$f$ 的周期便是所有满足条件的 $T$ 中的最小值；频率则为该周期的倒数。

---

#### 练习 2

观察下列代码并验证第三个信号（例如 $\sin(3\pi t)+\sin(4\pi t)$）确实是周期为2的信号。  
（代码示例中有部分不完整，请自己补充完整）

```python
t = np.linspace(0, 10, 1000)
y = np.sin(3 * np.pi * t) + np.sin(4 * np.pi * t)

fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(t, y)
for s in [0, 2, 4, 6, 8, 10]:
    ax.axvline(s, color="k", linestyle=":")
plt.show()
```

通过图中每隔2秒标一条竖线，可以直观看出信号在每个周期内重复，从而验证周期为2秒。

---

## 正弦余弦分解

### 在区间 $[0,T]$ 上讨论

在整个这一部分中，我们规定时间区间为 $[0,T]$，这里令 $T=2$ 秒。注意，函数可以看做是周期为 $T$ 的信号在此区间内的截取，也可以只是定义在有限区间上的信号（例如录音信号）。

首先，我们定义离散时间及步长：

```python
T = 2
nb_points = 200
t = np.linspace(0, T, nb_points, endpoint=False)
step = T / nb_points  # 两个点之间的间隔
```

### 信号的光滑性

**定义：**  
一个函数 $f$ 称为“分段光滑的”，当且仅当 $f$ 及其导数 $f'$ 在各个区间上都是连续的。

由于自然界中的所有信号大多都是分段光滑的，故此后所称的“信号”均指“分段光滑的函数”。

另外，还可能讨论两种更高的正则性：

- 信号 $f$ 是否整体连续（无跳跃）。
- 信号是否满足 $f(0)=f(T)$（边界端点的函数值相同）。

#### 练习 3

**题目：**  
请说明当我们希望将一个信号周期延拓到整个实数轴时，为什么要求 $f(0)=f(T)$ 有多重要？

**答案提示：**  
如果 $f$ 在 $t=0$ 和 $t=T$ 处的值不同，那么周期延拓时在这些衔接处会产生突变或不连续，从而导致傅里叶级数收敛性和计算结果不理想。因此，为了获得良好的周期延拓效果与傅里叶级数逼近，通常要求 $f(0)=f(T)$。

---

下面给出一个不连续信号（方波信号）的例子。需要注意，因为存在跳跃，其绘图时可能会出现垂直线段（计算机绘图默认连接断点），我们可以通过改变绘图样式来让图形更真实地反映跳跃现象。

```python
def square_signal(t):
    posi = (np.sin(4 * np.pi * t) > 0)
    f = np.empty(len(t))
    f[posi] = 1
    f[~posi] = -1
    return f

t = np.linspace(0, 2, 2000)
plt.plot(t, square_signal(t))  # 绘图时可通过设置参数“drawstyle='steps-post'”去除垂直连接线
```

#### 练习 4

1. 修改上面的绘图代码，使得垂直连接线不再出现（利用 `plt.plot(..., drawstyle='steps-post')` 或其他样式参数）。
2. 为函数 `square_signal()` 添加一个参数，使得可以指定周期 $T$。

---

#### 练习 5

编写一个函数 `triangle_signal()` 生成锯齿信号（周期性连续的三角波）。提示：  
可以利用辅助函数  
```python
def frac(t):
    return t - np.floor(t)
```
来返回 $t$ 的小数部分，然后构造分段线性函数得到锯齿形状。

例如：
```python
def triangle_signal(t, period):
    # 先将 t 映射到 [0, period) 区间内
    t_mod = np.mod(t, period)
    # 生成上升和下降的线性部分
    # 这里给出一种可能实现方式：
    return 2 * np.abs(2 * (t_mod / period) - 1) - 1

fig, ax = plt.subplots()
t = np.linspace(0, 10, 1000)
plt.plot(t, triangle_signal(t, 2))
```

---

## 标量积

对于定义在 $[0,T]$ 上的两个信号 $f$ 和 $g$，我们定义其标量积为：
$$
\mathtt{dot}(f, g) = \frac{2}{T} \int_{0}^{T} f(t)g(t)\, dt
$$
这个标量积满足下列性质：

1. **对称性：** $\mathtt{dot}(f, g) = \mathtt{dot}(g, f)$  
2. **双线性性：** 对 $f$ 和 $g$ 分别为线性映射  
3. **正定性：**  
   $$
   \mathtt{dot}(f, f) \ge 0 \quad\text{且}\quad \mathtt{dot}(f, f) = 0 \iff f \equiv 0.
   $$
  
#### 练习 6

**题目：**  
验证上面三个性质对给定的 $\mathtt{dot}()$ 定义都成立。（对于第三条性质，可利用信号的分段连续性）

---

### 数值积分方法

在实际计算中，积分将用数值方法近似来替代。下面的代码展示了几种积分数值近似方法（矩形法、中点法、梯形法、辛普森法）的示意图：

```python
def one_approx(x, a, b, f, kind):
    if kind == "rectangle":
        return f(a) * np.ones_like(x)
    elif kind == "mid-point":
        return f((a+b)/2) * np.ones_like(x)
    elif kind == "trapeze":
        fa = f(a)
        fb = f(b)
        return (fa + fb) / 2 + (fb - fa) / (b - a) * (x - (a+b)/2)
    elif kind == "simpson":
        fa = f(a)
        m = (a+b)/2
        fm = f(m)
        fb = f(b)
        return 2*(fa - 2*fm + fb)/(b - a)**2 * (x - m)**2 + (fb - fa)/(b - a)*(x - m) + fm

f = lambda x: np.sin(3*x) * np.sin(2*x)

fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True, sharey=True)
points = np.arange(0, 11)

for a, b in zip(points[:-1], points[1:]):
    x = np.linspace(a, b, 50)
    for i, kind in enumerate(["rectangle", "mid-point", "trapeze", "simpson"]):
        axs[i].plot(x, one_approx(x, a, b, f, kind))
        axs[i].plot(x, f(x), "k")
        axs[i].set_title(kind)
fig.tight_layout()
```

#### 练习 7

**题目：**  
各个积分方法的收敛速度分别是多少？  
**答案提示：**  
- 矩形法和中点法都是一阶收敛，即误差与步长呈线性关系；  
- 梯形法通常也是二阶收敛；  
- 辛普森法则是四阶收敛。这可以通过误差估计公式验证。

---

### 我们采用的数值积分方法

在后续计算中，我们使用下面这种简单的积分近似方法，其实质是将积分用和替代：

```python
def dot(f, g):
    return 2 * np.sum(f * g) / len(f)
```

#### 练习 8

**题目：**  
在上述 `dot` 函数中我们采用了哪种数值积分方法？请通过简单计算说明你的答案。  
**答案提示：**  
注意到积分被近似为离散求和，因而乘上了 $T/\text{(点数)}$（在这里 $T=2$），这正是矩形积分法的应用。

另外，运行下面代码：
```python
t = np.linspace(0, 2*np.pi, 2000)
f = np.sin(t)
g = np.ones_like(t)
print(dot(f, g))
```
会发现结果非常接近0。原因在于 $\sin(t)$ 关于区间 $[0,2\pi]$ 的正负区域相互抵消，使得其积分结果为0。

---

## 正弦余弦基底族

在区间 $[0, T]$ 上，定义一组特殊的函数：
$$
\begin{aligned}
\sin_n(t) &= \sin\Big(2\pi \frac{n t}{T}\Big) \quad \text{对于 } n\ge 1,\$$1mm]
\cos_n(t) &= \cos\Big(2\pi \frac{n t}{T}\Big) \quad \text{对于 } n\ge 1.
\end{aligned}
$$
另外，加入常数函数 $sc_0(t)=\frac{1}{\sqrt{2}}$ 后，我们定义完整的正弦余弦基底族为：
$$
\begin{cases}
sc_0(t)= \frac{1}{\sqrt{2}}\$$1mm]
sc_{2i-1}(t) = \sin_i(t),\quad i\ge1\$$1mm]
sc_{2i}(t) = \cos_i(t),\quad i\ge1.
\end{cases}
$$
后续我们会证明这组函数在前面定义的标量积下是正交归一的。

下面是构造该基底族的代码：
  
```python
def compute_sinCos_basis(t, T, M):
    """
    参数：
        t (1d-array): 离散化时间点
        T (int): 信号总时长
        M (int): 最大频率数，即基底中正弦和余弦的个数为 M，其总数为 2*M+1 个。
    """
    basis_sc = np.empty([2*M+1, len(t)])
    basis_sc[0] = np.ones_like(t) / np.sqrt(2)
    for i in range(1, M+1):
        basis_sc[2*i-1, :] = np.sin(i * 2 * np.pi * t / T)
        basis_sc[2*i, :] = np.cos(i * 2 * np.pi * t / T)
    return basis_sc

T = 2
n_point = 200
M = 5
t = np.linspace(0, T, n_point, endpoint=False)
basis_sc = compute_sinCos_basis(t, T, M)

for i in range(len(basis_sc)):
    plt.plot(t, basis_sc[i, :])
plt.show()
```

验证正交归一性（计算不同基底的内积）：
  
```python
many_dot_products = np.empty([len(basis_sc), len(basis_sc)])
for i in range(len(basis_sc)):
    for j in range(len(basis_sc)):
        many_dot_products[i, j] = dot(basis_sc[i, :], basis_sc[j, :])
print(many_dot_products)
```

#### 练习 9

**题目：**  
上面的代码实现是出于教学目的，但效率不高。试着将 `dot` 函数去掉，通过矩阵乘法（用 `@` 操作符）计算内积矩阵。  
**答案提示：**  
注意到
$$
\text{many\_dot\_products}[i,j] = \sum_k \text{basis\_sc}[i,k] \cdot \text{basis\_sc}[j,k] \times \frac{T}{n\_point},
$$
因此可以写成：
```python
many_dot_products = basis_sc @ basis_sc.T * (T / n_point)
```

---

### 用正弦余弦基底逼近信号

任意信号 $f$ 都可以写成无限多正弦余弦基底的线性组合：
$$
f(t)=\sum_{n\ge0} a_n\, sc_n(t)
$$
由于正交归一性，系数容易计算：
$$
a_n = \mathtt{dot}(f,\, sc_n)
$$
因此，使用有限项逼近时，我们有：
$$
f(t) \simeq \sum_{n=0}^{2M+1} a_n\, sc_n(t)
$$

例如，考虑以下代码，将函数 $f(t)=(t-1)^2$ 在基底下分解：

```python
T = 2
nb_points = 30
t = np.linspace(0, 2, nb_points, endpoint=False)
f = (t - 1)**2
plt.plot(t, f)

M = 10
basis_sc = compute_sinCos_basis(t, T, M)
print(basis_sc.shape)  # (2*M+1, nb_points)

# 计算在正弦余弦基底下的系数
coordinates = np.empty(len(basis_sc))
for i in range(len(basis_sc)):
    coordinates[i] = dot(f, basis_sc[i, :])
plt.plot(range(len(basis_sc)), coordinates, ".")
```

#### 练习 10

**题目：**  
观察上述系数图像，试解释它们有什么特殊之处？  
**答案提示：**  
- 系数图显示出低频系数（较小的 $n$）往往贡献较大，而高频系数逐渐变小。这反映出函数本身是比较平滑的；  
- 对称性与信号的中心对称性也会反映在傅里叶系数上。

进一步，通过如下代码，我们分别利用前 $N$ 项系数构造逼近信号：

```python
approximations = np.empty([len(basis_sc), nb_points])
for i in range(len(basis_sc)):
    coor_troncated = coordinates.copy()
    coor_troncated[i:] = 0  # 只保留前 i 项
    approximations[i, :] = coor_troncated @ basis_sc
```

绘制部分逼近结果：

```python
nb_plots = 4
fig, axs = plt.subplots(nb_plots, figsize=(5, nb_plots*3), sharex=True)
for i in range(nb_plots):
    axs[i].plot(t, approximations[2*i, :], label='Approximation')
    axs[i].plot(t, f, label='Original')
    axs[i].set_title("使用 " + str(2*i) + " 个正弦余弦项")
    axs[i].legend()
plt.tight_layout()
```

#### 练习 11

**题目：**  
作图研究 $L_2$ 误差和 $L_\infty$ 误差如何随着逼近项数增加而变化。  
- $L_2$ 误差利用 `dot()` 计算：  
  $$
  L_2 = \sqrt{\mathtt{dot}(f - f_N,\, f - f_N)}
  $$
- $L_\infty$ 误差利用 `np.max()` 计算：  
  $$
  L_\infty = \max |f - f_N|
  $$

示例代码如下：

```python
fig, ax = plt.subplots()

errors_L2 = []
errors_Loo = []

for approximation in approximations:
    err = f - approximation
    errors_L2.append(np.sqrt(dot(err, err)))
    errors_Loo.append(np.max(np.abs(err)))

ax.plot(errors_L2, label=r"误差 $L_2$")
ax.plot(errors_Loo, label=r"误差 $L_\infty$")
ax.set_yscale("log")
ax.legend()
plt.show()
```

---

## Gipps 现象与理论

### 整体封装代码（面向对象）

为了结构更清晰，同时便于多次使用，我们将前面的代码封装到一个类中。下面给出的 `Decomposer` 类负责计算基底展开及逼近误差。

```python
class Decomposer:
    def __init__(self, t, basis, dot_fn, dtype=np.float64):
        self.t = t
        self.basis = basis
        self.dot = dot_fn
        self.dtype = dtype
        self.OUT_approximations = None  # 用于保存逼近结果
        
    def check_ortho(self):
        nb = len(self.basis)
        res = np.empty([nb, nb], self.dtype)
        for i in range(nb):
            for j in range(nb):
                res[i, j] = self.dot(self.basis[i, :], self.basis[j, :])
        print(res)
    
    def compute_coordinates(self, f, plotThem=False):
        coordinates = np.empty(len(self.basis), dtype=self.dtype)
        for i in range(len(self.basis)):
            coordinates[i] = self.dot(f, self.basis[i, :])
        if plotThem:
            plt.plot(range(len(self.basis)), coordinates, ".")
        return coordinates
    
    def compute_approximations(self, f, approx_indexes, plotThem=False):
        assert max(approx_indexes) <= len(self.basis), "逼近项数不能超过基底总数"
        coordinates = self.compute_coordinates(f)
        approximations = np.empty([len(approx_indexes), len(self.t)], dtype=self.dtype)
        for i, j in enumerate(approx_indexes):
            coor_troncated = coordinates.copy()
            coor_troncated[j:] = 0
            approximations[i, :] = coor_troncated @ self.basis
        if plotThem:
            nb = len(approx_indexes)
            if nb <= 1: nb = 2
            fig, axs = plt.subplots(nb, 1, figsize=(8, nb*2))
            for i in range(len(approx_indexes)):
                axs[i].plot(self.t, f, label="原信号")
                axs[i].plot(self.t, approximations[i, :], label="逼近")
                axs[i].set_title("使用 " + str(approx_indexes[i]) + " 项逼近")
                axs[i].legend()
            fig.tight_layout()
        self.OUT_approximations = approximations
        self.approx_indexes = approx_indexes
        self.f = f
        return approximations
    
    def compute_L2_error(self):
        assert self.OUT_approximations is not None, "请先调用 compute_approximations()"
        L2_error = np.empty(len(self.approx_indexes), dtype=self.dtype)
        for i in range(len(self.approx_indexes)):
            L2_error[i] = np.sqrt(dot(self.f - self.OUT_approximations[i], self.f - self.OUT_approximations[i]))
        return L2_error
    
    def compute_Loo_error(self, plotThem=False):
        assert self.OUT_approximations is not None, "请先调用 compute_approximations()"
        Loo_error = np.max(np.abs(self.f - self.OUT_approximations), axis=1)
        return Loo_error
```

#### 练习 12

**题目：**  
为何将上面的代码封装成一个类，而不是写成单独的函数？  
**答案提示：**  
- 类可以保存状态（例如基底、逼近结果等），避免重复计算；  
- 面向对象的方式使得代码结构更加清晰和可维护；  
- 将相关功能封装在同一个对象中便于扩展和调用。

例如使用该类进行逼近：

```python
T = 2
t = np.linspace(0, T, 2000, endpoint=False)
basis_sc = compute_sinCos_basis(t, T, 10)
decomposer = Decomposer(t, basis_sc, dot)
decomposer.compute_approximations((t - 1)**2, [1, 5, 7], plotThem=True)
decomposer.compute_approximations(t**2, [1, 5, 7], plotThem=True)
```

**问题：** 为什么对 $f(t)=t^2$ 的逼近效果远不如 $(t-1)^2$ 的逼近？  
**答案提示：**  
- 因为 $t^2$ 在边界处（例如 $t=0$ 或 $t=T$）不满足 $f(0)=f(T)$，延拓后产生周期不连续，从而导致傅里叶级数逼近效果较差，尤其在最大误差（$L_\infty$）上会出现较大的震荡（也就是Gipps现象）。

### Gipps现象

下面我们用方波信号示例来展示Gipps现象：即在信号的不连续处，傅里叶级数会出现明显的振铃或振荡现象，虽然整体 $L_2$ 误差下降，但在不连续点附近误差仍然较大。

```python
f_discont = square_signal(t)
plt.plot(t, f_discont)
plt.show()

basis_sc_big = compute_sinCos_basis(t, T, 500)
print(basis_sc_big.shape)
decomposer = Decomposer(t, basis_sc_big, dot)
approx_indices = [10, 50, 100, 150, 200, 500, 1000]
decomposer.compute_approximations(f_discont, approx_indices, plotThem=True)

error_L2 = decomposer.compute_L2_error()
error_Loo = decomposer.compute_Loo_error()

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xscale('log')
ax.plot(approx_indices, error_L2, label="L2 误差")
ax.plot(approx_indices, error_Loo, label="L∞ 误差")
ax.legend()
plt.show()
```

图中可以看出：  
- $L_2$ 误差随逼近项数增加而下降；  
- 而 $L_\infty$ 误差在不连续点附近始终很高，这就是Gipps现象。

### 总结定理

下面的定理总结了正弦余弦级数的主要结论：

> [!theorem]
> **定理（正弦余弦版）：**  
> 1. 基底 $(sc_n)$ 在上述标量积下正交归一。  
> 2. 任一信号 $f$ 都可以写成  
>    $$
>    f(t)=\sum_{n\ge 0} a_n\, sc_n(t)
>    $$
>    其中  
>    $$
>    a_n = \mathtt{dot}(f, sc_n)
>    $$
>    特别地写成  
>    $$
>    f(t) = \frac{a_0}{\sqrt{2}} + \sum_{i\ge1} \Big(a_{2i-1}\sin\Big(2\pi\frac{i t}{T}\Big) + a_{2i}\cos\Big(2\pi\frac{i t}{T}\Big)\Big)
>    $$
> 3. 当 $t\to f(t)$ 连续时，级数处处收敛；当 $f$ 连续且满足 $f(0)=f(T)$ 时，级数一致收敛。  
>  
> **证明要点：**  
> - 利用正交归一性证明傅里叶系数满足  
>   $$
>   \mathtt{dot}\Big(f, sc_n\Big)= \mathtt{dot}\Big(\sum_m a_m\, sc_m, sc_n\Big)= a_n
>   $$
> - 如果 $f$ 在不连续处存在跳跃，则无法取得一致收敛，因为在跳跃处傅里叶级数会出现振铃现象，从而导致任意小邻域内都不能保证误差任意小。

---

## 仅用余弦展开

现在考虑在对称区间 $\left[-\frac{T}{2}, \frac{T}{2}\right]$ 上，只用余弦函数构造傅里叶级数。对偶函数的标量积仍定义为：
$$
\mathtt{dot}(f, g) = \frac{2}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}} f(t)g(t)\, dt.
$$

对于**偶函数** $f_{even}(t)$，由于正弦函数为奇函数，有：
$$
\frac{1}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}} f_{even}(t)\sin(2\pi\frac{nt}{T})dt = 0,
$$
因此偶函数只需要用余弦函数展开。观察下面代码：

```python
T = 2
t = np.linspace(-T/2, T/2, 200, endpoint=False)

def compute_cos_basis(t, T, nb_freq):
    basis = np.empty([nb_freq + 1, len(t)])
    basis[0] = np.ones_like(t) / np.sqrt(2)
    for i in range(nb_freq):
        basis[i+1, :] = np.cos((i+1) * 2 * np.pi * t / T)
    return basis

basis_cos = compute_cos_basis(t, T, 10)
for i in range(7):
    plt.plot(t, basis_cos[i, :])
plt.show()

f_even = t**2
plt.plot(t, f_even)
plt.show()

decomposer = Decomposer(t, basis_cos, dot)
decomposer.compute_approximations(f_even, [2, 5, 7, 11], plotThem=True)
```

**思考：**

#### 练习 13

1. **题目：** 解释为何对于定义在 $[0, \frac{T}{2}]$ 上的信号，可以通过对称延拓成为偶函数，从而仅使用余弦展开。
   
   **答案提示：**  
   若原信号定义在 $[0, \frac{T}{2}]$ 上，则可以构造偶延拓，即令  
   $$
   f_{even}(t)=
   \begin{cases}
   f(t), & t\in[0,\frac{T}{2}]\$$1mm]
   f(-t), & t\in[-\frac{T}{2}, 0)
   \end{cases}
   $$
   由于偶延拓后的函数为偶函数，其傅里叶正弦项必然为零，因此只用余弦基函数即可逼近。另外，余弦展开由于消除了因奇偶项带来的高频振荡，通常较不容易出现 Gipps 现象，使得逼近更为平滑。

2. **题目：** 编写程序计算在 $[0, \frac{T}{2}]$ 上定义的信号（例如 $f(t)=t$）的余弦级数展开。注意：需要修改之前的代码以避免不必要计算。
   
   **实现思路：**  
   - 先在 $[0, \frac{T}{2}]$ 取采样点。  
   - 构造余弦基底时，利用偶延拓的思想，或者直接对信号做对称延拓；  
   - 利用定义好的标量积（注意积分范围相应改变）计算傅里叶系数；  
   - 用余弦基底重构信号并绘图比较。

例如：
```python
# 定义时间区间为 [0, T/2]
T = 2
t_half = np.linspace(0, T/2, 200, endpoint=False)
# 对于余弦级数展开，我们构造一个偶延拓后的 t 坐标：
t_full = np.linspace(-T/2, T/2, 400, endpoint=False)

# 定义原始信号 f(t)=t 在 [0, T/2] 上的值，并构造偶延拓：
f_half = t_half  # 在正半轴上
# 构造偶延拓：
f_full = np.concatenate((f_half[::-1], f_half))

# 构造余弦基底：
def compute_cos_basis_full(t, T, nb_freq):
    basis = np.empty([nb_freq + 1, len(t)])
    basis[0] = np.ones_like(t) / np.sqrt(2)
    for i in range(nb_freq):
        basis[i+1, :] = np.cos((i+1) * 2 * np.pi * t / T)
    return basis

basis_cos_full = compute_cos_basis_full(t_full, T, 20)
# 计算傅里叶系数：
coefficients = np.array([dot(f_full, basis_cos_full[i, :]) for i in range(len(basis_cos_full))])

# 重构信号：
f_recons = coefficients @ basis_cos_full
plt.plot(t_full, f_full, label='原始偶延拓信号')
plt.plot(t_full, f_recons, label='余弦级数重构信号')
plt.legend()
plt.show()
```

---

以上即为完整的中文讲义，涵盖了傅里叶正弦余弦级数的基本概念、数值实现、正交基的构造、信号逼近及误差分析，并补全了笔记中遗留的练习题解答。希望这份笔记能帮助你更好地理解傅里叶级数在信号处理中的应用。