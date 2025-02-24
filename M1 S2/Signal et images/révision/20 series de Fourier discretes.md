# 离散傅里叶变换（DFT）与快速傅里叶变换（FFT）笔记

本笔记主要介绍如何用 Python 进行傅里叶级数和离散傅里叶变换（DFT）的计算，并讨论实信号的傅里叶变换、谱的解释、采样定理以及混叠（aliasing）现象。

---

## 1. 导入数据与环境初始化

首先，我们从 GitHub 上获取一些示例信号数据。如果本地不存在目录 `assets_signal`，则克隆；如果已存在则更新。

```python
import os

if not os.path.exists("assets_signal"):
    print("目录 assets_signal 不存在，正在创建并克隆数据...")
    !git clone https://github.com/vincentvigon/assets_signal
else:
    print("目录 assets_signal 已存在，正在更新数据...")
    %cd assets_signal
    !git pull https://github.com/vincentvigon/assets_signal
    %cd ..
  
!pwd
```

接下来重置环境并导入必要的库：

```python
%reset -f
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

np.set_printoptions(linewidth=500, precision=3, suppress=True)
```

---

## 2. 傅里叶级数回顾（连续区间下的情形）

### 2.1 定义区间和离散点

设定一个非对称区间 \([ -1, 3 )\)（注意这里不要求对称），并用离散点来采样信号。令：

- 最大频率：\( M = 5 \)
- 指数基底个数：\( N = 2M+1 \)
- 离散点个数：`nb_points = 101`（后续也会有其他参数组合的练习）

```python
M = 5           # 最大频率
N = 2 * M + 1   # 指数基底个数
nb_points = 101 # 离散采样点个数

left = -1
right = 3
T = right - left  # 区间长度

t = np.linspace(left, right, nb_points, endpoint=False)
plt.plot(t, np.zeros_like(t), '+');
```

### 2.2 构造连续傅里叶指数基底

对于每个 \( n \in \{-M, -M+1, \ldots, M\} \)，我们构造函数
\[
e_n(t) = \exp\left(2i\pi \frac{n\, t}{T}\right)
\]
将这些基底按从 \( n=-M \) 到 \( n=M \) 的顺序排列到矩阵 `basis_exp` 中（行数为 \( N \)，列数为采样点个数）。

```python
basis_exp = np.empty([N, len(t)], dtype=np.complex64)

for n in range(-M, M+1):
    print("创建对应于 n=%d 的指数波" % n)
    basis_exp[M + n, :] = np.exp(2 * 1j * np.pi * t * n / T)

print("basis_exp 的形状为：", basis_exp.shape)
```

### 2.3 绘制基底的实部与虚部

将每个基底函数的实部和虚部分别画出，方便直观理解各个波形。

```python
fig, axs = plt.subplots(N, 2, figsize=(8, N), sharex=True, sharey=True)

for n in range(-M, M+1):
    m = n + M
    axs[m, 0].plot(t, np.real(basis_exp[m, :]))
    axs[m, 1].plot(t, np.imag(basis_exp[m, :]))

    axs[m, 0].set_title("实部, n=%d" % n)
    axs[m, 1].set_title("虚部, n=%d" % n)

fig.tight_layout()
```

### 2.4 对信号进行傅里叶分解

选取一个信号：
\[
f(t) = (t-1)(t-3)^2
\]
并在区间上画出其图像。

```python
f = (t - 1) * (t - 3)**2
plt.plot(t, f);
```

计算该信号在基底 `basis_exp` 上的傅里叶系数：
\[
\alpha[n] \approx \frac{1}{N} \sum_{j} f(t_j) \, \overline{e_n(t_j)}
\]
这里我们用矩阵乘法实现：

```python
alpha = basis_exp.conj() @ f / len(t)
```

画出从 \( n = -M \) 到 \( n = M \) 的傅里叶系数的幅值谱：

```python
fig, ax = plt.subplots()
ax.plot(range(-M, M+1), np.abs(alpha), ".")
ax.set_xticks(range(-M, M+1));
```

也可以只看正频率部分（半谱）：

```python
fig, ax = plt.subplots()
ax.plot(range(0, M+1), np.abs(alpha[M:]), ".")
ax.set_xticks(range(0, M+1));

# 利用频率作为 x 轴标签
frequencies = np.arange(0, M+1) / T
ax.set_xticklabels(frequencies)
ax.set_xlabel("频率 (Hz)");
```

信号重构：
\[
f_{\text{approx}}(t) = \sum_{n=-M}^{M} \alpha[n] e^{2i\pi \frac{n\,t}{T}}
\]
用代码实现：

```python
f_approx = alpha @ basis_exp
plt.plot(t, f, label="原信号")
plt.plot(t, np.real(f_approx), label="傅里叶重构")
plt.legend();
```

### 2.5 练习：改变参数

1. **情况1：** 令 `nb_points = 11`，保持 `M = 5`；此时基底矩阵 `basis_exp` 是一个 \( (2*5+1) \times 11 \) 的矩阵。
2. **情况2：** 令 `M = 50`，并重置 `nb_points = 101`。由于基底数量较多（共 101 个），不建议画出所有波形，只需计算傅里叶系数并重构信号。

在这两种情况下，由于采样点个数与基底数相同，所以矩阵是方阵，从而可以保证傅里叶分解和重构完全一致，即 \( f_{\text{approx}} = f \)。

---

## 3. 离散傅里叶变换（DFT）与 FFT

在离散情形下，我们把信号看作一个仅由整数索引的向量，并直接定义离散指数基底。

### 3.1 离散指数基底的定义

给定 \( N \) 个点，我们定义离散基底：
\[
d_n(k) = \exp\left(2i\pi \frac{n\,k}{N}\right), \quad k = 0,1,\ldots,N-1.
\]
这与连续情况中的 \( e_n(t) = \exp\left(2i\pi \frac{n\,t}{T}\right) \) 相对应（注意：后续可以讨论二者的联系）。

构造基底矩阵（这里先以 \( M=4 \) 为例，则 \( N = 2M+1 = 9 \)）：

```python
M = 4
N = 2 * M + 1
k = np.arange(0, N)

basis_dis = np.empty([N, N], dtype=np.complex64)
for n in range(N):
    basis_dis[n, :] = np.exp(2 * 1j * np.pi * n * k / N)

fig, axs = plt.subplots(N, 2, figsize=(8, N), sharex=True, sharey=True)

for n in range(N):
    axs[n, 0].plot(k, np.real(basis_dis[n, :]), ".")
    axs[n, 1].plot(k, np.imag(basis_dis[n, :]), ".")
    axs[n, 0].set_ylim(-1.1, 1.1)
    axs[n, 1].set_ylim(-1.1, 1.1)
```

### 3.2 正交性验证

定义自然的 Hermite 内积：
\[
\langle u, v \rangle = \frac{1}{N} \sum_{k=0}^{N-1} u_k \, \overline{v_k}
\]
证明基底 \(\{d_n\}\) 是正交归一的。利用矩阵乘法，可以计算基底矩阵与其共轭转置的乘积（归一化后应得到单位矩阵）。

*提示：* 你可以计算

```python
inner_products = basis_dis @ basis_dis.conj().T / N
print(np.round(inner_products, 3))
```

观察输出是否接近单位阵。

> **补充说明：**
> 这里利用离散傅里叶变换的正交性，其数学原理是：
> \[
> \frac{1}{N} \sum_{k=0}^{N-1} \exp\left(2i\pi \frac{(n-m)k}{N}\right) = \delta_{nm}.
> \]

### 3.3 基底的重新排列

在实际傅里叶分析中，我们常希望将负频率和正频率成对排列。令基底的下标从 \(-M\) 到 \(+M\)（共 \(N\) 个），构造“重排后的”离散基底：

```python
basis_dis_dec = np.empty([N, N], dtype=np.complex64)
for n in range(-M, M+1):
    basis_dis_dec[n+M, :] = np.exp(2 * 1j * np.pi * n * k / N)

fig, axs = plt.subplots(N, 2, figsize=(8, N), sharex=True, sharey=True)
for n in range(N):
    axs[n, 0].plot(k, np.real(basis_dis_dec[n, :]), ".")
    axs[n, 1].plot(k, np.imag(basis_dis_dec[n, :]), ".")
    axs[n, 0].set_ylim(-1.1, 1.1)
    axs[n, 1].set_ylim(-1.1, 1.1)
```

**验证关系：**

1. 对于任意 \( n \)，有
   \[
   d_n = d_{N+n}.
   \]
   这说明原始基底 \(\{d_0, d_1, \ldots, d_{N-1}\}\) 与重排后的基底仅是索引的平移。
2. 由于 \(\exp(-ia) = \overline{\exp(ia)}\)，有：
   \[
   d_{N-n} = d_{-n} = \overline{d_n}.
   \]

这些性质在实际理解傅里叶系数的对称性时十分重要。

---

## 4. 用 FFT 进行信号分解与重构

假设我们有一个离散信号（例如从文件 `assets_signal/signalToFilter.txt` 中读取），我们可以将其分解到离散指数基底中。

```python
f2 = np.loadtxt("assets_signal/signalToFilter.txt")
N = len(f2)

fig, ax = plt.subplots(figsize=(8, 2))
ax.plot(range(N), f2);
```

### 4.1 手动计算傅里叶系数

构造大小为 \( N \times N \) 的离散指数基底，并计算傅里叶系数：
\[
\beta_k = \frac{1}{N} \sum_{n=0}^{N-1} f2_n \, \exp\left(-2i\pi \frac{k\,n}{N}\right)
\]
注意到这里用的是共轭基底。

```python
basis_dis = np.empty([N, N], dtype=np.complex128)
x = np.arange(0, N)
for k in range(N):
    basis_dis[k, :] = np.exp(2 * 1j * np.pi * k * x / N)

%%time
alpha = basis_dis.conj() @ f2 / N
```

### 4.2 使用 FFT 的优势

FFT（快速傅里叶变换）是一种快速计算 DFT 的递归算法，其复杂度为 \(O(N \log N)\)。

> **注意：** 相比之下，直接使用矩阵乘法计算 DFT 的算法复杂度为 \(O(N^2)\)。

使用 `np.fft.fft` 来计算傅里叶系数：

```python
%%time
alpha_fft = np.fft.fft(f2)
```

对比手动计算和 FFT 得到的系数（注意 `np.fft.fft()` 并没有除以 \(N\)）：

```python
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 4))
ax0.plot(range(N), np.abs(alpha))
ax0.set_title("自实现傅里叶系数")
ax1.plot(range(N), np.abs(alpha_fft))
ax1.set_title("np.fft.fft 得到的傅里叶系数")
fig.tight_layout();
```

### 4.3 信号重构

重构信号有两种方法：

1. 利用我们手动计算的傅里叶系数：
   \[
   f2\_recons = \alpha @ \text{basis\_dis}
   \]
2. 利用 FFT 的逆变换 `np.fft.ifft`：
   ```python
   %%time
   f2_recons = alpha @ basis_dis

   %%time
   f2_recons_fft = np.fft.ifft(alpha_fft)

   fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 4))
   ax0.plot(range(N), np.abs(f2_recons))
   ax0.set_title("使用自实现方法重构的信号")
   ax1.plot(range(N), np.abs(f2_recons_fft))
   ax1.set_title("使用 np.fft.ifft 重构的信号")
   fig.tight_layout();
   ```

> **注意：** 这里一定要注意傅里叶系数和信号之间的比例问题。由于 `np.fft.fft()` 没有除以 \(N\)，所以在重构时要小心归一化问题。

---

## 5. 实数信号与 Hermitian 对称性

对于一个实值离散信号 \( u \)，其傅里叶系数 \( \beta \) 满足 Hermitian 对称性：
\[
\forall n \in \{0, \ldots, N-1\}:\quad \beta_{N-n} = \overline{\beta_n}
\]

**证明思路 1（利用重构公式）：**

- 信号重构公式：
  \[
  u_k = \sum_{n=0}^{N-1} \beta_n \, e^{2i\pi \frac{kn}{N}}
  \]
- 由于 \( u \) 为实信号，必有 \( u = \overline{u} \)：
  \[
  u_k = \overline{u_k} = \sum_{n=0}^{N-1} \overline{\beta_n} \, \overline{e^{2i\pi \frac{kn}{N}}}
  \]
- 注意到 \(\overline{e^{2i\pi \frac{kn}{N}}} = e^{-2i\pi \frac{kn}{N}} = e^{2i\pi \frac{k(N-n)}{N}}\)，作变换 \( n \to N-n \) 后，由于傅里叶分解的唯一性，得到：
  \[
  \overline{\beta_{N-n}} = \beta_n \quad \Rightarrow \quad \beta_{N-n} = \overline{\beta_n}.
  \]

**证明思路 2（利用傅里叶变换的定义）：**

从傅里叶系数定义出发：
\[
\beta_n = \frac{1}{N} \sum_{k=0}^{N-1} u_k \, e^{-2i\pi \frac{kn}{N}}
\]
由于 \( u_k \) 为实数，对比 \( \beta_n \) 与 \( \beta_{N-n} \) 可以得到同样的对称性。具体证明过程略（但关键就在于复指数函数的共轭关系）。

---

## 6. 只存储半谱：np.fft.rfft 与 np.fft.irfft

对于实值信号，由于 Hermitian 对称性，所有信息只包含在一半的傅里叶系数中。例如，当 \( N = 2M+1 \) 时，只需存储：

- \( \beta_0 \)
- \( \beta_1 \)（以及对应的 \( \beta_{N-1} = \overline{\beta_1} \)）
- …
- \( \beta_M \)（对应 \( \beta_{N-M} = \overline{\beta_M} \)）

当 \( N = 2M \) 时，类似地只需要存储 \( M+1 \) 个系数，其中 \( \beta_M \) 是实数（它等于其共轭）。

Python 中的 `np.fft.rfft` 就利用了这一点：输入实信号，返回大小为 \( N//2 + 1 \) 的复数数组，即“半谱”。例如：

```python
N = 4
for N in [4, 5]:
    u = np.ones([N])
    beta = np.fft.rfft(u)
    u_back = np.fft.irfft(beta, n=N)
    print(u_back)
```

> **练习：**
> 如果调用 `np.fft.irfft` 时不指定参数 `n=N`，numpy 会如何处理？
> **答案：** numpy 会根据输入的半谱长度自动推断原始信号的长度。具体来说，对于输入半谱长度为 \( K \) 的数组，默认输出长度为 \( 2 \times (K - 1) \)（适用于偶数情况；对于奇数情况，推断方式类似），以便正确重构原始信号。

例如：

```python
beta = np.fft.rfft(f2)
print("半谱大小：", beta.shape)
u_reconstructed = np.fft.irfft(beta)  # 未指定 n
print("重构后信号的长度：", u_reconstructed.shape)
```

---

## 7. 时间与频率的对应

### 7.1 改善谱图的横坐标

假设我们知道信号实际持续 \( T \) 秒，那么我们可以重构时间轴：

```python
f2 = np.loadtxt("assets_signal/signalToFilter.txt")
N = len(f2)
T = 2  # 信号时长 2 秒

t = np.linspace(0, T, N)
fig, ax = plt.subplots(figsize=(8, 2))
ax.plot(t, f2)
ax.set_xlabel("时间 (秒)");
```

同时，我们希望在傅里叶系数（或幅值谱）的横坐标上标出真实频率。对于用 `np.fft.rfft` 得到的半谱，其对应的频率可以通过：
\[
\text{frequencies} = \text{np.linspace}(0, \frac{N-1}{2T}, \text{len(half\_spectrum)})
\]
例如：

```python
half_spectrum = np.fft.rfft(f2)
frequencies = np.linspace(0, (N - 1) / (2 * T), len(half_spectrum))

fig, ax = plt.subplots(figsize=(8, 2))
ax.plot(frequencies, np.abs(half_spectrum))
ax.set_xlabel("频率 (Hz)");
```

### 7.2 练习：在 [0, 60 Hz] 范围内做精细的放大图

你需要编写一个“频率到索引”的函数，将给定频率 \( f \) 转换为对应的半谱数组中的索引。**提示：** 由于频率间隔为 \(\Delta f = \frac{1}{T}\) 或根据具体计算公式确定，所以可以写：

```python
def freq_to_index(f, T, half_spectrum_length):
    # 总采样点 N 可以由 half_spectrum_length 反推出 N = 2*(half_spectrum_length-1)（对于偶数情况）
    N = 2 * (half_spectrum_length - 1)
    df = 1 / T  # 或者 df = (N - 1) / (2*T) / (half_spectrum_length-1)
    return int(round(f / df))
  
# 示例：在 [0,60 Hz] 范围内
index_min = freq_to_index(0, T, len(half_spectrum))
index_max = freq_to_index(60, T, len(half_spectrum))

frequencies_zoom = frequencies[index_min:index_max]
spectrum_zoom = half_spectrum[index_min:index_max]

fig, ax = plt.subplots(figsize=(8, 2))
ax.plot(frequencies_zoom, np.abs(spectrum_zoom))
ax.set_xlabel("频率 (Hz) (0-60 Hz)");
```

---

## 8. 采样定理（Shannon 与 Nyquist）

### 8.1 基本概念

在信号离散化（采样）时，采样频率 \(\nu\) 表示每秒采样的点数。两个采样点之间的时间间隔为 \( \Delta t = 1/\nu \)。

**香农采样定理**指出：
为了完整地恢复一个带限信号（其最高频率为 \( f_{\max} \)），采样频率必须至少为 \( 2 f_{\max} \)（即奈奎斯特采样率）。
换句话说，对于一个采样频率 \(\nu\)（如 44100 Hz），信号中最高频率不得超过 \(\nu/2\)（奈奎斯特频率）。

### 8.2 示例信号与采样效果

设定一个周期性信号（但只在 \( T \) 秒内绘制）：
\[
\text{signal}(t) = \sin(8 \cdot 2\pi t) + 0.5 \sin(20 \cdot 2\pi t)
\]
可以看出，这个信号包含两个频率分量，最高频率为 20 Hz。

```python
def signal(t):
    return np.sin(8 * 2 * np.pi * t) + 0.5 * np.sin(20 * 2 * np.pi * t)

T = 4  # 信号时长 4 秒
t_smooth = np.linspace(0, T, 1000)
signal_smooth = signal(t_smooth)

fig, ax = plt.subplots(figsize=(10, 1))
ax.plot(t_smooth, signal_smooth);
```

利用平滑信号计算半谱：

```python
half_amplitude_spectrum = np.abs(np.fft.rfft(signal_smooth))
freqs = np.linspace(0, len(t_smooth) / (2 * T), len(half_amplitude_spectrum))
plt.plot(freqs, half_amplitude_spectrum);
plt.xlabel("频率 (Hz)");
```

### 8.3 不同采样率下的效果

对同一信号，用不同采样率（即不同每秒采样点数）进行采样，并观察时域图形与频域图形的变化。采样率越高，重构效果越好；采样率低于 \(2 \times 20 = 40\) Hz 时，会出现混叠现象。

```python
sampling_rates = [200, 100, 50, 44, 42, 40, 38, 36, 20, 10]
nb = len(sampling_rates)
fig, axs = plt.subplots(nb, 1, figsize=(8, nb), sharex=True)

for i, rate in enumerate(sampling_rates):
    t_sampled = np.linspace(0, T, rate * T, endpoint=False)
    axs[i].plot(t_sampled, signal(t_sampled), ".-")
    axs[i].set_title("采样率: %d Hz" % rate)

fig.tight_layout()
```

观察频谱：

```python
fig, axs = plt.subplots(nb, 1, figsize=(8, nb))
for i, rate in enumerate(sampling_rates):
    t_sampled = np.linspace(0, T, rate * T, endpoint=False)
    spectrum = np.abs(np.fft.rfft(signal(t_sampled))) / len(t_sampled)
    freqs_sampled = np.linspace(0, len(t_sampled) / (2 * T), len(spectrum))
    axs[i].plot(freqs_sampled, spectrum, ".")
    axs[i].set_title("采样率: %d Hz" % rate)

fig.tight_layout()
```

可以观察到，当采样率低于 40 Hz 时，频谱中最高频率成分出现“向右反弹”（混叠现象），高频分量被错误地映射到低频区，导致信息丢失。

### 8.4 理论推导与混叠现象解释

连续信号 \( f(t) \) 的傅里叶展开为：
\[
f(t) = \sum_{j \in \mathbb{Z}} \alpha_j \, e^{2i\pi \frac{j t}{T}}
\]
若对 \( f \) 在 \( t_n = \frac{nT}{N} \)（\( n=0,\ldots,N-1 \)）处采样，得到离散信号 \( u_n = f(t_n) \)。那么用离散傅里叶展开可以写为：
\[
u_n = \sum_{k=0}^{N-1} \beta_k \, e^{2i\pi \frac{k n}{N}},
\]
通过代入 \( f(t_n) \) 的傅里叶展开，并整理（令 \( j = k + qN \)），可得：
\[
\beta_k = \sum_{q \in \mathbb{Z}} \alpha_{k+qN}.
\]
这说明：如果原始连续谱中除 \( |j| < N/2 \) 外还有非零分量，那么在计算离散傅里叶变换时，相隔 \( N \) 的频率成分会“叠加”在一起，从而引起混叠。

> **练习 (1♥)：**
> **问题：** 为什么 44100 Hz 是一个合理的采样率？
> **答案：** 人耳通常能听到的频率范围约为 20 Hz 至 20000 Hz。根据采样定理，采样率至少应为最高频率的两倍，即至少 40000 Hz。44100 Hz 超过这一最低要求，因此能够较好地捕捉和重构可听频率范围内的所有信息。

> **练习 (2♥)：**
> **问题：** 在前面的混叠现象中，我们观察到半谱中存在一个“反弹”现象。请用完整的傅里叶变换（即 `np.fft.fft` 而不是 `np.fft.rfft`）重新绘制谱图，观察整个频谱（正负频率）的情况，从而更直观地看到混叠效应。
> **提示：** 使用 `np.fft.fft` 得到的频谱包含了负频率部分，由于实信号满足 Hermitian 对称性，其负频率部分是正频率部分的共轭。绘图时可以同时显示正负频率轴。

例如：

```python
# 计算完整的傅里叶变换
full_spectrum = np.fft.fft(signal(t_sampled))
# 对应的频率轴：采样率为 rate，长度为 len(t_sampled)
freqs_full = np.fft.fftfreq(len(t_sampled), d=1/rate)

# 将频率排序
idx = np.argsort(freqs_full)
plt.figure(figsize=(8,3))
plt.plot(freqs_full[idx], np.abs(full_spectrum)[idx], '.-')
plt.xlabel("频率 (Hz)")
plt.title("完整频谱（正负频率）");
```

---

这份笔记从连续傅里叶级数入手，详细介绍了离散傅里叶变换和 FFT 的计算、实信号的 Hermitian 对称性、如何仅存储半谱以及采样定理与混叠现象的理论和实践。希望这份笔记能帮助你深入理解傅里叶变换在信号处理中的应用。
