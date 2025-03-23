## 1. NumPy 中的 Strides 概念

在 NumPy 中，每个数组都有一个 `strides` 属性，用于描述在内存中，相邻元素之间的步长（即跳过多少字节可以到达下一个元素）。例如：

```python
import numpy as np

v = np.zeros([10], np.uint8)
print(v.strides)   # 输出 (1,)，表示每个元素占 1 个字节

v = np.zeros([10,15], np.uint8)
print(v.strides)   # 输出 (15, 1)，每行跳过15个字节，每个元素占1个字节

v = np.zeros([10], np.float32)
print(v.strides)   # 输出 (4,)，因为 float32 每个占4个字节

v = np.zeros([10,15], np.float32)
print(v.strides)   # 输出 (60, 4)，每行有15个 float32，总共60字节，每个元素4字节
```

通过这些例子，可以看到 stride 的意义：它告诉我们内存中相邻数据的间隔（以字节为单位）。

---

## 2. 利用 Strides 实现滚动窗口（Rolling Window）

下面的函数 `rolling_window()` 利用 NumPy 的 `as_strided` 技巧创建一个滚动窗口视图。对一维向量来说，输入数组长度为 N，选择一个窗口长度 `window_length` 后，输出数组的 shape 为：(N - window_length + 1, window_length)。

```python
def rolling_window(data, window_length):
    output_shape = data.shape[:-1] + (data.shape[-1] - window_length + 1, window_length)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=output_shape, strides=strides)

data = np.arange(0, 11, 1)
roll = rolling_window(data, 3)
print(roll)
print("输出的 shape:", roll.shape)
```

### 习题 (2♥)
**题目：** 给出输入和输出的 shape 之间的公式，并画一个示意图说明一个长向量中包含滚动窗口。

**答案：**

- 对于一维数组：  
  若输入数据 shape 为 `(N,)`，则输出的 shape 为 `(N - w + 1, w)`，其中 w 表示 `window_length`。  
  示意图如下（假设 N = 10，w = 3）：

  ```
  输入向量：
  [ x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 ]

  滚动窗口视图：
  [ [x0, x1, x2],
    [x1, x2, x3],
    [x2, x3, x4],
    [x3, x4, x5],
    [x4, x5, x6],
    [x5, x6, x7],
    [x6, x7, x8],
    [x7, x8, x9] ]
  ```

- 对于多维数组：  
  若输入数据 shape 为 `(..., N)`（即最后一维长度为 N），则输出 shape 为 `(..., N - w + 1, w)`。也就是说，函数始终在最后一个维度上进行窗口滑动。

### 习题 (1♥)
**题目：** 当输入 `data` 不仅仅是向量，而是一般的 `ndarray`（例如矩阵）时，`rolling_window()` 会做什么？

**答案：**  
函数会在数组最后一个维度上生成滚动窗口。例如，对于一个 shape 为 `(M, N)` 的矩阵，结果 shape 为 `(M, N - w + 1, w)`，即对每一行分别生成一个滚动窗口视图。

---

## 3. TensorFlow 版本的连续窗口生成及其 NumPy 转写

原始 TensorFlow 版本通过 reshape 和切片实现连续窗口生成。其大致流程是：
1. 将时间序列按照窗口大小进行分段（保证所有分段长度相同）。
2. 对于每个不同的起始位置（不完全对齐窗口边界），分别进行切片后 reshape。
3. 最后通过 `tf.stack` 和 reshape 得到所有可能的连续窗口。

**习题：**  
**题目：** 请将上述 TensorFlow 实现的函数 `make_consecutive_windows` 翻译为 NumPy 版本。

**参考答案（NumPy 实现）：**

下面是一种可能的实现思路：
- 对输入数组 `Z`（假设 shape 为 `(batch_size, nb_t, ...)`），首先舍弃掉不能完整分割成窗口的部分；
- 利用 `rolling_window()` 对每一行生成连续窗口（注意需要在时间维度上滑动）；
- 对不同起始偏移的情况，可以采用循环累积再拼接。

```python
def make_consecutive_windows_numpy(Z, window_size):
    # 假设 Z 的 shape 为 (batch_size, nb_t, ...) 
    batch_size, nb_t = Z.shape[0], Z.shape[1]
    # 为了简单起见，这里只处理时间维度，不考虑后续多余的维度（可以扩展）
    # 首先舍弃掉尾部不足一个完整窗口的部分
    nb_complete = nb_t - window_size + 1
    # 生成滑动窗口（沿第二个维度）
    windows = rolling_window(Z, window_size)  # 这里对 Z 的最后一维进行操作，如果 Z 是多维的，可以先调整维度
    # 如果 Z 是二维 (batch_size, nb_t) ，则 windows 的 shape 为 (batch_size, nb_complete, window_size)
    return windows

# 示例：二维数据
Z = np.arange(20).reshape(1, 20)
windows = make_consecutive_windows_numpy(Z, 5)
print("连续窗口 shape:", windows.shape)
print(windows)
```

注意：原 TensorFlow 版本中对窗口的偏移进行了循环处理，使得所有可能的连续窗口（包括重叠的不对齐窗口）都被提取；用纯 NumPy 实现时也可以采用类似方法，例如对不同的起始偏移做切片再拼接，但这里给出的是一种基本实现。

---

## 4. 局部极大值检测函数：findLocalMax

下面的函数通过滚动窗口的方式检测一维数据中的局部极大值：

```python
def findLocalMax(data, windowLength):
    dataRoll = rolling_window(data, windowLength)
    print("滚动窗口数组的 shape:", dataRoll.shape)
    armax = np.argmax(dataRoll, axis=1)  # 每个窗口内最大值的位置
    print("每个窗口最大值索引数组的 shape:", armax.shape)
    where = np.where(armax == (windowLength // 2))
    print("满足条件的位置:", where)
    return where[0] + windowLength // 2
```

### 习题 (2♥)
**题目：** 解释 `findLocalMax()` 的工作原理，说明 `windowLength` 的影响，并解释为什么最好选择奇数作为窗口长度。

**答案：**  
- **工作原理：**  
  该函数首先利用 `rolling_window()` 为输入数据构建一个滚动窗口视图，然后在每个窗口内寻找最大值的位置（使用 `np.argmax`）。如果某个窗口的最大值正好位于窗口的中心位置（即 `windowLength // 2`），则认为这个中心点是局部极大值。最后返回所有满足条件的位置（需要加上窗口偏移）。
  
- **窗口长度的影响：**  
  `windowLength` 决定了检测局部极大值时考虑的邻域范围。较大的窗口会考虑更大范围内的数据变化，而较小的窗口则只关注局部细节。

- **为什么选择奇数：**  
  选择奇数窗口可以保证窗口有一个明确的中心位置（例如，窗口长度 5 的中心位置就是索引 2）。如果窗口长度为偶数，就没有对称的中心，检测“中心极大值”就会出现歧义。

### 习题 (3♥)
**题目：** 目前的函数不能检测数据序列两端的峰值，请修改该函数，增加一个可选参数 `alsoDetectBorderPick=True`，使其也能检测边界上的极大值。

**参考修改方案：**

我们可以在函数内部增加对首尾数据的额外判断。例如：  
- 如果 `alsoDetectBorderPick` 为 True，则：
  - 检查数据的第一个点是否大于其邻近的几个点（可以只比较相邻一个或几个值）。
  - 检查数据的最后一个点是否大于其左侧相邻的几个点。

下面给出一种可能的实现：

```python
def findLocalMax_modified(data, windowLength, alsoDetectBorderPick=False):
    # 检测中间部分（利用滚动窗口检测中心极大值）
    dataRoll = rolling_window(data, windowLength)
    armax = np.argmax(dataRoll, axis=1)
    center_idx = windowLength // 2
    middle_idx = np.where(armax == center_idx)[0] + center_idx  # 补偿窗口偏移
    
    if alsoDetectBorderPick:
        border_idx = []
        # 检查第一个数据点
        if data[0] > data[1]:
            border_idx.append(0)
        # 检查最后一个数据点
        if data[-1] > data[-2]:
            border_idx.append(len(data) - 1)
        # 合并中间和边界检测结果，去重并排序
        all_idx = np.unique(np.concatenate([middle_idx, np.array(border_idx)]))
        return all_idx
    else:
        return middle_idx

# 测试示例
data = np.ones(20)
data[0] = 1.5
data[3] = 2
data[4] = 2.2
data[12] = 1.1
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(len(data)), data, ".")
plt.xticks(range(len(data)))
locMax = findLocalMax_modified(data, 5, alsoDetectBorderPick=True)
print("检测到的局部极大值位置:", locMax)
plt.plot(locMax, data[locMax], "+")
plt.show()
```

---

## 5. 物理信号处理实例

假设我们有一个包含左右声道的物理信号，并且在时域上看起来左右声道差异很大。但通过傅里叶变换，我们会发现两个通道在频域上具有相同的峰值，这说明它们实际上在频谱上是相关的。

### 信号读取与绘图

```python
import soundfile as sf

signal_phy, samplerate = sf.read('assets_signal/Sample.wav')
print("信号 shape 和采样率:", signal_phy.shape, samplerate)

def plot_a_part(x, y, index):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8,4), sharex=True)
    ax0.plot(x[:index], y[:index, 0])
    ax1.plot(x[:index], y[:index, 1])
    ax0.set_title("左声道")
    ax1.set_title("右声道")
    fig.tight_layout()
    plt.show()

N = len(signal_phy)
t = np.linspace(0, N * samplerate, N)
plot_a_part(t, signal_phy, 2000)
plot_a_part(t, signal_phy, 100)
```

### 傅里叶变换与对比

注意，如果直接对整个信号调用 `np.fft.rfft(signal_phy)` 得到的结果并非我们预期的二维结果，因此我们对左右声道分别计算傅里叶变换，再合并结果。

```python
spectrum_phy0 = np.fft.rfft(signal_phy[:,0])
spectrum_phy1 = np.fft.rfft(signal_phy[:,1])
spectrum_phy = np.stack([spectrum_phy0, spectrum_phy1], axis=1)
print("频谱 shape:", spectrum_phy.shape)

freqs = np.linspace(0, samplerate/2, len(spectrum_phy0))
plt.figure()
plt.plot(freqs, np.abs(spectrum_phy[:,0]), label="左声道频谱")
plt.plot(freqs, np.abs(spectrum_phy[:,1]), label="右声道频谱")
plt.title("左右声道频谱对比")
plt.xlabel("频率 (Hz)")
plt.legend()
plt.show()
```

### 习题
1. **检测信号中的频谱峰值**  
   请检测左右声道频谱中的峰值，并要求：  
   - (2♥) 将检测到的峰值叠加在频谱图上；
   - (2♥) 将左右声道的峰值叠加在同一图中进行对比。

   **参考思路：**  
   可使用类似 `findLocalMax_modified` 的方法，对每个通道的频谱幅值（例如 `np.abs(spectrum_phy[:,0])`）构造滚动窗口检测局部极大值，找到峰值位置后使用 `plt.plot` 叠加标记。

2. **声音指纹压缩技术简介**  
   请描述一种对声音进行极端压缩的方法，该方法能够判断两个声音是否相近。  
   例如：你的方法可以判断两个信号是否实际上是 Britney Spears 的《toxic》的不同录音。如果你在2012年前有这个想法，就有可能成为著名软件 “Shazam” 的发明者。

   **答案示例：**  
   声音指纹技术的基本思想是：  
   - 首先对音频信号进行傅里叶变换，得到频谱信息；  
   - 然后提取频谱中的局部峰值（或称“地标”），这些峰值能代表信号在时间和频率上的特征；  
   - 接下来对这些峰值信息进行哈希编码，生成一个紧凑的指纹；  
   - 最后，将两个声音的指纹进行比较，若匹配度高，则说明两个声音相似。  
     
   这种方法不仅大幅降低了数据存储的需求，而且可以高效地在海量数据库中进行查找，从而实现快速识别和匹配。

---

以上就是对原法语笔记内容的中文重写，详细介绍了 NumPy 中 stride 的概念、滚动窗口的实现方法、TensorFlow 与 NumPy 的连续窗口生成、局部极大值的检测（包括对边界情况的改进）以及物理信号的频谱分析与声音指纹技术的基本原理。