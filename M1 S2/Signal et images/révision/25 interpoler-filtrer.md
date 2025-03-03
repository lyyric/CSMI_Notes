下面提供一份用中文、通俗语言重写的笔记，内容保留了原法语笔记中的所有细节与逻辑，并补全了留空的习题部分。

---

# 插值与滤波——周期信号的采样、傅里叶变换、插值与滤波

在这里，我们讨论的是定义在区间 [0, T] 上的周期信号（这里取 T=5），并探讨如何对其进行采样、傅里叶变换、插值以及滤波。整个过程实际上揭示了采样过程与频谱“干涉”（或混叠）的关系。

---

## 1. 信号采样

### 1.1 信号函数定义

首先，我们定义一个信号函数：
  
$$
\text{signal\_fn}(t) = \sin(2\times2\pi t) + 0.5\sin(4\times2\pi t) + 0.2\sin(10\times2\pi t)
$$

这意味着信号由三个正弦波组成，其频率分别为 2Hz、4Hz 和 10Hz（注意：这里系数前面的 2、4、10 是“谐波数”，而内部又有一个 2π，用来保证周期性）。

### 1.2 高采样率（平滑采样）

我们先用非常高的采样率（每秒 1000 个点，即 `sampling_rate_smooth=1000`）在 [0, T) 上采样，得到一个平滑且精确的信号表示：

```python
T = 5
sampling_rate_smooth = 1000
t = np.linspace(0, T, sampling_rate_smooth * T, endpoint=False)
signal_smooth = signal_fn(t)
```

绘制出的曲线平滑连续，能精确反映信号细节。

### 1.3 低采样率（子采样）

接下来，我们模拟一种“次级”采样情况，比如原始记录经过压缩后，只留下较少的采样点（每秒仅 25 个点，即 `sampling_rate_raw=25`）：

```python
sampling_rate_raw = 25
t = np.linspace(0, T, sampling_rate_raw * T, endpoint=False)
signal_raw = signal_fn(t)
```

由于采样点稀疏，信号的图形会显得“断断续续”，这也为后续讨论的混叠问题埋下伏笔——低采样率可能会导致高频成分“卷入”低频区域，即发生频谱折叠（aliasing）。

---

## 2. 观察傅里叶变换

为了直观了解采样对信号频谱的影响，我们分别对平滑信号和低采样信号进行快速傅里叶变换（FFT）。

这里定义了一个辅助函数 `plot_half_amplitude_spectrum`，它计算实数信号的正频率部分（使用 `np.fft.rfft`）并归一化，然后绘制出各频率处的振幅：

```python
def plot_half_amplitude_spectrum(ax, signal, title, max_freq_plotted):
    N = len(signal)
    # 计算显示的最大频率索引，注意这里 max_freq_plotted 与 T 的关系
    n_max_for_plot = int(max_freq_plotted * T)
    
    half_spectrum = np.fft.rfft(signal) / N
    freqs = np.linspace(0, (N-1)/2/T, len(half_spectrum))
    
    ax.plot(freqs[:n_max_for_plot], np.abs(half_spectrum)[:n_max_for_plot])
    ax.set_title(title)
```

绘图结果会显示：  
- **平滑信号**的频谱图细节丰富，各个正弦波成分清晰可见；  
- **低采样信号**的频谱图则可能出现额外的峰值，这正是由于采样不足导致的混叠现象。

---

## 3. 插值：利用傅里叶零填充恢复高分辨率信号

### 3.1 思路

对于周期信号，我们可以利用其傅里叶展开：  
- 计算原始信号的傅里叶系数；  
- 在频域上进行“零填充”（即延长傅里叶系数数组，未提供信息的部分补零）；  
- 通过逆傅里叶变换（IFFT）恢复出更多采样点的信号，从而达到插值的目的。

这种方法基于信号在圆周（周期）上的定义，故称为**圆形谐波插值**。

### 3.2 插值类的实现

下面给出一个 `CircularHarmonicInterpolator` 类，其中包括：

- `compute_coef(x__x, x__value)`：对给定采样点和对应信号值计算傅里叶系数（使用 `np.fft.rfft` 并归一化）。
- `reconstruction(n_x_)`：给定目标采样点数（要求比原始采样点多），先进行零填充，再用 `np.fft.irfft` 得到插值后的信号。

```python
class CircularHarmonicInterpolator:
    def compute_coef(self, x__x, x__value):
        self.coef = np.fft.rfft(x__value) / len(x__x)
        self.xmax = x__x[-1]
        self.nx = len(x__x)

    def reconstruction(self, n_x_):
        assert n_x_ > self.nx
        coef_prolongated = np.zeros([n_x_ // 2 + 1], dtype=np.complex128)
        coef_prolongated[:len(self.coef)] = self.coef
        x__value_ = np.fft.irfft(coef_prolongated, n=n_x_) * n_x_
        x__x_ = np.linspace(0, self.xmax, n_x_, endpoint=False)
        return x__x_, x__value_
```

例如，我们利用低采样率（25Hz）数据计算傅里叶系数，然后插值恢复到 10000 个采样点，可以同时观察原始采样点与重构后平滑信号的对比。

---

## 4. 滤波：去除高频噪声

### 4.1 带噪信号

假设我们在原有信号上加入一个额外的高频噪声成分（100Hz）：

$$
\text{noisy\_signal\_fn}(t) = \sin(2\times2\pi t) + 0.5\sin(4\times2\pi t) + 0.2\sin(10\times2\pi t) + 0.15\sin(100\times2\pi t)
$$

这样，高频成分会使得信号受到噪声干扰。我们同样分别以高采样率和平采样率记录此信号，并用 FFT 观察其频谱。通常会在频谱中看到一个异常的峰值（对应 100Hz 处），特别是在低采样率数据中更明显。

### 4.2 插值与滤波结合

为了抑制高频噪声，我们可以只保留傅里叶系数中的低频部分。也就是说，在计算傅里叶系数后，我们仅保留前面一部分（例如前 40 个系数），然后再进行插值重构。这样实际上起到了**滤波**的作用，去掉了高频噪声。

为此，定义如下类：

```python
class CircularHarmonicInterpolatorAndFilter:
    def __init__(self, max_n_coef_kept):
        self.max_n_coef_kept = max_n_coef_kept

    def compute_coef(self, x__x, x__value):
        self.coef = np.fft.rfft(x__value) / len(x__x)
        # 只保留前 max_n_coef_kept 个系数，即低频部分
        self.coef = self.coef[:self.max_n_coef_kept]
        self.xmax = x__x[-1]
        self.nx = len(x__x)

    def reconstruction(self, n_x_):
        assert n_x_ > 2 * len(self.coef)
        coef_prolongated = np.zeros([n_x_ // 2 + 1], dtype=np.complex128)
        coef_prolongated[:len(self.coef)] = self.coef
        x__value_ = np.fft.irfft(coef_prolongated, n=n_x_) * n_x_
        x__x_ = np.linspace(0, self.xmax, n_x_, endpoint=False)
        return x__x_, x__value_
```

使用时，例如设置 `max_n_coef_kept=40`，这样就能有效去除高于这一频率范围的噪声，使重构信号更加平滑。

---

## 5. 采样与干涉（混叠）的关系

在许多实际应用中，采样（尤其是低采样率采样）会引起 aliasing，也可以看作是一种“干涉”现象。下面通过一个例子来说明这一点。

### 5.1 天文信号案例

假设你正在做天文研究，导师给你一个微弱信号进行分析，但由于能量非常低，采样仪器只能以 100Hz 的采样率记录数据。对话可能如下：

- **你**：老师，这个采样率太低了，肯定发生了谱折叠（aliasing）。
- **导师**：没错，但你注意到，这个信号理论上的频率范围是在 60Hz 到 90Hz。
- **你**：哦，我明白了……

在这种情况下，虽然仪器的采样率较低，但如果我们事先知道信号只存在于某个特定的频段内，就可以利用这一信息来重构或滤除不需要的成分。

### 5.2 利用 aliasing 产生干涉

实际上，在很多科学领域中，我们甚至会利用 aliasing 来“观察”那些原本超出检测器带宽的高频信号。方法包括：
- **乘性或加性干涉**：在采样前后对信号进行某种形式的调制，使得高频成分通过频谱的折叠进入低频区域。
- **多传感器采样**：使用多个相隔一定距离的检测器，捕获信号的不同版本，从而通过干涉原理提取更多信息。

这种利用 aliasing 产生干涉效应的方法在成像、通信甚至天文观测中都有应用。

---

## 6. 练习题

为帮助大家更深入理解，下面给出两个思考题，并附上参考答案：

1. **为什么采样可以看作是一种干涉？**

   **参考答案：**  
   采样过程实际上是将连续信号与一个周期性的脉冲序列相乘（在时间上），这相当于在频域中将信号的频谱复制多次（周期性重复）。这种频谱的重复与重叠（如果采样率不够高，高频成分会“折叠”到低频区域）就形成了一种“干涉”现象，即不同频率成分相互叠加，导致原始信号信息混杂在一起。

2. **下图中网格的作用是什么？**  
   （图中网格的尺度为 $10\,\mu m \times 10\,\mu m$，图片来源：[Nanosurf](https://www.nanosurf.com/en/application/photoresin-interference-grid)）

   **参考答案：**  
   这种网格通常用于校准或检测显微镜、扫描探针等设备的分辨率和成像精度。由于网格的尺寸已知，可以通过拍摄图像来验证设备的采样精度、光学畸变以及可能的干涉效应。换句话说，借助已知尺寸的标准网格，我们能确保测量系统在实际实验中能够准确捕捉细微结构。

---

## 总结

- **采样**：高采样率能准确还原信号，而低采样率可能引发混叠现象。
- **傅里叶变换**：通过 FFT 我们可以观察到信号的频谱，理解各频率成分的作用。
- **插值**：利用傅里叶零填充方法，可以在频域上延拓信号，实现高分辨率重构。
- **滤波**：在频域中仅保留低频成分，可以有效滤除高频噪声。
- **干涉与 aliasing**：采样过程本质上会导致频谱重复，从而产生类似于干涉的现象，而这在某些领域可以被巧妙利用。

希望这份详细的中文笔记能帮助大家更好地理解周期信号的采样、傅里叶变换、插值以及滤波的原理，同时认识到采样过程中 aliasing 与干涉之间的密切联系。