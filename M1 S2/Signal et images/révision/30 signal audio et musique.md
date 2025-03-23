# 音频信号笔记

本笔记主要介绍如何处理和分析音频信号，包括信号的读取、可视化、频谱分析、合成音生成、平滑（crescendos 与 decrescendos）过渡以及音频滤波处理。笔记中给出的代码大多为 Python 实现，并依赖于 NumPy、Matplotlib、SciPy、soundfile 等常用库。

---

## 1. 从 GitHub 获取数据

首先，我们通过 git 命令从 GitHub 上下载（或更新）音频数据。这里的仓库包含一些示例音频信号文件（例如 `bornwild.wav` 等）。

```python
import os

if not os.path.exists("assets_signal"):
    print("创建 assets_signal 目录")
    !git clone https://github.com/vincentvigon/assets_signal
else:
    print("更新 assets_signal 目录")
    %cd assets_signal
    !git pull https://github.com/vincentvigon/assets_signal
    %cd ..
```

---

## 2. 基本导入和设置

接下来重置工作环境，导入必要的包，并设置一些显示参数。注意这里我们还安装了 `PySoundFile` 包来读取音频文件。

```python
%reset -f
import numpy as np
import matplotlib.pyplot as plt
import IPython

np.set_printoptions(linewidth=500, precision=3, suppress=True)
plt.style.use("default")

# 安装并导入 soundfile
!pip install PySoundFile
import soundfile as sf
```

---

## 3. 读取和绘制音频信号

### 3.1 读取音频文件

使用 `soundfile` 读取 `assets_signal/bornwild.wav` 文件。该文件为立体声（左右声道各一列）。

```python
sound, samplerate = sf.read('assets_signal/bornwild.wav')
print(sound.shape, samplerate)
```

### 3.2 绘制波形图

由于采样率较高，直接绘制所有样本会呈现出快速振荡的情况，因此我们绘制的是波形轮廓图。下面代码分别绘制左、右两个声道，并增加了横坐标时间刻度。

- **练习 1 $(1\heartsuit)$：** 求该音频文件的时长。  
  **解答思路：** 时长 = 音频样本点数 ÷ 采样率。

- **练习 2 $(1\heartsuit)$：** 在波形图上添加横坐标时间刻度。  
  **解答思路：** 横坐标刻度根据采样率转换，将样本索引转换为时间（秒）。

例如，可计算时长并添加时间刻度如下：

```python
duration = sound.shape[0] / samplerate
print("音频时长：", duration, "秒")

# 绘制时域波形，并将横坐标转换为时间（秒）
time_axis = np.linspace(0, duration, sound.shape[0])
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 2), sharex=True)
ax0.plot(time_axis, sound[:,0])
ax1.plot(time_axis, sound[:,1])
ax0.set_title("左声道")
ax1.set_title("右声道")
ax1.set_xlabel("时间 (秒)")
fig.tight_layout()
plt.show()
```

### 3.3 绘制信号局部细节

为了观察信号的具体振荡情况，我们只绘制信号的开头部分（例如前 200 个样本），从而能清楚看到正弦波形的周期性波动。

```python
fig, ax = plt.subplots(figsize=(8, 1))
ax.plot(time_axis[:200], sound[:200, 0])
ax.set_title("信号起始部分（左声道）")
ax.set_xlabel("时间 (秒)")
plt.show()
```

---

## 4. 频谱分析

### 4.1 计算半幅频谱

对于左声道信号 `sound[:,0]`，使用快速傅里叶变换（FFT）计算其频谱。我们只计算非负频率部分，即半幅频谱。

```python
sound0 = sound[:, 0]
half_spectrum = np.fft.rfft(sound0)
N = len(half_spectrum)
freqs = np.linspace(0, samplerate/2, N)
fig, ax = plt.subplots(figsize=(8, 2))
ax.plot(freqs, np.abs(half_spectrum) / N)
ax.set_xlabel("频率 (Hz)")
ax.set_ylabel("幅度 (归一化)")
plt.show()
```

- **练习 3 $(1\heartsuit)$：** 在半幅频谱图中，观察 0 Hz 处的“尖峰”（pick at 0）的意义。  
  **解答：** 该尖峰表示直流分量（信号的平均值），若信号有直流偏移则会在 0 Hz 处出现明显峰值。

---

## 5. 生成合成音

### 5.1 生成带有谐波的音频

我们以参考音 A（440Hz）为例，生成正弦波，并叠加几个谐波（频率分别为 $2\times 440$、$3\times 440$、$5\times 440$），每个谐波的振幅依次降低。同时，生成一个“crescendo”效果，使声音逐渐增大。

```python
samplerate = 44100
duration = 3  # 秒
t = np.linspace(0, duration, duration * samplerate)

# 添加谐波：谐波频率与主频的倍数，以及对应的强度
harmonics = [2, 3, 5]
harmonics_intensity = [1/2, 1/4, 1/3]

# 基本正弦波：主频为 440Hz
signal = np.sin(2 * np.pi * 440 * t)
# 叠加谐波
for h, h_i in zip(harmonics, harmonics_intensity):
    signal += h_i * np.sin(h * 2 * np.pi * 440 * t)

# 绘制前200个样本观察波形
fig, ax = plt.subplots(figsize=(8, 1))
ax.plot(t[:200], signal[:200])
ax.set_title("合成音信号（前200个样本）")
plt.show()
```

为了使音量始终保持在合理范围（例如不超过 1），我们对信号进行归一化处理，同时制作一个“crescendo”效果，即声音逐渐变大：

```python
maxiSig = np.max(np.abs(signal))
# 制作一个从0到1的渐变音量，且音量缩放后不超过原来的强度范围
volume = t / t[-1] / maxiSig / 2
signal *= volume

# 绘制整个信号的波形
fig, ax = plt.subplots(figsize=(8, 1))
ax.plot(t, signal)
ax.set_title("添加渐强效果后的合成音信号")
ax.set_xlabel("时间 (秒)")
plt.show()

# 将生成的音频写入文件
sf.write('la440.wav', signal, samplerate)
IPython.display.Audio('la440.wav')
```

### 5.2 生成减弱音（Decrescendo）练习

**练习 4 $(1\heartsuit)$：** 请创建一个与“crescendo”相反的“decrescendo”（声音逐渐变弱）的音频效果。  
**思路提示：** 只需要将音量因子设置为从 1 逐渐减小到 0，例如使用 `volume = (1 - t/t[-1])` 来调制信号即可。  

示例代码如下：

```python
# 使用同样的合成信号，但音量从 1 渐减至 0
volume_dec = (1 - t / t[-1]) / maxiSig  # 此处可以根据需要调节衰减速率
signal_dec = np.sin(2 * np.pi * 440 * t)
for h, h_i in zip(harmonics, harmonics_intensity):
    signal_dec += h_i * np.sin(h * 2 * np.pi * 440 * t)
signal_dec *= volume_dec

fig, ax = plt.subplots(figsize=(8, 1))
ax.plot(t, signal_dec)
ax.set_title("减弱效果（decrescendo）的合成音信号")
ax.set_xlabel("时间 (秒)")
plt.show()

sf.write('la440_decrescendo.wav', signal_dec, samplerate)
IPython.display.Audio('la440_decrescendo.wav')
```

### 5.3 制作带平滑过渡的音乐

音乐制作中常常需要将不同的音符连接起来，而过渡部分需要平滑处理，否则会产生不自然的“爆破声”（tack）。下面给出一个例子：

1. 生成两段不同频率的正弦波（signal0 和 signal1），分别持续一定时间。  
2. 用窗口函数控制过渡区域。  
3. 初始设置为生硬过渡，再用渐变函数制作平滑过渡。

```python
samplerate = 11025
duration0 = 1.3
duration1 = 2.6

nb0 = int(duration0 * samplerate)
nb1 = int(duration1 * samplerate)
nb = nb0 + nb1
t = np.linspace(0, duration0 + duration1, nb)

signal0 = np.sin(2 * np.pi * t * 2)
signal1 = np.sin(2 * np.pi * t * 5)

window0 = np.zeros(nb)
window1 = np.zeros(nb)

# --- 生硬（abrupte）过渡 ---
window0[:nb0] = 1
window1[nb0:] = 1

def plot_all():
    plt.subplot(3, 1, 1)
    plt.plot(window0, label='window0')
    plt.plot(window0 * signal0, label='window0 * signal0')
    plt.ylim([-2, 2])
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(window1, label='window1')
    plt.plot(window1 * signal1, label='window1 * signal1')
    plt.ylim([-2, 2])
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(window0 + window1, label='window0+window1')
    plt.plot(window0 * signal0 + window1 * signal1, label='合成信号')
    plt.ylim([-2, 2])
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_all()

sf.write("raw_transi.wav", window0 * signal0 + window1 * signal1, samplerate)
IPython.display.Audio("raw_transi.wav")
```

以上声音过渡由于直接拼接，过渡处噪声明显。接下来用渐变函数使过渡变得平滑：

```python
demiTransition = int(0.1 * samplerate)
montee = np.linspace(0, 1, 2 * demiTransition)
descente = 1 - montee

# 修改窗口函数，实现平滑过渡
window0 = np.zeros(nb)
window1 = np.zeros(nb)
window0[:nb0 - demiTransition] = 1
window0[nb0 - demiTransition:nb0 + demiTransition] = descente
window1[nb0 - demiTransition:nb0 + demiTransition] = montee
window1[nb0 + demiTransition:] = 1

plot_all()

sf.write("soft_transi.wav", window0 * signal0 + window1 * signal1, samplerate)
IPython.display.Audio("soft_transi.wav")
```

- **练习 5 $(2\heartsuit)$：** 请重新制作两个过渡（分别对应上面的生硬与平滑过渡），但这次要求过渡过程中能听到明显的声音变化（即过渡区域内保留一定的“音符信息”而非直接归零），使过渡本身具有音乐性。  
  **解答思路：**  
  - 生硬过渡可通过直接拼接各个音符来实现；  
  - 平滑过渡则可以在过渡区内采用逐渐增大和逐渐减小的窗口函数，但同时保持音符的基本频率信息，使过渡既平滑又具有可听性。  
  可尝试在窗口函数中保留部分原始信号幅度，而非直接将一个音符完全衰减到0，再逐渐增大另一个音符。  

此外，还可以试着写一个小旋律练习：  
- **Bonus：** 写一段五个音符的旋律，要求过渡平滑。你可以先用一个长向量在对应的区间内修改，然后拼接各段，或用多个短向量拼接。  
  *评分规则：*  
  - 5个音符、平滑过渡：4 星  
  - 若能辨识出具体旋律（比如简短的名曲片段）：再加 4 星  
  - 若优化代码，只使用一个长向量进行拼接处理：再加 4 星  

（此处可根据你的创意写出具体代码实现）

---

## 6. 谱图（Spectrogram）分析

谱图是对长时段信号进行时频分析的一种方法。其基本步骤为：  
1. 将信号切成若干短时段，每一段内信号可以近似看作是稳定的正弦波组合；  
2. 对每一段计算 FFT；  
3. 将每段的频谱堆叠成矩阵，然后用颜色映射（colormap）显示出来。

下面是一个简单的示例，先对一段信号生成谱图，再对 `bornwild.wav` 中的信号生成谱图。

```python
import scipy.signal

# 构造一个示例信号，前后两部分频率不同
epsilon = 0.0001
t_small = np.arange(0, 1, epsilon)
sig_debut = 0.5 * np.sin(t_small * 2 * np.pi * 440) + np.sin(t_small * 2 * np.pi * 220)
sig_fin   = 0.5 * np.sin(t_small * 2 * np.pi * 880) + np.sin(t_small * 2 * np.pi * 440)
sig = np.concatenate((sig_debut, sig_fin))

# 计算谱图
f, t_spec, Sxx = scipy.signal.spectrogram(sig, 1/epsilon)
plt.pcolormesh(t_spec, f, Sxx)
plt.ylabel('频率 [Hz]')
plt.xlabel('时间 [秒]')
plt.title("示例信号谱图")
plt.show()
```

- **练习 6 $(3\heartsuit)$：**  
  1. 请对你自己选择的短音频文件生成谱图，并尝试描述听到的声音和谱图上看到的特征之间的联系。  
  2. 以 `bornwild.wav` 为例，观察谱图中哪些部分能体现出歌手使用了“声效”（例如回声、混响、失真等）。  
     
  **解答提示：**  
  - 分析谱图时，注意频率分布的变化。如果在某些时段出现额外的频率成分或频谱分布变得模糊，这可能反映出声效处理。  
  - 例如，在人声部分，如果看到频谱中出现宽频带噪声或者不规则的谐波分布，则可能是使用了特殊的效果处理。

```python
# 以 bornwild.wav 为例生成谱图
f, t_born, Sxx = scipy.signal.spectrogram(sound[:,0], 1/epsilon)
plt.pcolormesh(t_born, f, Sxx)
plt.ylabel('频率 [Hz]')
plt.xlabel('时间 [秒]')
plt.title("bornwild.wav 左声道谱图")
plt.show()
```

观察谱图时，若在某个时间段频率分布异常（例如能量分布较宽、边缘不清），则可推测该处可能运用了“声效”（如变声、混响、调制效果等）。

---

## 7. 声音滤波练习

接下来我们针对 `assets_signal/sound_surprise.wav` 进行一些滤波练习。首先读取该文件，并了解其基本信息：

```python
sound2, samplerate2 = sf.read('assets_signal/sound_surprise.wav')
print(sound2.shape, samplerate2)
IPython.display.Audio('assets_signal/sound_surprise.wav')
```

### 7.1 分析信号

- **练习 7：**  
  1. **信号时长：** 计算时长 = 样本数 ÷ 采样率，并说明计算依据。  
  2. **全局时域图：** 绘制整个声音信号的波形，并确保横坐标显示正确的时间单位（秒）。  
  3. **局部细节图：** 绘制信号开头部分，观察振荡细节。  
  4. **半幅频谱图：** 对信号进行 FFT，绘制半幅频谱，并设置合适的频率刻度。  
  5. **声音特征分析：** 请判断该声音的乐音名称（例如某个乐器或音符），以及该声音包含多少个谐波。  
  6. **滤波处理：** 利用 FFT，将除基频之外的所有谐波滤除，仅保留“纯净”的音调，并听取效果。  
  7. **渐强效果：** 将原始信号做一个“crescendo”处理，使其声音逐渐增大。  
  8. **谱图分析：** 绘制处理后信号的谱图，并观察变化。

**解题思路：**  
- 时长可以用 `sound2.shape[0] / samplerate2` 得到；  
- 绘制时域图时，将横坐标转换为秒；  
- 局部细节图类似前面的绘图，只选取前几个采样点；  
- FFT 处理时注意归一化及频率轴构造；  
- 判断乐音名称需要通过听觉和频谱信息对比标准频率；  
- 滤波处理可通过将 FFT 后非基频部分置零，再做逆 FFT；  
- 渐强效果与前面合成音类似，利用一个从 0 到 1 的音量包络进行调制。

可以参考如下代码框架（具体实现细节由你根据需要调整）：

```python
# 计算时长
duration2 = sound2.shape[0] / samplerate2
print("sound_surprise.wav 时长：", duration2, "秒")

# 绘制全局时域图（转换为秒）
time_axis2 = np.linspace(0, duration2, sound2.shape[0])
plt.figure(figsize=(8, 3))
plt.plot(time_axis2, sound2)
plt.xlabel("时间 (秒)")
plt.title("sound_surprise.wav 全局波形")
plt.show()

# 绘制信号起始部分，观察细节
plt.figure(figsize=(8, 2))
plt.plot(time_axis2[:200], sound2[:200])
plt.xlabel("时间 (秒)")
plt.title("sound_surprise.wav 局部波形（起始部分）")
plt.show()

# 半幅频谱图
sound2_mono = sound2[:,0] if len(sound2.shape) > 1 else sound2
spectrum2 = np.fft.rfft(sound2_mono)
N2 = len(spectrum2)
freqs2 = np.linspace(0, samplerate2/2, N2)
plt.figure(figsize=(8, 2))
plt.plot(freqs2, np.abs(spectrum2)/N2)
plt.xlabel("频率 (Hz)")
plt.title("sound_surprise.wav 半幅频谱")
plt.show()

# 关于乐音名称与谐波数量：
# 观察频谱中是否有明显的主频（基频）峰值和多个谐波峰值，
# 如果基频为 f0 ，而谱图中出现 f0, 2f0, 3f0, ... 则可认为该音具有 n 个谐波。
# 具体乐音名称可参照标准频率对照表。

# 使用 FFT 滤波，仅保留基频
spectrum_filtered = np.copy(spectrum2)
# 假设第一个峰值为基频，其他置零
spectrum_filtered[1:] = 0
pure_sound = np.fft.irfft(spectrum_filtered)
sf.write('sound_surprise_pure.wav', pure_sound, samplerate2)
IPython.display.Audio('sound_surprise_pure.wav')

# 制作渐强效果：乘以一个从 0 到 1 的包络
envelope = np.linspace(0, 1, sound2.shape[0])
sound2_crescendo = sound2_mono * envelope
sf.write('sound_surprise_crescendo.wav', sound2_crescendo, samplerate2)
IPython.display.Audio('sound_surprise_crescendo.wav')

# 绘制渐强处理后的信号谱图
f_c, t_c, Sxx_c = scipy.signal.spectrogram(sound2_crescendo, samplerate2)
plt.pcolormesh(t_c, f_c, Sxx_c)
plt.ylabel('频率 [Hz]')
plt.xlabel('时间 [秒]')
plt.title("渐强后信号的谱图")
plt.show()
```

各步骤中，需结合自己的观察进行解释。例如：  
- 当我们仅保留基频时，听到的音将变得非常纯净，谐波信息被滤除；  
- 渐强处理后，音量由静至响，谱图中能看到能量逐渐增强的趋势；  
- 乐音名称及谐波数量需要结合频谱分析和听感综合判断。

---

## 总结

本笔记详细讲解了如何通过 Python 处理音频信号，从数据获取、时域与频域分析，到音频合成、平滑过渡以及滤波处理。每一部分均附有相应的练习题，旨在帮助你更深入理解音频信号的物理意义与数字处理方法。请结合代码运行结果与谱图观察，进一步理解信号处理中的细节及背后的数学原理。
