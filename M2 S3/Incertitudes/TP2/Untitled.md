### 第一步：定制你的“积木” (高级分布)

**对应文件：** `2-1-Approfondissement-distributions-1D.ipynb`

有时候标准的高斯分布不够用。比如，河流的流量不可能小于 0，但高斯分布是包含负无穷的。这时你需要“截断”它。或者你需要自己写一个奇怪的分布。

* **核心技能**：截断分布（切掉不要的部分）、混合分布、自定义分布。

**核心代码：**

```python
import openturns as ot

# 1. 截断分布 (比如：数值必须大于 0)
dist = ot.Normal(0.0, 1.0)
# 创建一个只取 [0, +无穷] 范围的截断正态分布
truncated_dist = ot.TruncatedDistribution(dist, 0.0, ot.TruncatedDistribution.LOWER)

# 2. 分布的算术运算 (比如 Z = 2X + 3)
# 之前的 1-5 是用函数做的，这里直接对对象操作，更简单
X = ot.Normal()
Z = 2.0 * X + 3.0 

```

> **通俗理解**：以前你只能买标准的乐高积木，现在学会了怎么把积木切开、或者把两个积木粘在一起用。

---

### 第二步：逆向工程 (数据拟合)

**对应文件：** `2-2-Ajustement-distribution-1D.ipynb`

这是工业界最常用的功能。你手头有一份 Excel 数据（比如 100 个零件的尺寸），你想知道它服从什么分布（是正态？还是威布尔？）。

* **Factory (工厂类)**：专门负责把数据变成分布。
* **FittingTest (拟合测试)**：用来打分，告诉你这个分布拟合得好不好。

**核心代码：**

```python
# 假设 sample 是你的一堆实验数据
sample = ot.Sample([[10.2], [10.5], [9.8], [10.1], [10.3]])

# 1. 自动拟合正态分布 (告诉我不确定性参数是多少)
factory = ot.NormalFactory()
fitted_dist = factory.build(sample)
print(fitted_dist) # 输出：Normal(mu=10.18, sigma=0.23)

# 2. 检验一下拟合得好不好 (Kolmogorov-Smirnov 测试)
# 结果主要看 p-value，如果 > 0.05 通常表示接受假设
result = ot.FittingTest.Kolmogorov(sample, fitted_dist)
print(f"P-value: {result.getPValue()}")

```

> **通俗理解**：这就是“猜谜语”。给你一串数字，让你猜它们背后的生成规律（分布）是什么。

---

### 第三步：实战演练 (拟合练习)

**对应文件：** `2-3-cantilever-beam-fit-distribution.ipynb`

这是一个具体的工程案例。

* **场景**：有一根悬臂梁，你要分析它的杨氏模量 。
* **任务**：给你一堆  的测量数据，找出最适合它的分布。
* **方法**：对比了“直方图”、“参数拟合（Beta分布）”和“核密度估计（Kernel Smoothing）”。

**核心代码思路：**

```python
# 1. 核密度估计 (万能钥匙)
# 当你不知道数据服从什么形状时，用这个能画出一条很贴合的曲线，但不一定有解析公式
ks = ot.KernelSmoothing()
fitted_dist_ks = ks.build(sample)

# 2. 对比画图
graph = fitted_dist_ks.drawPDF()
# 将原始数据的直方图叠加上去看重合度

```

---

### 第四步：它会断吗？ (应力-强度干涉)

**对应文件：** `2-4-Exercice-convolution-axial-stressed-beam.ipynb`

这是可靠性分析中最经典的模型：**应力 (Load) vs 强度 (Strength)**。

* **场景**：如果 应力  > 强度 ，结构就坏了。
* **目标**：计算失效概率 。
* **方法**：这就是所谓的**卷积 (Convolution)** 问题，即两个随机变量相减。

**核心代码：**

```python
# R 是强度分布 (比如均值 300)，S 是应力分布 (比如均值 100)
R = ot.LogNormal(...) 
S = ot.Normal(...)

# 定义余量 Z = R - S (结构还能承受多少力)
Z = R - S 

# 计算失效概率 (即 Z <= 0 的概率)
# computeCDF(0.0) 就是计算累积概率分布在 0 处的值
prob_failure = Z.computeCDF(0.0) 
print(f"失效概率: {prob_failure}")

```

> **通俗理解**：你的盾牌强度是不确定的，敌人的攻击力也是不确定的。算出你的盾牌被打碎的概率。

---

### 第五步：不仅是正态分布 (Copula 联合分布)

**对应文件：** `2-5-Exercice-copule-gaussienne.ipynb`

在 1-2 中，我们假设变量是独立的。但现实中，**变量往往是相关的**（比如身高高的人通常体重也大）。

* **Copula (连接函数)**：它是把单纯的“边缘分布”缝合在一起的“胶水”，用来描述相关性。
* **Gaussian Copula**：最常用的胶水，用来模拟类似高斯分布的相关性结构。

**核心代码：**

```python
# 1. 定义相关矩阵 (比如 0.8 的强相关)
R = ot.CorrelationMatrix(2)
R[0, 1] = 0.8 

# 2. 定义 Copula (胶水)
copula = ot.NormalCopula(R)

# 3. 定义边缘分布 (材料 A 和 材料 B 的特性)
marginals = [ot.Exponential(1.0), ot.Exponential(1.0)]

# 4. 组合！(有相关性的多变量分布)
distribution = ot.ComposedDistribution(marginals, copula)

# 5. 生成数据 (你会发现画出来的点是沿对角线分布的)
sample = distribution.getSample(100)

```

> **通俗理解**：如果  和  没关系，它们画出来像一团散沙。如果用了 Copula 加上相关性，它们画出来可能像一个压扁的椭圆。

---

### 总结：你的进阶学习路线

1. **先看 `2-2**`：学会用 `Factory` 把手中的数据变成分布，这是数据分析的第一步。
2. **接着看 `2-4**`：学会算 ，这是工程中最核心的“算风险”的逻辑。
3. **看 `2-1**`：当你发现标准分布不够用时，回来查这个“字典”。
4. **最后看 `2-5**`：当你需要处理多变量且它们之间有关系时，再来研究 Copula。

这套教程从**基础建模**（第1部分）通过，现在带你进入了**数据驱动与风险评估**（第2部分）的大门。