这组文件（6-1 到 6-3）带你进入了**代理模型 (Metamodeling)** 的高级领域，特别是 **克里金法 Kriging / Gaussian Process Regression**。

**为什么要用它？**
之前的教程里，我们直接用物理模型  算结果。但如果  运行一次需要 1 小时，你不可能跑 10,000 次蒙特卡洛模拟。
**克里金法**的核心思想是：用几百个样本点训练一个**“数学替身”**（Metamodel），这个替身算得飞快，而且它还能告诉你它哪里算得准、哪里不准。

我同样把它拆解为**三个进阶步骤**：

---

### 第一步：理解核心引擎 (高斯过程)

**对应文件：** `6-1-Processus-gaussiens-Solution.ipynb`

克里金的背后是**高斯过程 (Gaussian Process)**。简单说，我们不仅假设数据点服从正态分布，我们假设**整个函数曲线**都服从某种概率分布。

* **协方差模型 (Covariance Model)**：这是核心。它定义了点与点之间的“相似度”。离得近的点，函数值应该也接近（平滑性）。
* `SquaredExponential` (高斯核)：生成的曲线非常平滑。
* `MaternModel`：生成的曲线可以粗糙一些，更符合物理现实。



**核心代码：**

```python
import openturns as ot

# 1. 定义协方差模型 (决定曲线的光滑程度)
# scale=[1.5] 控制相关长度，amplitude=[3.5] 控制波动幅度
myModel = ot.SquaredExponential([1.5], [3.5])

# 2. 定义时间网格 (在哪里观察)
myTimeGrid = ot.RegularGrid(0.0, 0.1, 100)

# 3. 创建高斯过程并生成一条随机轨迹
process = ot.GaussianProcess(myModel, myTimeGrid)
sample = process.getRealization() # 生成一次可能的函数形态

```

> **通俗理解**：你在画图，你告诉计算机：“画一条波浪线，波动幅度大概是 3.5，每隔 1.5 秒波形可能会变化一次。”

---

### 第二步：训练替身 (克里金模型)

**对应文件：** `6-2-Krigeage-Solution.ipynb`

这是工业界最常用的部分。你有一组昂贵的实验数据（训练集），你要训练一个克里金模型来预测未知点。

* **趋势 (Trend)**：数据的整体走向（比如常数、线性）。通常用 `ConstantBasisFactory`。
* **训练**：`KrigingAlgorithm` 会自动调整协方差模型的参数，让模型最贴合你的数据。
* **验证**：用  $Q-squared$ 分数来衡量预测准不准（类似于 ，越接近 1 越好）。

**核心代码：**

```python
# 1. 准备数据 (X_train, Y_train 是你已有的昂贵实验数据)
dimension = 4
basis = ot.ConstantBasisFactory(dimension).build() # 假设均值是常数
covModel = ot.SquaredExponential([1.0]*dimension, [1.0]) # 初始猜测

# 2. 训练克里金模型
algo = ot.KrigingAlgorithm(X_train, Y_train, covModel, basis)
algo.run()
result = algo.getResult()
metamodel = result.getMetaModel() # 这就是你的“数学替身”

# 3. 验证模型质量 (Q2 score)
# 用测试集 (X_test, Y_test) 来看它预测得准不准
val = ot.MetaModelValidation(Y_test, metamodel(X_test))
print(f"Q2 Score: {val.computePredictivityFactor()}") 

```

---

### 第三步：用替身解决难题 (Ishigami 敏感性分析)

**对应文件：** `6-3-Exercice-Ishigami-Solution.ipynb`

这一步展示了为什么要用克里金。

* **场景**：我们要对 Ishigami 函数（一个高度非线性的测试函数）做敏感性分析（算 Sobol 指数）。
* **问题**：Sobol 分析需要调用函数几万次，如果原函数很慢，根本算不出来。
* **方案**：
1. 先跑几百次原函数，训练好克里金模型。
2. 用克里金模型（算一次只要 0.0001秒）去跑那几万次 Sobol 分析。



**核心代码逻辑：**

```python
# 1. 训练好 metamodel (同上一步)
# ...

# 2. 用 metamodel 代替真函数进行 Sobol 分析
# 注意：这里传给 Sobol 算法的是 krigeageMM (替身)，而不是原函数 g
estimator = ot.SaltelliSensitivityAlgorithm()
sobol_algo = ot.SobolSimulationAlgorithm(input_distribution, metamodel, estimator)

# 3. 运行大量模拟 (现在非常快)
sobol_algo.setMaximumOuterSampling(1000) # 内部其实跑了 N * (2d+2) 次
sobol_algo.run()
print(sobol_algo.getResult().getFirstOrderIndices())

```

### 总结：你的 Level Up

恭喜你，你已经接触到了 OpenTURNS 最强大的功能之一。

* **Level 1-3**: 只是在算简单的统计量。
* **Level 4**: 找到了谁是关键变量（敏感性）。
* **Level 6 (现在)**: **学会了“偷懒”**。当模型跑不动时，造一个克里金替身来帮你在几秒钟内完成成千上万次的计算任务。这是处理复杂工程问题（如有限元仿真、流体计算）的必备技能。