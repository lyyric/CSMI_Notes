
这组文件（4-1 到 4-5）带你进入了**敏感性分析 (Sensitivity Analysis)** 的世界。

如果说之前的教程是在算 **“输出结果是多少”**（均值、概率），那么这组教程就是在问 **“谁是幕后推手？”**。

比如，梁弯得太厉害，到底是因为**材料太软 ()**？还是**受力太大 ()**？还是**梁太长 ()**？敏感性分析就是给这些输入变量“排座次”，看谁对结果的影响最大。

我同样把它拆解为**三个进阶步骤**：

---

### 第一步：线性世界的简单法则 (SRC)

**对应文件：** `4-1` (理论) 和 `4-2` (悬臂梁实战)

如果你的物理模型比较简单（接近线性，比如 ），那么看谁的系数大，谁就更重要。

* **Pearson 此时**：看输入和输出的相关性。
* **SRC (Standardized Regression Coefficients)**：标准化的回归系数。如果  的 SRC 是 -0.8，说明  越大，输出  越小，且影响很大。

**核心代码逻辑：**

```python
# 1. 准备数据 (输入 sampleX, 输出 sampleY)
sampleX = inputRV.getSample(100)
sampleY = my_function(sampleX)

# 2. 计算 SRC (标准回归系数)
# 这一步就像在做一个线性回归，看斜率
src_indices = ot.CorrelationAnalysis.SRC(sampleX, sampleY)
print(src_indices) 
# 结果可能像 [0.1, 0.8, 0.05, 0.05]，说明第二个变量是主导因素

```

> **警示**：在 `4-2` 的练习中，你会发现当输入变化范围变大时，线性模型 () 变得很差。这时候 SRC 就**失效**了，哪怕算出来也不准。

---

### 第二步：非线性世界的真相 (Sobol 指数)

**对应文件：** `4-3` (理论) 和 `4-4` (悬臂梁实战)

现实世界往往是复杂的（非线性的）。比如公式  里， 是三次方，它的一点点变化会被放大很多倍。这时候必须用 **Sobol 指数**。

* **一阶指数 (First Order)**：变量独自贡献的影响力。
* **总阶指数 (Total Order)**：变量独自的影响 + 它和其他变量**“合伙”**（交互作用）产生的影响。

**核心代码逻辑：**

```python
# 1. 设置算法 (Saltelli 或 Martinez)
# 这需要生成特殊的实验设计 (DOE)，通常需要很多样本 (比如 N=1000)
# 这是一个专门设计的“轰炸”方案，用来探测模型的每个角落
sensitivity_algo = ot.SaltelliSensitivityAlgorithm(input_distribution, N=1000, model=my_function)

# 2. 计算指数
first_order = sensitivity_algo.getFirstOrderIndices() # 独自贡献
total_order = sensitivity_algo.getTotalOrderIndices() # 总贡献

# 3.看图说话
ot.SobolIndicesAlgorithm.DrawSobolIndices(first_order, total_order)

```

> **通俗理解**：如果  的总指数是 0.6，说明 60% 的输出波动都是  造成的。**抓主要矛盾就抓它！**

---

### 第三步：对比与验证 (什么时候用哪个？)

**对应文件：** `4-5` (轴向受力梁练习)

这个练习对比了两种方法。

* **案例**： （强度 - 应力）。这是一个纯线性公式（减法）。
* **结果**：你会发现 **SRC 指数** 和 **Sobol 指数** 的结果几乎一模一样。
* **结论**：如果模型是线性的，用 SRC 就够了（计算快）；如果模型是非线性的（如悬臂梁），必须用 Sobol（计算慢但准）。

### 总结：你的学习金字塔

这四组文件构成了完整的 **OpenTURNS 不确定性量化 (UQ)** 学习路径：

1. **基础 $1-x$**：学会造积木（定义分布 `Normal`，定义函数 `Function`）。
2. **数据 $2-x$**：学会逆向工程（用 `Factory` 把数据拟合成分布）。
3. **预测 $3-x$**：学会算命（用 `MonteCarlo` 算均值和失效概率）。
4. **诊断 $4-x$**：学会看病（用 `Sobol` 找出哪个零件是导致问题的罪魁祸首）。

**现在的你，已经掌握了从“建立模型”到“分析问题”的整套核心逻辑。**