### 第一步：准备数据容器 (基础)

**对应文件：** `1-1-Point-et-Sample.ipynb`

在开始计算概率之前，你需要知道数据在 OpenTURNS 里长什么样。

* **Point (点)**：就是一个向量，比如坐标 。
* **Sample (样本)**：就是一张表（矩阵），有很多行数据。

**核心代码：**

```python
import openturns as ot

# 1. 创建一个点 (比如二维坐标 [2.5, 4.0])
p = ot.Point([2.5, 4.0])

# 2. 创建一个样本 (比如做了5次实验，每次记录2个数据)
# 方法：可以直接从点创建，或者生成空的
data = ot.Sample(5, 2)  # 5行，2列
data[0, 0] = 1.5        # 修改第1行第1列的数据

```

> **通俗理解**：`Point` 就是一行数据，`Sample` 就是 Excel 里的一张数据表。

---

### 第二步：描述不确定性 (定义分布)

**对应文件：** `1-2-Distributions.ipynb`

现实世界是不确定的。你不能只给一个固定的数，你要给一个“分布”。

* **单变量**：比如“身高”服从正态分布。
* **多变量**：比如“身高”和“体重”有相关性。

**核心代码：**

```python
# 1. 定义常见的分布
dist1 = ot.Normal(0.0, 1.0)      # 正态分布：均值0，标准差1
dist2 = ot.Uniform(0.0, 10.0)    # 均匀分布：0到10之间

# 2. 画图看看 (查看概率密度)
dist1.drawPDF() 

# 3. 组合成多变量分布 (比如 X 和 Y 相互独立)
# Collection 相当于把两个分布打包
myCollection = [dist1, dist2] 
jointDist = ot.ComposedDistribution(myCollection) # 形成一个二维分布

```

---

### 第三步：定义你的模型 (函数)

**对应文件：** `1-3- Fonctions.ipynb`

如果不确定性是输入，那你的物理公式或模拟器就是“函数”。

* **PythonFunction**：把你写好的普通 Python 函数变成 OpenTURNS 能用的。
* **SymbolicFunction**：直接用字符串写数学公式（速度快，简单）。

**核心代码：**

```python
# 场景：计算 Z = X + Y

# 方法A：符号函数 (推荐简单公式用)
# 输入是 x,y；输出是 x+y
f = ot.SymbolicFunction(["x", "y"], ["x + y"]) 

# 方法B：Python 函数 (复杂逻辑用)
def my_python_func(X):
    return [X[0] + X[1]]

# 转换成 OpenTURNS 函数 (需要指定输入和输出维度)
f_py = ot.PythonFunction(2, 1, my_python_func)

```

---

### 第四步：连接输入与模型 (随机向量)

**对应文件：** `1-4-RandomVector.ipynb`

这是 OpenTURNS 最核心的概念。你有了分布（Step 2）和函数（Step 3），现在要把它们连起来。

* **输入随机向量**：直接来自分布。
* **输出随机向量**：通过函数计算出来的结果。

**核心代码：**

```python
# 1. 定义输入的随机向量 X (假设它服从上面定义的 jointDist)
inputRV = ot.RandomVector(jointDist)

# 2. 定义输出的随机向量 Y = f(X)
# 这步很关键：它告诉系统，Y 的随机性是由 X 通过函数 f 传导过来的
outputRV = ot.CompositeRandomVector(f, inputRV)

```

---

### 第五步：实战演练 (运行模拟)

**对应文件：** `1-5-Exercice-Somme-de-deux-gaussiennes-Solution.ipynb`

这是最后一个文件，也是一次综合练习。它演示了如何用**蒙特卡洛模拟 (Monte Carlo)** 来解决问题。

* **问题**： 是正态分布， 是正态分布，求  的均值和方差。

**核心代码流程总结：**

```python
# 1. 生成样本 (做实验)
# 让计算机随机生成 1000 次可能的输入，并自动算出 1000 个输出
sample_Z = outputRV.getSample(1000) 

# 2. 统计结果
mean_Z = sample_Z.computeMean()       # 算平均值
std_Z = sample_Z.computeStandardDeviation() # 算标准差

print(f"模拟结果: 均值={mean_Z}, 标准差={std_Z}")

```

### 总结：你的学习路线图

既然你“什么都不会”，建议按照以下顺序阅读代码（不要试图一次看完所有细节，只看大意）：

1. **先看 `1-1**`：学会怎么建 `Point` 和 `Sample`，这是基础积木。
2. **重点看 `1-2**`：学会怎么 `ot.Normal()`，因为一切概率计算都源于分布。
3. **跳过 `1-3` 和 `1-4` 的理论细节**：直接看 **`1-5` (练习题)**。
* `1-5` 这个文件是**集大成者**。它完整展示了：
* 定义两个高斯分布 (Step 2)
* 定义加法公式 (Step 3)
* 把它们连起来 (Step 4)
* 抽取 1000 个样本看结果 (Step 5)





这套代码非常规范，一旦你理解了 **分布 (Distribution) + 函数 (Function) = 随机结果 (RandomVector)** 这个公式，你就学会了 80%。