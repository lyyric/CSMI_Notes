# 数据学习的基本原理

灵感来源：《The Elements of Statistical Learning》

## 随机变量与取值

* 随机变量通常用大写字母表示，例如：$X$ 代表一个未观测到的通用数据。我们不知道它的具体值，但可以推测它的概率分布（即发生不同结果的概率）。
* 常量用小写字母表示，例如：$x$ 表示 $X$ 的某个特定数值。
* 可以写成 $\mathbf P[X=x]$ 来表示概率。

有时很难判断该使用大写还是小写。以下是一个示例：

**示例**：假设在一个小区内所有房屋价格都是随机变量，记作 $X_1, X_2, X_3, \dots$。这些价格是不确定的，因为它们取决于未来的买卖谈判。而每间房屋的出售情况独立于其他房屋。因此，$X_1, X_2, \dots$ 可以被视为相同随机变量 $X$ 的“独立副本”，表示房屋的“通用”价格。假设前 10 套房已售出。此时，这些房价已不再随机，可以用小写 $x_1, x_2, \dots, x_{10}$ 表示。可以通过均值 $\frac 1 {10} (x_1+...+x_{10})$ 来估计通用房价 $X$ 的期望值。

本课程的重点是一个随机变量对 $(X, Y)$，其中 $X \in \mathbb R^p$，$Y \in \mathbb R$。我们拥有独立的样本 $(X_i, Y_i)$，称为样本集。当我们观测到 $(X_i, Y_i)$ 的具体实现 $(x_i, y_i)$ 时，可以用它们来估计期望：$\mathbf E[\phi(X,Y)] \simeq \frac 1 n \sum_{i=1}^n \phi(x_i, y_i)$。

---

## 学习的原则

### 构建未知函数 $f^?$ 的估计器 $\hat f$

我们考虑以下变量：

* 解释变量（输入变量）：$X=(X^1, ..., X^p) \in \mathbb R^p$，如面积、区位、房间数。
* 响应变量（输出变量）：$Y$，如房价。

假设存在某种关系：$Y = f^?(X) + \text{噪声}$。

我们未知 $f^?$，但拥有独立的观测 $X_i$ 和 $Y_i = f^?(X_i) + \text{噪声}_i$。

我们的目标是从这些观测数据中构建估计器 $\hat f$，希望最终 $\hat f$ 能够接近 $f^?$。

![functionApprox|400](https://github.com/vincentvigon/public/blob/main/data/functionApprox.png?raw=true)

---

### 两个目标

* **预测**：给定一个未知输出的输入 $x=(x^1, ..., x^p)$，通过 $\hat f(x)$ 进行预测。
* **解释**：理解输入变量对输出的影响。例如，$X^1$ 表示烟草消费量，$X^2$ 表示牛奶消费量，$Y \in \{0,1\}$ 表示是否患有肺癌。我们想了解输入变量 $X^1$ 和 $X^2$ 对 $Y$ 的影响程度。

---

### 变量类型

变量主要分为两类：

* **定量变量**：如年龄、价格、面积、灰度等级。
* **定性变量**：如性别、地区、年龄组别。

此外：
* 若输出 $Y$ 为定量变量，则称为**回归**问题。
* 若输出 $Y$ 为定性变量，则称为**分类**问题。

**示例1**：MNIST 手写数字识别
* 定量输入：$X \in \mathbb R^{28 \times 28}$，表示灰度图像。
* 定性输出：$Y \in \{0,1,2,3,4,5,6,7,8,9\}$，表示数字类别。
* 问题：对每张图像进行数字分类。是回归还是分类？

**示例2**：保险定价
* 定量输入：驾驶员年龄、保险系数、车辆价格。
* 定性输入：驾驶员性别、地区、车辆品牌。
* 定量输出：保险费用。
* 问题：为每个驾驶员计算合理的保险费用。是回归还是分类？

---

## 构建和评估 $\hat f$

### 评估 $\hat f$

将样本 $(X_i, Y_i)$ 分为两部分：

* 训练数据集：用来构建 $\hat f$。
* 测试数据集：用于评估 $\hat f$ 的表现。评价标准是**损失函数**，即 $\text{Loss}_{\text{Test}} = \sum_{i \in \text{Test}} \text{dist}(Y_i, \hat f(X_i))$，例如 $\text{dist}(a, b) = (a - b)^2$。

若损失小，则模型效果较好。对于分类问题，可能使用其他损失函数，因为定性输出不能直接用距离度量。

---

### 通过最小化训练损失来构建 $\hat f$

由于估计器用损失函数来评估，那么构建估计器时也可以使用同样的函数。给定函数族 $\mathbb F$，我们可以定义：

$$
\hat f = \text{argmin}_{f \in \mathbb F} \sum_{i \in \text{Train}} \text{dist}(Y_i, f(X_i))
$$

有时我们希望 $\hat f$ 的模型更简单，可以对不规则函数加以惩罚：

$$
\hat f = \text{argmin}_{f \in \mathbb F} \left(\sum_{i \in \text{Train}} \text{dist}(Y_i, f(X_i)) + \lambda \, \text{Penalization}(f)\right)
$$

这样我们在拟合训练数据和保持模型简单之间做出平衡。

---

### 在不最小化训练损失的情况下构建 $\hat f$

**k 最近邻法（k-Nearest Neighbors，简称 k-NN）**：

这个方法基于计算新数据点 $x$ 周围 $k$ 个最近邻点的平均值来预测。记 $V_k(x)$ 为包含 $x$ 附近 $k$ 个最近邻样本点的集合，则：

$$
\hat f(x) = \frac{1}{k} \sum_{i: X_i \in V_k(x)} Y_i
$$

这是一个非参数化方法，因为我们没有预先描述可能的函数集合 $\mathbb F$。

![knn](https://github.com/vincentvigon/public/blob/main/data/knn.png?raw=true)

**核方法（Kernel Method）**：我们定义一个核函数 $N(x, y)$，例如 $N(x, y) = e^{-\frac{1}{2} \left(\frac{x - y}{\sigma}\right)^2}$，其中 $\sigma > 0$ 为常数。则模型可以表示为：

$$
\hat f(x) = \frac{\sum_{\text{Train}} N(x, X_i) Y_i}{\sum_{\text{Train}} N(x, X_i)}
$$

这类方法属于插值和平滑技术。

---

### 什么是建模

建模意味着我们对函数 $f^?$、噪声、或联合分布 $(X, Y)$ 进行假设。这个假设可以基于我们对问题的了解（专家意见）或对数据的观察（描述性统计）。建模自然会引导我们选择某种方法来构建 $\hat f$。

**示例**：假设房价 $Y$ 与面积 $X$ 存在线性关系：

$$
Y = w^?_0 + w^?_1 X + \text{噪声}
$$

这引导我们选择一个线性估计器：

$$
\hat f(x) = \hat w_0 + \hat w_1 x
$$

其中 $(\hat w_0, \hat w_1)$ 通过最小化损失函数确定：

$$
(\hat w_0, \hat w_1) = \text{argmin}_{(w_0, w_1) \in \mathbb R^2} \sum_{\text{Train}} (w_0 + w_1 X_i - Y_i)^2
$$

建模可以进一步假设噪声为均值为 0、方差为 $\sigma^2$ 的高斯变量。这一假设还可以用来构建置信区间、进行统计检验等。

对于线性回归，我们假设数据分布在某条直线、平面或超平面附近。而使用最近邻方法时，我们假设函数 $f^?$ 是局部常数的，即 $f^?(x)$ 与邻域内 $f^?(x_i)$ 值相近。

---

### 模型选择

一种更高级的技术是选择多个模型并最终选出最佳模型。为此，我们将数据集 $(X_i, Y_i)$ 分为三部分：**训练集（Train）**、**验证集（Validation）** 和 **测试集（Test）**。

* **训练集**：用于构建多个候选估计器 $(\hat f_1, \hat f_2, \dots)$。我们可以设置多个可能的函数集 $\mathbb F_1, \mathbb F_2, \dots$ 和多个惩罚参数 $\lambda_1, \lambda_2, \dots$，并定义每个估计器 $\hat f_k$ 为：

$$
\hat f_k = \text{argmin}_{f \in \mathbb F_k} \left(\sum_{\text{Train}} \text{dist}(f(X_i), Y_i) + \lambda_k \, \text{penalization}(f)\right)
$$

* **验证集**：用于选择最优估计器 $\hat f_k$，即：

$$
\hat f = \hat f_{\hat k} \quad \text{其中 } \hat k = \text{argmin}_k \sum_{\text{Validation}} \text{dist}(\hat f_k(X_i), Y_i)
$$

* **测试集**：最终对模型进行测试，计算最终损失：

$$
\sum_{\text{Test}} \text{dist}(\hat f(X_i), Y_i)
$$

---

## 线性回归

### 简单线性回归

**模型**：

* 输入 $p$ 个定量特征：$X=(X^1, ..., X^p) \in \mathbb R^p$。
* 输出为定量变量：$Y \in \mathbb R$。

我们假设 $Y$ 近似为 $X^j$ 的线性组合，定义一个参数化的函数集合：

$$
\mathbb F = \left\{f_w(x) = w_0 + \sum_{j=1}^p w_j x^j : w \in \mathbb R^{p+1}\right\}
$$

我们寻求该集合中最优的函数：

$$
\hat f = f_{\hat w} \quad \text{其中 } \hat w = \text{argmin}_{w} \sum_{\text{Train}} \left(Y_i - f_w(X_i)\right)^2 = \text{argmin}_{w} \text{Loss}(w)
$$

![linearRegression|400](https://github.com/vincentvigon/public/blob/main/data/linearRegression.png?raw=true)

找到 $\hat w$ 有两种方法：

* 使用梯度下降法最小化 $w \to \text{Loss}(w)$。
* 直接计算，详细见下一节。

---

### 直接计算 $\hat w$

令 $\mathbf{X}=\mathbf{X}_{\text{train}}$ 为数据矩阵，其中第 0 列为全 1，其他列为解释变量 $X^j_i$。令 $\mathbf{Y}$ 为响应变量的列向量，$\mathbf{Y}_i = Y_i$。将参数 $w=(w_0, w_1, ..., w_p)$ 视为列向量。

可用矩阵运算表示损失函数：

$$
\text{Loss}(w) = \sum_i \left(Y_i - f_w(X_i)\right)^2 = (\mathbf Y - \mathbf X w)^T (\mathbf Y - \mathbf X w)
$$

为找到损失函数的极小值，我们计算其微分：

$$
d \, \text{Loss}(w) = -2 \mathbf Y^T \mathbf X + 2 w^T \mathbf X^T \mathbf X
$$

解得：

$$
\hat w = (\mathbf X^T \mathbf X)^{-1} \mathbf X^T \mathbf Y
$$

因此估计为：

$$
\hat{\mathbf{Y}} = \mathbf{X} \hat{w} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
$$

对于测试数据集 $\mathbf{X}_{\text{test}}$，其预测值为：

$$
\hat{\mathbf{Y}}_{\text{test}} = \mathbf{X}_{\text{test}} \hat{w} = \mathbf{X}_{\text{test}} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
$$

---

### 最小值的唯一性？

通过分析，可以证明唯一性条件为：$\mathbf{X}$ 的秩等于其列数，换言之，$\mathbf{X}$ 的列是线性无关的。

---

### 正则化（缩减）

#### 岭回归（Ridge Regression）

当解释变量 $X^1, \dots, X^p$ 之间存在相关性时（例如，房屋的总面积和可居住面积通常高度相关），直接求解线性回归模型可能会遇到问题：

* 使用直接法时，矩阵 $\mathbf{X}^T \mathbf{X}$ 可能难以求逆（称为病态矩阵）。
* 使用梯度法时，损失函数的最小值可能会对应多个参数值。

例如，当两个变量几乎相等 $X^1 \approx X^2$ 时：

$$
\text{Loss}(w) = \sum_{\text{Train}} \left(w_0 + w_1 X^1_i + w_2 X^2_i - Y_i \right)^2 \approx \sum_{\text{Train}} \left(w_0 + (w_1 + w_2) X^1_i - Y_i \right)^2
$$

在这种情况下，损失只依赖于 $w_1$ 和 $w_2$ 的和，因此梯度下降可能会产生任意大的 $w_1$ 和 $w_2$ 值组合。

为减少这种不稳定性，我们可以引入惩罚项：

$$
\text{Loss}_\alpha(w) = \sum_{i \in \text{Train}} \left(y_i - w_0 - \sum_j w_j X^j_i\right)^2 + \alpha \sum_{j > 0} (w_j)^2
$$

这样，梯度下降会更倾向于较小的 $w_1$ 和 $w_2$ 值组合，从而提高模型的稳定性。

**练习**：用矩阵形式重新表示 $\text{Loss}_\alpha$，并证明最小化这个损失的解为：

$$
\hat w = (\mathbf{X}^T \mathbf{X} + \alpha I)^{-1} \mathbf{X}^T \mathbf{Y}
$$

---

#### 套索回归（Lasso Regression）

另一种正则化方法是引入 $L_1$ 惩罚项：

$$
\text{Loss}_\alpha(w) = \sum_{i \in \text{Train}} \left(y_i - \sum_j w_j X^j_i\right)^2 + \alpha \sum_{j > 0} |w_j|
$$

这里的 $L_1$ 惩罚会进一步将小的权重 $w_j$ 推向零，从而导致较多的权重为零，这种稀疏性使得模型更易于解释，因为不重要的特征变量被移除。

---

#### 岭回归与主成分分析（PCA）：相似之处

岭回归和主成分分析在某些方面具有相似性。设 $\mathbf{X}$ 的奇异值分解为 $\mathbf{X} = \mathbf{V} \mathbf{S} \mathbf{W}$，则：

$$
\hat Y_i = \sum_{j=1}^p \sum_{n=1}^N V_{ij} \frac{s_j^2}{s_j^2 + \alpha} V_{nj} Y_n
$$

增大 $\alpha$ 值会减少对较小奇异值的影响，类似于主成分分析中的降维过程。

---

## 增加/转换输入变量

在建模时，可以选择引入新的变量 $\phi^1(X), \phi^2(X), \dots$ 来丰富特征集。以预测房价为例，假设我们已有以下变量：

* $X^1$ = 面积
* $X^2$ = 可变成本（如暖气、水电等）
* $X^3$ = 固定成本（如清洁、电梯等）
* $X^4$ = 采光度
* $X^5 \in \{0, 1\}$ = 是否在高档小区（0 表示普通小区，1 表示高档小区）

可以引入以下新的变量：

* $\phi^1(X) = X^2 + X^3$：总成本
* $\phi^2(X) = X^1 / \phi^1(X)$：面积与总成本的比值
* $\phi^3(X) = X^1 X^4 X^5$：高档小区的面积*采光度
* $\phi^4(X) = X^1 X^4 (1 - X^5)$：普通小区的面积*采光度

构建的模型为：

$$
f_{w}(x) = w_0 + \sum_j w_j x^j + \sum_k w_k \phi^k(x)
$$

---

# 偏差与方差

## 偏差-方差分解

### 通过转换输入变量的线性模型示例

考虑定量输入 $X \in [0,1]$ 和定量输出 $Y \in \mathbb{R}$。定义一个参数 `freqMax`，模型表示为：

$$
Y = w_0 + \sum_{n=1}^{\texttt{freqMax}} w_{2n} \cos(2 \pi n X) + w_{2n-1} \sin(2 \pi n X)
$$

随着 `freqMax` 增大，模型的拟合能力增强，$f_{\hat w}$ 更接近 $f^?$，但 `freqMax` 过大可能导致过拟合。

---

### 模型的灵活性

灵活性的定义如下：

* **简单线性模型**：灵活性低，仅能拟合接近于超平面的数据。
* **SinCos 模型**：若 `freqMax` 较大，则灵活性高。
* 当模型通过最小化带有惩罚项的损失函数得到时，$\mathbb{F}$ 越大，模型越灵活。
* **k 最近邻法**：当 $k=1$ 时模型非常灵活，$k$ 较大时灵活性较低。

一般而言，越是局部拟合数据的模型，灵活性越高，但也更复杂且难以解释。

---

### 过拟合

以下为 SinCos 模型在不同 `freqMax` 参数下的表现：

![频率变化图](https://github.com/vincentvigon/public/blob/main/data/freqMax2.png?raw=true)

随着 `freqMax` 增大，模型在训练集上的拟合效果越来越好，但测试集上逐渐偏离，发生了过拟合，说明模型未能很好地**泛化**。

---

### 偏差-方差分解

灵活的模型倾向于在测试集上偏离，这种现象称为“方差”大，即估计函数 $\hat f_{T_r}$ 对于训练集 $T_r$ 的变化敏感。偏差-方差分解如下：

对于给定的观测 $(x, y)$，其均方误差可以分解为：

$$
\mathbb{E}\left[\left(\hat f_{T_r}(x) - y\right)^2\right] = \underbrace{\mathbb{E}\left[\left(\hat f_{T_r}(x) - \mathbb{E}\left[\hat f_{T_r}(x)\right]\right)^2\right]}_{\text{方差}} + \underbrace{\left(\mathbb{E}\left[\hat f_{T_r}(x)\right] - y\right)^2}_{\text{偏差}}
$$

方差-偏差的分解解释了模型的泛化性能：方差大的模型对不同训练集非常敏感，而偏差大的模型则未能有效学习数据的趋势。

### 用图示表示

偏差和方差之间的关系如下图所示：

![偏差-方差图示](https://github.com/vincentvigon/public/blob/main/data/biaisVarianceSchema.png?raw=true)

我们可以通过调节模型的超参数（如 `freqMax`）来平衡偏差与方差。

---

## 附录

### 微分

设 $f : \mathbb{R}^p \to \mathbb{R}, w \to f(w)$。$f$ 在 $w$ 处的微分 $\ell$ 定义为：

$$
f(w+\epsilon) = f(w) + \ell(\epsilon) + o(\epsilon)
$$

练习：

1. 设 $S$ 为对称矩阵，证明 $f(w) = w^T S w$ 的微分为 $\ell(\epsilon) = 2 w^T S \epsilon$。
2. 设 $A$ 为行向量，证明 $f(w) = A w$ 的微分为 $\ell(\epsilon) = A \epsilon$。

---

### Hessian 矩阵

$f$ 在 $w$ 处的 Hessian 矩阵 $H$ 满足：

$$
f(w+\epsilon) = f(w) + L \

epsilon + \epsilon^T H \epsilon + o(\epsilon^2)
$$

练习：

1. 设 $S$ 为对称矩阵，证明 $f(w) = w^T S w$ 的 Hessian 矩阵为 $2 S$。
2. 设 $A$ 为行向量，证明 $f(w) = A w$ 的 Hessian 矩阵为 0。