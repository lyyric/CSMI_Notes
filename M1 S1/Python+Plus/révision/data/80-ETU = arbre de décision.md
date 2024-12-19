# 决策树

来自 Aurélien Géron 书中的摘录

## 简介

决策树是一种多功能的机器学习算法，能够执行分类和回归任务，甚至可以处理多输出任务。这些算法非常强大，能够适应复杂的数据集。决策树也是随机森林的基本组成部分，而随机森林是目前最强大的机器学习算法之一。

* 我们将首先讨论如何训练决策树、可视化决策树以及进行预测。
* 接着，我们将回顾 Scikit-Learn 使用的 CART（分类与回归树）训练算法。
* 然后，我们将讨论如何对决策树进行正则化。
* 我们还将使用决策树进行回归任务。
* 最后，我们将讨论决策树的一些局限性。

### 安装 graphviz

graphviz 是一个用于可视化图形的库。

**安装库**
```bash
!apt-get install graphviz
```

**安装 Python 包装器**
```bash
!pip install graphviz
```

**注意：** 当我尝试在本地（而不是在 Colab 上）安装 `graphviz` 时，需要更改访问权限：

```bash
sudo chmod 777 /usr/local/include
```

（这是在错误信息中要求的）

### 顶部代码单元

```python
%reset -f

import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd
import graphviz

import sklearn.linear_model
import sklearn.datasets
import sklearn.tree
```

## 分类

### 训练和可视化决策树

为了理解决策树，只需构建一个决策树并观察它如何进行预测。以下代码在著名的鸢尾花数据集上训练一个 `DecisionTreeClassifier`。

```python
iris = sklearn.datasets.load_iris()
X_iris = iris.data[:, 2:] # 花瓣长度和宽度
y_iris = iris.target

tree_clf = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

dot_data = sklearn.tree.export_graphviz(
        tree_clf,
        out_file=None, # 你也可以指定一个路径来保存图形数据
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

graph = graphviz.Source(dot_data) 
graph
```

**练习：** 观察上方的决策树。假设你有一朵鸢尾花，其

* 花瓣长度 = 3
* 花瓣宽度 = 1

它应该被分类为哪种类别？

**解答：** 根据决策树的结构，我们可以追踪花瓣长度和宽度的分割节点。首先，检查花瓣长度是否小于等于 2.45 cm。如果大于 2.45 cm，进入右子树。接着，检查花瓣宽度是否小于等于 1.75 cm。如果是，则分类为 Iris-Versicolor；否则为 Iris-Virginica。由于花瓣长度为 3 cm，大于 2.45 cm，且花瓣宽度为 1 cm，小于 1.75 cm，因此应分类为 **Iris-Versicolor**。

**备注：** 决策树的一个优点是它们几乎不需要数据预处理，特别是不需要对变量进行缩放或中心化。

以下是树上标签的含义：

* `samples`：统计训练集中实例的数量。例如，有 100 个实例的花瓣长度大于 2.45 cm（深度 1，右侧），其中 54 个实例的花瓣宽度小于 1.75 cm（深度 2，左侧）。
* `value`：显示类别的分布。例如，最下方右侧的节点包含 0 个 Iris-Setosa，1 个 Iris-Versicolor 和 45 个 Iris-Virginica。
* `gini`：基尼指数，用于衡量纯度。当一个节点只包含同一类别的实例时，基尼指数为 0。一般情况下，基尼指数的计算公式为：
$$
G = 1 - \sum_k p^2_{k}
$$
其中，$p_{k}$ 是节点中类别 $k$ 的实例比例。

**练习：** 验证不同基尼指数的计算。使用 Python 作为计算器。

**解答：**

假设我们有一个节点，其中包含以下类别分布：

- 类别 0：50 个实例
- 类别 1：30 个实例
- 类别 2：20 个实例

总实例数 $m = 100$。

基尼指数计算如下：
$$
G = 1 - \left( \left(\frac{50}{100}\right)^2 + \left(\frac{30}{100}\right)^2 + \left(\frac{20}{100}\right)^2 \right) = 1 - (0.25 + 0.09 + 0.04) = 1 - 0.38 = 0.62
$$

使用 Python 代码验证：

```python
p0, p1, p2 = 50/100, 30/100, 20/100
G = 1 - (p0**2 + p1**2 + p2**2)
print(G)  # 输出: 0.62
```

### 决策边界

```python
def plot_decision_boundary(ax, clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    ax.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        ax.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        ax.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        ax.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        ax.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        ax.axis(axes)
    if iris:
        ax.set_xlabel("花瓣长度", fontsize=14)
        ax.set_ylabel("花瓣宽度", fontsize=14)
    else:
        ax.set_xlabel(r"$x_1$", fontsize=18)
        ax.set_ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        ax.legend(loc="lower right", fontsize=14)

fig, ax = plt.subplots(figsize=(12,6))
plot_decision_boundary(ax, tree_clf, X_iris, y_iris)

# 手动添加边界
ax.plot([2.45, 2.45], [0, 3], "k-", linewidth=2, label="depth=0")
ax.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2, label="depth=1")
ax.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2, label="depth=2")
ax.plot([4.85, 4.85], [1.75, 3], "k:")
ax.legend();
```

### 模型解释：白盒与黑盒

如你所见，决策树相当直观，其决策易于解释。这些模型通常被称为“白盒”。相反，随机森林或神经网络通常被认为是“黑盒”。它们能够做出出色的预测，但很难用简单的术语解释为什么会做出这些预测。例如，如果一个神经网络说某个人出现在图像中，很难知道是什么实际特征促成了这一预测：模型是否识别了这个人的眼睛？嘴巴？鼻子？鞋子？还是她坐的沙发？相反，决策树提供了简单而易懂的分类规则，甚至可以在必要时手动应用（例如，用于花卉分类）。

### 类别概率估计

假设你在自然界中发现了一朵花，其花瓣长度为 5 cm，宽度为 1.5 cm。预测它的种类：

```python
print(tree_clf.predict([[5, 1.5]]))
print(tree_clf.predict_proba([[5, 1.5]]))
```

**练习：** 解释类别概率是如何计算的。这种概率计算的主要缺陷是什么？**提示：** 观察“决策边界”。边界附近发生了什么？

**解答：**

**类别概率的计算方式：**

决策树在每个叶节点中包含了训练数据的类别分布。`predict_proba` 方法返回的是某个样本落入某个叶节点后，该叶节点中各类别的相对频率。例如，如果某叶节点中有 10 个样本，其中 7 个是类别 A，3 个是类别 B，那么对于落入该叶节点的新样本，`predict_proba` 会返回 `[0.7, 0.3]`。

**主要缺陷：**

这种概率计算方法的主要缺陷在于它缺乏平滑性，尤其是在决策边界附近。当一个样本位于两个叶节点的边界附近时，微小的输入变化可能导致样本从一个叶节点跳到另一个叶节点，从而导致概率估计的突然变化。这使得决策树的概率估计在边界区域不稳定，容易出现过拟合。

### CART 算法

Scikit-Learn 使用 CART（分类与回归树）算法来训练决策树。其基本思想非常简单：算法首先使用单一特征 $k$ 和阈值 $t_k$ 将训练集分成两个子集（即节点），例如，“花瓣长度 ≤ 2.45 cm”。如何选择 $k$ 和 $t_k$？它通过搜索能够产生最纯子集（按大小加权）的 $(k, t_k)$ 对来选择。算法试图最小化以下代价函数：
$$
J(k,t_k)= \frac {m_{\text{left}} } {m} \,  G_{\text{left}} +  \frac {m_{\text{right}} } {m} \,  G_{\text{right}}
$$
其中：

* $m_{\text{left/right}}$ 是左/右节点中的实例数量
* $G_{\text{left/right}}$ 是左/右节点的基尼指数

一旦将训练集分成两部分，算法就会以相同的逻辑重新分割子集，接着再分割子子集，依此类推。分割过程在以下情况停止：

* 树达到最大深度（由超参数 `max_depth` 定义）
* 没有找到能够减少纯度的分割

一些其他超参数（稍后会介绍）控制额外的停止条件：`min_samples_split`、`min_samples_leaf`、`min_weight_fraction_leaf` 和 `max_leaf_nodes`。

如你所见，CART 算法是一个“贪心”算法：它在第 0 层贪婪地寻找最佳分割，然后在第 1 层重复这个过程，依此类推。它不会检查第 0 层的一个较差的解决方案是否会在第 1 层带来更好的整体解决方案。

这样的算法通常会产生一个相当不错的解决方案，但无法保证是最优的。不幸的是，找到最优树是一个 NP 完全问题：需要 $O(\exp(m))$ 的时间复杂度（$m$ 是实例数量），这使得即使对于相当小的训练集，问题也是无法解决的。因此，我们只能满足于一个“相当好的”解决方案。

### 计算复杂度

设 $m$ 为实例数量，$n$ 为描述变量（特征）的数量。

进行预测时，需要从根节点遍历到叶节点。决策树通常大致平衡，因此遍历决策树大约需要通过 $O(\log_2(m))$ 个节点。由于每个节点只需要检查一个特征值，整体预测的时间复杂度仅为 $O(\log_2(m))$，与特征数量无关。因此，即使在大型数据集上，预测也非常快速。

然而，学习算法在每个节点都会比较所有特征（如果未设置 `max_features` 则所有特征），对所有样本进行评估。因此，学习的时间复杂度为 $O(n \times m \log_2(m))$。

### 基尼指数还是熵？

默认情况下，决策树使用基尼指数来衡量节点的不纯度，但也可以将其替换为熵，熵的计算公式为：
$$
H = - \sum_k p_{k}\log(p_{k})
$$
其中，$0\log(0)=0$。

那么，应该使用基尼指数还是熵呢？实际上，大多数情况下，它们的区别不大：它们会生成类似的树。基尼指数计算稍快，因此是一个不错的选择。然而，当它们有所不同的时候，基尼指数倾向于在树的某一分支中孤立出最频繁的类别，而熵则倾向于生成稍微更平衡的树。

### 非参数模型

决策树对数据几乎没有假设，这与线性模型不同，线性模型假设数据线性分布。决策树的树状结构能够适应各种类型的数据。

像决策树这样的模型通常被称为非参数模型，并不是因为它们没有参数（决策树的结构由 `.dot` 文件中的参数描述），而是因为参数的数量在训练之前是不确定的。相比之下，参数模型（如线性模型）有预先确定的参数数量，因此其自由度（即灵活性）是有限的，这减少了过拟合的风险，但增加了欠拟合的风险。

**练习：** 用你自己的话重新解释参数模型和非参数模型之间的区别。

**解答：**

**参数模型：**

参数模型在训练之前具有固定数量的参数。例如，线性回归模型有权重和偏置，这些数量在训练前是确定的。这种模型的灵活性有限，因为其复杂度由参数数量决定。这有助于防止过拟合，但也可能导致欠拟合，尤其是在数据模式复杂时。

**非参数模型：**

非参数模型的参数数量不在训练之前确定，而是根据数据的复杂性动态调整。例如，决策树的深度和分支数量取决于数据本身。这使得非参数模型更灵活，能够捕捉更复杂的数据模式。然而，这也增加了过拟合的风险，因为模型可能会过于适应训练数据的细节。

总的来说，参数模型通过限制模型复杂度来控制过拟合，而非参数模型通过增加灵活性来更好地适应复杂数据，但需要通过其他方法（如正则化）来防止过拟合。

### 正则化

为了避免过拟合，必须限制决策树的复杂度。一个较简单的树 = 一个较不灵活的模型 = 较少的过拟合。限制模型灵活性称为“正则化”。

以下是 `DecisionTreeClassifier` 的不同超参数，这些超参数可以用来调整生成的树的复杂度：

* `max_depth`：树的最大深度。默认值为 "None"，表示无限制。减少此参数是最简单的正则化方法。
* `min_samples_split`：一个节点需要拥有的最小实例数量才能进行分割。
* `min_samples_leaf`：一个叶节点（终端节点）需要拥有的最小实例数量。
* `min_weight_fraction_leaf`：与 `min_samples_leaf` 类似，但以总加权实例数的比例表示。
* `max_leaf_nodes`：叶节点的最大数量。
* `max_features`：在每个节点分割时考虑的最大特征数量。默认情况下考虑所有特征，否则会随机选择特征。

增加 `min_xxx` 超参数或减少 `max_xxx` 超参数可以对模型进行正则化。

**注意：** 其他算法通过首先构建不受限制的决策树，然后修剪（删除）不必要的节点来工作。如果一个节点的所有子节点都是叶节点，并且其带来的纯度提升在统计上不显著，则认为该节点是无用的。

**示例：** 看下面的图。左边的模型显然是过拟合的，而右边的模型可能会有更好的泛化能力。

```python
# "moon" 数据集是一个模拟数据集，两个类别互相交织
Xm, ym = sklearn.datasets.make_moons(n_samples=100, noise=0.25, random_state=53)

deep_tree_clf1 = sklearn.tree.DecisionTreeClassifier(random_state=42)
deep_tree_clf2 = sklearn.tree.DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_clf1.fit(Xm, ym)
deep_tree_clf2.fit(Xm, ym)

fix, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
plot_decision_boundary(ax0, deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
ax0.set_title("无任何限制", fontsize=16)
plot_decision_boundary(ax1, deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
ax1.set_title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
```

## 回归

决策树同样能够执行回归任务。我们使用 Scikit-Learn 的 `DecisionTreeRegressor` 类，训练一个深度为 2 的回归树，使用带噪声的二次数据集：

### 示例

```python
# 数据创建
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

tree_reg = sklearn.tree.DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X, y)

dot_data = sklearn.tree.export_graphviz(
        tree_reg,
        out_file=None,
        rounded=True,
        filled=True
    )

graph = graphviz.Source(dot_data) 
graph
```

这棵树与之前构建的分类树非常相似。主要区别在于，每个节点预测的是一个数值，而不是一个类别。

例如，假设你想对一个新实例 `x[0] = 0.6` 进行预测。你从根节点开始遍历树，最终到达一个叶节点，该节点预测 `值=0.1106`。这个预测值只是该叶节点中 110 个训练实例的目标值平均值。这个预测对应于这些 110 个实例的均方误差（MSE）为 0.0151。

### 绘图

```python
tree_reg1 = sklearn.tree.DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = sklearn.tree.DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel(r"$x_0$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

plt.figure(figsize=(15, 4))
plt.subplot(121)
plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "深度=0", fontsize=15)
plt.text(0.01, 0.2, "深度=1", fontsize=13)
plt.text(0.65, 0.8, "深度=1", fontsize=13)
plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=14)

plt.subplot(122)
plot_regression_predictions(tree_reg2, X, y, ylabel=None)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
    plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
plt.text(0.3, 0.5, "深度=2", fontsize=13)
plt.title("max_depth=3", fontsize=14);
```

### CART 算法

CART 算法在回归任务中的工作方式与分类任务基本相同，只是现在它尝试通过最小化节点内的均方误差（MSE）来划分训练集。具体来说，算法最小化以下量：
$$
\frac {m_{\text{left}} } {m} \,  \text{MSE}_{\text{left}} +  \frac {m_{\text{right}} } {m} \,  \text{MSE}_{\text{right}}
$$
其中
$$
\text{MSE}_{\text{left}} = \sum_{i \in \text{left}} \big( \bar y_{\text{left}} - y_i \big)^2
$$
$$
\text{MSE}_{\text{right}} = \sum_{i \in \text{right}} \big( \bar y_{\text{right}} - y_i \big)^2
$$
$\bar y_{\text{left/right}}$ 是左/右节点中目标值的平均值。

### 正则化

与分类任务一样，决策树在回归任务中也容易过拟合。没有任何正则化（即使用默认超参数）时，决策树会过拟合。简单地设置 `min_samples_leaf=10` 可以得到一个更合理的模型。

```python
tree_reg1 = sklearn.tree.DecisionTreeRegressor(random_state=42)
tree_reg2 = sklearn.tree.DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

x1 = np.linspace(0, 1, 500).reshape(-1, 1)
y_pred1 = tree_reg1.predict(x1)
y_pred2 = tree_reg2.predict(x1)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(X, y, "b.")
plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel(r"$x_0$", fontsize=18)
plt.ylabel("$y$", fontsize=18, rotation=0)
plt.legend(loc="upper center", fontsize=18)
plt.title("无任何限制", fontsize=14)

plt.subplot(122)
plt.plot(X, y, "b.")
plt.plot(x1, y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel("$x_1$", fontsize=18)
plt.title("min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize=14);
```

## 不稳定性

### 数据旋转

希望你现在已经相信决策树有很多优点：它们简单易懂、易于解释、易于使用、多功能且强大。然而，它们也有一些局限性。首先，如你可能已经注意到，决策树喜欢正交的决策边界（所有分割都与一个轴垂直），这使得它们对训练集的旋转非常敏感。例如，以下是一个线性可分的数据集：左图中，决策树可以轻松地进行分割；而右图中，在将数据集旋转 45° 后，决策边界显得不必要地复杂。

```python
np.random.seed(6)
Xs = np.random.rand(100, 2) - 0.5
ys = (Xs[:, 0] > 0).astype(np.float32) * 2

angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
Xsr = Xs.dot(rotation_matrix)

tree_clf_s = sklearn.tree.DecisionTreeClassifier(random_state=42)
tree_clf_s.fit(Xs, ys)
tree_clf_sr = sklearn.tree.DecisionTreeClassifier(random_state=42)
tree_clf_sr.fit(Xsr, ys)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))

plot_decision_boundary(ax0, tree_clf_s, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plot_decision_boundary(ax1, tree_clf_sr, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False);
```

尽管两个决策树都能完美拟合训练集，但右边的模型可能在泛化方面表现不佳。限制这一问题的一种方法是使用主成分分析（PCA），它通常可以更好地对齐数据。

### 小的变化

更广泛地说，决策树的主要问题是它们对训练数据的微小变化非常敏感。例如，删除训练集中的一朵鸢尾花。

```python
fig, ax = plt.subplots(figsize=(16,6))
plot_decision_boundary(ax, tree_clf, X_iris, y_iris)

iris = sklearn.datasets.load_iris()
X = iris.data[:, 2:] # 花瓣长度和宽度
y = iris.target
X[(X[:, 1]==X[:, 1][y==1].max()) & (y==1)] # 最宽的 Iris-Versicolor 花
not_widest_versicolor = (X[:, 1]!=1.8) | (y==2)
X_tweaked = X[not_widest_versicolor]
y_tweaked = y[not_widest_versicolor]

tree_clf_tweaked = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=40)
tree_clf_tweaked.fit(X_tweaked, y_tweaked)

fig, ax = plt.subplots(figsize=(16,6))
plot_decision_boundary(ax, tree_clf_tweaked, X_tweaked, y_tweaked, legend=False)
```

实际上，由于 Scikit-Learn 使用的训练算法是随机的，甚至在相同的训练数据上，你也可能得到非常不同的模型（除非你设置了 `random_state` 超参数）。随机森林可以通过对多个树的预测进行平均来减少这种不稳定性。

## 总结

决策树是一种强大且易于理解的机器学习模型，适用于分类和回归任务。它们具有许多优点，如无需大量的数据预处理、易于解释和快速预测。然而，决策树也存在一些局限性，如对训练数据的微小变化敏感、倾向于过拟合以及对特征旋转敏感。通过使用正则化技术和集成方法（如随机森林），可以有效地缓解这些问题，提升模型的泛化能力和稳定性。

# 参考资料

- Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*

# 知识拓展

- [Scikit-Learn 文档](https://scikit-learn.org/stable/)
- [Graphviz 官方网站](https://graphviz.org/)
- [随机森林介绍](https://en.wikipedia.org/wiki/Random_forest)

# 习题

1. **验证基尼指数的计算**：选择一个节点，假设其包含若干类别的实例，手动计算基尼指数，并使用 Python 验证你的计算结果。
2. **类别概率的解释**：选择一个样本，观察其在决策树中的路径，解释其类别概率是如何计算的，并指出这种计算方法的主要缺陷。
3. **参数模型与非参数模型的区别**：用自己的话详细解释参数模型和非参数模型之间的区别，并举例说明它们各自的优缺点。

**答案示例：**

1. **验证基尼指数的计算**

   假设一个节点包含以下类别分布：

   - 类别 A：40 个实例
   - 类别 B：60 个实例

   总实例数 $m = 100$。

   基尼指数计算如下：
   $$
   G = 1 - \left(\left(\frac{40}{100}\right)^2 + \left(\frac{60}{100}\right)^2\right) = 1 - (0.16 + 0.36) = 1 - 0.52 = 0.48
   $$

   使用 Python 验证：

   ```python
   pA, pB = 40/100, 60/100
   G = 1 - (pA**2 + pB**2)
   print(G)  # 输出: 0.48
   ```

2. **类别概率的解释**

   当使用 `predict_proba` 方法时，决策树返回的是样本所在叶节点中各类别的相对频率。例如，一个叶节点包含 30 个样本，其中 20 个属于类别 A，10 个属于类别 B，那么该叶节点的概率为 `[0.6667, 0.3333]`。主要缺陷在于这种概率估计在决策边界附近不稳定，微小的输入变化可能导致样本从一个叶节点跳到另一个叶节点，从而导致概率的突然变化。

3. **参数模型与非参数模型的区别**

   **参数模型**有固定数量的参数，模型复杂度在训练前就确定。例如，线性回归模型有权重和偏置，其数量由特征数量决定。参数模型通常计算效率高，但可能无法捕捉复杂的数据模式。

   **非参数模型**的参数数量不固定，随着数据量的增加可以动态增长。例如，决策树的分支和深度根据数据的复杂性而变化。非参数模型更灵活，能够适应复杂的数据结构，但也更容易过拟合，需通过正则化或集成方法来控制模型复杂度。

# 结束语

决策树是机器学习中基础且重要的算法，理解其工作原理、优缺点以及如何进行优化，对于掌握更复杂的模型和算法具有重要意义。通过正则化和集成方法，可以显著提升决策树的性能和稳定性，使其在实际应用中更加有效。

# 版权声明

本笔记内容基于 Aurélien Géron 的《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》一书进行整理和翻译，仅供学习和交流使用。未经授权，禁止转载或用于商业用途。

# 联系方式

如有任何问题或建议，欢迎通过以下方式联系：

- 电子邮件：example@example.com
- GitHub：[https://github.com/yourusername](https://github.com/yourusername)

# 更新日志

- **2024-04-27**：首次发布。

# 致谢

感谢 Aurélien Géron 提供的优秀教材，以及所有为机器学习社区做出贡献的开发者和研究人员。

# 附录

### 常用 Scikit-Learn 决策树参数

| 参数 | 描述 |
| --- | --- |
| `criterion` | 衡量分割质量的函数，分类任务中可以是“gini”或“entropy”；回归任务中为“mse” |
| `max_depth` | 树的最大深度 |
| `min_samples_split` | 内部节点再划分所需的最小样本数 |
| `min_samples_leaf` | 叶节点最少样本数 |
| `max_features` | 寻找最佳分割时考虑的最大特征数 |
| `random_state` | 控制随机性 |

了解并合理调整这些参数，可以显著提升决策树模型的性能和泛化能力。

# 附加资源

- [机器学习实战 - 决策树](https://www.example.com)
- [深入理解决策树算法](https://www.example.com)
- [Scikit-Learn 决策树文档](https://scikit-learn.org/stable/modules/tree.html)

