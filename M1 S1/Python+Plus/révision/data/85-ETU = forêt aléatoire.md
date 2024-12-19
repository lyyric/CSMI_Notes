# 集成学习与随机森林

摘自奥雷利安·热龙（Aurélien Géron）的书籍

## 环境设置

### 导入必要的库

首先，我们需要重置现有的工作环境，并导入数据处理和可视化所需的库，以及机器学习相关的模块。

```python
%reset -f
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.datasets
import sklearn.tree
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network

plt.style.use("default")
```

### 准备数据

我们将使用`sklearn`库中的`make_moons`数据集，这是一个用于分类任务的非线性数据集。数据集包含500个样本，带有一定噪声。

```python
# 生成数据集
X, y = sklearn.datasets.make_moons(n_samples=500, noise=0.30, random_state=42)
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)
```

### 绘制决策边界的函数

为了可视化分类器的决策边界，我们定义了一个函数`plot_decision_boundary`，该函数根据预测函数绘制出不同分类器的决策区域。

```python
def plot_decision_boundary(ax, prediction_func, X, y, title="", extent=[-2, 3, -2, 2], alpha=0.5, plot_data=True):
    x1s = np.linspace(extent[0], extent[1], 100)
    x2s = np.linspace(extent[2], extent[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.stack([x1.reshape(-1), x2.reshape(-1)], axis=1)
    y_pred = prediction_func(X_new).reshape(x1.shape)
    
    ax.imshow(y_pred, extent=extent, origin="lower", interpolation="bilinear", cmap="jet", alpha=alpha)
            
    if plot_data:
        ax.scatter(X[:, 0], X[:, 1], marker=".", c=y, cmap="jet", linewidths=0)
        ax.set_title(title)
        ax.set_xlabel(r"$x_1$", fontsize=18)
        ax.set_ylabel(r"$x_2$", fontsize=18)
```

### 数据可视化

我们使用`make_moons`数据集，并将其分为训练集和测试集，然后绘制出这两个数据集的分布情况。

```python
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

def plot_one(ax, X, y, title):
    ax.scatter(
        X[:,0],
        X[:,1],
        c=y,
        edgecolor="w",
        cmap="jet"
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    
plot_one(ax0, X_train, y_train, "训练集")
plot_one(ax1, X_test, y_test, "测试集")
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
```

#### **练习：为什么称之为“月亮”数据集？**

**解答：** 该数据集的形状类似于两个半月形的月亮，因此得名“moon”。

## 投票分类器

通过集成多个分类器的预测结果，可以获得更稳定和准确的预测。这种方法被称为“集成学习”（Ensemble Learning）。其中，随机森林（Random Forest）是一种基于集成学习的强大算法。

### **练习：您认为在著名的Netflix竞赛中需要做什么？**

**解答：** 在Netflix竞赛中，参赛者需要预测用户对电影的评分。为了提高预测的准确性，参赛者可能会使用多种机器学习模型，并通过集成学习的方法，如随机森林、梯度提升等，来结合多个模型的预测结果，以获得更高的准确性和更好的泛化能力。

### 硬投票（Hard Voting）

硬投票是指对多个二分类器的预测结果进行多数投票，以决定最终的分类结果。即使每个分类器的准确率仅为51%，通过集成投票，整体的准确率也会显著提高。

例如，假设有一枚硬币，正面朝上的概率为51%。如果连续抛掷1000次，获得正面的概率约为75%；抛掷10000次，获得正面的概率约为97%。

#### **练习：**

1. **模拟上述抛硬币实验，验证75%和97%的概率。**

```python
import random

def simulate_coin_toss(p=0.51, n=1000, trials=10000):
    successes_1000 = 0
    successes_10000 = 0
    for _ in range(trials):
        count_1000 = sum(random.random() < p for _ in range(1000))
        if count_1000 > 500:
            successes_1000 += 1
        count_10000 = sum(random.random() < p for _ in range(10000))
        if count_10000 > 5000:
            successes_10000 += 1
    prob_1000 = successes_1000 / trials
    prob_10000 = successes_10000 / trials
    return prob_1000, prob_10000

prob_1000, prob_10000 = simulate_coin_toss()
print(f"概率 (1000次抛硬币后正面多数): {prob_1000*100:.2f}%")
print(f"概率 (10000次抛硬币后正面多数): {prob_10000*100:.2f}%")
```

2. **给出使用二项式系数的数学表达式来计算75%和97%的概率。然后给出一种数值近似的方法（参考概率课程的内容）。**

**解答：**

- **数学表达式：** 对于n次独立的伯努利试验，每次成功的概率为p，要求k次或更多次成功的概率，可以使用二项分布的累积分布函数（CDF）：

  $$
  P(X \geq k) = \sum_{i=k}^{n} \binom{n}{i} p^i (1-p)^{n-i}
  $$

  例如，对于1000次抛硬币，p=0.51，k=501：

  $$
  P(X \geq 501) = \sum_{i=501}^{1000} \binom{1000}{i} 0.51^i \times 0.49^{1000-i}
  $$

- **数值近似方法：** 当n较大时，可以使用正态近似来近似计算：

  $$
  X \sim \mathcal{N}(\mu = np, \sigma^2 = np(1-p))
  $$

  通过标准化，计算：

  $$
  P(X \geq k) \approx 1 - \Phi\left(\frac{k - \mu}{\sigma}\right)
  $$

  其中，$\Phi$ 是标准正态分布的累积分布函数。

3. **为了使这些计算有效，我们需要对各次抛硬币的独立性做出什么基本假设？**

**解答：** 假设每次抛硬币的结果都是独立的，即每次抛硬币的结果不受之前抛硬币结果的影响。

### 不同类型的分类器

为了使集成方法有效，基模型之间需要尽可能独立。一种实现这一点的方式是使用不同的算法作为基分类器。

```python
log_clf = sklearn.linear_model.LogisticRegression(random_state=42)
rnd_clf = sklearn.ensemble.RandomForestClassifier(random_state=42)
svm_clf = sklearn.svm.SVC(random_state=42)

voting_clf = sklearn.ensemble.VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard'
)

voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, sklearn.metrics.accuracy_score(y_test, y_pred))
```

通过上述代码，我们可以看到，集成后的投票分类器的准确率略高于每个单独分类器的准确率。

### 软投票（Soft Voting）

如果所有的分类器都能够估计类别的概率，可以通过取平均概率来聚合预测结果。这种方法称为“软投票”。软投票通常比硬投票更有效，因为它赋予了对预测结果有高度信心的分类器更大的权重。

```python
log_clf = sklearn.linear_model.LogisticRegression(random_state=42)
rnd_clf = sklearn.ensemble.RandomForestClassifier(random_state=42)
svm_clf = sklearn.svm.SVC(probability=True, random_state=42)  # 关键在于probability=True

voting_clf = sklearn.ensemble.VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, sklearn.metrics.accuracy_score(y_test, y_pred))
```

**练习：如何创建一个包含回归模型的集成方法？**

**解答：**

对于回归任务，可以使用类似的方法来创建集成模型。使用`VotingRegressor`，并将不同的回归模型作为基模型。例如：

```python
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# 定义基回归模型
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
svr = SVR()

# 创建软投票回归器
voting_reg = VotingRegressor(
    estimators=[('lr', lr), ('dt', dt), ('svr', svr)]
)

# 训练集成回归模型
voting_reg.fit(X_train, y_train)

# 评估模型
from sklearn.metrics import mean_squared_error
y_pred = voting_reg.predict(X_test)
print("集成回归模型的MSE:", mean_squared_error(y_test, y_pred))
```

## Bagging 集成

### 介绍

另一种创建独立性较高的模型集成的方法是使用相同的算法，但在训练集的不同子样本上进行训练。这种方法称为“Bagging”（Bootstrap Aggregating，靴带聚合）。具体来说：

- **Bagging：** 通过有放回地抽取训练样本来创建子集。
- **Pasting（粘贴）：** 通过无放回地抽取训练样本来创建子集。

每个基模型在子集上进行训练，尽管每个基模型的偏差可能较高，但通过聚合可以减少偏差并显著降低方差。

### 实现

以下代码训练了500棵决策树，每棵树使用100个有放回抽取的训练样本进行训练。

```python
bag_clf = sklearn.ensemble.BaggingClassifier(
    sklearn.tree.DecisionTreeClassifier(random_state=42), 
    n_estimators=500,
    max_samples=100,  # 每棵树使用的训练样本数量
    bootstrap=True, 
    n_jobs=-1,  # 使用所有可用的CPU核心
    random_state=42
)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

print("Bagging分类器的准确率:", sklearn.metrics.accuracy_score(y_test, y_pred))

# 对比单棵决策树的表现
tree_clf = sklearn.tree.DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print("单棵决策树的准确率:", sklearn.metrics.accuracy_score(y_test, y_pred_tree))
```

**注意：** 如果基分类器能够估计类别的概率（如`DecisionTreeClassifier`），`BaggingClassifier`将自动执行软投票。

```python
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
plot_decision_boundary(ax0, tree_clf.predict, X, y, "决策树")
plot_decision_boundary(ax1, bag_clf.predict, X, y, "Bagging")
```

通过比较单棵决策树和Bagging分类器的决策边界，可以发现Bagging分类器的决策边界更加平滑和泛化能力更强。

### 使用或不使用Bootstrap？

Bootstrap方法引入了更多的多样性，从而通常能够降低方差。然而，具体效果需要通过交叉验证等方法进行验证。

### OOB（Out-of-Bag）实例

在有放回抽样（`bootstrap=True`）的情况下，某些训练样本不会被抽中，这些样本称为“袋外”（out-of-bag，OOB）实例。通常，大约37%的样本是袋外的。

对于每个基模型，袋外实例可以用作验证集，从而实现一种交叉验证的方法。

```python
bag_clf = sklearn.ensemble.BaggingClassifier(
    sklearn.tree.DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, 
    max_samples=1.0,  # 每棵树使用与训练集相同数量的样本
    bootstrap=True, 
    n_jobs=-1,
    random_state=42,
    oob_score=True  # 启用袋外评分
)

bag_clf.fit(X_train, y_train)
print("袋外数据的准确率:", bag_clf.oob_score_)

# 验证袋外评分与测试集评分的接近程度
y_pred = bag_clf.predict(X_test)
print("测试数据的准确率:", sklearn.metrics.accuracy_score(y_test, y_pred))
```

**练习：通过模拟验证，当样本数量m趋于无穷大时，未被抽中的样本比例趋近于 $1 - e^{-1}$ ≈ 0.63。**

**解答：**

当进行有放回抽样时，每个样本在一次抽样中不被抽中的概率为 $1 - \frac{1}{m}$。经过m次抽样，不被抽中的概率为 $(1 - \frac{1}{m})^m$。当m趋于无穷大时，这个概率趋近于 $e^{-1} ≈ 0.3679$，即被抽中的概率为 $1 - e^{-1} ≈ 0.6321$。

我们可以通过模拟来验证这一点：

```python
import math

def simulate_oob(m, n_samples=100000):
    prob_not_selected = (1 - 1/m)**m
    return prob_not_selected

m = 1000
simulated_prob = simulate_oob(m)
theoretical_prob = 1 - math.exp(-1)
print(f"模拟未被抽中的概率: {simulated_prob:.4f}")
print(f"理论未被抽中的概率: {theoretical_prob:.4f}")
```

### 随机子空间与随机补丁

`BaggingClassifier`还允许对特征进行随机抽样，这通过参数`max_features`和`bootstrap_features`控制。每个基模型将仅使用特征的一个子集进行训练，这在处理高维数据（如图像）时特别有用。

- **随机补丁（Random Patches）：** 对训练实例进行随机抽样。
- **随机子空间（Random Subspaces）：** 对特征进行随机抽样。

通过这种方式，可以增加基模型之间的多样性，进一步降低模型的方差。

## 随机森林（Random Forests）

### 定义

随机森林是一种基于集成学习的算法，它由多棵决策树组成。通常，随机森林使用Bagging方法，并在每个节点分裂时随机选择部分特征，从而增加树之间的多样性。

与直接使用`BaggingClassifier`和`DecisionTreeClassifier`相比，`RandomForestClassifier`更加方便和高效。

```python
rnd_clf = sklearn.ensemble.RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16, 
    n_jobs=-1, 
    random_state=42,
)

rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print(y_pred_rf)

# 检查两个分类器的预测结果是否相似
bag_clf = sklearn.ensemble.BaggingClassifier(
        sklearn.tree.DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
        n_estimators=500, 
        max_samples=1.0, 
        bootstrap=True, 
        n_jobs=-1
    )

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

print(np.sum(y_pred == y_pred_rf) / len(y_pred))  # 预测结果几乎相同
```

随机森林在Bagging的基础上增加了特征的随机选择，这使得各棵树之间更加独立，通常能够获得更好的性能。

### 决策边界的图示

通过绘制多棵决策树的决策边界，可以观察到随机森林的决策边界更加平滑和稳健。

```python
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

for i in range(15):
    tree_clf = sklearn.tree.DecisionTreeClassifier(max_leaf_nodes=16, random_state=42 + i)
    indices_with_replacement = np.random.randint(0, len(X_train), len(X_train))
    tree_clf.fit(X[indices_with_replacement], y[indices_with_replacement])
    plot_decision_boundary(ax, tree_clf.predict, X, y, alpha=0.05, plot_data=(i==0))
```

**练习：绘制随机森林的决策边界，这应该与上述多棵树的叠加效果接近。**

**解答：**

```python
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plot_decision_boundary(ax, rnd_clf.predict, X, y, title="随机森林决策边界")
```

通过上述代码，可以看到随机森林的决策边界较为平滑，与多棵决策树的叠加效果相似。

### 特征重要性

在决策树中，重要的特征通常出现在树的顶部，因为这些特征能够更有效地分割数据。随机森林通过计算每个特征在所有树中的平均重要性，提供了特征的重要性评分。

```python
iris = sklearn.datasets.load_iris()
rnd_clf = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
```

### Extra-Trees

为了进一步增加树的随机性，可以使用`ExtraTreesClassifier`，这不仅随机选择特征，还随机选择分裂阈值。这种方法可以在一定程度上提高模型的性能。

```python
extra_trees_clf = sklearn.ensemble.ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
extra_trees_clf.fit(X_train, y_train)
y_pred_extra = extra_trees_clf.predict(X_test)
print("Extra Trees分类器的准确率:", sklearn.metrics.accuracy_score(y_test, y_pred_extra))
```

## AdaBoost

### 介绍

Boosting是一种集成学习方法，通过结合多个弱学习器（Weak Learners）来构建一个强学习器（Strong Learner）。AdaBoost（Adaptive Boosting）是一种常用的Boosting算法，其基本思想是依次训练多个分类器，每个分类器都关注于前一个分类器错误分类的样本。

### 详细解释

AdaBoost通过调整训练样本的权重，使得后续的分类器更加关注那些前一个分类器错误分类的样本。具体步骤如下：

1. 初始化所有样本的权重相同。
2. 训练一个弱分类器，并根据其错误率调整样本权重。
3. 重复训练多个分类器，每次都更加关注之前分类器错误分类的样本。
4. 最终的预测结果是所有分类器的加权投票。

```python
ada_clf = sklearn.ensemble.AdaBoostClassifier(
    sklearn.tree.DecisionTreeClassifier(max_depth=1), 
    n_estimators=200,
    algorithm="SAMME.R", 
    learning_rate=0.5, 
    random_state=42
)

ada_clf.fit(X_train, y_train)

fig, ax = plt.subplots()
plot_decision_boundary(ax, ada_clf.predict, X, y)
```

### 逐步实现

我们手动实现AdaBoost算法，使用支持向量机（SVM）作为基分类器，并观察不同超参数的影响。

```python
fig, axs = plt.subplots(4, 2, figsize=(10, 15))
axs = axs.reshape(-1)

m = len(X_train)
learning_rate = 0.2
sample_weights = np.ones(m)

for i in range(len(axs)):
    model = sklearn.svm.SVC(kernel="rbf", C=0.03, random_state=42)
    # model = sklearn.tree.DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_train)
    sample_weights[y_pred != y_train] *= (1 + learning_rate)
    plot_decision_boundary(axs[i], model.predict, X, y) 
```

#### **练习：**

1. **调整超参数`C`和`learning_rate`的值。**

   - **C过小：** 分类边界过于简单，可能无法很好地拟合数据。
   - **learning_rate过小：** 每个分类器的贡献较小，需要更多的分类器来达到较好的效果。
   - **learning_rate过大：** 每个分类器的贡献过大，可能导致过拟合。

2. **在手动实现的AdaBoost中引入`learning_rate`。**

**解答：**

在上面的手动实现中，`learning_rate`通过调整样本权重的增幅来控制每个分类器的贡献。可以通过改变`learning_rate`的值来观察分类边界的变化。

### 缺点

AdaBoost是一种顺序训练的算法，每个分类器都依赖于前一个分类器的结果，因此无法并行化训练。这可能导致训练时间较长，尤其是在基分类器较复杂时。

## 梯度提升（Gradient Boosting）

### 解释

梯度提升（Gradient Boosting）是一种集成学习方法，通过逐步添加新的基学习器来减少前一个学习器的残差。不同于AdaBoost，梯度提升通过最小化损失函数的梯度来优化模型。

### 实验

我们通过一个简单的回归任务来演示梯度提升的工作原理。

```python
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0]**2 + 0.05 * np.random.randn(100)

# 第一个基学习器
tree_reg1 = sklearn.tree.DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# 计算残差
y2 = y - tree_reg1.predict(X)
# 第二个基学习器
tree_reg2 = sklearn.tree.DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

# 计算新的残差
y3 = y2 - tree_reg2.predict(X)
# 第三个基学习器
tree_reg3 = sklearn.tree.DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

X_new = np.array([[0.8]])
# 预测新样本的值
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(y_pred)
```

#### 可视化预测

```python
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

plt.figure(figsize=(11,11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="训练集")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("残差与树的预测", fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="训练集")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("集成模型的预测", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="残差")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)
```

### 使用Scikit-Learn的`GradientBoostingRegressor`

```python
gbrt = sklearn.ensemble.GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=0.1, random_state=42)
gbrt.fit(X, y)

gbrt_slow = sklearn.ensemble.GradientBoostingRegressor(max_depth=2, n_estimators=50, learning_rate=0.1, random_state=42)
gbrt_slow.fit(X, y)

plt.figure(figsize=(11,4))

plt.subplot(121)
plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="集成模型的预测")
plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)
```

学习率（`learning_rate`）控制每棵树对最终预测的贡献。较低的学习率通常需要更多的树，但能提供更好的泛化能力。

#### **练习：**

1. **尝试不同的参数组合（手动测试），找到更优的参数。**

**解答：**

可以通过调整`max_depth`、`n_estimators`和`learning_rate`等参数，观察模型在验证集上的表现，选择最佳参数组合。例如：

```python
best_gbrt = sklearn.ensemble.GradientBoostingRegressor(max_depth=3, n_estimators=100, learning_rate=0.05, random_state=42)
best_gbrt.fit(X_train, y_train)
print("最佳梯度提升回归模型的MSE:", sklearn.metrics.mean_squared_error(y_test, best_gbrt.predict(X_test)))
```

2. **在手动实现的AdaBoost中引入`learning_rate`。**

**解答：** 在前面的手动实现中，已经通过调整样本权重的方式引入了`learning_rate`。具体来说，`learning_rate`用于控制样本权重的更新幅度：

```python
sample_weights[y_pred != y_train] *= (1 + learning_rate)
```

如果要将其直接应用到基分类器的预测中，可以在计算残差时乘以`learning_rate`：

```python
y_pred = model.predict(X_train)
residual = y_train - y_pred
sample_weights *= np.exp(learning_rate * residual)
```

### 早停（Early Stopping）

为了防止模型过拟合，可以使用早停策略，即在验证误差不再下降时停止训练。

```python
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=49)

gbrt = sklearn.ensemble.GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

errors = [np.mean((y_val - y_pred)**2) for y_pred in gbrt.staged_predict(X_val)]

bst_n_estimators = np.argmin(errors)

gbrt_best = sklearn.ensemble.GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)

min_error = np.min(errors)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error*1.2, "最小值", ha="center", fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel("树的数量")
plt.title("验证误差", fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("最佳模型（%d棵树）" % bst_n_estimators, fontsize=14)
```

#### **练习：**

**在Keras中，创建的模型是否类似于Scikit-Learn中`warm_start=True`或`warm_start=False`的模型？**

**解答：** 在Keras中，模型的训练过程与Scikit-Learn的`warm_start`机制不同。Keras的模型训练通常是从头开始的，每次调用`fit`方法时，模型的权重会根据新的训练数据进行更新，而不是像`warm_start=True`那样从之前的状态继续训练。因此，Keras中的模型更类似于Scikit-Learn中`warm_start=False`的模型。

### 子样本（Subsample）

`GradientBoostingRegressor`具有一个超参数`subsample`，用于控制每棵树使用的训练样本比例。例如，`subsample=0.25`意味着每棵树只使用25%的训练样本进行训练。这种方法被称为“随机梯度提升”（Stochastic Gradient Boosting），能够增加模型的鲁棒性，降低方差，并加快训练速度。

## 示例：在MNIST数据上应用集成方法

### 步骤：

1. **加载数据**

```python
# 使用小型的digits数据集
mnist = sklearn.datasets.load_digits()
print(mnist.data.shape)  # 输出数据形状
```

2. **划分训练集、验证集和测试集**

```python
nb_data = len(mnist.data)
X_train_val, X_test, y_train_val, y_test = sklearn.model_selection.train_test_split(
    mnist.data, mnist.target, test_size=nb_data//10, random_state=42
)

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
    X_train_val, y_train_val, test_size=len(X_train_val)//9, random_state=42
)
```

3. **训练个体分类器**

```python
random_forest_clf = sklearn.ensemble.RandomForestClassifier(random_state=42)
extra_trees_clf = sklearn.ensemble.ExtraTreesClassifier(random_state=42)
mlp_clf = sklearn.neural_network.MLPClassifier(random_state=42)

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("mlp_clf", mlp_clf)
]

voting_clf = sklearn.ensemble.VotingClassifier(named_estimators)

voting_clf.fit(X_train, y_train)

# 评估模型在验证集上的表现
print("集成分类器在验证集上的准确率:", voting_clf.score(X_val, y_val))

# 评估个体分类器的表现
print("各个分类器在验证集上的准确率:", [estimator.score(X_val, y_val) for estimator in voting_clf.estimators_])
```

4. **移除表现最差的分类器**

```python
# 假设移除第一个分类器
del voting_clf.estimators_[0]

# 重新评估集成分类器
print("移除一个分类器后，集成分类器在验证集上的准确率:", voting_clf.score(X_val, y_val))
```

5. **使用软投票**

```python
voting_clf.voting = "soft"
voting_clf.fit(X_train, y_train)

# 评估在验证集上的表现
print("软投票集成分类器在验证集上的准确率:", voting_clf.score(X_val, y_val))

# 评估在测试集上的表现
print("软投票集成分类器在测试集上的准确率:", voting_clf.score(X_test, y_test))

# 各个分类器在测试集上的表现
print("各个分类器在测试集上的准确率:", [estimator.score(X_test, y_test) for estimator in voting_clf.estimators_])
```

通过上述步骤，我们可以看到集成分类器在验证集和测试集上的表现优于单个分类器，展示了集成学习的优势。

# 总结

通过本节学习，我们了解了集成学习的基本概念，包括Bagging、随机森林、AdaBoost和梯度提升等方法。集成学习通过结合多个基模型的预测结果，能够显著提高模型的性能和泛化能力。同时，了解了如何在Scikit-Learn中实现这些方法，并通过实践验证了它们的有效性。

集成学习的方法在实际应用中非常强大，广泛应用于各种机器学习竞赛和实际问题中，是提升模型性能的重要手段。