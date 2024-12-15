## 二值化

**词汇说明：** 在本实验中，我将使用“类别变量”一词，作为“定性变量”的同义词。类别将始终用整数表示。对于目标变量（输出），我将使用“类别”一词来代替“类别变量”。

我们将使用一些 TensorFlow 的功能，因为分类损失函数在 TensorFlow 中命名得更清晰。

### 独热向量（One Hot Vector）

考虑有 $k$ 个类别。与第 $i$ 类别对应的独热向量是一个长度为 $k$ 的零向量，其中第 $i$ 个分量被置为 $1$。

**示例：**假设有 3 个可能的类别：

- 类别 0 → `[1., 0., 0.]`
- 类别 1 → `[0., 1., 0.]`
- 类别 2 → `[0., 0., 1.]`

这些向量可以看作是概率向量，将所有权重集中在第 $i$ 类别上。注意，这些向量由浮点数组成（因为它们表示概率）。

**类别变量的二值化**就是将其转换为独热向量。

下面是将一个类别变量样本转换为独热向量的示例，假设该变量有 5 个可能的取值：

```python
import tensorflow as tf

N = [3, 1, 2, 4, 0, 1, 1, 0, 4]
N_proba = tf.one_hot(N, 5)
print(N_proba)
```

### 对于描述性变量

当我们有一个类别描述性变量时，必须对其进行二值化。让我们通过一个例子来理解这一点：

假设我们有两个描述性变量：

- $X$：驾驶员的酒精消费量。
- $N$：车辆所在的省编号。

我们希望创建一个神经网络来预测事故数量。如果不进行二值化，一个神经元在第二层接收到的激活值形式如下：

$$
\text{relu}(X \cdot w_0 + N \cdot w_1 + b)
$$

**问题**：对于 Var 省（$N=83$），$N \cdot w_1$ 的值比对于 Ain 省（$N=1$）大得多（绝对值上）。这显然是不合理的！这会导致模型难以训练。

如果我们对 $N$ 进行二值化，则需要引入与省份数量相同数量的参数 $w$。第二层的一个神经元将接收到：

$$
\text{relu}(X \cdot w_0 + 1_{N=1} \cdot w_1 + 1_{N=2} \cdot w_2 + \dots + b) = \text{relu}(X \cdot w_0 + w_N + b)
$$

这样，变量 $N$ 可以选择一个特定于省份的偏置 $w_N$。这很自然，因为我们可以假设不同省份对酒精的容忍度不同。

**注意**：通过二值化，我们考虑的是“属于某个省份”这一特性，而不是“省份的编号”。

### 另一种技术

当类别数量非常大时，二值化会变得非常耗费计算资源。这在处理单词时尤为常见，因为词汇表中的单词数量可能达到数万个。

在这种情况下，有另一种技术叫做**嵌入（Embedding）**，我们将在处理语言模型时详细介绍。

## 多类别分类

### 回顾

多类别分类模型的构建步骤如下：

1. **选择函数 $f_\theta$**：这是一个将输入映射到 $\mathbb{R}^k$ 的函数，其中 $k$ 是类别数量。函数的输出被称为“logits”：
    $$
    \hat{Y}_{\text{logits}} := f_\theta(X)
    $$
   
2. **应用 Softmax 函数**：将 logits 转换为概率向量：
    $$
    \hat{Y}_{\text{proba}} := \text{Softmax}(f_\theta(X)) = \text{model}_\theta(X)
    $$

### 稀疏交叉熵（Sparse Cross Entropy）

考虑一个类别目标变量 $Y \in \{0, 1, \dots, k-1\}$。

记 $Y_{\text{proba}}$ 为其二值化后的独热向量。我们可以使用交叉熵（Cross-Entropy）来比较真实概率 $Y_{\text{proba}}$ 和预测概率 $\hat{Y}_{\text{proba}}$：

$$
\text{CE}(Y_{\text{proba}}, \hat{Y}_{\text{proba}}) = -\sum_{i=0}^{k-1} Y_{\text{proba}}[i] \log(\hat{Y}_{\text{proba}}[i])
$$

由于 $Y_{\text{proba}}$ 是独热向量，只有一个分量为 $1$，其余为 $0$，所以求和可以简化为：

$$
\text{CE}(Y_{\text{proba}}, \hat{Y}_{\text{proba}}) = -\log(\hat{Y}_{\text{proba}}[Y])
$$

这就是**稀疏交叉熵（Sparse Cross Entropy）**：

$$
\text{SCE}(Y, \hat{Y}_{\text{proba}}) = -\log(\hat{Y}_{\text{proba}}[Y])
$$

### 仅使用 logits 进行计算

在训练完成后，为了预测类别，我们选择概率最大的类别：

$$
\hat{Y} = \text{argmax}_i \hat{Y}_{\text{proba}}[i] = \text{argmax}_i [\text{Softmax}(f_\theta(X))]_i
$$

由于 Softmax 函数是基于指数函数的，且指数函数是单调递增的，因此：

$$
\hat{Y} = \text{argmax}_i f_\theta(X)_i
$$

因此，在构建模型时，我们可以选择不在最后一层显式添加 Softmax 函数。这在评估模型时可以节省计算时间（对于在智能手机上运行模型尤为重要）。

**但需要注意**：如果不添加 Softmax 函数，我们必须在损失函数中添加 Softmax。这是因为交叉熵需要比较概率向量。因此，我们使用基于 logits 的交叉熵损失函数（Cross-Entropy-from-Logits），也称为稀疏基于 logits 的交叉熵（Sparse-Cross-Entropy-from-Logits）：

$$
\text{SCEL}(Y, \hat{Y}_{\text{logits}}) = -\log(\text{Softmax}(\hat{Y}_{\text{logits}})[Y])
$$

### 练习

**问题：** 验证如果 $\hat{Y}_{\text{proba}}$ 是一个概率向量，那么：

$$
\text{SCE}(Y, \hat{Y}_{\text{proba}}) = \text{SCEL}(Y, \log(\hat{Y}_{\text{proba}}))
$$

### TensorFlow 代码示例

```python
import tensorflow as tf
import numpy as np

# 定义类别和概率
Y = tf.constant([1])  # 类别 1
Y_proba = tf.constant([[0., 1, 0]])  # 独热向量
Y_proba_pred = tf.constant([[0.1, 0.8, 0.1]])  # 预测概率

# 计算稀疏交叉熵
loss = tf.keras.losses.sparse_categorical_crossentropy(Y, Y_proba_pred)
print(loss.numpy())  # 输出: 0.22314354

# 验证
print(-np.log(0.8))  # 输出: 0.22314355131420976

# 使用非稀疏交叉熵
loss = tf.keras.losses.categorical_crossentropy(Y_proba, Y_proba_pred)
print(loss.numpy())  # 输出: [0.22314354]

# 使用 logits 版本的交叉熵
loss = tf.keras.losses.sparse_categorical_crossentropy(Y, tf.math.log(Y_proba_pred), from_logits=True)
print(loss.numpy())  # 输出: 0.22314354
```

**问题：** 为什么会出现以下警告？

```
UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a sigmoid or softmax activation and thus does not represent logits. Was this intended?
```

**尝试触发警告：**

```python
# 定义 logits 和经过 Softmax 的概率
logits = tf.constant([[0.1, 0.8, 0.1]])
proba = tf.nn.softmax(logits)

# 使用 from_logits=True 但输出已是概率
loss = tf.keras.losses.sparse_categorical_crossentropy(Y, proba, from_logits=True)
```

**解释：** 当 `from_logits=True` 时，损失函数期望输入是未经过 Softmax 的 logits。然而，如果输入已经经过 Softmax（即是概率向量），则会导致计算不正确，并触发上述警告。

### PyTorch 代码示例

```python
import torch

# 定义类别和概率
Y = torch.tensor([1], dtype=torch.int64)  # 类别 1
Y_proba = torch.tensor([[0., 1, 0]], dtype=torch.float32)  # 独热向量
Y_proba_pred = torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32)  # 预测概率

# 定义交叉熵损失函数（基于 logits）
loss_fn = torch.nn.CrossEntropyLoss()

# 计算损失
loss = loss_fn(torch.log(Y_proba_pred), Y)  # 正确用法（假设输入是 logits）
print(loss)

# 错误用法：使用概率而非 logits
loss = loss_fn(torch.log(Y_proba_pred), Y_proba)  # 会报错，因为 Y 是独热向量
print(loss)
```

### Hinge 损失

**Hinge 损失**不依赖于 Softmax 函数，直接在 logits 上计算。它源自支持向量机（SVM），在神经网络流行之前，SVM 是主流的分类模型。

定义一个常数 $\Delta > 0$，Hinge 损失定义为：

$$
\text{Hinge}(Y, \hat{Y}_{\text{logits}}) = \sum_{i \neq Y} \max(0, \hat{Y}_{\text{logits}}[i] - \hat{Y}_{\text{logits}}[Y] + \Delta)
$$

为了使损失为零，需要正确类别的 logit $\hat{Y}_{\text{logits}}[Y]$ 至少比其他类别的 logit 大 $\Delta$。

常见情况下，$\Delta$ 取值为 1。

Hinge 损失有时也会采用“平方 hinge”，即：

$$
\sum_{i \neq Y} \left(\max(0, \hat{Y}_{\text{logits}}[i] - \hat{Y}_{\text{logits}}[Y] + \Delta)\right)^2
$$

## 二元分类

### 两种方法

假设我们有一个二分类问题（例如：猫/狗）。有两种方法来处理这种情况：

1. **Softmax 技术**：选择一个函数 $f_\theta$，输出在 $\mathbb{R}^2$ 中，然后定义：
    $$
    \text{model}_\theta(X) = \text{Softmax}(f_\theta(X)) = [1 - p, p]
    $$
    这里，$p$ 表示属于第二类的概率。

2. **Sigmoid 技术**：选择一个函数 $g_\theta$，输出在 $\mathbb{R}$ 中，然后定义：
    $$
    \text{model}_\theta(X) = \sigma(g_\theta(X))
    $$
    其中，$\sigma$ 是 Sigmoid 函数：
    $$
    \sigma(x) = \frac{1}{1 + e^{-x}}
    $$
    这将输出一个在 $[0, 1]$ 范围内的概率，表示属于某一类的概率（例如，类别 1）。

### 两者等价

我们证明这两种方法在数学上是等价的。

假设 Softmax 技术中的函数 $f_\theta$ 输出为 $[f_0, f_1]$：

$$
f_\theta(x) = [f_0, f_1]
$$

定义 $g = f_1 - f_0$，则 Softmax 的输出为：

$$
\text{Softmax}(f_\theta(x)) = \left[ \frac{e^{f_0}}{e^{f_0} + e^{f_1}}, \frac{e^{f_1}}{e^{f_0} + e^{f_1}} \right] = \left[ 1 - \frac{1}{1 + e^{f_0 - f_1}}, \frac{1}{1 + e^{f_0 - f_1}} \right] = [1 - \sigma(g), \sigma(g)]
$$

这表明，在 Softmax 技术中，真正影响输出的是 logits 的差异 $f_1 - f_0$。因此，Softmax 技术存在一种“过参数化”（over-parameterization），即引入了不必要的参数冗余。

相比之下，Sigmoid 技术通过单一的 $g_\theta(x)$ 函数避免了这种冗余。

**注意**：参数的冗余在神经网络中并不严重，甚至是网络设计的一个特点。对于多类别分类（$k > 2$），同样存在参数冗余，但通常不需要去除。

### 关于损失函数

需要注意的是，如果使用 Sigmoid 技术进行二分类，必须更改损失函数：

在 Softmax 技术中，我们有多个输出对应多个类别。而在 Sigmoid 技术中，模型只输出一个概率值 $\hat{Y}$，表示属于某一类的概率。

**二元交叉熵（Binary Cross-Entropy）**定义为：

$$
\text{BCE}(Y, \hat{Y}) = - (1 - Y) \log(1 - \hat{Y}) - Y \log(\hat{Y})
$$

这与多类别交叉熵类似，但适用于二分类问题。

**问题：** 如果我们创建一个预测概率 $\hat{Y}$ 的二分类模型，并将损失函数简单定义为：

$$
- Y \log(\hat{Y})
$$

**结果会怎样？**

**解释：** 这种定义忽略了 $Y=0$ 的情况，相当于只关注正类的预测，而不考虑负类。这会导致模型在负类上表现不佳，因为它没有受到任何惩罚。此外，缺少 $(1 - Y) \log(1 - \hat{Y})$ 项会导致损失函数在负类时不正确，影响模型的整体性能。

## 计算方面的技巧

### 对数的技巧

每当损失函数中出现 $\log(x)$ 时，实际上计算机会使用 $\log(x + \epsilon)$，其中 $\epsilon$ 是一个极小的正数。这是为了避免数值计算中 $\log(0)$ 的情况。

例如：

$$
\text{CE}(Y_{\text{proba}}, \hat{Y}_{\text{proba}}) = -\sum_{i=0}^{k-1} Y_{\text{proba}}[i] \log(\hat{Y}_{\text{proba}}[i] + \epsilon)
$$

这不会造成问题，因为当预测概率 $\hat{Y}_{\text{proba}}[i]$ 恰好为零时，真实标签 $Y$ 也应该为零，从而符合数学上的 $0 \log(0) = 0$ 约定。

### Softmax 的计算技巧

为了计算 Softmax，库通常使用以下公式：

$$
\frac{e^{x_i - a}}{\sum_j e^{x_j - a}}
$$

其中，$a = \max(x_i)$。

**问题：** 为什么这是一个更好的计算方法？

**解释：** 这种方法通过减去最大值 $a$ 来防止指数函数导致的数值溢出（即避免 $e^{x_i}$ 过大）。这在计算机中处理大数时尤为重要，可以提高计算的稳定性和准确性。

### Binary Cross-Entropy 的计算技巧

二元交叉熵（Binary Cross-Entropy）也有基于 logits 的版本，当模型输出未经过 Sigmoid 函数时使用。这称为“基于 logits 的二元交叉熵（BCEL from logits）”。

定义如下：

$$
\text{BCEL}(Y, \ell) = - (1 - Y) \log\left(1 - \sigma(\ell)\right) - Y \log\left(\sigma(\ell)\right)
$$

其中，$\ell$ 是模型的输出 logits，$\sigma$ 是 Sigmoid 函数。

这个表达式可以简化为：

$$
\text{BCEL}(Y, \ell) = \max(\ell, 0) - \ell Y + \log\left(1 + e^{-|\ell|}\right)
$$

这种简化方式在实际计算中有助于提高数值稳定性，避免因 $\ell$ 值过大或过小时导致的计算问题。

## 总结

通过以上内容，我们学习了如何处理类别变量的二值化、理解多类别和二元分类的损失函数以及相关的计算技巧。这些基础知识对于构建和训练有效的分类模型至关重要。

### 关键点回顾：

- **独热向量**用于将类别变量转换为适合神经网络处理的格式。
- **交叉熵损失函数**是分类任务中常用的损失函数，适用于多类别和二元分类。
- **Softmax 技术**和 **Sigmoid 技术**是处理多类别和二元分类的两种主要方法，它们在数学上是等价的，但实现上有所不同。
- **计算技巧**（如防止指数溢出的 Softmax 计算方式和稳定的交叉熵计算）在实际应用中非常重要，可以提高模型的数值稳定性和准确性。

通过掌握这些概念和技巧，可以更有效地构建和训练分类模型，提升模型的性能和泛化能力。