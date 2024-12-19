# 二元分类

本章结合了以下两本书的内容：
* François Chollet 的 [**Deep Learning with Python**](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff)
* Aurélien Géron 的 **Hands-On Machine Learning with Scikit-Learn and TensorFlow**

我们将对电影评论进行二分类：将评论归为“正面评论”或“负面评论”两类。

```python
%reset -f

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## 数据

### IMDB 数据集

我们将下载 IMDB 数据集。
* 数据是电影评论
* 标签是整数 0 或 1，表示评论是负面还是正面

```python
num_words = 10_000

# 下载数据集（80MB）：仅第一次下载时执行
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
```

上面的 `num_words=10000` 意味着我们只保留了前 10,000 个最常见的单词，其他单词被删除。

### 数据集规模

整个数据集中共有 50,000 条评论。

```python
len(train_data), len(test_data)
```

### 数据内容

每个数据项是一个索引列表，每个索引代表一个单词。

```python
# 评论的长度不一
train_data
```

### 前十条评论的长度

```python
# 显示前 10 条评论的长度
for i in range(10):
    print(len(train_data[i]))
```

**练习1：** 所有列表都以 1 开头。为什么？0 被使用了吗？

**解答：**
在 IMDB 数据集中，索引 `0` 通常被保留用于填充（padding），而 `1` 被用作序列的起始标记（start of sequence）。因此，所有评论列表都以 `1` 开头，以表示评论的开始。索引 `0` 用于在需要时对评论进行填充，使所有评论具有相同的长度。

### 评论长度的直方图

```python
# 评论长度的直方图
length_sentences = [len(sentence) for sentence in train_data]
val, count = np.unique(length_sentences, return_counts=True)
fig, ax = plt.subplots(figsize=(15,2))
ax.bar(val, count);
```

可以看到，两类标签是平衡的。

```python
train_labels
```

### 标签分布

```python
val, count = np.unique(train_labels, return_counts=True)
plt.bar(val, count);
```

**练习2：** 为什么会出现 9999？

**解答：**
`num_words=10,000` 意味着我们只保留了前 10,000 个单词。IMDB 数据集中，索引 `0`、`1` 和 `2` 被保留用于特殊用途（如填充、序列起始和未知单词）。因此，索引 `9999` 实际上代表第 10,000 个最常见的单词。这是因为索引从 `0` 开始计数，因此最大索引为 `9999`。

### 词汇频率的直方图

**练习3：** 请绘制词汇（即转换为整数的单词）的直方图，以查看哪些单词最常见。

```python
# 词汇频率的直方图
word_counts = np.bincount([word for sentence in train_data for word in sentence])
plt.figure(figsize=(15,5))
plt.bar(range(len(word_counts)), word_counts)
plt.xlim(0, 100)
plt.xlabel('Word Index')
plt.ylabel('Frequency')
plt.title('Word Frequency Distribution')
plt.show()
```

### 标签的数据类型转换

```python
# 将标签的数据类型转换为浮点数
y_train = train_labels.astype(np.float32)
y_test = test_labels.astype(np.float32)
```

### 解码评论

将索引转换回单词：

```python
# 获取词汇索引
word_index = imdb.get_word_index()
# 反转字典，映射整数索引到单词
reverse_word_index = {value: key for (key, value) in word_index.items()}
# 解码评论；注意，我们的索引偏移了 3，因为 0, 1, 2 被保留
decoded_review = ' '.join([reverse_word_index.get(i - 3, '£') for i in train_data[0]])

decoded_review
```

**练习4：** 这条评论显然是正面的。请找到一条负面评论并将其解码。

```python
# 找到一条负面评论并解码
negative_index = np.where(train_labels == 0)[0][0]
decoded_negative_review = ' '.join([reverse_word_index.get(i - 3, '£') for i in train_data[negative_index]])
print(decoded_negative_review)
```

### 数据准备

索引表示的单词是类别变量，需要进行数值化。有两种方法：
* **Word2Vec**：将每个单词表示为高维向量，使得单词之间的语义关系在向量空间中体现。稍后会详细介绍。
* **One-Hot Encoding**：例如，评论 `[3, 5, 1]` 将被转换为一个长度为 10,000 的向量，除了索引 3、5 和 1 对应的位置为 1，其余位置为 0。

```python
def vectorize_sequences(sequences, dimension):
    # 创建一个全零矩阵，形状为 (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension), dtype=np.int32)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # 将 results[i] 的特定索引位置设置为 1
    return results
```

**练习5：** 编写一个函数 `vectorize_sequences_with_count`，考虑单词出现的次数。例如：

```python
vectorize_sequences_with_count([[3,1,1,3], [1,2,2]], 10)
```

将返回：

```
[[0 2 0 2 0 0 0 0 0 0]
 [0 1 2 0 0 0 0 0 0 0]]
```

**解答：**

```python
def vectorize_sequences_with_count(sequences, dimension):
    results = np.zeros((len(sequences), dimension), dtype=np.int32)
    for i, sequence in enumerate(sequences):
        for index in sequence:
            if index < dimension:
                results[i, index] += 1
    return results

# 测试函数
print(vectorize_sequences_with_count([[3,1,1,3], [1,2,2]], 10))
```

### 向量化训练和测试数据

```python
x_train = vectorize_sequences(train_data, num_words)
x_test = vectorize_sequences(test_data, num_words)

x_train.shape, x_test.shape
```

```python
y_train.shape, y_test.shape
```

## 模型

### 数学原理

1. 我们创建一个模型：它是一个函数 $x \to \text{model}_w(x)$，值在 $[0,1]$ 之间，由参数 $w$ 决定。这个函数是由几个简单函数组成的：
    * 线性函数
    * ReLU 函数（引入非线性）
    * Sigmoid 函数（将输出限制在 $[0,1]$）
2. 我们的直觉是：对于某个参数 $w$，对于每对数据 $(x, y)$，有：
    $$
    \text{model}_w(x) = \hat{y} \in [0,1] \quad \text{接近} \quad y \in \{0,1\}
    $$
3. 我们选择一种“距离”度量来衡量 $\hat{y}$ 和 $y$ 之间的差距，即二元交叉熵损失（Binary Crossentropy, BCE）：
    $$
    \text{BCE}(y, \hat{y}) := - y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})
    $$
    然后在所有样本上求和：
    $$
    \text{loss}_w = \sum_{i \in \text{Batch}} \text{BCE}(y_i, \text{model}_w(x_i))
    $$
    我们要求优化算法找到使损失最小化的参数 $\hat{w}$。
4. 函数 $x \to \text{model}_{\hat{w}}(x)$ 将成为一个可以用于预测新输入 $x$ 的工具。

### 构建模型

我们将构建一个全连接网络：

```python
class ModelClassif(tf.keras.Model):

    def __init__(self):
        super().__init__()
        # 指定输入维度，但这不是必须的！
        self.layer1 = tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,))
        self.layer2 = tf.keras.layers.Dense(20, activation='relu')
        self.layer3 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, X):
        Y = self.layer1(X)      # = relu(X @ W1 + b1)
        Y = self.layer2(Y)      # = relu(Y @ W2 + b2)
        return self.layer3(Y)   # = sigmoid(Y @ W3 + b3)
```

```python
model = ModelClassif()
res = model(tf.random.uniform([3, 10000]))
res.shape
```

**练习6：** 各个可训练变量的形状是什么？请通过执行以下代码进行验证：

```python
for tens in model.trainable_variables:
    print(tens.shape)
```

**解答：**
假设第一层有 16 个神经元，输入维度为 10,000，则：
- 第一层权重形状为 `(10000, 16)`
- 第一层偏置形状为 `(16,)`
- 第二层权重形状为 `(16, 20)`
- 第二层偏置形状为 `(20,)`
- 第三层权重形状为 `(20, 1)`
- 第三层偏置形状为 `(1,)`

模型的总参数数量为：
- 第一层：10000 * 16 + 16 = 160,016
- 第二层：16 * 20 + 20 = 340
- 第三层：20 * 1 + 1 = 21
- **总计：160,377 个参数**

可以通过以下代码确认：

```python
for tens in model.trainable_variables:
    print(tens.shape)
```

简化模型的定义：

```python
model = keras.models.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(keras.layers.Dense(20, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```

我们还可以单独访问各层并获取它们的权重：

```python
for index in [0, 1, 2]:
    weights, bias = model.layers[index].get_weights()
    print(f"层 {index}")
    print("权重形状:", weights.shape)
    print("偏置形状:", bias.shape)
    print()
```

```python
model.summary()
```

**备注：** 对于每个隐藏层：神经元的数量（即维度=单元数量）对应于“我们赋予神经网络的自由度数量”。更多的自由度允许网络学习更复杂的表示，但也需要更多的计算，并且可能学习到训练数据中特有的、不可泛化的模式。

### 模型重建：绘制一个小型模型

以下是一个更小的网络。

```python
model = keras.models.Sequential()
model.add(keras.layers.Dense(8, activation='relu', input_shape=(16,)))
model.add(keras.layers.Dense(4, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```

让我们进行图形表示：每条箭头代表一个参数。

**练习7：**
1. 黑色箭头（$\heartsuit 1$）表示什么？
2. 这个网络有多少个参数？

**解答：**
1. 黑色箭头通常表示权重连接或偏置项。
2. 计算参数数量：
   - 第一层：16（输入） * 8（神经元） + 8（偏置） = 136
   - 第二层：8 * 4 + 4 = 36
   - 第三层：4 * 1 + 1 = 5
   - **总计：177 个参数**

```python
nb_input = 16
nb_hidden1 = 8
nb_hidden2 = 4
input_points = [(1, i) for i in range(nb_input)]
hidden1_points = [(2, 2*i + 0.5) for i in range(nb_hidden1)]
hidden2_points = [(3, 4*i + 1) for i in range(nb_hidden2)]

for x, y in input_points:
    for a, b in hidden1_points:
        plt.plot([x, a], [y, b])

for a, b in hidden1_points:
    plt.plot([1.1, a], [16, b], "k")
    for x, y in hidden2_points:
        plt.plot([x, a], [y, b])

for x, y in hidden2_points:
    plt.plot([2.1, x], [16, y], "k")
    plt.plot([x, 4], [y, 8])

plt.plot([3.1, 4], [16, 8], "k");
```

## 训练准备

### 批量数据生成器

```python
""" 批量数据分发器。 """
def oneEpoch(X_all, Y_all, batch_size):
    nb_batches = len(X_all) // batch_size
    shuffle_index = np.random.permutation(len(X_all))
    X_all_shuffle = X_all[shuffle_index]
    Y_all_shuffle = Y_all[shuffle_index]

    for i in range(nb_batches):
        yield X_all_shuffle[i*batch_size:(i+1)*batch_size], Y_all_shuffle[i*batch_size:(i+1)*batch_size]
```

```python
for x, y in oneEpoch(x_train, y_train, 256):
    print(x_train.shape)
    print(y_train.shape)
    break
```

### 损失和准确率

```python
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-6
    return tf.reduce_mean(-y_true * tf.math.log(y_pred + epsilon) - (1 - y_true) * tf.math.log((1 - y_pred) + epsilon))

y_true = np.array([[1., 0]])
y_pred = np.array([[0.9, 0.1]])
print(y_true)

binary_cross_entropy(y_true, y_pred).numpy()
```

Keras 已经实现了这个函数，只是它不会取平均值：

```python
keras.losses.binary_crossentropy(y_true, y_pred).numpy()
```

```python
def accuracy(y_true, y_pred):
    y_pred = np.round(y_pred).astype(bool)
    y_true = y_true.astype(bool)
    return np.mean(y_pred == y_true)

accuracy(y_true, y_pred)
```

### 训练代理

```python
class Agent:

    def __init__(self, model, learning_rate, X, Y, batch_size, alpha=0, ridge_vs_lasso=True):
        self.loss_fn = binary_cross_entropy
        self.model = model
        self.batch_size = batch_size
        self.alpha = alpha
        self.ridge_vs_lasso = ridge_vs_lasso

        nb_data_train = int(len(X) * 0.8)
        self.X_train = X[:nb_data_train]
        self.Y_train = Y[:nb_data_train]
        self.X_val = X[nb_data_train:]
        self.Y_val = Y[nb_data_train:]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.losses = []
        self.val_steps = []
        self.val_losses = []
        self.step_count = -1

    def compute_penalization_term(self):
        all_kernels = []

        if self.ridge_vs_lasso:
            fn = lambda kernel: kernel**2
        else:
            fn = lambda kernel: tf.abs(kernel)

        for tv in self.model.trainable_variables:
            if len(tv.shape) == 2:
                all_kernels.append(tf.reshape(fn(tv), [-1]))

        return self.alpha * tf.reduce_sum(tf.concat(all_kernels, axis=0))

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y[:, None], y_pred)
            if self.alpha > 0:
                loss += self.compute_penalization_term()

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables,))
        return loss

    @tf.function
    def val_step(self, x, y):
        y_pred = self.model(x, training=False)
        loss = self.loss_fn(y[:, None], y_pred)  # 注意，与 PyTorch 顺序相反
        return loss

    def train(self, nb_epochs):
        for _ in range(nb_epochs):
            for x, y in oneEpoch(self.X_train, self.Y_train, self.batch_size):
                self.step_count += 1
                loss = self.train_step(x, y)
                self.losses.append(loss.numpy())

            val_loss = self.val_step(self.X_val, self.Y_val).numpy()
            self.val_losses.append(val_loss)

            if val_loss <= np.min(self.val_losses):
                print(f"⬊{val_loss:.2f}", end="")
                # get_weights() 进行复制，无需像 PyTorch 一样使用 deepcopy
                self.best_weights = self.model.get_weights()
            else:
                print("⬈", end="")

            self.val_steps.append(self.step_count)

    def set_model_at_best(self):
        self.model.set_weights(self.best_weights)
```

```python
def test():
    learning_rate = 1e-3
    batch_size = 256
    model = ModelClassif()
    agent = Agent(model, learning_rate, x_train, y_train, batch_size, alpha=0.1, ridge_vs_lasso=False)
    agent.train(1)
test()
```

## 训练

### 初始训练

```python
learning_rate = 1e-3
batch_size = 256
agent = Agent(ModelClassif(), learning_rate, x_train, y_train, batch_size)

agent.train(10)
```

绘制损失和验证损失：

```python
plt.plot(agent.losses, label="loss")
plt.plot(agent.val_steps, agent.val_losses, ".", label="val_loss")
plt.legend();
```

* 损失持续下降：优化器正常工作。
* 但验证损失在 2-3 个 epoch 后上升。

这典型地表明过拟合：优化器学习到了训练数据中特有的模式，无法泛化到验证集、测试集或未来的数据。

### 在测试数据上评估模型

```python
binary_cross_entropy(y_test[:, None], agent.model(x_test))
```

```python
accuracy(y_test[:, None], agent.model(x_test))
```

```python
agent.set_model_at_best()
binary_cross_entropy(y_test[:, None], agent.model(x_test))
```

```python
accuracy(y_test[:, None], agent.model(x_test))
```

**结果：** 88% 的准确率，虽然不错，但“最先进的方法”（state-of-the-art）可达 95%。

**练习8：** 对于我们使用的协议，是否真的需要将数据分为验证集和测试集？

**解答：**
在机器学习中，验证集和测试集的分离是为了确保模型的泛化能力。验证集用于模型选择和超参数调整，而测试集用于最终评估模型性能。如果仅使用一个测试集，可能会导致模型对测试集过拟合，无法真实反映其泛化能力。因此，分离验证集和测试集是必要的。

### 预测分析

```python
hat_y_test_proba = agent.model(x_test).numpy()
hat_y_test = (hat_y_test_proba > 0.5).astype(int)
print("proba:")
print(hat_y_test_proba[:10])
print("估计类别")
print(hat_y_test[:10])
print("真实类别")
print(y_test[:10])
```

**练习9：** 找到一条分类错误的评论，并将其解码为英文。

```python
# 找到分类错误的索引
incorrect_indices = np.where(hat_y_test.flatten() != y_test)[0]
# 选择第一条错误分类的评论
incorrect_review_index = incorrect_indices[0]
decoded_incorrect_review = ' '.join([reverse_word_index.get(i - 3, '£') for i in test_data[incorrect_review_index]])
print(decoded_incorrect_review)
```

**练习10：** 从上述两个向量中计算出 88% 的准确率。

**解答：**
准确率是预测正确的样本数除以总样本数。假设有 `n` 个测试样本，其中 `m` 个预测正确，则准确率为 `m / n = 0.88`。具体计算可以通过以下代码实现：

```python
correct_predictions = np.sum(hat_y_test.flatten() == y_test)
total_predictions = len(y_test)
accuracy_score = correct_predictions / total_predictions
print(f"Accuracy: {accuracy_score * 100:.2f}%")
```

### 概率分布的直方图

```python
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8,2))
ax0.hist(hat_y_test_proba[y_test == 0], bins=40, edgecolor="k")
ax1.hist(hat_y_test_proba[y_test == 1], bins=40, edgecolor="k");
```

可以看到，网络对预测相当自信：大多数概率接近于 0 或 1。

### 练习11：进行自己的实验

* **练习11.1:** 尝试使用 `vectorize_sequences_with_count` 对序列进行向量化。
* **练习11.2:** 我们使用了两层隐藏层。尝试使用 1 层或 3 层。
* **练习11.3:** 更改每层的神经元数量（units）。
* **练习11.4:** 尝试使用 `mse` 损失函数。
* **练习11.5:** 尝试在中间层使用 `tanh` 激活函数。

## 对抗过拟合

### 减少模型规模

```python
smaller_model = keras.models.Sequential()
smaller_model.add(keras.layers.Dense(4, activation='relu', input_shape=(10000,)))
smaller_model.add(keras.layers.Dense(4, activation='relu'))
smaller_model.add(keras.layers.Dense(1, activation='sigmoid'))

agent_smaller = Agent(smaller_model, learning_rate, x_train, y_train, batch_size)
agent_smaller.train(10)

plt.plot(agent.val_steps, agent.val_losses, ".-", label="original")
plt.plot(agent_smaller.val_steps, agent_smaller.val_losses, ".-", label="smaller")
plt.legend();
```

较小的模型在更少的 epoch 后开始过拟合，其性能下降更缓慢。

现在尝试一个更大规模的模型：

```python
bigger_model = keras.models.Sequential()
bigger_model.add(keras.layers.Dense(512, activation='relu'))
bigger_model.add(keras.layers.Dense(512, activation='relu'))
bigger_model.add(keras.layers.Dense(1, activation='sigmoid'))

agent_bigger = Agent(bigger_model, learning_rate, x_train, y_train, batch_size)
agent_bigger.train(10)

plt.plot(agent.val_steps, agent.val_losses, ".-", label="original")
plt.plot(agent_smaller.val_steps, agent_smaller.val_losses, ".-", label="smaller")
plt.plot(agent_bigger.val_steps, agent_bigger.val_losses, ".-", label="bigger")
plt.legend();
```

**结果：** 大模型过拟合更严重，验证损失更加波动。

### 正则化

**奥卡姆剃刀（Occam's Razor）原则：** 在有两个有效的解释时，最好的解释是最简单的。

这也适用于机器学习模型：简单模型具有更低的参数熵。为正则化，我们可以强制权重取较小的值，这使得权重分布更平滑。这称为“权重正则化”或“权重惩罚”。

为了实现这一点，我们在损失函数中添加一个项，当权重较大时，这个项也较大。两种主要技术是：
* **L1 正则化（Lasso）**：
    $$
    \text{loss}_\alpha = \text{loss} + \alpha \sum_i |w_i|
    $$
* **L2 正则化（Ridge）**：
    $$
    \text{loss}_\alpha = \text{loss} + \alpha \sum_i w_i^2
    $$
其中，求和是对所有权重 $w_i$ 进行的。

**注意：** 通常不惩罚偏置（biases）。

**练习12：** 岭正则化（Ridge）也被称为权重衰减（weight decay）。这是为什么？考虑未正则化的损失函数 $\text{loss}(w)$。梯度下降的规则为：
$$
w_i \leftarrow w_i - \ell \cdot \frac{\partial \text{loss}}{\partial w_i}
$$
其中 $\ell$ 是学习率。考虑正则化后的损失函数：
$$
\text{loss}_\alpha(w) = \text{loss}(w) + \alpha \sum_i w_i^2
$$
现在梯度下降的更新规则是什么？这如何证明在岭正则化下，系数 $\alpha$ 有时被称为 `loss_decay`？

**解答：**
对于正则化后的损失函数：
$$
\text{loss}_\alpha(w) = \text{loss}(w) + \alpha \sum_i w_i^2
$$
梯度下降的更新规则为：
$$
w_i \leftarrow w_i - \ell \left( \frac{\partial \text{loss}}{\partial w_i} + 2\alpha w_i \right)
$$
这可以重写为：
$$
w_i \leftarrow w_i - \ell \cdot \frac{\partial \text{loss}}{\partial w_i} - 2\ell\alpha w_i
$$
即：
$$
w_i \leftarrow (1 - 2\ell\alpha) w_i - \ell \cdot \frac{\partial \text{loss}}{\partial w_i}
$$
这表明每次更新时，权重 $w_i$ 都会乘以一个小于 1 的因子 $(1 - 2\ell\alpha)$，从而导致权重逐渐衰减。这就是“权重衰减”（weight decay）的由来。

**备注：**
- 必须适当调整 $\alpha$：
  * $\alpha$ 太小，正则化效果不明显。
  * $\alpha$ 太大，模型无法学习（所有权重趋近于 0）。
- 通常，通过交叉验证选择 $\alpha$，尝试 $0.1, 0.01, 0.001, 0.0001$ 并比较验证损失曲线。

### 实现 L2 正则化

```python
agent_ridge = Agent(ModelClassif(), learning_rate, x_train, y_train, batch_size, alpha=0.001, ridge_vs_lasso=False)
agent_ridge.train(10)

plt.plot(agent.val_steps, agent.val_losses, ".-", label="original")
plt.plot(agent_ridge.val_steps, agent_ridge.val_losses, ".-", label="ridge")
plt.legend();
```

可以看到，正则化后的模型更能抵抗过拟合。

### 权重分布的可视化

```python
layer_index = 0

l2_weights, l2_biases = agent_ridge.model.layers[layer_index].get_weights()
original_weights, original_biases = agent.model.layers[layer_index].get_weights()

fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10,10))

bins = np.linspace(-0.1, 0.1, 40)

axs[0,0].hist(l2_weights.reshape(-1), bins=bins)
axs[0,0].set_title("权重正则化")
axs[0,0].set_xlim(-0.1, 0.1)
axs[0,1].hist(l2_biases.reshape(-1), bins=bins)
axs[0,1].set_title("偏置正则化")
axs[0,1].set_xlim(-0.1, 0.1)

axs[1,0].hist(original_weights.reshape(-1), bins=bins)
axs[1,0].set_title("权重")
axs[1,1].hist(original_biases.reshape(-1), bins=bins);
axs[1,1].set_title("偏置")
```

**练习13：** `Agent` 类中的 `ridge_vs_lasso` 选项允许切换不同的正则化。请运行程序，选择另一种正则化方式（如 L1），并观察权重的直方图。

```python
# 使用 L1 正则化
agent_lasso = Agent(ModelClassif(), learning_rate, x_train, y_train, batch_size, alpha=0.001, ridge_vs_lasso=True)
agent_lasso.train(10)

# 可视化权重分布
l1_weights, l1_biases = agent_lasso.model.layers[layer_index].get_weights()

fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10,10))

axs[0,0].hist(l1_weights.reshape(-1), bins=bins)
axs[0,0].set_title("权重 L1 正则化")
axs[0,0].set_xlim(-0.1, 0.1)
axs[0,1].hist(l1_biases.reshape(-1), bins=bins)
axs[0,1].set_title("偏置 L1 正则化")
axs[0,1].set_xlim(-0.1, 0.1)

axs[1,0].hist(original_weights.reshape(-1), bins=bins)
axs[1,0].set_title("权重")
axs[1,1].hist(original_biases.reshape(-1), bins=bins);
axs[1,1].set_title("偏置")
```

**结果：** L1 正则化倾向于产生稀疏权重，即许多权重变为零，这有助于特征选择。

正则化 L1 通常用于线性模型，以选择重要特征：查看哪些变量对应的权重不为零，这些就是重要的变量。

我们还可以结合两种正则化，称为“弹性网”（elastic-net）。

### 总结

为了对抗过拟合，我们可以：
* 收集更多数据或增强数据
* 减少模型的复杂度（参数数量）
* 对权重进行正则化
* 添加 dropout（将在下学期介绍）

研究人员还发现，减少浮点数精度（如使用 float32 而非 float64）引入的模糊可以限制过拟合。进一步的研究包括使用贝叶斯神经网络，通过随机化参数来增加模型的泛化能力。

然而，有时我们希望过拟合：
* 当我们想测试模型是否足够复杂时，可以验证其是否能够过拟合训练数据
* 当我们希望在无噪声的数据上进行插值时（训练集等于测试集）

此时，我们会尽可能减少训练损失。

## 评估指标

总结：
* 模型基于 `x_test` 预测出概率 `hat_y_test_proba`
* 设定阈值 0.5，得到预测类别 `hat_y_test = (hat_y_test_proba > 0.5)`
* 现在，我们将以不同的方式衡量 `hat_y_test` 和真实标签 `y_test` 之间的差异

在二元分类中，两类之一称为“正类”，另一类称为“负类”。传统上，正类是我们最关注的类别，例如疾病的存在，通常是少数类。在这里，我们将类别 1 定义为正类，即正面电影评论。

```python
print(hat_y_test_proba)
print(hat_y_test)
print(y_test)
```

### 混淆矩阵

* 行表示真实类别
* 列表示预测类别
* 例如，第一行第一列的交集表示被正确分类为负类的数量

```python
import sklearn.metrics
C = sklearn.metrics.confusion_matrix(y_test, hat_y_test)
# 为了展示，我们使用 DataFrame
C_df = pd.DataFrame(data=C, columns=[r"^-", r"^+"], index=[r"-", r"+"])
C_df
```

### 真正类与假正类

我们为混淆矩阵的每个格子命名：
$$
\begin{array}{c|cc}
& \hat{-} & \hat{+} \\
\hline
- & TN & FP \\
+ & FN & TP   
\end{array}
$$

- **TN** (True Negative)：真实为负类，预测为负类
- **FP** (False Positive)：真实为负类，预测为正类
- **FN** (False Negative)：真实为正类，预测为负类
- **TP** (True Positive)：真实为正类，预测为正类

```python
TN = C[0, 0]
FN = C[1, 0]
FP = C[0, 1]
TP = C[1, 1]
```

**练习14：**
1. 如果执行 `confusion_matrix(y_test, y_test)`，结果会是什么？
2. 如果执行 `confusion_matrix(y_test, y_rand)`，其中 `y_rand` 是一个随机的 0 和 1 的向量，结果会是什么？

**解答：**
1. `confusion_matrix(y_test, y_test)`：所有预测都正确，因此混淆矩阵的对角线元素为样本数量，非对角线元素为 0。
   ```
      ^-  ^+
   -  TN  FP
   +  FN  TP
   ```
   结果将是：
   ```
      ^-  ^+
   -  N   0
   +  0   P
   ```
   其中 N 和 P 分别是负类和正类的样本数量。

2. `confusion_matrix(y_test, y_rand)`：由于 `y_rand` 是随机的 0 和 1，因此混淆矩阵的元素将接近于随机分布。具体数值取决于随机生成的 `y_rand`。

### 精确率与召回率

$$
\begin{align}
\text{精确率 (Precision)} & = \frac{TP}{TP + FP} = \frac{+\cap \hat{+}}{\hat{+}} = \text{正类预测的准确率} \\
\text{召回率 (Recall)} & = \frac{TP}{TP + FN} = \frac{+\cap \hat{+}}{+} = \text{正确检测出的正类比例}
\end{align}
$$

* 精确率接近 1：大多数预测为正类是正确的。
* 召回率接近 1：大多数正类被正确检测出来。

如果我们的模型用于检测疾病，我们需要高召回率，尤其是在可以进行后续筛查以消除假阳性的情况下。

```python
print("精确率: %.2f" % (TP / (TP + FP)))
print("召回率: %.2f" % (TP / (TP + FN)))
```

### F1 分数

F1 分数是精确率和召回率的调和平均数：
$$
F_1 = \frac{2}{\frac{1}{\text{精确率}} + \frac{1}{\text{召回率}}}
$$
当精确率和召回率都较高时，模型具有良好的 F1 分数。

## 调整阈值

我们不必将阈值固定为 0.5。通常，我们可以根据需求调整阈值，尤其是在需要平衡精确率和召回率时。

在选择阈值之前，绘制所有可能阈值的结果曲线是有益的。

### 精确率/召回率权衡

```python
precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(y_test, hat_y_test_proba)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="精确率", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="召回率", linewidth=2)
    plt.xlabel("阈值", fontsize=16)
    plt.legend(loc="lower center", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([0, 1])
```

* 高阈值 → 预测为正类的数量少 → 精确率高，召回率低。
* 极端情况：阈值 = 1 → 预测为正类为空 → 精确率 = 1，召回率 = 0。

```python
# 极端情况
threshold = 1
y_hat = (hat_y_test_proba >= threshold).astype(int)
print("精确率:", TP / (TP + FP) if (TP + FP) > 0 else 1)
print("召回率:", TP / (TP + FN))
```

* 低阈值 → 预测为正类的数量多 → 精确率低，召回率高。
* 极端情况：阈值 = 0 → 预测为正类 = 所有样本 → 精确率 = 正类比例，召回率 = 1。

```python
threshold = 0
y_hat = (hat_y_test_proba >= threshold).astype(int)
print("精确率:", TP / (TP + FP) if (TP + FP) > 0 else 1)
print("召回率:", TP / (TP + FN))
```

可以根据精确率和召回率曲线选择合适的阈值，例如选择召回率为 0.8 时的阈值。

### ROC 曲线

ROC（Receiver Operating Characteristic）曲线是评估分类模型的另一种常用方法。

* **TPR** (True Positive Rate) = 召回率
* **FPR** (False Positive Rate) = $\frac{FP}{FP + TN}$

```python
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, hat_y_test_proba)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('假正率 (FPR)', fontsize=16)
    plt.ylabel('真正率 (TPR)', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)

# 计算 AUC 分数
print("AUC 分数:", sklearn.metrics.roc_auc_score(y_test, hat_y_test_proba))
```

### 手动绘制 ROC 曲线

```python
def TPR_FPR(threshold: float, y: np.ndarray, scores: np.ndarray):
    """
    该函数返回：
        * 真正率 (TPR)：正确分类为正类的比例
        * 假正率 (FPR)：错误分类为正类的比例
    """
    # 将概率转换为预测类别
    y_hat = (scores >= threshold).astype(int)

    # 计算正类和负类的索引
    index_P = (y == 1)
    index_N = (y == 0)

    TPR = np.sum(y_hat[index_P] == 1) / np.sum(index_P)
    FPR = np.sum(y_hat[index_N] == 1) / np.sum(index_N)

    return TPR, FPR

# 测试函数
a = [
    [0.49, 0.51],
    [0.3, 0.7],
    [0.1, 0.9],
    [0.6, 0.4],
    [0.55, 0.45],
    [0.2, 0.8]
]
probas = np.array(a)
print("概率:\n", probas)
true_cl = np.array([1, 1, 1, 0, 1, 0])
print("真实类别:\n", true_cl)

# 使用阈值 0.5
estimated_cl = (probas > 0.5).astype(int)[:,1]
print("估计类别:\n", estimated_cl)

# 计算 TPR 和 FPR
TPR, FPR = TPR_FPR(threshold=0.5, y=true_cl, scores=probas[:,1])
print(f"TPR: {TPR}, FPR: {FPR}")
```

**练习15：** 优化 `TPR_FPR` 函数：
1. **练习15.1:** 如果输入 `y = np.zeros(6)`，会出现什么数学问题？
2. **练习15.2:** 如果用户输入 `y = [1,1,1,0,1,0]`（列表而非 NumPy 数组），会有什么问题？

**解答：**
1. 如果 `y = np.zeros(6)`，即所有样本都是负类，那么计算 `TPR` 时分母为零（因为没有正类），会导致数学上的未定义（除以零）。
2. 如果用户输入 `y = [1,1,1,0,1,0]`（列表），`y == 1` 会返回一个列表而非布尔数组，导致后续的索引操作失效。因此，函数应确保 `y` 是 NumPy 数组。

优化后的函数：

```python
def TPR_FPR(threshold: float, y: np.ndarray, scores: np.ndarray):
    """
    该函数返回：
        * 真正率 (TPR)：正确分类为正类的比例
        * 假正率 (FPR)：错误分类为正类的比例
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)
    
    # 将概率转换为预测类别
    y_hat = (scores >= threshold).astype(int)

    # 计算正类和负类的索引
    index_P = (y == 1)
    index_N = (y == 0)

    # 防止除以零
    TPR = np.sum(y_hat[index_P] == 1) / np.sum(index_P) if np.sum(index_P) > 0 else 0
    FPR = np.sum(y_hat[index_N] == 1) / np.sum(index_N) if np.sum(index_N) > 0 else 0

    return TPR, FPR
```

```python
thresholds = np.linspace(np.min(hat_y_test_proba), np.max(hat_y_test_proba), 100)
fpr_list = []
tpr_list = []
for th in thresholds:
    fpr_val, tpr_val = TPR_FPR(th, y_test, hat_y_test_proba)
    fpr_list.append(fpr_val)
    tpr_list.append(tpr_val)
print(fpr_list)

plt.plot(tpr_list, fpr_list);
```

### 与随机森林的比较

我们将在下一次实验中详细介绍随机森林的工作原理。

随机森林通过 `predict_proba` 方法返回概率（而非 `predict` 方法）。

```python
import sklearn.ensemble
forest_clf = sklearn.ensemble.RandomForestClassifier(random_state=42)
forest_clf.fit(x_train, y_train)
hat_y_test_proba_forest = forest_clf.predict_proba(x_test)

print(hat_y_test_proba_forest[:10,:])
```

为了绘制 ROC 曲线，我们需要一个评分而不是两个概率。一个简单的解决方案是使用正类的概率作为评分：

```python
y_scores_forest = hat_y_test_proba_forest[:, 1]  # 评分 = 正类的概率
fpr_forest, tpr_forest, thresholds_forest = sklearn.metrics.roc_curve(y_test, y_scores_forest)
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr_forest, tpr_forest, "随机森林")
plot_roc_curve(fpr, tpr, "神经网络")

plt.legend(loc="lower right", fontsize=16);
```

**结果：** 随机森林的表现较差。
* 我们没有调整随机森林的超参数。
* 文本数据对随机森林不太友好。

## 使用嵌入（Embedding）

### 原理

我们将使用更先进的技术：嵌入（embedding）。每个整数（代表单词）将与一个高维向量关联；这些向量是可训练的变量：在训练过程中，单词之间的语义关系将通过向量关系体现。

### 小示例

```python
import pandas as pd
vocab_size = 10  # 可能的单词数量
embed_dim = 3    # 每个向量的维度
embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_dim)
input_seq = tf.constant([1, 0, 2, 2, 1, 8, 8])
res = embedding_layer(input_seq)
pd.DataFrame(index=input_seq, data=res.numpy())
```

查看嵌入层的可训练变量：

```python
embedding_layer.trainable_variables
```

### 添加填充（Padding）

```python
def make_full_matrix(data, max_sentence_length=600):
    nb = len(data)
    res = np.zeros([nb, max_sentence_length], dtype=np.int32)
    for i, sentence in enumerate(data):
        keep = min(len(sentence), max_sentence_length)
        res[i, :keep] = sentence[:keep]
    return res

x_train2 = make_full_matrix(train_data)
x_test2 = make_full_matrix(test_data)

x_train2.shape
```

```python
plt.imshow(x_train2[:500] > 0);
```

### 在批量上测试嵌入层

```python
batch = x_train2[:5]
batch.shape

embed_dim = 32
embedding_layer = tf.keras.layers.Embedding(10_000, embed_dim)
batch_embedded = embedding_layer(batch)
batch_embedded.shape
```

绘制嵌入向量的部分维度：

```python
nb = 5
fig, axs = plt.subplots(nb, 1, figsize=(10, nb), sharex="all")
for i in range(nb):
    axs[i].plot(batch_embedded[0,:,i])
```

```python
nb = 5
fig, axs = plt.subplots(nb, 1, figsize=(10, nb), sharex="all")
for i in range(nb):
    axs[i].plot(batch_embedded[1,:,i])
```

**练习16：** 为什么序列的部分会变得恒定？

**解答：**
当序列达到一定长度后，填充的部分（即索引为 0 的位置）被映射到同一个嵌入向量（可能是全零向量或其他固定向量），因此这些部分的嵌入向量会变得恒定。

### 构建嵌入模型

```python
class ModelClassifEmbed(tf.keras.Model):

    def __init__(self):
        super().__init__()
        # 指定嵌入层参数
        self.embed_dim = 32
        self.vocab_size = num_words
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embed_dim)

        self.layer0 = tf.keras.layers.Dense(32, activation='relu')
        self.layer1 = tf.keras.layers.Dense(32, activation='relu')
        self.layer2 = tf.keras.layers.Dense(32, activation='relu')
        self.layer3 = tf.keras.layers.Dense(1, activation='sigmoid')

        rate = 0.3
        self.drop_layer0 = tf.keras.layers.Dropout(rate)
        self.drop_layer1 = tf.keras.layers.Dropout(rate)
        self.drop_layer2 = tf.keras.layers.Dropout(rate)
        self.drop_layer3 = tf.keras.layers.Dropout(rate)

    def call(self, X, training):
        batch_size, nb_word = X.shape

        X = self.embedding_layer(X)  # (batch_size, nb_word, embed_dim)
        X = tf.reshape(X, [batch_size, nb_word * self.embed_dim])

        X = self.drop_layer0(X, training)
        X = self.layer0(X)
        X = self.drop_layer1(X, training)
        X = self.layer1(X)
        X = self.drop_layer2(X, training)
        X = self.layer2(X)
        X = self.drop_layer3(X, training)
        return self.layer3(X)
```

### 训练嵌入模型

```python
learning_rate = 5e-4
batch_size = 512
agent = Agent(ModelClassifEmbed(), learning_rate, x_train2, y_train, batch_size)

agent.train(10)
```

**结果：** 尝试了多种变体，但损失无法降到 0.32 以下。

```python
plt.plot(agent.losses, label="loss")
plt.plot(agent.val_steps, agent.val_losses, ".", label="val_loss")
plt.legend();
```

```python
agent.set_model_at_best()
y_pred = agent.model(x_test2, training=False)
binary_cross_entropy(y_test[:, None], y_pred)
```

```python
accuracy(y_test[:, None], y_pred)
```

## 总结

本章我们学习了如何使用神经网络对 IMDB 数据集进行二元分类。我们探讨了数据准备、模型构建、训练、评估以及如何应对过拟合等关键步骤。此外，我们介绍了嵌入层的使用，以更好地表示文本数据中的单词关系。通过实验和练习，加深了对模型性能评估指标的理解，如混淆矩阵、精确率、召回率、F1 分数和 ROC 曲线。