# 优化器

### 清除当前环境

```python
%reset -f
```

### 导入必要的库

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
```

## 原理

我们将不再手动进行梯度下降，而是使用优化器来自动减去可训练变量的梯度。实际上，最常用的优化器是 `Adam`。它不仅仅是简单地减去梯度，还对梯度进行自适应缩放。我们将在后续的实验中详细探讨这一点。

### 优化单个变量

以下是使用 `Adam` 优化单个变量的示例：

```python
x = torch.tensor(10., requires_grad=True)
opt = torch.optim.Adam([x], lr=1.)  # 指定优化器要操作的变量

for _ in range(10):
    y = x**2
    y.backward()
    opt.step()
    opt.zero_grad()
    print(f"x:{x}")
```

### 优化两个变量

接下来，我们来看优化两个变量的例子：

```python
x = torch.tensor(10., requires_grad=True)
y = torch.tensor(10., requires_grad=True)
opt = torch.optim.Adam([x, y], lr=1.)

for _ in range(10):
    loss = torch.abs(x + y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"loss:{loss} -> x:{x}, y:{y}")
```

#### 问题：

**问题**：解释为什么 `x` 和 `y` 在每次迭代中都恰好减少 1。如果继续迭代，是否能够达到损失函数的全局最小值？如果我们降低学习率，会发生什么？

**解答**：
- 由于损失函数 `loss = |x + y|`，其梯度对于 `x` 和 `y` 都是 `sign(x + y)`。
- 初始时 `x = 10`，`y = 10`，所以 `loss = 20`，梯度对于 `x` 和 `y` 都是 `1`。
- 优化器以学习率 `lr=1` 进行更新，所以每次迭代 `x` 和 `y` 都会减少 `1`。
- 继续迭代，`x` 和 `y` 会逐渐接近 `-y` 和 `-x` 的平衡点，使得 `x + y = 0`，达到损失函数的最小值。
- 如果降低学习率，更新步长会变小，`x` 和 `y` 会更缓慢地接近最小值。

**启示**：这解释了为什么在某些情况下，均方误差（MSE）比平均绝对误差（MAE）更受欢迎，因为 MSE 的梯度在优化过程中更稳定，有助于更快地收敛。

### 类似操作，但使用更大的张量

```python
xy = torch.tensor([10., 10.], requires_grad=True)
opt = torch.optim.Adam([xy], lr=1.)

for _ in range(10):
    loss = torch.abs(torch.sum(xy))
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"loss:{loss} -> x:{xy[0]}, y:{xy[1]}")
```

## 分离（Detaching）

### 问题出现

当我们尝试绘图时，可能会遇到如下错误：

```python
x = torch.tensor(3., requires_grad=True)
y = torch.tensor(5., requires_grad=True)
z = x + y
try:
    plt.scatter(x, z)
except Exception as e:
    print(e)
```

**错误原因**：
- 由于 `x`, `y`, `z` 都有 `requires_grad=True`，它们被附加到计算图中。
- `plt.scatter` 在尝试将这些张量转换为 NumPy 数组时，会失败，因为带有梯度的张量无法直接转换。

### 解决方法

为了避免这个问题，我们需要将张量从计算图中分离出来，并移动到 CPU 上：

```python
x.detach().cpu().numpy()
```

这通常在使用其他库进行绘图或数据处理时非常有用。

### 优化部分变量

以下示例展示了如何优化部分变量，而不影响其他变量：

```python
x = torch.tensor(10., requires_grad=True)
y = torch.tensor(10., requires_grad=True)
opt = torch.optim.Adam([x, y], lr=1.)

for _ in range(10):
    x_ = x.detach()
    loss = torch.abs(x_ + y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"x:{x}, x':{x_}, y:{y}, loss:{loss}")
```

**解释**：
- `x_ = x.detach()` 创建了 `x` 的一个副本，但不附加到计算图中。
- 在计算损失时，`x_` 不会影响损失函数的梯度，因此优化器不会更新 `x`，只会更新 `y`。

### 深入理解 `.detach()`

需要注意的是，`.detach()` 方法创建了一个表面副本，移除了所有与计算图相关的信息，但数据本身仍然共享：

```python
x = torch.tensor(10., requires_grad=True)
y = torch.tensor(10., requires_grad=True)
opt = torch.optim.Adam([x, y], lr=1.)

for _ in range(10):
    x_ = x.detach()
    loss = torch.abs(x + y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"x:{x}, x':{x_}, y:{y}, loss:{loss}")
```

**注意事项**：
- `x.detach()` 和 `x` 共享同一份数据，改变 `x` 会影响 `x_`。
- 如果需要深拷贝，可以使用 `.clone()` 方法。
- 对于标量张量，可以使用 `.item()` 获取纯 Python 数值。

## 回归

当然，我们会在模型的所有变量上使用优化器。因此，优化器的语法通常如下：

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### 数据准备

```python
def make_data(nb):
    X = torch.rand(nb) * 2
    Y = torch.exp(-X) * torch.sin(5 * X) + torch.randn(nb) * 0.005
    return X, Y

X_train, Y_train = make_data(10000)
X_train.shape, Y_train.shape

plt.plot(X_train, Y_train, ".");
```

### 模型定义

```python
class Model_1d_to_1d(torch.nn.Module):
    def __init__(self, hidden_dim=20):
        super().__init__()
        self.lay1 = torch.nn.Linear(1, hidden_dim)
        self.lay2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lay3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.final_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.lay1(x))
        x = torch.tanh(self.lay2(x))
        x = torch.tanh(self.lay3(x))
        return self.final_layer(x)

model = Model_1d_to_1d()
model(X_train[:13, None]).shape
```

**备注**：如果要在 GPU 上运行，需要将数据和模型移动到 GPU：

```python
X_train = X_train.to("cuda")
Y_train = Y_train.to("cuda")
model = model.to("cuda")
```

### 训练

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
batch_size = 126
nb_steps = 300

for _ in range(nb_steps):
    indices = np.random.randint(0, len(X_train), size=batch_size)
    x = X_train[indices, None]
    y = Y_train[indices, None]
    y_pred = model(x)
    loss = torch.mean((y_pred - y) ** 2)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(losses);
```

### 测试

```python
X_test = torch.linspace(0, 2, 500)
Y_test = model(X_test[:, None])[:, 0].detach()
plt.plot(X_train, Y_train, ".")
plt.plot(X_test, Y_test);
```

#### 问题：

**问题**：你是如何得到下面的图的？

**解答**：
- 通过在训练过程中记录损失值并绘制损失曲线，可以观察模型的收敛情况。
- 使用训练好的模型在测试数据上进行预测，并将预测结果与训练数据一起绘制，观察模型的拟合效果。

### 使用 ReLU 激活函数的模型

```python
class Model_1d_to_1d_relu(torch.nn.Module):
    def __init__(self, hidden_dim=20):
        super().__init__()
        self.lay1 = torch.nn.Linear(1, hidden_dim)
        self.lay2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lay3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.final_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.lay1(x))
        x = torch.relu(self.lay2(x))
        x = torch.relu(self.lay3(x))
        return self.final_layer(x)

model = Model_1d_to_1d_relu()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
batch_size = 126
nb_steps = 300

for _ in range(nb_steps):
    indices = np.random.randint(0, len(X_train), size=batch_size)
    x = X_train[indices, None]
    y = Y_train[indices, None]
    y_pred = model(x)
    loss = torch.mean((y_pred - y) ** 2)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(losses);

X_test = torch.linspace(0, 2, 500)
Y_test = model(X_test[:, None])[:, 0].detach()
plt.plot(X_train, Y_train, ".")
plt.plot(X_test, Y_test);
```

#### 问题：

**问题**：重新运行所有代码，将激活函数 `torch.arctan` 替换为 `torch.relu`，并分析得到的图。

**解答**：
- 使用 ReLU 激活函数后，模型在处理数据时更倾向于保持线性关系。
- 由于 ReLU 只对正值有反应，可能会导致模型在拟合非线性数据时出现欠拟合或过拟合。
- 对比使用 `torch.tanh` 和 `torch.relu` 的模型，ReLU 可能在某些区域表现更好，但在处理对称非线性时可能不如 `tanh`。

### 一个隐性错误

需要注意的是，模型输入和输出通常是矩阵形式。如果我们使用一维向量进行操作，可能会引发隐性错误。例如：

```python
# 错误示例
X_train.shape = (batch_size,)
Y_train.shape = (batch_size,)

Y_pred = model(X_train[:, None])
loss = torch.mean((Y_pred - Y_train) ** 2)
```

**问题**：
- `X_train[:, None]` 将输入变为二维矩阵 `(batch_size, 1)`。
- `Y_train` 的形状是 `(batch_size,)`，在计算损失时会自动扩展为 `(1, batch_size)`。
- 这样，`Y_pred` 和 `Y_train` 的形状不匹配，导致计算错误，但不会抛出显式错误，只会得到一个无意义的损失值。

**解决方法**：
- 确保输入和输出的维度匹配。
- 可以使用 `Y_train[:, None]` 来保持形状一致。

```python
Y_pred = model(X_train[:, None])
loss = torch.mean((Y_pred - Y_train[:, None]) ** 2)
```

这样可以避免隐性错误，确保损失函数的计算正确。

## 分类

### 数据准备

我们将使用 TensorFlow 提供的 Fashion MNIST 数据集，因为它提供了方便的 NumPy 格式数据，便于使用。Fashion MNIST 数据集包含 10 个类别的服饰图片，每张图片大小为 28x28 像素。

```python
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images.shape, type(train_images)
```

绘制部分训练样本：

```python
fig, axs = plt.subplots(5, 5, figsize=(10, 10), sharex="all", sharey="all")
for i in range(5):
    for j in range(5):
        axs[i, j].imshow(train_images[i*5 + j, :, :], cmap='gray')
        axs[i, j].set_title(f"类别:{train_labels[i*5 + j]}")
fig.tight_layout()
```

**类别含义**：

```
0    T恤/上衣
1    裤子
2    套头衫
3    连衣裙
4    外套
5    凉鞋
6    衬衫
7    运动鞋
8    包
9    短靴
```

我们将每张图片视为一个简单的 784 维向量，忽略其二维结构。需要对数据进行归一化处理，使其数值不至于过大。通常采用归一化到 `[0, 1]` 区间。

```python
X_train = np.reshape(train_images / 255., [-1, 28 * 28])
X_test = np.reshape(test_images / 255., [-1, 28 * 28])

Y_train = train_labels
Y_test = test_labels

X_train.shape, X_test.shape
```

### 训练

我们将使用与 TensorFlow 训练相同的超参数：

```python
batch_size = 128
learning_rate = 1e-3
hidden_dim = 20
nb_steps = 1_000

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)

Y_train_torch = torch.tensor(Y_train, dtype=torch.int64)
Y_test_torch = torch.tensor(Y_test, dtype=torch.int64)
```

#### 模型定义

```python
class Model_classif(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lay1 = torch.nn.Linear(784, hidden_dim)
        self.lay2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lay3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.final_layer = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = torch.relu(self.lay1(x))
        x = torch.relu(self.lay2(x))
        x = torch.relu(self.lay3(x))
        return self.final_layer(x)
```

#### 训练步骤定义

```python
def train_step_torch(x, y, model, optimizer, loss_fn):
    pred = model(x)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
```

#### 训练过程

```python
loss_fn = torch.nn.CrossEntropyLoss()
model = Model_classif(hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []

%%time
for _ in range(nb_steps):
    indices = np.random.randint(0, len(X_train), size=batch_size)
    x = X_train_torch[indices, :]
    y = Y_train_torch[indices]

    loss = train_step_torch(x, y, model, optimizer, loss_fn)
    losses.append(loss)
```

### 测试

```python
Y_test_pred_logits = model(X_test_torch).detach().cpu().numpy()
Y_test_pred = np.argmax(Y_test_pred_logits, axis=1)
Y_test_pred
```

**计算准确率**：

```python
np.mean(Y_test_pred == Y_test)
```

#### 问题：

**问题**：延长训练时间，观察是否能够提高准确率。

**解答**：
- 增加训练步数可以让模型有更多机会学习数据的特征，从而提高准确率。
- 需要监控损失和准确率，避免过拟合或欠拟合。
- 适当调整学习率和隐藏层维度，可以进一步优化模型性能。

## 总结

通过以上步骤，我们学习了如何使用 PyTorch 实现优化器，进行回归和分类任务的训练。重点内容包括：

- 使用 `torch.optim.Adam` 优化器自动进行梯度下降。
- 理解 `requires_grad`、`.detach()` 和优化器的工作原理。
- 实现简单的回归模型，并观察不同激活函数对模型性能的影响。
- 处理分类任务，使用交叉熵损失函数训练分类模型，并计算准确率。

这些基础知识为深入理解和构建复杂的神经网络模型打下了坚实的基础。