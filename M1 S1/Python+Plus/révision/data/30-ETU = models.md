## 实现一个神经网络

### 清除当前环境

```python
%reset -f
```

### 导入必要的库

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
```

## 使用 `__call__` 方法

### 定义参数化函数

在数学中，我们通常区分参数和变量：
$$
f_\lambda (x) = e^{\lambda x}
$$
在这个表示中，$\lambda$ 是参数，$x$ 是变量。参数的变化性较低，而变量则更为灵活。

当一个类拥有 `__call__` 方法时，这个类的实例可以像函数一样被调用（即使用括号）。这在定义参数化函数时非常实用。以下是一些简单的示例：

```python
class Expo:
    def __init__(self, lamb):
        self.lamb = lamb

    def __call__(self, x):
        return torch.exp(self.lamb * x)

# 定义类的一个实例
exp1 = Expo(1)
# 由于实现了 __call__ 方法，可以像函数一样调用
exp1(torch.tensor(50))
```

```python
class Beta:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x):
        return (x)**(self.alpha - 1) * (1 - x)**(self.beta - 1) * 10
```

定义一个绘图函数来展示这些参数化函数：

```python
def trace_01(fn):
    x = torch.linspace(0, 1, 100)
    plt.plot(x, fn(x))

exp1 = Expo(1)
exp2 = Expo(2)
beta2_2 = Beta(2, 2)

trace_01(exp1)
trace_01(exp2)
trace_01(beta2_2)
```

### 定义回调函数

以下是一个需要 `__call__` 方法的例子。

假设你使用一个优化器库，比如 `scipy`，或者下面这个自定义的优化器 `FunctionOptimizer`。

`FunctionOptimizer` 的特点：
- 构造函数接受一个要最小化的函数。
- 唯一的方法 `iter(self, n_iter, callback=None)` 只需调用一次，进行 `n_iter` 次优化迭代。
- `FunctionOptimizer` 有两个公共属性：
  - `self.mini_value`：当前最小值的位置。
  - `self.mini_state`：当前最小值的值。

```python
class FunctionOptimizer:
    def __init__(self, func):
        self.func = func

    def iter(self, n_iter, callback=None):
        self.mini_value = float("inf")
        self.mini_state = -0.5
        for _ in range(n_iter):
            if callback is not None:
                callback()
            current_state = self.mini_state + (np.random.rand() - 0.5)
            current_value = self.func(current_state)
            if current_value < self.mini_value:
                self.mini_value = current_value
                self.mini_state = current_state
```

典型的使用示例：

```python
def func_to_opt(x):
    return np.sin(10 * (x - 3)) * np.sin(5 * (x + 1))

x = np.linspace(-2, 2, 300)
plt.plot(x, func_to_opt(x));

opti = FunctionOptimizer(func_to_opt)
opti.iter(100)
print(f"we find the minimum {opti.mini_value} at the point {opti.mini_state}")
```

**问题**：`FunctionOptimizer` 无法访问 `self.mini_value` 和 `self.mini_state` 的历史记录，但你需要这些信息。

解决方法：`iter` 方法允许传入一个回调函数 `callback`，在每次迭代时调用。你可以通过回调函数记录每一步的状态和数值。这个回调函数可以是一个对象，而这个对象需要实现 `__call__` 方法，使其可以被调用。

示例实现：

```python
class MyCallback:
    def __init__(self, functionOptimizer):
        self.states = []
        self.values = []
        self.functionOptimizer = functionOptimizer

    def __call__(self):
        self.states.append(self.functionOptimizer.mini_state)
        self.values.append(self.functionOptimizer.mini_value)

np.random.seed(123)
opti = FunctionOptimizer(func_to_opt)
callback = MyCallback(opti)
opti.iter(100, callback)

fig, ax = plt.subplots()
x = np.linspace(min(callback.states), max(callback.states), 500)
y = func_to_opt(x)
ax.plot(x, y)
ax.plot(callback.states, callback.values, ".-");
```

## 多维回归

### 数据的经典维度

在大多数数据处理问题中，每个数据点是一个向量。输入张量的形状通常为：

```
X.shape = (nb_data, dim_in)
```

输出张量的形状为：

```
Y.shape = (nb_data, dim_out)
```

例如，我们可能想根据小鼠的蛋白质、脂肪、碳水化合物的摄入量来预测它们的身高和体重。在这种情况下：
- `nb_data` = 小鼠数量
- `dim_in` = 3（蛋白质、脂肪、碳水化合物）
- `dim_out` = 2（身高和体重）

生成数据的函数：

```python
def make_data(dim_in, dim_out, nb_data=1000):
    TRUE_W = np.random.randint(1, 10, size=(dim_in, dim_out)).astype(np.float32)
    TRUE_b = np.random.randint(1, 10, size=(dim_out,)).astype(np.float32)
    TRUE_W = torch.tensor(TRUE_W)
    TRUE_b = torch.tensor(TRUE_b)

    X = torch.rand(nb_data, dim_in)
    noise = torch.rand(dim_out) * 0.01

    Y = X @ TRUE_W + TRUE_b + noise * 0.05
    return X, Y, TRUE_W, TRUE_b

X, Y, _, _ = make_data(3, 4)
X.shape, Y.shape
```

### 模型

#### 实现多维线性模型

```python
class ModelLineaireMultiD:
    def __init__(self, dim_in, dim_out):
        W_data = torch.rand(dim_in, dim_out) * 0.2 - 0.1
        self.W = W_data.clone().requires_grad_(True)
        self.b = torch.zeros([1, dim_out], requires_grad=True)

    def __call__(self, X):
        return X @ self.W + self.b

def test():
    dim_in, dim_out = 3, 4
    X, Y, _, _ = make_data(dim_in, dim_out)
    model = ModelLineaireMultiD(dim_in, dim_out)
    Y_pred = model(X)
    print(Y_pred.shape)
    assert Y_pred.shape == Y.shape

test()
```

输出：

```
torch.Size([1000, 4])
```

### 训练

#### 定义损失函数和训练函数

```python
def loss_fn(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def train(model, inputs, outputs, learning_rate):
    loss = loss_fn(model(inputs), outputs)
    loss.backward()

    dW = model.W.grad
    db = model.b.grad

    with torch.no_grad():
        model.W -= learning_rate * dW
        model.b -= learning_rate * db

        model.W.grad.zero_()
        model.b.grad.zero_()

    return loss.item()
```

训练模型：

```python
dim_in, dim_out = 2, 5
X, Y, W_true, b_true = make_data(dim_in, dim_out)

model = ModelLineaireMultiD(dim_in, dim_out)
losses = []

fig, ax = plt.subplots()
for epoch in range(500):
    loss = train(model, X, Y, learning_rate=0.3)
    losses.append(loss)
ax.plot(losses)
ax.set_yscale("log")
```

### 比较学习到的参数与真实参数

```python
model.W, W_true
model.b, b_true
```

两者非常接近。我们可以通过均方误差（MSE）来量化这种接近程度：

```python
torch.mean((model.W - W_true) ** 2), torch.mean((model.b - b_true) ** 2)
```

计算预测误差：

```python
torch.mean((model(X) - X @ W_true - b_true) ** 2)
```

这与以下等价：

```python
torch.mean((model(X) - (X @ W_true + b_true)) ** 2)
```

### 唯一性问题

**问题**：学习到的参数 `model.W` 和 `model.b` 是否可能与真实参数 `W_true` 和 `b_true` 有很大差异？观察并解释原因。

尝试高维度低样本量：

```python
dim_in, dim_out = 20, 2
X, Y, W_true, b_true = make_data(dim_in, dim_out, nb_data=18)

model = ModelLineaireMultiD(dim_in, dim_out)
losses = []

fig, ax = plt.subplots()
for epoch in range(1000):
    loss = train(model, X, Y, learning_rate=0.1)
    losses.append(loss)
ax.plot(losses)
ax.set_yscale("log")

torch.mean((model.W - W_true) ** 2), torch.mean((model.b - b_true) ** 2)
```

这时 `model.W` 和 `model.b` 远离 `W_true` 和 `b_true`，但模型的预测误差仍然较低：

```python
torch.mean((model(X) - X @ W_true - b_true) ** 2)
```

**解释**：当试图通过优化来恢复 `W_true` 和 `b_true` 时，如果样本数量 `nb_data` 小于参数数量 `dim_in * dim_out + dim_out`，则无法唯一确定参数。这意味着存在多个参数组合可以达到相同的预测效果，因此即使参数与真实值差异很大，模型的预测性能仍然良好。

在深度学习中，由于模型参数众多，通常存在多个最佳参数组合，只要预测误差低即可。因此，参数本身的唯一性并不重要，关键在于模型的预测性能。

**解决方法**：使用正则化方法，如岭回归，可以减少高维度问题带来的不稳定性。

## 神经网络

### 介绍

神经网络是一个参数化函数，通常是多个中间函数（即层）的组合。在本实验中，我们将学习如何使用 PyTorch 创建这些函数。接下来，我们将学习如何调整参数，使网络完成特定任务。

首先，我们介绍全连接网络（Dense Network），也称为多层感知器（MLP）或前馈网络（Feedforward Network）。

在全连接网络中，一个神经元接收来自前一层所有神经元的输入信号，进行加权求和，然后应用一个非线性激活函数 $S$，得到输出：
$$
y_j = S\Big( \sum_{i: i\to j} x_i\, w_{ij} \Big)
$$
然后，将新的信号 $y_j$ 传递给下一层的所有神经元。

**激活函数** $S$ 的作用是对输入进行非线性变换，通常选择以下几种函数之一：
- **Sigmoid**: $\frac{1}{1 + e^{-x}}$
- **ReLU**: $x \cdot 1_{\{x > 0\}}$
- **Arctangent**

激活函数的选择通常基于其非线性特性和对小值的抑制作用。常见的选择是单调递增函数，但在某些情况下，也会使用非单调函数，如 SIREN 网络中的正弦函数。

### 激活函数示例

```python
x = torch.linspace(-4, 4, 100)

for S, name in [
    (torch.relu, "relu"),
    (torch.sigmoid, "sigmoid"),
    (torch.tanh, "tanh"),
    (torch.selu, "selu"),
    (torch.nn.functional.softplus, "softplus"),
    (torch.nn.functional.gelu, "gelu"),
    (torch.sin, "sin"),
]:
    y = S(x)
    plt.plot(x, y, label=name)
plt.legend();
```

注意，大多数激活函数都是单调递增的，除了某些例外。

### 连接两层

神经元按层分组。在全连接网络中：
- 一层的神经元接收前一层所有神经元的信号。
- 将信号传递给下一层的所有神经元。

用数学表示：
$$
\forall j \in \text{Layer}_1 \quad y_j = S\Big( \sum_{i} x_i\, w_{ij} + b_j \Big)
$$
$$
\forall k \in \text{Layer}_2 \quad z_k = S\Big( \sum_{j} y_j\, w'_{jk} + b'_k \Big)
$$

也可以用矩阵乘法表示：
$$
y = S(X @ W + b)
$$
$$
z = S(Y @ W' + b')
$$

## 实现

### 随机初始化矩阵

```python
def rand_mat(dim_in, dim_out):
    return torch.rand(dim_in, dim_out) * 0.2 - 0.1
```

### 使用一个类实现神经网络

#### 示例实现

```python
class ModelNN:
    def __init__(self, dim_in, dim_out, dim_hidden=20):
        self.W1 = rand_mat(dim_in, dim_hidden).requires_grad_(True)
        self.b1 = torch.zeros([1, dim_hidden], requires_grad=True)

        self.W2 = rand_mat(dim_hidden, dim_out).requires_grad_(True)
        self.b2 = torch.zeros([1, dim_out], requires_grad=True)

    def __call__(self, X):
        X = torch.relu(X @ self.W1 + self.b1)
        X = torch.relu(X @ self.W2 + self.b2)
        return X

def test():
    dim_in, dim_out = 3, 4
    batch_size = 1
    X = torch.rand(batch_size, dim_in)
    model = ModelNN(dim_in, dim_out)
    Y_pred = model(X)
    print(Y_pred.shape)
    assert Y_pred.shape == (batch_size, dim_out)

test()
```

输出：

```
torch.Size([1, 4])
```

**注意**：上述模型的最后一层使用了 `ReLU` 激活函数，导致输出值只能为正数。如果需要学习带符号的数据，最后一层的激活函数需要根据目标数据的特点进行选择。

#### 填空题

> 在上述实现中，如果没有添加激活函数，模型将是一个线性模型。添加激活函数后，模型具有非线性表达能力，这对于复杂任务至关重要。

### 使用子类实现层

通过将层封装到独立的类中，可以提高代码的模块化和可复用性。

```python
class LayerLinear:
    def __init__(self, dim_in, dim_out):
        self.W = rand_mat(dim_in, dim_out).requires_grad_(True)
        self.b = torch.zeros([1, dim_out], requires_grad=True)

    def __call__(self, X):
        return X @ self.W + self.b

class ModelNN:
    def __init__(self, dim_in, dim_out, dim_hidden=20):
        self.layer1 = LayerLinear(dim_in, dim_hidden)
        self.layer2 = LayerLinear(dim_hidden, dim_out)

    def __call__(self, X):
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        return X

def test():
    dim_in, dim_out = 3, 4
    batch_size = 1
    X = torch.rand(batch_size, dim_in)
    model = ModelNN(dim_in, dim_out)
    Y_pred = model(X)
    assert Y_pred.shape == (batch_size, dim_out)

test()
```

### 使用现有的层

PyTorch 提供了现成的层（如 `torch.nn.Linear`），使用这些层可以简化代码并提高效率。

```python
class ModelNN:
    def __init__(self, dim_in, dim_out, dim_hidden=20):
        self.layer1 = torch.nn.Linear(dim_in, dim_hidden)
        self.layer2 = torch.nn.Linear(dim_hidden, dim_out)

    def __call__(self, X):
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        return X

def test():
    dim_in, dim_out = 3, 4
    batch_size = 1
    X = torch.rand(batch_size, dim_in)
    model = ModelNN(dim_in, dim_out)
    Y_pred = model(X)
    assert Y_pred.shape == (batch_size, dim_out)

test()
```

### 更好的初始化方法

使用 `torch.nn.Linear` 时，权重矩阵的初始化范围为 `-lim` 到 `+lim`，其中
$$
\text{lim} = \frac{1}{\sqrt{\text{dim\_in}}}
$$

验证这一点：

```python
dim_in = 100
layer = torch.nn.Linear(dim_in, 50)
kernel = list(layer.parameters())[0]
kernel.shape

kernel_flat = torch.reshape(kernel, [-1]).detach().numpy()
kernel_flat.shape

fig, ax = plt.subplots()
ax.hist(kernel_flat)
ax.axvline(1 / np.sqrt(dim_in), color="k")
ax.axvline(-1 / np.sqrt(dim_in), color="k")
```

**原因**：这种初始化方法确保信号在经过多层网络时，方差保持恒定，防止激活函数过度饱和或梯度爆炸/消失的问题。

数学推导：
考虑一层神经网络的第 $j$ 个输出：
$$
(xW)_j = \sum_{i=1}^{\text{dim\_in}} x_i W_{ij}
$$
假设：
- 输入 $x_i$ 是标准化的（均值为 0，方差为 1）。
- 权重 $W_{ij}$ 是独立同分布（iid）的，均值为 0，方差待定。
- 输入 $x$ 与权重 $W$ 独立。

计算 $(xW)_j$ 的方差：
$$
\mathbf{V}[(xW)_j] = \mathbf{E}\left[\left(\sum_{i=1}^{\text{dim\_in}} x_i W_{ij}\right)^2\right] = \sum_{i=1}^{\text{dim\_in}} \mathbf{E}[x_i^2] \mathbf{E}[W_{ij}^2] = \text{dim\_in} \cdot \mathbf{E}[W_{ij}^2]
$$
为了使方差保持为 1，需：
$$
\mathbf{E}[W_{ij}^2] = \frac{1}{\text{dim\_in}}
$$
如果 $W_{ij}$ 服从均匀分布 $[-\text{lim}, \text{lim}]$，则方差为：
$$
\mathbf{V}[W_{ij}] = \frac{(2 \cdot \text{lim})^2}{12} = \frac{4 \cdot \text{lim}^2}{12} = \frac{\text{lim}^2}{3}
$$
因此，为了满足：
$$
\frac{\text{lim}^2}{3} = \frac{1}{\text{dim\_in}} \quad \Rightarrow \quad \text{lim} = \frac{\sqrt{3}}{\text{dim\_in}^{0.5}}
$$

PyTorch 的 `torch.nn.Linear` 使用了一个简化的初始化方法，但接近上述值。如果需要更精确的初始化，可以自定义层。

**进一步说明**：在反向传播过程中，梯度也会经过权重矩阵的转置乘积，因此有些作者建议权重的初始化应考虑输出维度 `dim_out`，例如：
$$
\text{lim} = \frac{1}{\sqrt{\text{dim\_out}}}
$$
为了同时满足前向和反向传播的方差恒定，可以取：
$$
\text{lim} = \frac{1}{\sqrt{\text{dim\_in} + \text{dim\_out}}}
$$
这种初始化方法被称为 Xavier 初始化（也称为 Glorot 初始化）。

### 矩阵乘法的顺序

示例：

```python
dim_in, dim_hidden, dim_out = 5, 10, 4
batch_size = 1
X = torch.rand(batch_size, dim_in)

model = ModelNN(dim_in, dim_out, dim_hidden=dim_hidden)
model(X)
```

在上述程序中，数据流的维度如下：

```
dim_in --> dim_hidden --> dim_out
  5            10           4
```

观察权重矩阵的维度：

```python
for p in model.layer1.parameters():
    print(p.shape)

for p in model.layer2.parameters():
    print(p.shape)
```

输出：

```
torch.Size([10, 5])
torch.Size([4, 10])
```

这表明权重矩阵的形状为 `(dim_hidden, dim_in)` 和 `(dim_out, dim_hidden)`，即它们是右乘在数据上的。

```
dim_in --> dim_hidden --> dim_out
  5            10           4
  X           Y = X @ W1 + b1     Z = Y @ W2 + b2
```

然而，在我们的 `LayerLinear` 实现中，矩阵的乘法顺序是相反的，这看起来更优雅（也是 TensorFlow 的选择）。

## 收集所有参数

为了训练模型，必须能够修改模型的所有参数。这需要知道所有参数的列表。在之前的实现中，这并不容易，需要手动创建列表。PyTorch 提供了一个解决方案：

### 继承 `torch.nn.Module` 很方便

修改前面的实现，继承自 `torch.nn.Module`：

```python
class ModelNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=20):
        super().__init__()
        self.layer1 = torch.nn.Linear(dim_in, dim_hidden)
        self.layer2 = torch.nn.Linear(dim_hidden, dim_out)

    def forward(self, X):
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        return X
```

现在，可以轻松获取模型的所有参数：

```python
dim_in, dim_out = 3, 4
batch_size = 1
X = torch.rand(batch_size, dim_in)
model = ModelNN(dim_in, dim_out)

for p in model.parameters():
    print(p.shape)
```

**注意**：在 PyTorch 中，习惯上实现一个 `forward` 方法，而 `torch.nn.Module` 的 `__call__` 方法会自动调用用户定义的 `forward` 方法。

### 参数化激活函数

你也可以在构造函数中定义激活函数，就像定义层一样。例如，使用带参数的 ReLU（PReLU）激活函数。

```python
class ModelNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=20):
        super().__init__()
        self.layer1 = torch.nn.Linear(dim_in, dim_hidden)
        self.layer2 = torch.nn.Linear(dim_hidden, dim_out)

        self.S1 = torch.nn.PReLU()
        self.S2 = torch.nn.PReLU()

    def forward(self, X):
        X = self.S1(self.layer1(X))
        X = self.S2(self.layer2(X))
        return X

dim_in, dim_out = 3, 4
batch_size = 1
X = torch.rand(batch_size, dim_in)
model = ModelNN(dim_in, dim_out)

for p in model.parameters():
    print(p.shape)
```

观察 PReLU 的参数：

```python
S = torch.nn.PReLU()
x = torch.linspace(-3, 3, 100)
plt.plot(x, S(x).detach());

for p in S.parameters():
    print(p)
```

**输出**：
- 参数具有 `requires_grad=True` 属性，意味着它们是可训练的。
- 参数存储在正确的设备上（如 CPU 或 GPU）。
- 参数的形状符合预期。

### 使用 `nn.ModuleList` 管理多个层

```python
class ModelTorchDeep(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, nb_layer):
        super().__init__()
        self.initial_layer = torch.nn.Linear(dim_in, dim_hidden)

        # 注意，使用普通的列表无法正确注册子模块
        self.lays = torch.nn.ModuleList()
        for _ in range(nb_layer):
            self.lays.append(torch.nn.Linear(dim_hidden, dim_hidden))

        self.final_layer = torch.nn.Linear(dim_hidden, dim_out)

    def forward(self, X):
        X = torch.relu(self.initial_layer(X))

        for lay in self.lays:
            X = torch.relu(lay(X))

        return self.final_layer(X)

model = ModelTorchDeep(2, 20, 2, 10)
for p in model.parameters():
    print(p.shape)
```

### 添加自定义参数

可以在模型中添加自定义参数，这些参数将自动被注册并参与训练。

```python
class ModelSpecial(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=20):
        super().__init__()
        self.layer1 = torch.nn.Linear(dim_in, dim_hidden)
        self.layer2 = torch.nn.Linear(dim_hidden, dim_out)

        self.mult_param1 = torch.nn.Parameter(torch.rand(1, dim_hidden))
        self.mult_param2 = torch.nn.Parameter(torch.rand(1, dim_hidden))

    def forward(self, X):
        X = torch.relu(self.layer1(X)) * self.mult_param1
        X = torch.relu(self.layer2(X)) * self.mult_param2
        return X

model = ModelSpecial(2, 2)
for p in model.parameters():
    print(p.shape, p.requires_grad)
```

```python
model = ModelSpecial(2, 2)
for k, p in model.state_dict().items():
    print(k, p.shape)
```

**注意**：如果仅仅这样定义：
```python
self.mult_param1 = torch.tensor(torch.rand(1, dim_hidden), requires_grad=True)
```
这个参数将不会被包含在 `model.parameters()` 中，因为它没有使用 `torch.nn.Parameter` 进行声明。

### 一次性修改模型

可以一次性修改模型中所有参数的属性。例如，将所有参数设置为不可训练：

```python
model = ModelSpecial(2, 2)
model.requires_grad_(False)

for p in model.parameters():
    print(f"p.requires_grad:{p.requires_grad}, p.device:{p.device}, p.dtype:{p.dtype}")
```

将所有参数移动到 GPU：

```python
if torch.cuda.is_available():
    model = model.to("cuda")
    for p in model.parameters():
        print(f"p.requires_grad:{p.requires_grad}, p.device:{p.device}, p.dtype:{p.dtype}")
```

将所有参数转换为 `float16` 类型（量化）：

```python
model = model.to(dtype=torch.float16)
for p in model.parameters():
    print(f"p.requires_grad:{p.requires_grad}, p.device:{p.device}, p.dtype:{p.dtype}")
```

**注意**：量化通常在模型训练完成后进行，因为 `float16` 的精度不足以支持梯度下降的精确计算。

## 一切皆有可能

假设你需要一个模型来分析以下情况：

- **输入**：
  - `x1`：男性的描述向量，维度为 5，包含以下特征：
    - 体重（kg）
    - 身高（cm）
    - 葡萄酒年消费量（升/年）
    - 啤酒年消费量（升/年）
    - 水年消费量（升/年）
  - `x2`：女性的描述向量，维度同上。

- **输出**：
  - `y1`：拥有 0、1、2、3、4+ 个孩子的概率（4+ 表示 4 个或更多孩子）
  - `y2`：夫妻关系的寿命（共同生活的年数）
  - `y3`：在波尔多地区旅行的概率

### 任务 1：完成模型构造函数

**任务**：在不修改 `forward` 方法的情况下，完成 `ModelCoupleWine` 类的构造函数，使得以下测试程序能够通过。

```python
class ModelCoupleWine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessing_layers1 = torch.nn.Linear(5, 16)  # 男性描述向量
        self.preprocessing_layers2 = torch.nn.Linear(5, 16)  # 女性描述向量

        self.common_layers = torch.nn.ModuleList()
        for i in range(3):
            self.common_layers.append(torch.nn.Linear(32, 32))

        self.separation_layer = torch.nn.Linear(32, 96)  # 分离为 y1, y2, y3

        self.postprocessing_layers = torch.nn.ModuleList()
        for i in range(4):
            self.postprocessing_layers.append(torch.nn.Linear(32, 32))

        self.final_layer1 = torch.nn.Linear(32, 5)  # y1
        self.final_layer2 = torch.nn.Linear(32, 1)  # y2
        self.final_layer3 = torch.nn.Linear(32, 1)  # y3

    def forward(self, x1, x2):
        x1 = torch.relu(self.preprocessing_layers1(x1))
        x2 = torch.relu(self.preprocessing_layers2(x2))

        x12 = torch.cat([x1, x2], dim=1)

        for lay in self.common_layers:
            x12 = torch.relu(lay(x12))

        y123 = self.separation_layer(x12)

        y1 = y123[:, :32]
        y2 = y123[:, 32:64]
        y3 = y123[:, 64:]

        for lay in self.postprocessing_layers:
            y1 = torch.relu(lay(y1))
            y2 = torch.relu(lay(y2))
            y3 = torch.relu(lay(y3))

        y1 = self.final_layer1(y1)
        y2 = self.final_layer2(y2)
        y3 = self.final_layer3(y3)

        return y1, y2, y3

model = ModelCoupleWine()

x1 = torch.tensor([[78., 178, 34, 54, 679]])  # 男性描述向量
x2 = torch.tensor([[56., 154, 56, 12, 1043]])  # 女性描述向量

y1, y2, y3 = model(x1, x2)
assert y1.shape == (1, 5)
assert y2.shape == (1, 1)
assert y3.shape == (1, 1)
```

### 任务 2：添加适当的激活函数

**任务**：复制之前的代码，并修改 `forward` 方法，添加适当的激活函数（参考题目说明）。

```python
class ModelCoupleWine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessing_layers1 = torch.nn.Linear(5, 16)  # 男性描述向量
        self.preprocessing_layers2 = torch.nn.Linear(5, 16)  # 女性描述向量

        self.common_layers = torch.nn.ModuleList()
        for i in range(3):
            self.common_layers.append(torch.nn.Linear(32, 32))

        self.separation_layer = torch.nn.Linear(32, 96)  # 分离为 y1, y2, y3

        self.postprocessing_layers = torch.nn.ModuleList()
        for i in range(4):
            self.postprocessing_layers.append(torch.nn.Linear(32, 32))

        self.final_layer1 = torch.nn.Linear(32, 5)  # y1
        self.final_layer2 = torch.nn.Linear(32, 1)  # y2
        self.final_layer3 = torch.nn.Linear(32, 1)  # y3

    def forward(self, x1, x2):
        x1 = torch.relu(self.preprocessing_layers1(x1))
        x2 = torch.relu(self.preprocessing_layers2(x2))

        x12 = torch.cat([x1, x2], dim=1)

        for lay in self.common_layers:
            x12 = torch.relu(lay(x12))

        y123 = self.separation_layer(x12)

        y1 = y123[:, :32]
        y2 = y123[:, 32:64]
        y3 = y123[:, 64:]

        for lay in self.postprocessing_layers:
            y1 = torch.relu(lay(y1))
            y2 = torch.relu(lay(y2))
            y3 = torch.relu(lay(y3))

        y1 = self.final_layer1(y1)  # 对于 y1，使用 Softmax 激活函数更合适，因为它表示概率
        y2 = self.final_layer2(y2)  # 对于 y2，没有激活函数，因为它是回归任务
        y3 = torch.sigmoid(self.final_layer3(y3))  # 对于 y3，使用 Sigmoid 激活函数，因为它表示概率

        return y1, y2, y3

# 测试代码
model = ModelCoupleWine()

x1 = torch.tensor([[78., 178, 34, 54, 679]])  # 男性描述向量
x2 = torch.tensor([[56., 154, 56, 12, 1043]])  # 女性描述向量

y1, y2, y3 = model(x1, x2)
print(f"y1.shape:{y1.shape}, y2.shape:{y2.shape}, y3.shape:{y3.shape}")
```

**输出**：

```
y1.shape: torch.Size([1, 5]), y2.shape: torch.Size([1, 1]), y3.shape: torch.Size([1, 1])
```

**解释**：
- 对于 `y1`（孩子数量的概率分布），可以使用 `Softmax` 激活函数以确保输出为概率分布。但在本例中，为简单起见，未显式添加 `Softmax`。
- 对于 `y2`（夫妻关系的寿命），是一个回归任务，因此不需要激活函数。
- 对于 `y3`（旅行概率），使用 `Sigmoid` 激活函数将输出限制在 `[0, 1]` 之间，表示概率。

## 总结

通过以上步骤，我们了解了如何使用 PyTorch 实现一个简单的神经网络，包括参数初始化、层的定义、激活函数的选择、模型的训练以及参数的管理。通过继承 `torch.nn.Module`，可以更方便地管理模型参数，并利用 PyTorch 提供的丰富功能来构建和训练复杂的神经网络模型。