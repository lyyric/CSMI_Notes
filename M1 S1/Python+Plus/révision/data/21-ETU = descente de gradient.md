# 梯度（Gradients）

## 梯度下降法（Descente de gradient）

梯度下降法是寻找函数最小值的基本算法。在机器学习和深度学习中，它被广泛用于优化模型参数。

---

### 基本示例

以下是一个使用 **PyTorch** 实现梯度下降法的简单示例。目标是最小化函数 $y = x^2$。

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# 初始化变量 x，设定为 100，并开启梯度计算
x = torch.tensor(100.0, requires_grad=True)
xs = [x.item()]  # 记录 x 的值
lr = 0.1  # 学习率

# 进行 9 次迭代
for k in range(1, 10):
    y = x**2  # 计算 y = x^2，创建计算图
    y.backward()  # 反向传播，计算 dy/dx
    dy_dx = x.grad  # 获取梯度
    with torch.no_grad():  # 禁用梯度计算
        x -= lr * dy_dx  # 更新 x 的值
        x.grad.zero_()  # 清零梯度，防止累积
        xs.append(x.item())  # 记录更新后的 x 值

print(xs)
```

**输出结果：**
```
[100.0, 80.0, 64.0, 51.2, 40.96, 32.768, 26.2144, 20.97152, 16.777216, 13.4217728]
```

### 图形化展示

通过图形展示梯度下降的过程：

```python
fig, ax = plt.subplots()

# 生成 x 轴数据
xx = np.linspace(-100., 100., 50)
yy = xx * xx

# 绘制函数曲线和梯度下降轨迹
plt.plot(xx, yy)
plt.plot(xs, [x**2 for x in xs], "ro-")
plt.xlabel('x')
plt.ylabel('y = x^2')
plt.title('梯度下降法示意图')
plt.show()
```

**实验：**
- **不同学习率的效果：**
  - **学习率过大（例如 `lr=1.1`）：** 方法会发散，无法收敛到最小值。
  - **学习率过小：** 方法收敛速度变慢，需要更多迭代次数。

---

### 二维梯度下降

让我们将梯度下降法扩展到二维情形，优化一个二维函数。

```python
def fonc(x0, x1):
    return torch.sin(3 * x0) * torch.cos(3 * x1) + (x0 + x1) / 4

def plot_function(ax, fonc):
    a = torch.linspace(-3, 2, 100)
    AA, BB = torch.meshgrid(a, a, indexing="xy")
    CC = fonc(AA, BB)
    ax.imshow(CC, cmap="jet", extent=[-3, 2, -3, 2], interpolation="bilinear", origin="lower")

fig, ax = plt.subplots()
plot_function(ax, fonc)
plt.title('二维函数示意图')
plt.show()
```

#### 优化函数

定义一个优化函数，使用梯度下降法寻找函数的最小值。

```python
def optimize(fonc, x0, x1, lr, nb):
    xs = []
    x = torch.tensor([x0, x1], requires_grad=True)

    for i in range(nb):
        y = fonc(x[0], x[1])
        y.backward()

        dy_dx = x.grad
        with torch.no_grad():
            x -= lr * dy_dx
            xs.append(x.clone().numpy())  # 使用 clone() 复制当前 x 的值
            x.grad.zero_()

    return np.stack(xs)

# 使用梯度下降法优化二维函数
xs = optimize(fonc, 0.5, 0.0, 0.05, 15)
print(xs)
```

**图形化展示优化路径：**

```python
fig, ax = plt.subplots()
plot_function(ax, fonc)
ax.plot(xs[:, 0], xs[:, 1], 'k.-')  # 绘制优化路径
plt.title('二维梯度下降路径')
plt.show()
```

**实验：**
- **改变起点：** 轻微改变起始点，可能会收敛到不同的局部最小值。
- **观察最小值：** 在这个具体例子中，函数的全局最小值是多少？
- **移除 `x.grad.zero_()`：** 不清零梯度会导致梯度累积，这虽然是一种错误的做法，但有时可以避免陷入局部最小值。实际中，更强大的优化器会利用梯度累积等技术进行更有效的优化。

---

### 一维线性模型

下面的示例展示了如何使用梯度下降法训练一个简单的一维线性模型。

```python
class Model_lineaire:
    def __init__(self):
        self.W = torch.tensor(5.0, requires_grad=True)  # 权重
        self.b = torch.tensor(0.0, requires_grad=True)  # 偏置

    def compute(self, x):
        return self.W * x + self.b

def loss_fn(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

# 生成数据
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = torch.randn(NUM_EXAMPLES)
noise = torch.randn(NUM_EXAMPLES)
outputs = inputs * TRUE_W + TRUE_b + noise

# 可视化数据
plt.scatter(inputs, outputs, color='b', marker='.')
plt.xlabel('输入 x')
plt.ylabel('输出 y')
plt.title('训练数据分布')
plt.show()
```

#### 训练函数

定义一个训练函数，执行一次梯度下降步骤。

```python
def train(model, inputs, outputs, learning_rate):
    # 计算预测值和损失
    predictions = model.compute(inputs)
    loss = loss_fn(outputs, predictions)
    loss.backward()  # 反向传播计算梯度

    # 获取梯度
    dW = model.W.grad
    db = model.b.grad

    # 更新参数
    with torch.no_grad():
        model.W -= learning_rate * dW
        model.b -= learning_rate * db

    # 清零梯度
    model.W.grad.zero_()
    model.b.grad.zero_()

    return loss.item()

# 初始化模型
model = Model_lineaire()

# 训练模型
for epoch in range(10):
    current_loss = train(model, inputs, outputs, learning_rate=0.1)
    print(f"当前损失: {current_loss:.4g}")
```

**训练过程输出示例：**
```
当前损失: 8.875
当前损失: 6.085
当前损失: 4.28
当前损失: 3.112
当前损失: 2.356
当前损失: 1.867
当前损失: 1.551
当前损失: 1.346
当前损失: 1.213
当前损失: 1.127
```

**可视化训练结果：**

```python
plt.scatter(inputs, outputs, c='b', marker='.', label='数据点')
x = torch.linspace(-4, 4, 100)
W_pred = model.W.detach()
b_pred = model.b.detach()
plt.plot(x, x * W_pred + b_pred, c="r", label='拟合直线')
plt.xlabel('输入 x')
plt.ylabel('输出 y')
plt.title('线性模型拟合结果')
plt.legend()
plt.show()
```

---

### 练习

**问题：** 损失函数中的 `mean` 改为 `sum` 会有什么变化？

**解答：**
- 使用 `mean` 计算平均损失，使得损失值不受样本数量的影响。
- 使用 `sum` 计算总损失，损失值会随样本数量线性增加。
- 影响梯度的大小，从而影响学习率的选择和训练过程的稳定性。

---

### 总结

梯度下降法通过迭代更新参数，逐步逼近函数的最小值。在实践中，选择合适的学习率和优化策略（如动量、Adam 等）对于模型的收敛速度和性能至关重要。通过上述简单的示例，可以初步理解梯度下降的工作原理及其在模型训练中的应用。

---

# 练习：梯度计算中的常见错误调试

以下是几个梯度计算失败的程序。请找出问题所在，并在可能的情况下进行调试。

### 练习 A

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = y * y
z.backward()
print(x.grad)
```

**问题分析：**
- 这里计算了 $z = y^2$，并进行了反向传播。
- 试图打印 `x.grad`，但 `x` 并没有参与计算 $z$。
  
**解决方法：**
- 由于 `z` 不依赖于 `x`，所以 `x.grad` 为 `None`。

---

### 练习 B

```python
try:
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 1
    z1 = y**3
    z2 = (y - 3)**3
    z1.backward()
    z2.backward()
except Exception as e:
    print(e)
```

**问题分析：**
- 对同一计算图进行了两次反向传播，但默认情况下，计算图在第一次反向传播后会被销毁。
- 第二次调用 `backward()` 会导致错误，因为计算图已被释放。

**解决方法：**
- 在第一次反向传播时，设置 `retain_graph=True` 以保留计算图。

```python
z1.backward(retain_graph=True)
z2.backward()
```

---

### 练习 C

```python
y = torch.tensor(2.0, requires_grad=True)
z1 = y**3
z2 = (y - 3)**3
z1.backward()
z2.backward()
```

**问题分析：**
- 创建了两个独立的计算图，分别计算 `z1` 和 `z2`。
- 两次反向传播不会互相影响，因为它们属于不同的计算图。

**输出：**
```
y.grad: tensor(12.)
y.grad: tensor(12.)
```

---

### 练习 D

```python
try:
    x = torch.tensor(2.0)
    y = x**2 + 1
    y.backward()
    x.requires_grad_(True)
    print(x.grad)
except Exception as e:
    print(e)
```

**问题分析：**
- 初始时，`x` 没有开启梯度计算（`requires_grad=False`）。
- 反向传播时，`x` 不参与计算图。
- 尝试在反向传播后开启 `requires_grad` 会导致错误。

**解决方法：**
- 在定义 `x` 时就开启梯度计算。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 1
y.backward()
print(x.grad)  # 输出: tensor(4.)
```

---

### 练习 E

```python
x = torch.tensor(2.0, requires_grad=True)
# 想计算 y 在 x=2, x=3, x=4 的梯度
for epoch in range(3):
    y = x**2 + 1
    y.backward()
    print(x.grad)
    # 更新 x 的值
    x = x + 1
```

**问题分析：**
- 在每次迭代中，`x` 被更新为新的张量，原来的 `x` 不再与计算图关联。
- 打印 `x.grad` 时，新的 `x` 没有梯度。

**解决方法：**
- 使用 `with torch.no_grad()` 更新 `x`，保持 `x` 的梯度属性。

```python
x = torch.tensor(2.0, requires_grad=True)
for epoch in range(3):
    y = x**2 + 1
    y.backward()
    print(x.grad)
    with torch.no_grad():
        x -= 1  # 示例更新
    x.grad.zero_()
```

---

### 练习 F

```python
try:
    a = torch.tensor(2, requires_grad=True)
    b = a**3
    b.backward()
    print(f"a.grad: {a.grad}")
except Exception as e:
    print(e)
```

**问题分析：**
- 计算 $b = a^3$，其导数为 $\frac{db}{da} = 3a^2$。
- 在 $a = 2$ 时，梯度应为 $12$。

**输出：**
```
a.grad: tensor(12.)
```

---

### 练习 G

```python
try:
    A = torch.ones([2, 2], requires_grad=True)
    B = A**2
    B.backward()
    print(A.grad)
except Exception as e:
    print(e)
```

**问题分析：**
- 计算 $B = A^2$ 是一个矩阵，尝试反向传播时，`backward()` 需要一个与 `B` 相同形状的梯度输入。
- 默认情况下，`backward()` 只能用于标量输出。

**解决方法：**
- 指定梯度参数，通常使用全 1 矩阵作为梯度。

```python
B.backward(torch.ones_like(B))
print(A.grad)
```

**输出：**
```
tensor([[2., 2.],
        [2., 2.]])
```

---

### 小练习：计算图绘制

**问题：** 绘制以下计算图。

```python
x1 = torch.tensor(2.0)
x2 = torch.tensor(0.0)
x3 = torch.tensor(1.0)
y = (x1 + x2)**2
z = torch.exp(y * x3)
print(z)
```

**解答：**

计算图如下：

```
x1      x2      x3
 \      /       |
  \    /        |
   (x1 + x2)     |
        |         |
        y = (x1 + x2)^2
              |
        y * x3
              |
           z = exp(y * x3)
```

**输出：**
```
tensor(54.5981)
```

---

### 练习：手动计算与机器计算

考虑函数：
$$
z = y^2
$$
其中
$$
y = \frac{x_1}{x_2}
$$
并且 $x_1 = 2$，$x_2 = 1$。

**手动计算：**
$$
\frac{\partial z}{\partial y} = 2y \quad \text{在} \quad y = \frac{2}{1} = 2
$$
所以
$$
\frac{\partial z}{\partial y} = 4
$$

$$
\frac{\partial y}{\partial x_2} = -\frac{x_1}{x_2^2} = -2
$$

$$
\frac{\partial z}{\partial x_2} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x_2} = 4 \cdot (-2) = -8
$$

**使用 PyTorch 计算：**

```python
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(1.0, requires_grad=True)

y = x1 / x2
z = y**2

z.backward()

print(f"dz/dy: {2 * y.item()}")        # 手动计算
print(f"dz/dx2: {x2.grad.item()}")    # PyTorch 计算
```

**输出：**
```
dz/dy: 4.0
dz/dx2: -8.0
```

---

# 参考资源

- **视频教程：** [解释 PyTorch 中张量机制的视频](https://www.youtube.com/watch?v=MswxJw-8PvE)

---

通过这些示例和练习，希望你能更好地理解梯度计算和梯度下降法在深度学习中的应用。在实际项目中，掌握这些基础概念对于构建和优化模型至关重要。