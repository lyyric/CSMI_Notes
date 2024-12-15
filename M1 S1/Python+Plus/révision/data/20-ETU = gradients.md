# 梯度（Gradients）

梯度是深度学习中非常重要的概念，它帮助我们优化模型的参数。下面我们将通过 **TensorFlow** 和 **PyTorch** 两种库来讲解梯度的基本概念和计算方法。

---

## 反向传播与梯度计算

### 什么是反向传播？

假设我们要计算函数 $y=3x \cdot (x^2 + 4)$ 的导数 $\frac{dy}{dx}=3(x^2+4) + 3x \cdot 2x$，并求出在 $x=3$ 的值。

反向传播分为两个阶段：

1. **前向传播（Forward Pass）**  
   按顺序计算函数的值，同时记录每一步的计算过程。比如：  
   - 计算 $x^2$，存储结果；
   - 计算 $x^2 + 4$，存储结果；
   - 计算 $3x$；
   - 最后计算 $y$。

2. **反向传播（Backward Pass）**  
   从最后一步往回计算每一步的梯度，依次应用链式法则（Chain Rule）。

在 **TensorFlow** 中，我们使用 `tf.GradientTape` 对前向传播进行记录，随后通过 `tape.gradient()` 实现反向传播。

示例代码：

```python
x = tf.Variable(3.0)

# 前向传播
with tf.GradientTape() as tape:
    y = 3 * x * (x**2 + 4)

# 反向传播
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())  # 输出结果为 51
```

---

### 链式法则（Chain Rule）

链式法则描述了复合函数的导数计算方式。例如，对于函数 $h(g(f(x)))$：

- 计算图：
$$
x \xrightarrow{f} y \xrightarrow{g} z \xrightarrow{h} t
$$

- 链式法则公式：
$$
\frac{\partial t}{\partial x} = \frac{\partial t}{\partial z} \cdot \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

假设：

- $f(x) = \sin(x)$，则 $f'(x) = \cos(x)$；
- $g(x) = 4x^2$，则 $g'(x) = 8x$；
- $h(x) = \tanh(x)$，则 $h'(x) = 1 - \tanh^2(x)$。

我们希望计算在 $x=7$ 时的 $\frac{\partial (h \circ g \circ f)}{\partial x}$。

代码实现：

```python
x = tf.Variable(7.0)

# 前向传播
with tf.GradientTape() as tape:
    y = tf.sin(x)
    z = 4 * y**2
    t = tf.tanh(z)

# 反向传播
print(tape.gradient(t, x).numpy())
```

---

## 累积规则（Accumulation Rule）

当变量 $x$ 多次出现在计算图中时（例如作为多个函数的输入），它对目标函数的影响需要累加。

例如：
$$
z = g[f_1(x), f_2(x), ...] = g[y_1, y_2, ...]
$$

计算图：
$$
x \xrightarrow{f} \begin{bmatrix} y_1 \\ y_2 \\ \vdots \end{bmatrix} \xrightarrow{g} z
$$

梯度累积公式：
$$
\frac{\partial z}{\partial x} = \sum_i \frac{\partial z}{\partial y_i} \cdot \frac{\partial y_i}{\partial x}
$$

**验证累积规则的代码：**

```python
x = tf.Variable(7.0)

with tf.GradientTape(persistent=True) as tape:
    y1 = x**2
    y2 = tf.cos(x)
    y3 = tf.atan(x)
    z = y1 * y2 / y3

# 梯度计算
dz_dx = tape.gradient(z, x)
print(dz_dx.numpy())
```

---

## PyTorch 中的梯度计算

### 计算图（Computation Graph）

PyTorch 的计算图会动态构建，并支持自动反向传播。

示例 1：
$$
z = (x^2 \cdot \cos(x))^2
$$

代码实现：

```python
x = torch.tensor(torch.pi, requires_grad=True)
y = x**2 * torch.cos(x)
z = y**2

# 反向传播
z.backward()
print(x.grad)  # 输出梯度值
```

### 非叶子节点的梯度

默认情况下，PyTorch 只会记录叶子节点（leaf nodes）的梯度。如果需要计算非叶子节点的梯度，需要显式调用 `retain_grad()`。

```python
x = torch.tensor(1.0, requires_grad=True)
a = torch.tensor(2.0, requires_grad=True)
y = a * x
y.retain_grad()  # 保留非叶子节点的梯度
z = y**2

# 反向传播
z.backward()
print(f"x.grad: {x.grad}, a.grad: {a.grad}, y.grad: {y.grad}")
```

---

### 多次反向传播

PyTorch 中，每次调用 `.backward()` 都会销毁部分计算图数据。为了避免数据销毁，可以使用 `retain_graph=True`。

```python
x = torch.tensor(1.0, requires_grad=True)
y = x**2
y.backward(retain_graph=True)  # 保留计算图
z = y**3
z.backward()
print(x.grad)  # 梯度会累加
```

如需避免梯度累加，可以在每次计算后调用 `x.grad.zero_()`：

```python
x.grad.zero_()  # 清零梯度
```

---

## 禁用梯度计算

某些情况下，我们不需要计算梯度（如推理阶段），可以通过以下方式禁用梯度计算：

1. 使用 `torch.no_grad()`：
   ```python
   with torch.no_grad():
       result = x**2 + 3*x + 2
   ```

2. 使用 `x.requires_grad_(False)`：将变量的 `requires_grad` 属性设为 `False`。

---

## GPU 内存管理

PyTorch 提供工具监控 GPU 内存的使用情况。例如：

```python
torch.cuda.reset_peak_memory_stats()  # 重置内存统计
torch.cuda.max_memory_allocated()    # 获取当前最大内存分配量
```

测试不同数据类型和大小对 GPU 内存的影响：

```python
size = 1024
A = torch.ones(size, dtype=torch.float32, device="cuda")
print(torch.cuda.max_memory_allocated())  # 输出内存占用
```

张量大小是 2 的幂时，内存管理效率更高。

---

## 总结

梯度计算的核心在于利用 **链式法则** 和 **计算图**。现代框架（如 TensorFlow 和 PyTorch）为我们自动处理了这些细节，只需关注实现模型和优化目标即可。在开发中：

1. **深度学习：** 使用 `requires_grad=True` 定义需要优化的参数，使用 `.backward()` 或 `.grad()` 进行梯度计算。
2. **推理阶段：** 禁用梯度计算以提高性能。
3. **GPU 使用：** 注意张量的类型和大小，以优化内存管理。