# 张量（Tensors）

张量是一个多维数组，具有特定的数据类型（`dtype`）和形状（`shape`）。在深度学习和科学计算中，张量是非常重要的工具。以下内容我们将对 **numpy**、**tensorflow** 和 **pytorch** 这三种主要的张量操作库进行比较，并深入分析它们的特点和用法。

---

## 张量库的比较

以下是这三种库的基本特点对比表：

| 特性               | numpy         | tensorflow   | pytorch      |
|--------------------|---------------|--------------|--------------|
| 默认精度           | 64 位         | 32 位        | 32float/64int 位 |
| 默认运行设备       | CPU           | GPU          | CPU          |
| 是否支持 GPU       | 否            | 是           | 是           |
| 张量是否可变       | 是            | 否           | 是           |
| 是否支持自动类型转换 | 是            | 否           | 是           |
| 是否支持原地操作    | 部分支持       | 不支持       | 自由选择      |
| 是否支持自动微分    | 否            | 是           | 是           |

**注意：** 
- Numpy 不能用来构建神经网络，但在数据预处理方面非常实用。
- TensorFlow 和 PyTorch 更适合深度学习。

---

### 张量的基本定义与操作

#### 张量的创建

我们可以用不同的库来创建张量。以下代码示例展示了如何用 numpy、tensorflow 和 pytorch 创建张量：

```python
data = [[1., 2], [3, 4], [5, 6]]
X_torch = torch.tensor(data)
X_tf = tf.constant(data)
X_np = np.array(data)

for X in [X_torch, X_tf, X_np]:
    print(X.shape, X.dtype)
```

**问题 1：** 如果将数据中的小数点去掉，比如 `data = [[1, 2], [3, 4], [5, 6]]`，结果会如何？

---

#### 张量的可变性测试

以下代码验证了张量在不同库中的可变性：

```python
def modify(X):
    X[0, 0] = 7

for title, X in [("torch", X_torch), ("tf", X_tf), ("np", X_np)]:
    print(title)
    try:
        modify(X)
        print(X)
    except Exception as e:
        print(e)
```

**观察点：**
- 如果修改张量中的值，结果是否会改变？
- 哪些库支持原地操作，哪些不支持？

---

### 张量之间的转换

有时，我们可能需要在不同库之间切换张量操作，比如用 numpy 的函数处理 pytorch 张量。但这样做可能会导致错误。以下代码展示了不同库函数对张量的兼容性：

```python
for title_tensor, X in [("torch", X_torch), ("tf", X_tf), ("np", X_np)]:
    print("title_tensor:", title_tensor)
    for title_func, fn in [("torch", torch.sin), ("tf", tf.sin), ("np", np.sin)]:
        print("\t title_func:", title_func)
        try:
            res = fn(X)
            print("\t\t", X.dtype)
        except Exception as e:
            print("\t\t", e)
```

---

### 自动类型转换

当不同类型的张量进行运算时，有些库会自动进行类型转换，而有些库会报错。以下示例说明了这种差异：

```python
np.zeros([3, 3], dtype=np.float32) + np.zeros([3, 3], dtype=np.float64) + np.zeros([3, 3], dtype=np.int32)

try:
    tf.zeros([3, 3], dtype=tf.float32) + tf.zeros([3, 3], dtype=np.float64)
except Exception as e:
    print(e)

torch.zeros([3, 3], dtype=torch.float32) + torch.zeros([3, 3], dtype=torch.float64)
```

---

### 张量的预处理与后处理

一个常见的开发流程是：
1. 使用 **numpy** 进行数据的预处理（因为其简单高效）。
2. 转换到 **tensorflow** 或 **pytorch**，执行复杂的张量操作。
3. 将结果转换回 numpy，进行后处理或可视化。

示例代码：

```python
# 预处理：用 numpy 生成并处理数据
X_np = np.random.normal(size=[1000])
X_np[X_np < 0] = 0
plt.hist(X_np, edgecolor='k')

# 处理：用 tensorflow 做复杂运算
X_tf = tf.constant(X_np, dtype=tf.float32)
X_tf = tf.sin(X_tf ** 3)

# 后处理：将结果转换回 numpy
plt.hist(X_tf.numpy(), edgecolor="k")
```

---

## GPU 加速

**tensorflow 和 pytorch** 支持 GPU 加速，而 numpy 不支持。

### TensorFlow 的 GPU 使用

以下代码检查是否启用了 GPU：

```python
tf.config.list_logical_devices()
```

创建张量后，可以通过 `.device` 查看它是否被分配到 GPU：

```python
X_tf = tf.ones([3, 3])
print(X_tf.device)
```

我们还可以显式地将张量分配到 CPU 或 GPU：

```python
with tf.device("CPU:0"):  # 或 "GPU:0"
    result = X_tf @ X_tf
```

---

### PyTorch 的 GPU 使用

PyTorch 通过 `.to` 方法将张量从 CPU 移动到 GPU：

```python
X_torch = torch.ones([3, 3])
X_torch = X_torch.to("cuda")
print(X_torch.device)
```

也可以用以下方法将张量移回 CPU：
```python
X_torch = X_torch.cpu()
```

为了简化操作，可以设定默认的设备和数据类型：
```python
torch.set_default_device("cuda")
torch.set_default_dtype(torch.float32)
```

---

## 广播机制（Broadcasting）

广播允许不同形状的张量进行运算，但需要满足以下规则：

**广播规则：**
对于每个维度 `i`，需满足以下条件之一：
1. `A.shape[i] == B.shape[i]`
2. `A.shape[i] == 1`
3. `B.shape[i] == 1`

示例：

```python
A = tf.constant([1, 10, 100])  # shape (3,)
B = tf.constant([2, 5])        # shape (2,)
A = A[None, :]  # shape (1, 3)
B = B[:, None]  # shape (2, 1)
C = A * B       # shape (2, 3)
print(C)
```

---

## 张量拼接

在实际应用中，我们经常需要将多个小张量拼接成一个大张量。以下代码展示了三种库的用法：

```python
for lib in [np, tf, torch]:
    A = []
    for i in range(5):
        A.append(i * lib.ones([2]))
    A = lib.stack(A)  # 默认沿第 0 维拼接
    print(A)
```

如果需要更灵活的拼接方式，可以使用以下函数：
- **tensorflow:** `tf.concat`
- **numpy:** `np.concatenate`
- **pytorch:** `torch.cat`

示例代码：

```python
A = []
for i in range(5):
    A.append(i * np.ones([1, 2]))
A = np.concatenate(A, axis=0)
print(A)
```

---

## 总结

在实际开发中：
1. 如果是常规科学计算，推荐使用 numpy。
2. 如果是深度学习项目，推荐使用 pytorch 或 tensorflow。
3. 确保在不同库之间切换时，清晰地管理张量类型和设备，以避免不必要的错误。