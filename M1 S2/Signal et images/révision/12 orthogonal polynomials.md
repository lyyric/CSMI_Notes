```python
%reset -f

import scipy.special as spe
import matplotlib.pyplot as plt
import numpy as np
```

## 勒让德多项式

除了正弦-余弦基和指数基之外，还有其他关于通常内积正交的基。特别是，勒让德多项式常用于数值分析中的数值积分：其思想是用一个多项式来逼近被积函数，然后精确计算积分（因为多项式的原函数很容易求出）。

但我们将看到，获得勒让德基的完美离散化实际上更为复杂。

**定义：** 勒让德多项式定义在区间 $[-1, +1]$ 上，通过递归公式定义：
$$
\begin{align*}
n P_n(x) &= (2n - 1) x P_{n-1}(x) - (n - 1) P_{n-2}(x) \\
P_0(x) &= 1 \\
P_1(x) &= x
\end{align*}
$$

它们对于通常的内积是正交的：
$$
\mathtt{dot} (f,g) = \int_{-1}^{+1} f(x)g(x) \, dx
$$

### → ♡

```python
x = np.linspace(-1, 1, 100)
basis_cont = [np.ones_like(x), x]

for n in range(2, 10):
    P_n_minus_1 = basis_cont[n-1]
    P_n_minus_2 = basis_cont[n-2]
    P_n = ((2*n - 1)*x*P_n_minus_1 - (n - 1)*P_n_minus_2) / n
    basis_cont.append(P_n)

for i in range(len(basis_cont)):
    plt.plot(x, basis_cont[i], label=f'P_{i}(x)')
plt.legend()
plt.title('连续勒让德多项式基')
plt.show()
```

我们将这些数据转换为一个 NumPy 矩阵：

### → ♡

```python
basis_matrix = np.array(basis_cont)
print(basis_matrix.shape)
```
```
(10, 100)
```

### 验证正交性

### → ♡♡

```python
np.set_printoptions(precision=2, suppress=True, linewidth=1000)
# 计算基矩阵的内积
sca = basis_matrix @ basis_matrix.T * (2 / (basis_matrix.shape[1] - 1))
print(sca)
```
```
[[100.  ,   0.  ,   1.01,   0.  ,   1.03,   0.  ,   1.07,   0.  ,   1.12,   0.  ],
 [  0.  ,  34.01,   0.  ,   1.02,   0.  ,   1.05,   0.  ,   1.1 ,   0.  ,   1.15],
 [  1.01,   0.  ,  20.82,   0.  ,   1.04,   0.  ,   1.08,   0.  ,   1.13,   0.  ],
 [  0.  ,   1.02,   0.  ,  15.18,   0.  ,   1.07,   0.  ,   1.11,   0.  ,   1.17],
 [  1.03,   0.  ,   1.04,   0.  ,  12.07,   0.  ,   1.1 ,   0.  ,   1.15,   0.  ],
 [  0.  ,   1.05,   0.  ,   1.07,   0.  ,  10.1 ,   0.  ,   1.14,   0.  ,   1.2 ],
 [  1.07,   0.  ,   1.08,   0.  ,   1.1 ,   0.  ,   8.76,   0.  ,   1.19,   0.  ],
 [  0.  ,   1.1 ,   0.  ,   1.11,   0.  ,   1.14,   0.  ,   7.79,   0.  ,   1.24],
 [  1.12,   0.  ,   1.13,   0.  ,   1.15,   0.  ,   1.19,   0.  ,   7.06,   0.  ],
 [  0.  ,   1.15,   0.  ,   1.17,   0.  ,   1.2 ,   0.  ,   1.24,   0.  ,   6.51]]
```

可以看到，矩阵是对角占优的，但非对角项并不可以忽略。

这种情况让我们联想到在区间 $[0,T]$ 上的余弦-正弦基，当我们使用不正确的离散化点时，比如使用：
```
np.linspace(0, T, 100, endpoint=True)
```
而正确的正弦-余弦离散化点应该是：
```
# 正确的离散化点
x_correct = np.pi * np.linspace(0, 1, 100, endpoint=False)
```

### → ♡

接下来，正确的勒让德基离散化方法。

## 勒让德基的正确离散化

我们现在要创建一个离散基，该基源自这些多项式的离散化。我们希望这个基对于给定的离散内积是完全正交的。

要创建这样的基，必须使用特殊的离散化点：高斯点。这些点是勒让德多项式 $P_{n+1}$ 的根。记这些点为 $(x_0, ..., x_n)$。

考虑向量族 $V^0, V^1, \dots$，定义为：
$$
V^k_i = P_k(x_i)
$$
这个向量族对于加权离散内积是正交的：
$$
\mathtt{dot_{dis}} (u, v) = \sum_{i} w_i u_i v_i
$$
其中 $w_i$ 是给定的权重。

我们将使用 `scipy` 来找到高斯点 $(x_i)$ 及其权重 $(w_i)$。

```python
n_point = 50
""" 高斯点及其相关权重 """
x, w = spe.roots_legendre(n_point)  # 注意：roots_legendre 返回的是点和权重
print("离散化点:", x)

plt.plot(x, w, 'o')
plt.title('高斯点及其权重')
plt.xlabel('x')
plt.ylabel('权重 w')
plt.show()

print(len(x), len(w))
```

```
离散化点: [ ... ]  # 输出的具体数值
50 50
```

### 创建离散基

我们将构建离散基，通过逐步填充矩阵（不同于之前构建 `basis_cont` 的方法）。

### → ♡

```python
n_poly = 19
basis = np.zeros([n_poly, n_point])
basis[0, :] = 1.
basis[1, :] = x
for n in range(2, n_poly):
    basis[n, :] = ((2*n - 1)*x*basis[n-1, :] - (n - 1)*basis[n-2, :]) / n

print(basis.shape)
```

```
(19, 50)
```

#### ♡♡♡♡

**习题：** 试着不使用循环，利用 NumPy 的向量化操作来计算 `basis` 矩阵。

### 绘制基函数

```python
for i in range(n_poly):
    plt.plot(x, basis[i, :], label=f'V^{i}(x)')
plt.legend()
plt.title('离散勒让德基函数')
plt.show()
```

### 验证加权正交性

### → ♡♡

```python
""" 权重被放置在矩阵的对角线上 """
W = np.diag(w)
sca = basis @ W @ basis.T
""" 勒让德族是正交的，但不是正交归一的 """
print(sca)
```
```
[[ 2.    0.   -0.   ... 0. ]
 [ 0.    0.67 -0.   ... 0. ]
 [ ... ]
 [0.    0.    0.05]]
```

### 正规化基族以获得正交归一基

### → ♡♡

```python
""" 我们对基族进行正规化 """
basisNor = np.zeros_like(basis)
for i in range(n_poly):
    norm = np.sqrt(basis[i, :] @ W @ basis[i, :])
    basisNor[i, :] = basis[i, :] / norm

print(basisNor @ W @ basisNor.T)
```
```
[[ 1. -0. -0. ... 0.]
 [ -0. 1. -0. ... 0.]
 [ ... ]
 [0. 0. ... 1.]]
```

## 函数近似

我们将考虑函数 $t \mapsto \sqrt{|t|}$，并在高斯点上进行离散化。我们将用刚刚创建的基来逼近这个向量。

### → ♡♡

```python
""" 要逼近的函数（离散化后） """
y = np.sqrt(np.abs(x))

""" 其坐标： """
y_coor = basisNor @ (W * y)

""" 逼近是基的线性组合： """
y_approx = basisNor.T @ y_coor

plt.plot(x, y, 'b.-', label="y")
plt.plot(x, y_approx, 'r.-', label="y_approx")
plt.legend()
plt.title('函数逼近')
plt.show()
```

注意，函数 $t \mapsto \sqrt{|t|}$ 在 $0$ 处图像是平的，这是由于离散化的原因。

### → ♡

**问题：** 在什么条件下，逼近是完美的？（即图中的蓝点和橙点重合）

**回答：** 当所使用的基能够完全表示目标函数时，即目标函数可以被基的有限线性组合精确表示。这通常发生在目标函数本身就是基的一个成员，或者基的维度足够高以完全表示目标函数的多项式展开。

## 总结

通过使用高斯点进行离散化，我们能够构建一个正交归一的离散勒让德基，从而有效地逼近目标函数。相比于简单的均匀离散化，高斯点的选择显著提高了逼近的精度和基的正交性。

**习题：**

1. 使用不同数量的高斯点（例如，n_point=10, 20, 50）重复上述过程，观察逼近的效果如何变化。
2. 尝试使用其他函数（如 $f(t) = t^2$ 或 $f(t) = \sin(\pi t)$）进行逼近，分析结果。
3. 实现一个不使用循环的正规化过程，以提高代码的效率。

```

# 笔记重写（中文）

## 勒让德多项式及其离散化

在数值分析中，除了常见的正弦-余弦基和指数基之外，还有其他关于通常内积正交的基。其中，勒让德多项式是一类重要的正交多项式，常用于数值积分。其基本思想是将被积函数用多项式逼近，然后通过计算多项式的积分来近似原函数的积分。

### 勒让德多项式的定义

勒让德多项式定义在区间 $[-1, +1]$ 上，并通过以下递归公式定义：

$$
\begin{align*}
n P_n(x) &= (2n - 1) x P_{n-1}(x) - (n - 1) P_{n-2}(x) \\
P_0(x) &= 1 \\
P_1(x) &= x
\end{align*}
$$

这些多项式对于通常的内积是正交的，内积定义为：

$$
\mathtt{dot} (f,g) = \int_{-1}^{+1} f(x)g(x) \, dx
$$

### 生成连续勒让德多项式基

我们首先在连续区间上生成勒让德多项式，并绘制它们的图形：

```python
x = np.linspace(-1, 1, 100)
basis_cont = [np.ones_like(x), x]

for n in range(2, 10):
    P_n_minus_1 = basis_cont[n-1]
    P_n_minus_2 = basis_cont[n-2]
    P_n = ((2*n - 1)*x*P_n_minus_1 - (n - 1)*P_n_minus_2) / n
    basis_cont.append(P_n)

for i in range(len(basis_cont)):
    plt.plot(x, basis_cont[i], label=f'P_{i}(x)')
plt.legend()
plt.title('连续勒让德多项式基')
plt.show()
```

### 验证正交性

将基函数转换为矩阵形式，并验证它们的正交性：

```python
basis_matrix = np.array(basis_cont)
print(basis_matrix.shape)
```
```
(10, 100)
```

计算内积矩阵：

```python
np.set_printoptions(precision=2, suppress=True, linewidth=1000)
# 计算基矩阵的内积
sca = basis_matrix @ basis_matrix.T * (2 / (basis_matrix.shape[1] - 1))
print(sca)
```

输出显示矩阵是对角占优的，但非对角项并不可以忽略，这表明基函数在离散化后并不完全正交。

### 高斯点离散化

为了获得完全正交的离散基，我们需要使用特殊的离散化点——高斯点。这些点是勒让德多项式 $P_{n+1}$ 的根。通过 `scipy` 库可以获得这些点及其权重：

```python
n_point = 50
""" 高斯点及其相关权重 """
x, w = spe.roots_legendre(n_point)  # 注意：roots_legendre 返回的是点和权重
print("离散化点:", x)

plt.plot(x, w, 'o')
plt.title('高斯点及其权重')
plt.xlabel('x')
plt.ylabel('权重 w')
plt.show()

print(len(x), len(w))
```
```
离散化点: [ ... ]  # 输出的具体数值
50 50
```

### 构建离散勒让德基

使用高斯点和权重构建正交归一的离散勒让德基：

```python
n_poly = 19
basis = np.zeros([n_poly, n_point])
basis[0, :] = 1.
basis[1, :] = x
for n in range(2, n_poly):
    basis[n, :] = ((2*n - 1)*x*basis[n-1, :] - (n - 1)*basis[n-2, :]) / n

print(basis.shape)
```
```
(19, 50)
```

### 绘制离散基函数

```python
for i in range(n_poly):
    plt.plot(x, basis[i, :], label=f'V^{i}(x)')
plt.legend()
plt.title('离散勒让德基函数')
plt.show()
```

### 验证加权正交性

计算加权内积矩阵以验证正交性：

```python
""" 权重被放置在矩阵的对角线上 """
W = np.diag(w)
sca = basis @ W @ basis.T
""" 勒让德族是正交的，但不是正交归一的 """
print(sca)
```

### 正规化基族

为了得到正交归一的基族，对基函数进行正规化：

```python
""" 我们对基族进行正规化 """
basisNor = np.zeros_like(basis)
for i in range(n_poly):
    norm = np.sqrt(basis[i, :] @ W @ basis[i, :])
    basisNor[i, :] = basis[i, :] / norm

print(basisNor @ W @ basisNor.T)
```
```
[[ 1. -0. -0. ... 0.]
 [ -0. 1. -0. ... 0.]
 [ ... ]
 [0. 0. ... 1.]]
```

### 函数逼近

使用正交归一的离散勒让德基逼近目标函数 $f(t) = \sqrt{|t|}$：

```python
""" 要逼近的函数（离散化后） """
y = np.sqrt(np.abs(x))

""" 其坐标： """
y_coor = basisNor @ (W * y)

""" 逼近是基的线性组合： """
y_approx = basisNor.T @ y_coor

plt.plot(x, y, 'b.-', label="y")
plt.plot(x, y_approx, 'r.-', label="y_approx")
plt.legend()
plt.title('函数逼近')
plt.show()
```

注意，由于离散化，函数在 $t=0$ 处显示为平的。

### 完美逼近的条件

**问题：** 在什么条件下，逼近是完美的？即图中的蓝点和红点完全重合。

**回答：** 当目标函数可以被基的有限线性组合精确表示时，逼近是完美的。具体来说，如果目标函数本身就是基中的一个成员，或者基的维度足够高，能够完全表示目标函数的多项式展开，那么逼近就会完美。

## 结论

通过使用高斯点进行离散化，我们成功地构建了一个正交归一的离散勒让德基，从而能够有效地逼近目标函数。与简单的均匀离散化相比，高斯点的选择显著提高了逼近的精度和基的正交性。

**习题：**

1. 使用不同数量的高斯点（例如，n_point=10, 20, 50）重复上述过程，观察逼近的效果如何变化。
2. 尝试使用其他函数（如 $f(t) = t^2$ 或 $f(t) = \sin(\pi t)$）进行逼近，分析结果。
3. 实现一个不使用循环的正规化过程，以提高代码的效率。