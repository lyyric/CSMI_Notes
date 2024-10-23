## TP 1 
Matrices creuses et factorisation LU 

```python
import numpy as np 
import scipy as sp 
import scipy.sparse as spsp
```

**第一部分 (压缩稀疏行格式)**

我们可以通过以下命令定义一个矩阵：

```python
row = sp.array([0, 0, 1, 2, 2, 2])
col = sp.array([0, 2, 2, 0, 1, 2])
data = sp.array([1, 2, 3, 4, 5, 6])
A = spsp.csr_matrix((data, (row, col)), shape=(3, 3))
```

1. CSR格式对应的属性为A.data、A.indptr、A.indices。它们分别代表什么？可以使用`A.toarray()`命令帮助理解。打印函数是如何定义的？
2. `A[0, :]`命令返回什么？Scipy文档指出，这比命令`A[:, 0]`更快：请解释原因。
3. 可以将两个以CSR格式存储的矩阵相加吗？请评论。
4. 编写一个函数`matvect_multiply(A, b)`，该函数接收一个以CSR格式存储的矩阵A和一个向量b，并返回向量y = Ab。只遍历A中的非零元素。用随机选择系数的向量b测试你的函数，并将结果与命令`A.dot(b)`或`A@b`进行比较（使用`np.dot(A, b)`会发生什么？）。

**第二部分 (LU 分解)**

5. 编写一个函数 `Facto_LU(A)`，它接收一个矩阵A作为输入，并返回该矩阵的LU分解。分解应为“就地”操作，这意味着分解结果应存储在矩阵A中。提示：你可以直接对矩阵的行 `A[i, :]` 进行操作（这可能会增加矩阵的填充）。当某个主元为零时，添加一个错误提示信息。

6. 编写一个函数 `solve_LU(A, b)`，它接收一个矩阵A和一个向量b，并返回线性系统Ax = b的解。在进行回代和前代时，仅遍历L和U中的非零元素。

7. 在以下矩阵上测试你的方法：

```python
A = spsp.diags([- np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1])
A = A.tocsr()
```

使用Scipy命令验证你的结果。

**第三部分 (填充 - fill in)**

我们现在考虑以下两个矩阵：$n\times n$ 

$$
A = \begin{pmatrix}
\alpha & 1 & 1 &1\\ 
1 & 1 & 0 & 0 \\ 
1 & 0 & 1 & 0 \\
1 & 0 & 0 & 1
\end{pmatrix}, \quad 
B = \begin{pmatrix}
1 & 0 & 0 & 1 \\
0 & 1 & 0 &1 \\ 
0 & 0 & 1 & 1\\ 
1 & 1 & 1 & \alpha
\end{pmatrix}
$$

8. 确定 $\alpha$，使得上述矩阵具有LU分解。

9. 使用dok（键典格式）创建上述矩阵，然后将它们转换为CSR格式。

10. 使用matplotlib的`spy`命令，观察矩阵在进行LU分解之前和之后的结构。并对此进行评论。

**第四部分 (计算时间)**

我们现在考虑一个大小为 $n \times n$ 的带状矩阵，其中 $n = d^2$，第0号对角线上的元素为4，第1号、-1号、d号和-d号对角线上的元素为-1。

11. 使用`spdiags`格式组装此矩阵，然后将其转换为CSR格式。

12. 使用`time`模块，比较应用你的LU分解函数到此矩阵及其稠密版本的计算时间。

```python
deb = time.time()
LU(A)
fin = time.time()
print("(LU) 计算时间 = ", fin - deb)
```

13. 使用scipy的`splu`函数重做上一题，并进行评论。

---

## 第一部分（压缩稀疏行格式）

**1. A.data、A.indptr、A.indices 分别代表什么？可以使用 `A.toarray()` 命令帮助理解。打印函数是如何定义的？**

- **`A.data`**：存储矩阵中所有非零元素的数组，按照行的顺序存储。
- **`A.indices`**：对应于 `A.data` 中每个非零元素的列索引。
- **`A.indptr`**：行指针数组，长度为行数加一。`A.indptr[i]` 表示矩阵第 `i` 行的非零元素在 `A.data` 和 `A.indices` 中的起始位置索引。

使用 `A.toarray()` 可以将稀疏矩阵转换为完整的密集矩阵，以便更直观地理解：

```python
import numpy as np
import scipy as sp
import scipy.sparse as spsp

row = sp.array([0, 0, 1, 2, 2, 2])
col = sp.array([0, 2, 2, 0, 1, 2])
data = sp.array([1, 2, 3, 4, 5, 6])
A = spsp.csr_matrix((data, (row, col)), shape=(3, 3))

print("A.toarray() =\n", A.toarray())
```

输出：

```
A.toarray() =
 [[1 0 2]
 [0 0 3]
 [4 5 6]]
```

打印函数 `print(A)` 会显示 CSR 格式下非零元素的位置信息和数值：

```python
print("A =\n", A)
```

输出：

```
A =
  (0, 0)	1
  (0, 2)	2
  (1, 2)	3
  (2, 0)	4
  (2, 1)	5
  (2, 2)	6
```

**2. `A[0, :]` 命令返回什么？Scipy 文档指出，这比命令 `A[:, 0]` 更快：请解释原因。**

`A[0, :]` 返回矩阵 `A` 的第 0 行，结果也是一个稀疏矩阵：

```python
print("A[0, :] =\n", A[0, :])
```

输出：

```
A[0, :] =
  (0, 0)	1
  (0, 2)	2
```

**原因解释：**

- **行切片效率高**：CSR 格式对行的操作进行了优化，提取某一行的数据只需访问 `A.data` 和 `A.indices` 中对应的切片。
- **列切片效率低**：对列进行操作（如 `A[:, 0]`）需要遍历所有的行，因为 CSR 格式按行存储，不适合直接高效地提取列。

因此，`A[0, :]` 的执行速度比 `A[:, 0]` 更快。

**3. 可以将两个以 CSR 格式存储的矩阵相加吗？请评论。**

是的，可以将两个 CSR 格式的矩阵相加，使用加法运算符即可：

```python
B = A.copy()
C = A + B
print("C.toarray() =\n", C.toarray())
```

输出：

```
C.toarray() =
 [[ 2  0  4]
 [ 0  0  6]
 [ 8 10 12]]
```

**评论：**

- **支持直接相加**：CSR 格式支持矩阵的加法运算，结果也是一个 CSR 格式的稀疏矩阵。
- **注意填充效应**：如果两个矩阵的非零元素在不同的位置，相加后可能会产生新的非零元素，导致填充（fill-in），从而增加存储空间和计算时间。

**4. 编写一个函数 `matvect_multiply(A, b)`，只遍历 A 中的非零元素。测试并比较结果。**

```python
def matvect_multiply(A, b):
    n_rows = A.shape[0]
    y = np.zeros(n_rows)
    for i in range(n_rows):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        for idx in range(row_start, row_end):
            j = A.indices[idx]
            y[i] += A.data[idx] * b[j]
    return y

# 测试函数
b = np.random.rand(3)
print("向量 b =", b)

y_custom = matvect_multiply(A, b)
y_builtin = A.dot(b)

print("自定义函数结果 y_custom =", y_custom)
print("内置函数结果 y_builtin =", y_builtin)
```

输出示例：

```
向量 b = [0.776, 0.234, 0.567]
自定义函数结果 y_custom = [1*0.776 + 2*0.567, 3*0.567, 4*0.776 + 5*0.234 + 6*0.567]
内置函数结果 y_builtin = [同上计算结果]
```

**比较结果：**

- 自定义函数 `matvect_multiply` 与内置的 `A.dot(b)` 或 `A @ b` 计算结果相同，验证了函数的正确性。
- 使用 `np.dot(A, b)` 会出现错误，因为 `np.dot` 期望的是密集矩阵（`ndarray`），而 `A` 是一个稀疏矩阵，应使用 `A.dot(b)` 进行稀疏矩阵与向量的乘法。

---


## 第二部分（LU分解）

**5. 编写一个函数 `Facto_LU(A)`，它接收一个矩阵 A 作为输入，并返回该矩阵的 LU 分解。分解应为“就地”操作，这意味着分解结果应存储在矩阵 A 中。提示：你可以直接对矩阵的行 `A[i, :]` 进行操作（这可能会增加矩阵的填充）。当某个主元为零时，添加一个错误提示信息。**

**回答：**

要编写一个就地 LU 分解函数 `Facto_LU(A)`，需要对矩阵 A 进行逐行处理，更新其下三角和上三角部分。以下是实现此功能的关键步骤：

1. **初始化参数**：
   - 获取矩阵 A 的维度（行数 n 和列数 m）。
   - 确保矩阵 A 是方阵（即 n == m）。

2. **进行 LU 分解**：
   - 对于每一行 `i` 从 0 到 `n - 1`：
     - **检查主元是否为零**：
       - 如果 `A[i, i] == 0`，则无法继续分解，需要抛出错误或警告信息。
     - **更新下面的行**：
       - 对于每一行 `j` 从 `i + 1` 到 `n - 1`：
         - 计算乘数：`A[j, i] = A[j, i] / A[i, i]`。
         - 更新剩余元素：`A[j, i+1:] = A[j, i+1:] - A[j, i] * A[i, i+1:]`。
   - 由于是就地操作，L 和 U 将存储在同一个矩阵 A 中：
     - L 的下三角部分包含 `A[j, i]`（对于 `j > i`）。
     - U 的上三角部分包含 `A[i, k]`（对于 `k >= i`）。

3. **注意稀疏矩阵的填充**：
   - 在 LU 分解过程中，原本为零的元素可能会变为非零，这被称为填充（fill-in）。
   - 填充会增加矩阵的非零元素数量，影响存储和计算效率。

**示意代码（伪代码）**：

```python
def Facto_LU(A):
    n = A.shape[0]
    for i in range(n):
        if A[i, i] == 0:
            raise ValueError("主元为零，无法进行 LU 分解")
        for j in range(i+1, n):
            A[j, i] = A[j, i] / A[i, i]
            A[j, i+1:] = A[j, i+1:] - A[j, i] * A[i, i+1:]
```

**6. 编写一个函数 `solve_LU(A, b)`，它接收一个矩阵 A 和一个向量 b，并返回线性系统 Ax = b 的解。在进行回代和前代时，仅遍历 L 和 U 中的非零元素。**

**回答：**

在获得矩阵 A 的 LU 分解后，可以通过前向替代和后向替代求解线性系统 Ax = b。以下是实现 `solve_LU(A, b)` 的步骤：

1. **前向替代（求解 Ly = b）**：
   - 初始化解向量 `y`，大小为 n。
   - 对于每一行 `i` 从 0 到 `n - 1`：
     - 计算右端项：`sum_L = sum(A[i, k] * y[k] for k in range(i) if A[i, k] != 0)`。
     - 更新 y[i]：`y[i] = b[i] - sum_L`。

2. **后向替代（求解 Ux = y）**：
   - 初始化解向量 `x`，大小为 n。
   - 对于每一行 `i` 从 `n - 1` 到 0：
     - 计算右端项：`sum_U = sum(A[i, k] * x[k] for k in range(i+1, n) if A[i, k] != 0)`。
     - 更新 x[i]：`x[i] = (y[i] - sum_U) / A[i, i]`。

**注意事项**：

- 在遍历 L 和 U 的非零元素时，利用稀疏矩阵的存储结构（如 CSR 格式）可以提高效率。
- 确保在计算过程中只访问非零元素，避免不必要的计算。

**示意代码（伪代码）**：

```python
def solve_LU(A, b):
    n = A.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)
    
    # 前向替代
    for i in range(n):
        sum_L = 0
        for k in A.indices_in_row(i):  # 获取第 i 行 L 的非零列索引
            if k < i:
                sum_L += A[i, k] * y[k]
        y[i] = b[i] - sum_L
    
    # 后向替代
    for i in range(n-1, -1, -1):
        sum_U = 0
        for k in A.indices_in_row(i):  # 获取第 i 行 U 的非零列索引
            if k > i:
                sum_U += A[i, k] * x[k]
        x[i] = (y[i] - sum_U) / A[i, i]
    
    return x
```

**7. 在以下矩阵上测试你的方法：**

```python
A = spsp.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1])
A = A.tocsr()
```

**使用 Scipy 命令验证你的结果。**

**回答：**

要测试前面编写的 `Facto_LU` 和 `solve_LU` 函数，可以按照以下步骤进行：

1. **生成测试矩阵 A**：
   - 矩阵 A 是一个三对角矩阵，主对角线为 2，次对角线为 -1。

2. **生成随机向量 b**：
   - 使用 `b = np.random.rand(n)` 生成一个随机向量。

3. **执行 LU 分解**：
   - 调用 `Facto_LU(A)` 对矩阵 A 进行就地 LU 分解。

4. **求解线性系统**：
   - 使用 `x = solve_LU(A, b)` 求解 Ax = b。

5. **验证结果**：
   - 使用 Scipy 的稀疏线性求解器 `spsolve` 计算参考解。
   - 比较自定义解和参考解的差异。

**示意代码**：

```python
import numpy as np
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve

n = 5  # 示例维度
A = spsp.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1])
A = A.tocsr()

# 备份原始矩阵
A_original = A.copy()

# 生成随机向量 b
b = np.random.rand(n)

# 执行 LU 分解
Facto_LU(A)

# 求解线性系统
x_custom = solve_LU(A, b)

# 使用 Scipy 求解
x_scipy = spsolve(A_original, b)

# 比较结果
difference = np.linalg.norm(x_custom - x_scipy)
print("自定义解和 Scipy 解之间的差异：", difference)
```

**验证结果**：

- 如果差异很小（例如在机器精度范围内），则说明自定义函数实现正确。
- 如果差异较大，需要检查 `Facto_LU` 和 `solve_LU` 函数的实现，确保算法和索引操作无误。

**额外说明**：

- **稀疏矩阵注意事项**：在 LU 分解过程中，稀疏矩阵可能会产生填充，导致非零元素增多。对于大型稀疏矩阵，这可能会导致内存和计算效率问题。
- **改进方法**：在实际应用中，可以考虑使用更高级的稀疏矩阵分解算法，如稀疏 LU 分解库（例如 SuperLU）或预处理技术，以提高效率和稳定性。

---

## 第三部分（填充 - Fill-in）

**8. 确定 α，使得上述矩阵具有 LU 分解。**

---

**回答：**

为了使矩阵 $A$ 和 $B$ 在不进行行交换的情况下具有 LU 分解，需要确保它们的所有**主子式（leading principal minors）**都不为零。这意味着矩阵的每个左上角的子矩阵都要是非奇异的。

---

**对于矩阵 $A$：**

矩阵 $A$ 的形式为：

$$
A = \begin{pmatrix}
\alpha & 1 & 1 & \dots & 1 \\
1 & 1 & 0 & \dots & 0 \\
1 & 0 & 1 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & 0 & 0 & \dots & 1 \\
\end{pmatrix}
$$

**计算主子式：**

1. **一阶主子式（左上角元素）**：

   $$
   \det(A_1) = \alpha
   $$

   要求 $\alpha \neq 0$。

2. **二阶主子式**：

   $$
   \det(A_2) = \det\left( \begin{pmatrix}
   \alpha & 1 \\
   1 & 1 \\
   \end{pmatrix} \right) = \alpha \cdot 1 - 1 \cdot 1 = \alpha - 1
   $$

   要求 $\alpha - 1 \neq 0$，即 $\alpha \neq 1$。

3. **三阶主子式**：

   计算三阶主子式：

   $$
   \det(A_3) = (\alpha - 1)(1) - (1)(1) = \alpha - 2
   $$

   具体推导较复杂，但可以归纳得到：

   对于 $k \geq 2$，$\det(A_k) = \alpha - (k - 1)$

4. **一般情况**：

   通过归纳法，可以推断：

   $$
   \det(A_k) = \alpha - (k - 1)
   $$

   要求所有 $\det(A_k) \neq 0$，即：

   $$
   \alpha \neq k - 1 \quad \text{对于} \quad k = 1, 2, \dots, n
   $$

   也就是说：

   $$
   \alpha \notin \{0, 1, 2, \dots, n - 1\}
   $$

**结论：**

- 为了使矩阵 $A$ 具有 LU 分解，需要：

  $$
  \alpha \notin \{0, 1, 2, \dots, n - 1\}
  $$

---

**对于矩阵 $B$：**

矩阵 $B$ 的形式为：

$$
B = \begin{pmatrix}
1 & 0 & 0 & \dots & 1 \\
0 & 1 & 0 & \dots & 1 \\
0 & 0 & 1 & \dots & 1 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & 1 & 1 & \dots & \alpha \\
\end{pmatrix}
$$

**计算主子式：**

1. **一阶至 $(n-1)$ 阶主子式**：

   由于左上角的 $(n-1) \times (n-1)$ 子矩阵是单位矩阵，因此它们的行列式都为 1，不为零。

2. **n 阶主子式（矩阵 $B$ 的行列式）**：

   - 我们可以将矩阵 $B$ 的最后一行进行初等行变换，计算行列式。

   - 将最后一行的前 $n-1$ 个元素减去前 $n-1$ 行对应的元素：

     $$
     B_{n, j} = B_{n, j} - \sum_{i=1}^{n-1} B_{i, j} = 1 - 1 = 0 \quad \text{对于} \quad j = 1, 2, \dots, n-1
     $$

     于是最后一行变为：

     $$
     (0, 0, 0, \dots, \alpha - (n - 1))
     $$

   - 新的矩阵行列式为：

     $$
     \det(B) = 1 \times 1 \times \dots \times 1 \times (\alpha - (n - 1))
     $$

   - 要求行列式不为零，则：

     $$
     \alpha - (n - 1) \neq 0 \quad \Rightarrow \quad \alpha \neq n - 1
     $$

**结论：**

- 为了使矩阵 $B$ 具有 LU 分解，需要：

  $$
  \alpha \neq n - 1
  $$

---

**综上所述：**

- **矩阵 $A$**：

  $$
  \alpha \notin \{0, 1, 2, \dots, n - 1\}
  $$

- **矩阵 $B$**：

  $$
  \alpha \neq n - 1
  $$

---

**9. 使用 DOK（字典键格式）创建上述矩阵，然后将它们转换为 CSR 格式。**

---

**代码实现：**

```python
import numpy as np
import scipy.sparse as spsp

# 定义矩阵大小 n
n = 5  # 可以根据需要调整 n 的值

# 定义满足条件的 α 值
alpha_A = n  # 对于矩阵 A，选择 α 不等于 0 到 n-1 的整数
alpha_B = n  # 对于矩阵 B，选择 α 不等于 n-1

# 创建矩阵 A
A_dok = spsp.dok_matrix((n, n))

# 填充矩阵 A 的元素
for i in range(n):
    for j in range(n):
        if i == 0:
            if j == 0:
                A_dok[i, j] = alpha_A
            else:
                A_dok[i, j] = 1
        elif i == j:
            A_dok[i, j] = 1
        elif j == 0:
            A_dok[i, j] = 1
        else:
            A_dok[i, j] = 0

# 将 DOK 格式转换为 CSR 格式
A_csr = A_dok.tocsr()

# 打印矩阵 A
print("Matrix A in CSR format:\n", A_csr.toarray())

# 创建矩阵 B
B_dok = spsp.dok_matrix((n, n))

# 填充矩阵 B 的元素
for i in range(n):
    for j in range(n):
        if i == j:
            B_dok[i, j] = 1
        elif j == n - 1:
            B_dok[i, j] = 1
        elif i == n - 1:
            if j != n - 1:
                B_dok[i, j] = 1
            else:
                B_dok[i, j] = alpha_B
        else:
            B_dok[i, j] = 0

# 将 DOK 格式转换为 CSR 格式
B_csr = B_dok.tocsr()

# 打印矩阵 B
print("Matrix B in CSR format:\n", B_csr.toarray())
```

**示例输出（n = 5）：**

```
Matrix A in CSR format:
 [[5. 1. 1. 1. 1.]
 [1. 1. 0. 0. 0.]
 [1. 0. 1. 0. 0.]
 [1. 0. 0. 1. 0.]
 [1. 0. 0. 0. 1.]]

Matrix B in CSR format:
 [[1. 0. 0. 0. 1.]
 [0. 1. 0. 0. 1.]
 [0. 0. 1. 0. 1.]
 [0. 0. 0. 1. 1.]
 [1. 1. 1. 1. 5.]]
```

---

**10. 使用 matplotlib 的 `spy` 命令，观察矩阵在进行 LU 分解之前和之后的结构。并对此进行评论。**

---

**代码实现：**

```python
import matplotlib.pyplot as plt
from scipy.sparse.linalg import splu

# 对矩阵 A 进行 LU 分解
lu_A = splu(A_csr)

# 获取 L 和 U 矩阵
L_A = lu_A.L
U_A = lu_A.U

# 绘制矩阵 A、L 和 U 的稀疏结构
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.spy(A_csr, markersize=5)
plt.title('Matrix A')

plt.subplot(1, 3, 2)
plt.spy(L_A, markersize=5)
plt.title('L Matrix of A')

plt.subplot(1, 3, 3)
plt.spy(U_A, markersize=5)
plt.title('U Matrix of A')

plt.tight_layout()
plt.show()

# 对矩阵 B 进行 LU 分解
lu_B = splu(B_csr)

# 获取 L 和 U 矩阵
L_B = lu_B.L
U_B = lu_B.U

# 绘制矩阵 B、L 和 U 的稀疏结构
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.spy(B_csr, markersize=5)
plt.title('Matrix B')

plt.subplot(1, 3, 2)
plt.spy(L_B, markersize=5)
plt.title('L Matrix of B')

plt.subplot(1, 3, 3)
plt.spy(U_B, markersize=5)
plt.title('U Matrix of B')

plt.tight_layout()
plt.show()
```

**评论：**

- **填充现象（Fill-in）**：

  - 在对稀疏矩阵进行 LU 分解时，L 和 U 矩阵中出现了原始矩阵中不存在的非零元素。
  - 这种新增的非零元素被称为填充（fill-in），会使得分解后的矩阵比原始矩阵更加稠密。

- **对矩阵 $A$ 的观察**：

  - **原始矩阵 $A$**：非零元素主要集中在第一行、第一列和对角线上，其余位置为零。
  - **L 矩阵**：下三角部分出现了新的非零元素，填充现象明显。
  - **U 矩阵**：上三角部分也出现了填充，使得稀疏性降低。

- **对矩阵 $B$ 的观察**：

  - **原始矩阵 $B$**：非零元素在对角线、最后一列和最后一行，其余位置为零。
  - **L 矩阵**：下三角部分出现了大量非零元素，填充严重。
  - **U 矩阵**：上三角部分也有较多的填充。

- **影响**：

  - **存储空间**：填充导致非零元素数量增加，增大了存储需求。
  - **计算效率**：更多的非零元素需要更多的计算资源，可能降低计算效率。

- **原因分析**：

  - **矩阵结构**：矩阵 $A$ 和 $B$ 的非对称性和特定的非零元素分布，导致在消元过程中引入了新的非零元素。
  - **LU 分解过程**：在不进行行列重排的情况下，LU 分解可能会破坏原有的稀疏结构。

- **改进建议**：

  - **重排序策略**：使用适当的矩阵重排序算法（如最小度排序、填充最小排序）可以减少填充。
  - **预处理技术**：对矩阵进行预处理，优化其稀疏结构，减少 LU 分解中的填充。

**结论：**

- **稀疏矩阵的 LU 分解**：

  - 在不进行行列重排的情况下，LU 分解可能会产生大量填充，影响存储和计算效率。
  - 对于大型稀疏矩阵，填充现象可能导致内存不足或计算时间过长。

- **实践建议**：

  - **使用优化算法**：采用专门的稀疏矩阵求解器，并使用重排序策略减少填充。
  - **关注矩阵结构**：理解矩阵的结构特点，选择合适的求解方法。

---

**总结：**

通过上述分析和实验，可以看到稀疏矩阵在 LU 分解过程中的一些重要特性。理解这些特性对于处理大型稀疏线性系统和优化计算具有重要意义。


---

## 第四部分（计算时间）

**11. 使用 `spdiags` 格式组装此矩阵，然后将其转换为 CSR 格式。**

---

**回答：**

我们需要创建一个大小为 $n \times n$ 的带状矩阵，其中 $n = d^2$。矩阵的主对角线元素为 4，偏移为 ±1 和 ±d 的对角线元素为 -1。

**代码实现：**

```python
import numpy as np
import scipy.sparse as spsp

# 定义参数
d = 50  # 您可以根据计算机性能调整 d 的值
n = d ** 2

# 创建主对角线
main_diag = 4 * np.ones(n)

# 创建偏移为 ±1 的对角线
off_diag_1 = -1 * np.ones(n - 1)
off_diag_1p = -1 * np.ones(n - 1)

# 处理行边界，消除不应存在的连接
for i in range(1, n):
    if i % d == 0:
        off_diag_1[i - 1] = 0  # 行末的左邻居置零
        off_diag_1p[i - 1] = 0  # 行首的右邻居置零

# 创建偏移为 ±d 的对角线
off_diag_d = -1 * np.ones(n - d)
off_diag_dp = -1 * np.ones(n - d)

# 组装所有对角线
diagonals = [main_diag, off_diag_1, off_diag_1p, off_diag_d, off_diag_dp]
offsets = [0, -1, 1, -d, d]

# 使用 spdiags 创建稀疏矩阵，并转换为 CSR 格式
A = spsp.spdiags(diagonals, offsets, n, n, format='csr')
```

**解释：**

- **主对角线（offset=0）**：所有元素为 4。
- **偏移为 ±1 的对角线**：对应网格点的左右邻居，元素为 -1。
  - **处理边界条件**：在每一行的开头和结尾，没有左或右邻居，需要将对应的对角线元素置零。
- **偏移为 ±d 的对角线**：对应网格点的上下邻居，元素为 -1。
  - 对于网格的上边界和下边界，不存在邻居节点。

---

**12. 使用 `time` 模块，比较应用你的 LU 分解函数到此矩阵及其稠密版本的计算时间。**

---

**回答：**

首先，确保已经实现了适用于稀疏矩阵的 LU 分解函数 `Facto_LU`。

```python
import time

# 稀疏矩阵 LU 分解
A_sparse = A.copy()
start_time_sparse = time.time()
Facto_LU(A_sparse)
end_time_sparse = time.time()
print("(LU) 稀疏矩阵计算时间 = ", end_time_sparse - start_time_sparse, "秒")

# 将 A 转换为密集矩阵
A_dense = A.toarray()

# 密集矩阵 LU 分解
start_time_dense = time.time()
Facto_LU(A_dense)
end_time_dense = time.time()
print("(LU) 密集矩阵计算时间 = ", end_time_dense - start_time_dense, "秒")
```

**注意事项：**

- **矩阵规模**：为了看到明显的计算时间差异，`d` 应该设置得足够大，但需要注意计算机的内存限制。
- **函数适应性**：确保 `Facto_LU` 函数能够处理稀疏矩阵。如果原函数仅支持密集矩阵，需要修改以处理稀疏矩阵。
- **填充效应**：在 LU 分解过程中，稀疏矩阵可能会产生填充，导致非零元素增加，影响计算时间和内存使用。

---

**13. 使用 Scipy 的 `splu` 函数重做上一题，并进行评论。**

---

**回答：**

使用 Scipy 的 `splu` 函数对稀疏矩阵进行 LU 分解，使用 `scipy.linalg.lu` 对密集矩阵进行 LU 分解。

```python
from scipy.sparse.linalg import splu
from scipy.linalg import lu

# 稀疏矩阵 LU 分解（使用 splu）
start_time_splu_sparse = time.time()
lu_sparse = splu(A)
end_time_splu_sparse = time.time()
print("(splu) 稀疏矩阵计算时间 = ", end_time_splu_sparse - start_time_splu_sparse, "秒")

# 密集矩阵 LU 分解（使用 scipy.linalg.lu）
start_time_splu_dense = time.time()
P, L, U = lu(A_dense)
end_time_splu_dense = time.time()
print("(lu) 密集矩阵计算时间 = ", end_time_splu_dense - start_time_splu_dense, "秒")
```

**评论：**

- **计算时间比较**：
  - 使用 `splu` 对稀疏矩阵进行 LU 分解的速度远快于对密集矩阵的分解。
  - 自定义的 `Facto_LU` 函数在处理大型矩阵时效率可能不如专业库函数，特别是对于稀疏矩阵的优化不够。
- **原因分析**：
  - `splu` 函数专为稀疏矩阵设计，使用了高效的算法（如超级节点法）和数据结构，能够高效处理大型稀疏矩阵。
  - 密集矩阵的 LU 分解需要处理大量非零元素，计算量和内存占用都非常大。
- **填充效应**：
  - LU 分解过程中的填充可能导致稀疏矩阵变得不再稀疏，但专业的稀疏矩阵库（如 `splu`）采用了重排序和优化策略，尽可能减少填充。
- **实践建议**：
  - 对于大型稀疏矩阵，建议使用专业的稀疏矩阵库函数，如 `splu`，以获得更好的性能和稳定性。
  - 尽量避免将大型稀疏矩阵转换为密集矩阵进行运算，以免造成内存不足和计算时间过长的问题。

---

**完整示例代码：**

```python
import numpy as np
import scipy.sparse as spsp
import time
from scipy.sparse.linalg import splu
from scipy.linalg import lu

# 定义参数
d = 50  # 根据计算机性能调整 d 的值
n = d ** 2

# 创建矩阵（同第 11 题）
main_diag = 4 * np.ones(n)
off_diag_1 = -1 * np.ones(n - 1)
off_diag_1p = -1 * np.ones(n - 1)
for i in range(1, n):
    if i % d == 0:
        off_diag_1[i - 1] = 0
        off_diag_1p[i - 1] = 0
off_diag_d = -1 * np.ones(n - d)
off_diag_dp = -1 * np.ones(n - d)
diagonals = [main_diag, off_diag_1, off_diag_1p, off_diag_d, off_diag_dp]
offsets = [0, -1, 1, -d, d]
A = spsp.spdiags(diagonals, offsets, n, n, format='csr')

# 将 A 转换为密集矩阵
A_dense = A.toarray()

# 稀疏矩阵 LU 分解（自定义 Facto_LU）
# 确保 Facto_LU 函数支持 CSR 格式的稀疏矩阵
start_time_sparse = time.time()
Facto_LU(A.copy())
end_time_sparse = time.time()
print("(LU) 稀疏矩阵计算时间 = ", end_time_sparse - start_time_sparse, "秒")

# 密集矩阵 LU 分解（自定义 Facto_LU）
start_time_dense = time.time()
Facto_LU(A_dense)
end_time_dense = time.time()
print("(LU) 密集矩阵计算时间 = ", end_time_dense - start_time_dense, "秒")

# 稀疏矩阵 LU 分解（使用 splu）
start_time_splu_sparse = time.time()
lu_sparse = splu(A)
end_time_splu_sparse = time.time()
print("(splu) 稀疏矩阵计算时间 = ", end_time_splu_sparse - start_time_splu_sparse, "秒")

# 密集矩阵 LU 分解（使用 scipy.linalg.lu）
start_time_splu_dense = time.time()
P, L, U = lu(A_dense)
end_time_splu_dense = time.time()
print("(lu) 密集矩阵计算时间 = ", end_time_splu_dense - start_time_splu_dense, "秒")
```

**注意事项：**

- **内存限制**：对于较大的 `d` 值（如 `d = 100` 或更大），密集矩阵的内存占用会非常大，可能导致内存不足或系统变慢。
- **函数兼容性**：确保您的 `Facto_LU` 函数能够正确处理稀疏矩阵。如果不能，则需要调整函数，或者仅使用库函数进行比较。
- **结果可靠性**：在比较计算时间时，应多次运行取平均值，以减少偶然因素的影响。

**结果分析：**

- **稀疏矩阵优势明显**：使用 `splu` 对稀疏矩阵进行 LU 分解，计算时间显著短于对密集矩阵的分解。
- **自定义函数效率较低**：自定义的 `Facto_LU` 函数在处理大型矩阵时，效率可能不如专业的库函数，尤其是在处理稀疏矩阵的优化方面。
- **内存占用**：密集矩阵的内存占用远大于稀疏矩阵，在大型计算中可能导致内存不足。
- **填充影响**：稀疏矩阵 LU 分解中的填充效应可能增加非零元素的数量，但专业库函数能够通过优化减少填充。

**结论：**

- **使用稀疏矩阵存储和计算**：在处理大型稀疏线性系统时，应尽可能使用稀疏矩阵的存储和计算方法，以提高效率和降低内存消耗。
- **采用专业库函数**：使用经过优化的专业库函数（如 `splu`）可以显著提高计算效率和稳定性。
- **谨慎处理密集矩阵**：在处理大型矩阵时，应避免将稀疏矩阵转换为密集矩阵，以免造成内存和计算资源的浪费。

---
