# 实验2：GMRES方法和共轭梯度法

本实验的目标是编程实现GMRES方法和共轭梯度法。首先导入Python模块：

```python
import numpy as np 
import scipy as sp 
import scipy.sparse as spsp 
import scipy.sparse.linalg as spsplin
```

对于矩阵乘法，使用命令 `@`：它与scipy的稀疏矩阵结构兼容。

## 第一部分 (GMRES)

1. 编写一个函数 `Arnoldi(A, V, H)`，它从矩阵 $A \in M_n(\mathbb{R})$，$V \in M_{n,p}(\mathbb{R})$（包含Krylov子空间 $K_{p-1}(A; r_0)$ 的基）和 $H \in M_{p,p-1}(\mathbb{R})$ 开始，返回矩阵 $V_p \in M_{n,p+1}(\mathbb{R})$（包含Krylov子空间 $K_p(A; r_0)$ 的基）和 $H_p \in M_{p+1,p}(\mathbb{R})$。我们回顾以下关系：

$$
w_p = A v_{p-1} - \sum_{j \leq p-1} \langle A v_{p-1}, v_j \rangle v_j, \quad v_p = \frac{w_p}{\|w_p\|},
$$

$$
A v_{p-1} = \|w_p\| v_p + \sum_{j \leq p-1} \langle A v_{p-1}, v_j \rangle v_j = h_{p,p-1} v_p + \sum_{j \leq p-1} h_{j,p-1} v_j.
$$

其中 $V = [v_0, \dots, v_{p-1}] \in M_{n,p}(\mathbb{R})$。根据这些关系，首先计算 $h_{j,p-1}$ 和 $w_p$，然后计算 $v_p$ 和 $h_{p,p-1}$。

2. 编写一个函数 `gmres(A, b, xexact)`，它从矩阵 $A \in M_n(\mathbb{R})$，向量 $b \in \mathbb{R}^n$ 和精确解 $x_{\text{exact}} \in \mathbb{R}^n$（如果可用）开始，返回通过GMRES算法获得的线性系统解 $x$，以及相对误差列表 $\|x_{\text{exact}} - x_p\|/\|x_{\text{exact}}\|$，和相对残差范数列表 $\|r_p\|/\|r_0\|$。我们在下面回顾算法：

- 给定 $x_0$
- $r_0 = b - A x_0$
- $v_0 = r_0 / \|r_0\|$, $V_0 = [v_0]$, $\hat{H}_{-1} = [\ ]$  
当条件未满足时：
- 从 $V_p, \hat{H}_{p-1}$ 计算 $V_{p+1}, \hat{H}_p$
- $Q_p R_p = \hat{H}_p$
- $(R_p)_{0 \leq i,j \leq p} y = |r_0| (Q_p^T e_0)_{0 \leq j \leq p}$
- $x_{p+1} = x_0 + V_p y$

您可以使用函数 `sp.linalg.qr` 进行QR分解，使用 `np.linalg.solve` 解三角系统。

3. 在以下矩阵上测试您的程序：

```python
A = np.diag(2*np.ones(n)) + 0.5 * np.random.rand(n, n)/np.sqrt(n)
```

取 $x_0 = 0$。以迭代次数为变量，绘制误差和残差（对数尺度）。可以使用numpy函数计算精确解。

4. （可选）编写一个函数 `gmres(A, b, xexact, p)`，它在每 $p \in \mathbb{N}^*$ 次迭代后重启GMRES函数。

## 第二部分 （共轭梯度法）

5. 编写一个函数 `gradient_conjugue(A, b, xexact)`，它从矩阵 $A \in M_n(\mathbb{R})$，向量 $b \in \mathbb{R}^n$ 和精确解 $x_{\text{exact}} \in \mathbb{R}^n$（如果可用）开始，返回通过共轭梯度算法获得的线性系统解 $x$，以及相对误差列表和相对残差范数列表。我们在下面回顾算法（左侧）：

**共轭梯度法**

- 给定 $x_0$
- $r_0 = b - A x_0$, $d_0 = r_0$
  
当条件未满足时：
- $s_p = \frac{(r_p, r_p)}{(A d_p, d_p)}$
- $x_{p+1} = x_p + s_p d_p$
- $r_{p+1} = r_p - s_p A d_p$
- $\beta_p = \frac{(r_{p+1}, r_{p+1})}{(r_p, r_p)}$
- $d_{p+1} = r_{p+1} + \beta_p d_p$

**预条件共轭梯度法**

- 给定 $x_0$
- $r_0 = b - A x_0$
- 解 $M z_0 = r_0$, $d_0 = z_0$
  
当条件未满足时：
- $s_p = \frac{(r_p, z_p)}{(A d_p, d_p)}$
- $x_{p+1} = x_p + s_p d_p$
- $r_{p+1} = r_p - s_p A d_p$
- 解 $M z_{p+1} = r_{p+1}$
- $\beta_p = \frac{(r_{p+1}, z_{p+1})}{(r_p, z_p)}$
- $d_{p+1} = z_{p+1} + \beta_p d_p$

6. 在以下矩阵上测试您的方法：

```python
B = spsp.diags([[4.]*n, [-1]*(n-1), [-1]*(n-1), [-1]*(n-d), [-1]*(n-d)], [0, 1, -1, d, -d])
```

其中 $n = d^2$。以迭代次数为变量，绘制误差和残差。可以使用 `scipy` 函数计算精确解。与使用GMRES方法获得的结果以及使用 `time` 模块测量的计算时间进行比较。评论一下。为什么这些算法非常适合稀疏矩阵结构？

## 第三部分 （预处理）

7. 将GMRES方法应用于以下矩阵：

```python
C = np.diag(2 + np.arange(n)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
```

与预处理系统 $M^{-1} C x = M^{-1} b$ 的结果进行比较，其中 $M$ 是 $C$ 的对角部分（对角预处理或雅可比预处理）。使用函数 `np.linalg.cond` 显示矩阵 $C$ 和 $M^{-1} C$ 的条件数。比较计算时间。观察不同矩阵大小 $n$ 的结果。评论一下。

8. 用以下矩阵重复上一个问题：

```python
D = np.diag(2*np.ones(n)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
```

评论一下。

9. 编写一个函数 `gradient_conjugue_precond(A, b, M, xexact)`，它从矩阵 $A \in M_n(\mathbb{R})$，向量 $b \in \mathbb{R}^n$，预处理矩阵 $M$，以及精确解 $x_{\text{exact}} \in \mathbb{R}^n$（如果可用）开始，返回通过预条件共轭梯度算法获得的线性系统解 $x$，以及相对误差列表和相对残差范数列表。算法在上面（右侧）回顾。

10. 在矩阵 $B$（问题6）上测试您的函数 `gradient_conjugue_precond`，使用 `spsplin.spilu` 提供的不完全LU分解作为预处理矩阵。与未预处理的共轭梯度方法的计算时间进行比较。评论一下。