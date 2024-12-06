### 科学计算 2 课程

### 第三次实践课：Lanczos 方法

本次实践课的目标是编程实现 **Lanczos 方法**，用于计算对称矩阵的特征值。

首先导入 Python 模块：

```
python
import numpy as np  
import scipy as sp  
import scipy.sparse as spsp  
import scipy.sparse.linalg as spsplin
```

---

#### 第一部分（Lanczos 方法）

1. **编写函数 `iter_Arnoldi_sym(A, v, vold, beta)`** 该函数用于在对称情况中执行 Arnoldi 算法的一个迭代步骤。根据向量 $v, vold \in \mathbb{R}^n$（对应于 $v_p$ 和 $v_{p-1}$）以及标量 $\beta \in \mathbb{R}$（对应于 $\beta_{p-1}$），函数将更新向量 $v, vold$ 为 $v_{p+1}, v_p$，并返回系数 $\beta$ 和 $\alpha$（对应于 $\beta_p$ 和 $\alpha_p$）。提醒：Lanczos 算法如下所示：

   - 初始化：

$$
v_0 \in \mathbb{R}^n
$$

$$
v_0 = \frac{v_0}{\|v_0\|}, \quad \beta_{-1} = 0, \quad v_{-1} = 0
$$

- 对于 $p = 0, \dots, N-1$：

$$
w_p = Av_p
$$

$$
\alpha_p = (w_p, v_p)
$$

$$
w_p = w_p - \alpha_p v_p - \beta_{p-1} v_{p-1}
$$


$$
\beta_p = \|w_p\|
$$

$$
v_{p+1} = \frac{w_p}{\beta_p}
$$

2. **编写函数 `Lanczos(A, nbiter)`**
   从矩阵 $A \in \mathbb{M}_n(\mathbb{R})$ 和整数 $nbiter$ 出发，返回一个大小为 $(nbiter, n)$ 的数组 `eigval`，每一行包含 Lanczos 算法获得的 **Ritz 值**，按升序排列。
   提醒：Lanczos 算法的完整流程如下：

   $v_0$ 随机初始化
   $v_0 = \frac{v_0}{\|v_0\|}$循环直到满足终止条件：

   - 计算 $\alpha_p, \beta_p$
   - 构造矩阵 $T_p$
   - 计算 $T_p$ 的特征值

我们有

$$
T_p =
\begin{bmatrix}
\alpha_0 & \beta_0 & 0 & \cdots & 0\\
\beta_0 & \alpha_1 & \beta_1 & \cdots & 0\\
0 & \beta_1 & \alpha_2 & \cdots & 0\\
\vdots & \vdots & \vdots & \ddots & \beta_{p-1}\\
0 & 0 & 0 & \beta_{p-1} & \alpha_p
\end{bmatrix}
$$

可以使用 `np.linalg.eig` 计算 $T_p$ 的特征值。

3. **测试程序**
   在以下矩阵上测试程序：

```python
   A = spsp.diags(
       [[4.]*n, [-1]*(n-1), [-1]*(n-1), [-1]*(n-d), [-1]*(n-d)], 
       [0, 1, -1, d, -d]
   )
```

   其中 $n = d^2, d = 10$，取 $nbiter = 40$。
   验证最大的 Ritz 值是否收敛到最大的特征值（绘制误差的对数图）。
   对比结果可以使用 `np.linalg.eig` 获得的精确特征值。
   **补充：** 增加对最小 Ritz 值收敛于最小特征值的验证，并估计矩阵的条件数。

   提醒：可以使用 `scipy.sparse.linalg.eigsh` 直接计算最大特征值。

4. **可选任务：** 显示所有 Ritz 值的演化。注意对于此类矩阵，算法的第 $p$ 次迭代提供了前 $(p+1)/2$ 个最大特征值和后 $(p+1)/2$ 个最小特征值的近似值（若 $p$ 为奇数）。
5. **考虑矩阵 B：**

```python
   B = spsp.diags(L, 0, dtype=float64)
```

   其中 $L = [0, 0.01, 0.02, \dots, 1.99, 2, 2.5, 3]$，大小为 203。
   在迭代过程中显示最大的两个 Ritz 值，观察并评论 **“伪特征值”现象**。

---

#### 第二部分（QR 方法）

6. **编写函数 `facto_QR_hessenberg(A)`** 从 Hessenberg 矩阵 $A \in \mathbb{M}_n(\mathbb{R})$ 出发，返回矩阵 $Q \in \mathbb{O}_n(\mathbb{R})$ 和下三角矩阵 $R \in \mathbb{M}_n(\mathbb{R})$，使得 $A = QR$，使用 **Givens 方法**。
   提醒：第 $k$ 步是通过左乘 Givens 矩阵 $G_k$ 消去 $A[k+1,k]$。
   Givens 矩阵形式为：

   $$
   G_k =
   \begin{bmatrix}
   1 & & & & \\
   & c & -s & & \\
   & s & c & & \\
   & & & \ddots & \\
   & & & & 1
   \end{bmatrix}
   $$

   其中 $c, s$ 满足：

   $$
   c^2 + s^2 = 1, \quad sA[k,k] + cA[k+1,k] = 0
   $$

   即：

   $$
   c = \pm \frac{A[k,k]}{\sqrt{A[k,k]^2 + A[k+1,k]^2}}, \quad s = \mp \frac{A[k+1,k]}{\sqrt{A[k,k]^2 + A[k+1,k]^2}}
   $$

   乘以 $G_k$ 实际上是对第 $k$ 行和第 $k+1$ 行进行旋转。
   **构造 $Q = (G_{n-1} \cdots G_1)^T$** 时需对单位矩阵进行相同的操作。
7. **测试程序**在小型随机矩阵上测试函数。可以使用 `np.triu` 构造 Hessenberg 矩阵。
8. **编写函数 `QR_hessenberg(A)`** 输入 Hessenberg 矩阵 $A \in \mathbb{M}_n(\mathbb{R})$，返回其特征值的近似值，使用 QR 方法。收敛标准：使用子对角线（-1）的最大绝对值和最大迭代次数（例如最多 2000 次迭代）。
9. **测试程序**在小型随机对称三对角矩阵上测试函数，并用 `np.linalg.eig` 验证结果。若矩阵为任意 Hessenberg 矩阵，观察并评论结果。
10. **在 Lanczos 算法中替换 `np.linalg.eig`**
    使用你的 `QR_hessenberg` 函数替换 `np.linalg.eig`，验证特征值计算是否有效，并比较计算时间。
