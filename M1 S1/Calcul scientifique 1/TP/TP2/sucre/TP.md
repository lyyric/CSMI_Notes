### **题目：热方程的数值求解**

**目标**：求解以下热传导方程的两种数值方法：

\[
u_t(x, t) - u_{xx}(x, t) = 0, \quad t > 0, \; 0 < x < L
\]

- 初始条件：\( u(x, 0) = u_0(x) \)
- 边界条件：\( u_x(0, t) = 0 \) 和 \( u_x(L, t) = 0 \)

其中 \( L > 0 \)，在应用中，我们取 \( L = 1 \) 并将初始条件定义为：

\[
u_0(x) = 
\begin{cases} 
1, & \text{if } x \in [1/2 - 1/8, 1/2 + 1/8], \\
0, & \text{otherwise}.
\end{cases}
\]

### **题目要求**

1. **形式上的推导**：通过启发式的方法，寻找 \( u \) 的解的形式：

    \[
    u(x, t) = \sum_{i=0}^{+\infty} c_i e^{-\frac{i^2 \pi^2}{L^2} t} \cos\left(\frac{i \pi}{L} x\right)。
    \]

2. **计算傅立叶系数 \( c_i \)**：利用初始条件 \( u_0(x) \) 来计算傅立叶系数 \( c_i \)。

3. **极限**：无需计算，直接给出当 \( t \) 趋向无穷大时，\( u(x, t) \) 的极限。

4. **截断级数**：利用截断级数近似 \( u(x, t) \)：

    \[
    u_N(x, t) = \sum_{i=0}^{N} c_i e^{-\frac{i^2 \pi^2}{L^2} t} \cos\left(\frac{i \pi}{L} x\right)。
    \]
    
    用 Rust 编写程序，在不同时间点 \( t = 0.001, 0.01, 0.1, 1 \) 下绘制 \( u_N(x, t) \)，并观察在 \( t \) 的小值情况下结果的变化。

5. **有限差分法求解**：编写一个 Rust 程序，使用时间积分的 θ-格式（\( \theta \)-scheme）求解方程（0.1）。要求：
    - 描述编程过程，解释如何利用矩阵的稀疏结构进行存储。
    - 使用 Rust 的 `sky.rs` 库来求解三对角矩阵方程。

6. **比较结果**：将第4部分中截断级数方法的结果与第5部分中有限差分法的结果进行比较。测试不同的 \( \theta \) 值以及不同的空间和时间离散化参数，验证稳定性条件的作用，特别是当 \( \theta = 0 \) 的情况。

