这份总结涵盖了 **SciML 2** 课程中关于 **时间步进方法、半拉格朗日法以及随机微分方程 (SDE/BSDE)** 的核心内容，同样采用考前 A4 纸速记布局。

---

## 📄 SciML 2: 高级时间方法与随机动力学 (A4版)

### 1. 神经网络 Galerkin 与 离散 PINNs

- **标准投影方法**：将 PDE 代入时间格式（如显式欧拉），得到估计值 $R(\mathbf{x})$，然后通过最小二乘投影到空间 $V_n$：$\mathbf{M}\theta(t_{n+1}) = \mathbf{M}\theta(t_n) + \Delta t \mathbf{b}$。
    
- **Neural Galerkin (NG)**：
    
    - **核心思想**：在 $\theta(t_n)$ 附近**线性化**网络：$nn_\theta \approx u_{\theta_n} + \nabla_\theta u_{\theta_n}(\theta - \theta_n)$。
        
    - **优点**：将每一步的非线性训练转化为求解一个**线性 Gram 矩阵方程**（质量矩阵 $\mathbf{M}$ 随时间演化）。
        
    - **高阶格式**：可结合 RK4 格式，每一步子步都需更新矩阵 $\mathbf{M}$ 和右端项 $\mathbf{b}$。
        
- **离散 PINNs**：
    
    - **显式**：$\theta_{n+1} = \arg \min_\theta \|nn_\theta - R(\mathbf{x}; u_{\theta_n})\|^2$。
        
    - **隐式**：$\theta_{n+1} = \arg \min_\theta \|nn_\theta - R(\mathbf{x}; nn_\theta)\|^2$（优化目标包含当前时刻的算子）。
        

---

### 2. 算子分裂法 (Splitting Schemes)

- **Lie-Trotter 分裂**：将 $\partial_t u = L_1 u + L_2 u$ 拆解，先解 $L_1$ 一步，以此为初值再解 $L_2$ 一步。
    
    - **误差**：$O(\Delta t^2)$ 局部误差，整体 1 阶精度。误差项取决于交换子 $[A_1, A_2]$。
        
- **在 SciML 中的应用**：可以将复杂的物理项（如平流+扩散）分开处理，分别训练或使用不同的数值方案。
    

---

### 3. 半拉格朗日法 (Semi-Lagrangian, SL)

- **原理**：利用**特征线方法**。对于 $\partial_t u + a \partial_x u = S$，沿特征线 $\dot{X} = a$ 有 $\frac{d}{dt} u(t, X(t)) = S(t, X(t), u)$。
    
- **算法步**：
    
    1. **溯源**：从网格点 $x_i$ 出发，反向解 ODE 找到上一时刻的位置 $X(t_n)$。
        
    2. **赋值**：$\hat{u}_{n+1}(x_i) = u_n(X(t_n)) + \text{源项积分}$。
        
    3. **投影**：优化 $\theta_{n+1}$ 使网络逼近 $\hat{u}_{n+1}$。
        
- **优势**：**无 CFL 稳定性条件限制**（大步长不炸），适用于平流主导问题。
    

---

### 4. 动力学方程与拉格朗日法 (PIC)

- **Vlasov/动力学方程**：描述粒子分布函数 $f(t, x, v)$。
    
- **粒子群算法 (PIC)**：将 $f$ 近似为 Dirac 质量之和 $\sum \delta(x-X_k)\delta(v-V_k)$。
    
- **Petrov-Galerkin 视角**：粒子运动方程（牛顿定律）可以通过测试函数与分布函数残差的正交性导出。
    

---

### 5. 随机方法与 BSDE (Deep Feynman-Kac)

- **SDE (前向)**：$dX_t = b dt + \sigma dW_t$（布朗运动）。
    
- **Ito 公式**：处理随机过程函数的微积分，多出二阶导项 $\frac{1}{2}\sigma^2 \partial_{xx} p$。
    
- **Feynman-Kac 公式**：将线性二阶偏微分方程（Fokker-Planck 类型）的解联系到随机过程的**期望值**。
    
- **Deep Feynman-Kac 算法**：
    
    1. **离散时间**：基于倒向随机微分方程 (BSDE)。
        
    2. **模拟轨迹**：模拟 $M$ 条前向随机轨迹。
        
    3. **递归学习**：从 $t=T$ 开始，利用网络学习 $p_{t_n}$，其标签由 $p_{t_{n+1}}$ 的采样值计算。
        
    4. **损失函数**：$L = \sum \| p_\theta(X^i_{t_n}) - (p_{t_{n+1}}(X^i_{t_{n+1}}) + \Delta t F) \|^2$。
        

---

### 6. JAX/Equinox 实现要点 (代码速记)

- **高阶导数**：`jax.hessian(u)(x)` 用于 PINNs 残差。
    
- **ENG (经验自然梯度)**：
    
    - 计算 Jacobian 矩阵 $J$。
        
    - 构建 Gram 矩阵 $G = J^T J + \epsilon I$。
        
    - 更新方向：`lstsq(G, grad)`。
        
- **Armijo Line Search**：确保非线性优化每一步都在下降，通过 $\eta$ 的缩减（beta因子）寻找最优学习率。
    

---

### 💡 考场核心对比总结

|**方法**|**解决的问题**|**核心工具**|**稳定性/精度**|
|---|---|---|---|
|**Neural Galerkin**|时间演化|线性化网络 + Gram 矩阵求解|取决于时间格式，收敛快|
|**Semi-Lagrangian**|输运/平流方程|特征线溯源|**无 CFL 限制**|
|**Deep Feynman-Kac**|高维偏微分方程|SDE 模拟 + 倒向递归训练|适用于避开维度灾难|
|**Splitting**|复杂多项 PDE|算子解耦|引入分裂误差|

---

**这两页 A4 总结（Part 1 & Part 2）构成了 E. Franck 教授 SciML 2 课程的完整知识体系。**