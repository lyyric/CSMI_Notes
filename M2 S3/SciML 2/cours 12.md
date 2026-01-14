这份总结涵盖了 **SciML 2: 生成模型 (Generative Models)** 的核心理论，重点解析了 **VAE、流形流 (Normalizing Flows)** 以及目前最前沿的 **流匹配 (Flow Matching)** 及其在 PINNs 中的应用。

---

## 📄 SciML 2: 生成模型与流匹配核心速记 (A4版)

### 1. 生成模型基础

- **目标**：从数据 $\mathbf{x}_1, \dots, \mathbf{x}_n$ 中估计分布 $p_{target}(\mathbf{x})$，并能采样生成新样本。
    
- **损失函数**：最大化对数似然 $\arg \max_\theta \sum \log p_\theta(\mathbf{x}_i)$，等价于最小化 **KL 散度** $D_{KL}(p_{target} \| p_\theta)$。
    
- **混合高斯 (GMM)**：$p_\theta(\mathbf{x}) = \sum \pi_k \mathcal{N}(\mathbf{x} | \mu_k, \Sigma_k)$。
    
- **变分自编码器 (VAE)**：
    
    - **Encoder**：将数据映射到潜在分布 $q_\phi(\mathbf{z}|\mathbf{x})$。
        
    - **Decoder**：从潜在空间采样并还原数据 $p_\theta(\mathbf{x}|\mathbf{z})$。
        
    - **训练**：通过极大化 **ELBO**（证据下界）。
        

---

### 2. 流变换模型 (Normalizing Flows, NF)

- **核心原理**：通过可逆变换 $\mathbf{x} = f(\mathbf{z})$ 将简单分布 $p_0(\mathbf{z})$ 映射为复杂目标分布。
    
- **变量替换公式**：$p_{target}(\mathbf{x}) = p_0(f^{-1}(\mathbf{x})) | \det J_{f^{-1}}(\mathbf{x}) |$。
    
- **架构设计要求**：
    
    1. **双射 (Bijective)** 且易于求逆。
        
    2. **雅可比行列式** (Jacobian determinant) 计算代价低。
        
- **经典层实现**：
    
    - **NICE / RealNVP**：利用分块耦合 (Coupling Layer)，行列式为对角线乘积。
        
    - **Residual Flow**：$\mathbf{x} \mapsto \mathbf{x} + \mathbf{v}_\theta(\mathbf{x})$，需满足谱归一化（收缩映射）以保证可逆。
        
    - **Neural Spline Flows**：利用单调有理二次样条插值，实现更灵活的非线性映射。
        

---

### 3. 连续流模型 (Continuous Normalizing Flows)

- **基于 ODE**：$\frac{d\mathbf{x}(t)}{dt} = \mathbf{v}_\theta(\mathbf{x}(t), t)$。
    
- **连续性方程**：$\frac{\partial p}{\partial t} + \nabla \cdot (p \mathbf{v}_\theta) = 0$。
    
- **概率对数演化**：$\frac{d \log p(\mathbf{x}(t), t)}{dt} = -\nabla \cdot \mathbf{v}_\theta$。
    
- **痛点**：训练时需数值求解 ODE，计算代价极高。
    

---

### 4. 流匹配 (Flow Matching, FM)

- **革新**：无需解 ODE 即可训练连续流。
    
- **目标函数**：最小化向量场误差 $\mathcal{L}_{FM} = \mathbb{E} \| \mathbf{v}_{\theta, t}(\mathbf{x}_t) - \mathbf{v}^*_t(\mathbf{x}_t) \|^2$。
    
- **条件流匹配 (Conditional FM)**：
    
    - 定义从 $t=0$ 到 $t=1$ 的高斯路径：$p_t(\mathbf{x}|\mathbf{x}_1) = \mathcal{N}(\mathbf{x}; \mu_t(\mathbf{x}_1), \sigma_t^2 \mathbf{I})$。
        
    - **条件向量场**：$\mathbf{v}_t(\mathbf{x}|\mathbf{x}_1) = \frac{d\mu_t}{dt} + \frac{d\sigma_t/dt}{\sigma_t}(\mathbf{x} - \mu_t(\mathbf{x}_1))$。
        
- **训练算法**：采样 $t, \mathbf{x}_0, \mathbf{x}_1$，构造插值 $\mathbf{x}_t$，通过回归拟合显式已知的 $\mathbf{v}_t(\mathbf{x}|\mathbf{x}_1)$。
    

---

### 5. 生成模型在 PINNs 中的应用

- **自适应采样**：在残差 $r(\mathbf{u}_\theta)$ 较大的区域增加采样点。
    
- **流程**：
    
    1. 计算权重 $w_i \propto |r(\mathbf{u}_\theta(\mathbf{x}_i))|$。
        
    2. **加权自助采样 (Weighted Bootstrap)**：根据权重重采样点集。
        
    3. **训练生成模型**：利用 NF 或 FM 学习新点的分布，从而在下轮训练中提供更高效的采样。
        
- **极小极大优化**：$\min_\theta \max_{p} \int r(\mathbf{u}_\theta)^2 p(\mathbf{x}) d\mathbf{x}$。通过增强学习率和正则化控制，迫使残差分布趋于均匀。
    

---

### 💡 核心对比与考点

|**特性**|**VAE**|**Normalizing Flows**|**Flow Matching**|
|---|---|---|---|
|**密度估计**|隐式 (ELBO 近似)|**显式** (精确计算)|隐式 / ODE 积分恢复|
|**计算复杂度**|低|高 (Jacobian 约束)|**中 (训练极快)**|
|**可逆性**|不完全可逆|严格可逆层|通过 ODE 轨迹可逆|
|**采样速度**|快|快|慢 (需 ODE 积分)|

---

**这份总结涵盖了生成式 AI 在物理科学计算（SciML）中的前沿结合点，特别是如何利用流匹配来优化偏微分方程的采样效率。**