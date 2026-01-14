这份总结涵盖了 **SciML 2: 生成扩散模型 (Diffusion Models)** 的核心理论，重点解析了 **前向/后向过程、SDE 视角、快速推理算法以及在逆问题中的应用**。

---

## 📄 SciML 2: 扩散模型与随机插值速记 (A4版)

### 1. 扩散模型核心原理 (DDPM)

- **前向过程 (Forward)**：逐渐向数据 $\mathbf{x}_0$ 中添加高斯噪声，使其最终变为纯噪声 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$。
    
    - **单步公式**：$\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}$。
        
    - **跨步公式**：$p(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\hat{\alpha}_t}\mathbf{x}_0, (1-\hat{\alpha}_t)\mathbf{I})$。
        
- **后向过程 (Backward)**：学习反转噪声的过程 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$。
    
    - **学习目标**：通常预测噪声 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$。
        
    - **损失函数**：$\mathcal{L}_\theta = \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\hat{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\hat{\alpha}_t}\boldsymbol{\epsilon}, t) \|^2$。
        

---

### 2. SDE 与 Score Matching 视角

- **连续时间 SDE**：前向扩散可视为 Ornstein-Uhlenbeck 过程。
    
    - **前向 SDE**：$d\mathbf{x}_t = -\frac{1}{2}\beta(t)\mathbf{x}_t dt + \sqrt{\beta(t)} d\mathbf{W}_t$。
        
    - **反向 SDE**：$d\mathbf{x}_t = -[\frac{1}{2}\beta(t)\mathbf{x}_t + \beta(t)\nabla_\mathbf{x} \log p_t(\mathbf{x}_t)] dt + \sqrt{\beta(t)} d\mathbf{W}_t$。
        
- **Score Function (分值函数)**：$\nabla_\mathbf{x} \log p(x)$。扩散模型本质上是在学习分值函数，用于引导样本向高概率密度区域移动。
    
- **与 PDE 的联系**：演化过程遵循 **Fokker-Planck 方程**。
    

---

### 3. 高效推理 (Fast Inference)

- **问题**：标准采样（如 Euler-Maruyama）需要上千步。
    
- **概率流 ODE (Probability Flow ODE)**：
    
    - 扩散过程可以由一个确定性的 ODE 描述：$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}(\mathbf{x}_t, t)$。
        
- **指数积分器 (Exponential Integrator)**：
    
    - 利用解析解处理线性部分，对非线性部分（神经网络预测值）进行一阶或二阶外推。
        
    - **优势**：在 10-20 步内即可获得高质量采样，极其稳定。
        

---

### 4. 逆问题与条件引导 (Guidance)

- **目标**：给定观测 $\mathbf{c} = \mathcal{A}(\mathbf{x})$，生成符合观测的 $\mathbf{x}$。
    
- **条件分值 (Bayes)**：$\nabla_\mathbf{x} \log p_t(\mathbf{x}_t | \mathbf{c}) = \nabla_\mathbf{x} \log p_t(\mathbf{x}_t) + \lambda \nabla_\mathbf{x} \log p_t(\mathbf{c} | \mathbf{x}_t)$。
    
- **常用引导策略**：
    
    - **Score ALD/SDE**：在每一步采样中通过投影算子向满足 $\mathcal{A}(\mathbf{x}) = \mathbf{c}$ 的方向修正。
        
    - **DPS (Diffusion Posterior Sampling)**：直接对测量误差项求梯度 $\nabla_{\mathbf{x}_t} \| \mathbf{c} - \mathcal{A}(\mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t]) \|^2$。
        
    - **Diffusion-PDE**：结合物理残差 $\mathcal{L}_{PDE}$ 进行引导，生成满足物理规律的解。
        

---

### 5. 随机插值 (Stochastic Interpolants)

- **统一框架**：一并涵盖了 Flow Matching 和 Diffusion。
    
- **插值定义**：$\mathbf{x}_t = I(\mathbf{x}_0, \mathbf{x}_1, t) + \gamma(t)\mathbf{z}$。
    
    - 当 $\gamma(t)=0$ 时退化为确定性的 Flow Matching。
        
    - 当 $\gamma(t)>0$ 时即为扩散模型。
        
- **学习内容**：同时学习速度场 $\mathbf{b}(\mathbf{x}, t)$ 和分值函数 $\mathbf{s}(\mathbf{x}, t)$。
    

---

### 💡 考点对比总结表

|**特性**|**DDPM (标准扩散)**|**Flow Matching**|**Stochastic Interpolant**|
|---|---|---|---|
|**路径类型**|随机 (SDE)|确定性 (ODE)|统一路径|
|**训练效率**|快 (噪声预测)|极快 (向量场回归)|中等 (多目标学习)|
|**采样复杂度**|高 (需多步迭代)|中 (ODE 积分)|灵活 (ODE 或 SDE)|
|**逆问题应用**|强 (鲁棒性高)|较强|理论最完备|

---

### 🚀 考前突击口诀

1. **扩散即加噪**：$\mathbf{x}_t$ 永远是 $\mathbf{x}_0$ 和高斯噪声的线性组合。
    
2. **分值即方向**：Score 指向数据存在的方向。
    
3. **逆问题靠引导**：在反向步中加入测量的梯度，让噪声“长成”我们要的样子。
    
4. **插值定天下**：Flow Matching 是没有噪声扰动的扩散。
    

---

**这份总结将复杂的随机微积分与前沿的 AI 算法结合，是理解生成式 SciML 核心逻辑的必备参考。**