这份总结涵盖了 **神经算子 (Neural Operators)** 的核心理论、离散化、以及主要架构（CNN, GreenNet, DeepONet, GNN），专为 SciML 2 考试复习设计。

---

## 📄 SciML 2: 神经算子 (Neural Operators) 核心速记表 (A4版)

### 1. 算子学习原理 (Operator Learning)

- **目标**：学习输入空间（如源项 $f$）到输出空间（如 PDE 解 $u$）的映射 $\mathcal{G}: \mathcal{A} \to \mathcal{U}$。
    
- **理论基础 (Green's Function)**：对于线性 PDE $L u = f$，解可表示为积分变换 $u(\mathbf{x}) = \int_\Omega G(\mathbf{x}, \mathbf{y})f(\mathbf{y})d\mathbf{y}$。
    
- **神经算子结构**：由**局部层**（Pointwise: $v_{l+1}(x) = \sigma(W v_l(x))$）和**非局部层**（Integral kernel: $v_{l+1}(x) = \int \kappa_\theta(x, y) v_l(y) dy$）组合而成。
    
- **分辨率无关性 (Resolution Invariance)**：神经算子是连续算子的收敛离散化。若离散格式满足 $\lim_{h \to 0} \|\mathcal{B}_\theta(v) - \mathcal{B}_h^\theta(v)\| = 0$，则在低分辨率训练的模型可直接推断高分辨率数据。
    

---

### 2. 卷积神经网络 (CNN) 与 对称性

- **平移等变性 (Equivariance)**：若输入平移，输出随之平移 $f(g \cdot x) = g \cdot f(x)$（如分割）。
    
- **平移不变性 (Invariance)**：输出不随输入平移改变 $f(g \cdot x) = f(x)$（如分类）。
    
- **卷积层**：是平移等变性的线性算子。其核函数满足 $\kappa(x, y) = \kappa_c(x - y)$。
    
- **局限性**：标准 CNN 不是严格的神经算子，因为其核尺寸（Pixel size）随网格步长 $h$ 改变，无法收敛到固定物理半径的积分算子。
    

---

### 3. 核方法架构 (Kernel-based)

- **GreenNet**：直接模拟格林函数积分 $u(x_i) = \frac{1}{n} \sum \kappa_\theta(x_i, x_j) f(x_j) + h_\theta(x_i)$。
    
    - **修正**：需引入点密度 $p(x_j)$ 权重以保证离散收敛性。
        
    - **激活函数**：使用 **Rational Activation (Padé)** 更好地捕捉格林函数的奇异性。
        
- **DeepONet**：基于积分算子的**低秩分解**（Schmidt 理论）。
    
    - **BranchNet**：编码输入函数（通过传感器 points $y_m$）生成系数 $c_j$。
        
    - **TrunkNet**：编码输出坐标 $x$ 生成基函数 $\phi_j(x)$。
        
    - **公式**：$u(x) = \sum_{j=1}^r c_j(\text{input}) \phi_j(x)$。
        
- **复杂度优化**：全核评估为 $O(n^2)$，低秩近似（DeepONet/PCA）可降至 $O(nr)$。
    

---

### 4. 空间图神经网络 (Spatial GNN)

- **图定义**：$G = [V, E, W]$。$V$ 节点，$E$ 边，$W$ 权重。
    
- **Message Passing (MPNN)**：
    
    1. **聚合 (Aggregation)**：$m_i = \sum_{j \in N(i)} w_{ij} \phi_\theta(h_i, h_j, e_{ij})$。
        
    2. **更新 (Update)**：$h'_i = \gamma_\theta(h_i, m_i)$。
        
- **图神经算子 (GNO)**：
    
    - 为使 GNN 成为 NO，聚合半径 $r$ 必须在物理空间固定，且需补偿点密度 $p(v_j)$。
        
- **坐标基方法 (MoNet/Anisotropic)**：使用极坐标 $(\rho, \theta)$ 或高斯核对邻域进行加权。
    

---

### 5. 训练与物理正则化 (PINO)

- **数据驱动**：$\mathcal{L}_{data} = \frac{1}{K} \sum \| \mathcal{G}_\theta(a_k) - u_k \|^2$。
    
- **物理约束 (Physics-Informed Neural Operator)**：
    
    - 将神经算子的输出代入 PDE 计算残差 $\mathcal{L}_{phy} = \| L(\mathcal{G}_\theta(a)) - f \|^2$。
        
    - **作用**：在小样本情况下利用物理定律引导学习，提高算子的泛化能力。
        

---

### 💡 核心架构对比表

|**架构**|**输入域**|**归纳偏置 (Inductive Bias)**|**计算效率**|
|---|---|---|---|
|**CNN**|笛卡尔网格|平移等变、局部性|极高 (GPU 优化)|
|**GreenNet**|任意点集|格林积分理论|$O(n^2)$ (高)|
|**DeepONet**|传感器点|基函数分解 (Trunk/Branch)|$O(n \times r)$ (中)|
|**GNN**|图/非结构网格|拓扑连接性、置换等变|$O(E)$ (随边线性)|

---

**本页涵盖了从格林函数到各种离散神经算子架构的完整逻辑，适合作为神经算子章节的考场速记。**