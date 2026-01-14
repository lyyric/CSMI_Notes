这份总结涵盖了 **神经算子的谱方法 (Spectral Approach)**，重点介绍了欧几里得空间下的 **FNO** 及其变体，以及流形和图上的 **谱图神经网络 (Spectral GNN)**。

---

## 📄 SciML 2: 谱神经算子与流形学习核心速记 (A4版)

### 1. 谱表征与算子学习 (Spectral Basics)

- **核心思想**：利用希尔伯特基（如 Fourier, Legendre, Chebyshev）将空间卷积转化为谱域的点乘 $\mathbf{u} = \mathbf{K}\mathbf{f}$。
    
- **优势**：学习谱核 $T_s$ 比空间核 $T_k$ 更高效，且天然具备分辨率无关性。
    
- **Fourier 基**：$\exp(i2\pi jx)$ 是拉普拉斯算子 $-\Delta$ 的特征函数。
    

---

### 2. 欧几里得空间：FNO 及其扩展

- **FNO (Fourier Neural Operator)**：
    
    - **公式**：$v_{l+1} = \sigma(\mathcal{F}^{-1}(K \cdot \mathcal{F}(v_l)) + W v_l)$。
        
    - **组件**：$\mathcal{F}$ (FFT 变换)，$K$ (学习复数滤波器)，$W$ (空间局部线性变换)。
        
    - **局限**：仅限于笛卡尔网格和周期性边界。
        
- **变体与扩展**：
    
    - **Factorized FNO**: 利用秩分解，将 $d$ 维卷积拆分为 $d$ 个 1D 卷积，降低显存 $O(r^d) \to O(dr)$。
        
    - **Geo-FNO**: 通过坐标映射 $\phi$ 处理复杂几何。
        
    - **GINO (Graph-Informed NO)**：利用局部平均算子 $\mathcal{E}_\epsilon$ 在物理流形与潜在欧几里得空间（网格）间转换。
        

---

### 3. 流形与图谱理论 (Manifold & Graph Spectral)

- **Laplace-Beltrami 算子 (LBO)**：在流形 $\mathcal{M}$ 上定义特征基底 $-\Delta \phi_j = \lambda_j \phi_j$。它是 Fourier 基在通用几何上的推广。
    
- **图拉普拉斯 (Graph Laplacian)**：
    
    - **非规范化**: $L = D - W$。
        
    - **随机游走**: $L_{rw} = I - D^{-1}W$。
        
    - **收敛性**：当 $n \to \infty$ 且采样密度 $p(x)$ 均匀时，图拉普拉斯收敛于 LBO。若 $p(x)$ 不均匀，需使用 **Markov 归一化 (Algorithm $\alpha=1$)** 来消除密度偏差。
        

---

### 4. 谱图神经网络 (Spectral GNN)

- **图卷积层**：$g = \sigma(\Phi K \Phi^T f + W f)$，其中 $\Phi$ 是图拉普拉斯的特征向量。
    
- **局部性与多项式滤波器**：
    
    - 直接对特征分解计算量巨大 $O(n^3)$。
        
    - **Chebyshev GCN**: 使用 $k$ 阶切比雪夫多项式近似滤波函数 $g(\lambda)$。
        
    - **公式**：$H(L)f = \sum \theta_k L^k f$。
        
    - **空间物理意义**：$L^k$ 表示在图中进行 $k$ 步跳跃的信息聚合，实现了谱域定义的**空间局部性**。
        
- **置换不变性 (Permutation Invariance)**：图滤波器满足 $H(PSP^T)(Pf) = P(H(S)f)$，确保节点编号顺序不影响结果。
    

---

### 5. 关键公式速记

- **Ito/Fourier 卷积定理**：$\mathcal{F}(k * f) = \mathcal{F}(k) \cdot \mathcal{F}(f)$。
    
- **Markov Normalization (权重修正)**：
    
    1. $Q_{ii} = \sum W_{ij}$
        
    2. $W_{new} = Q^{-1} W Q^{-1}$ (消除密度 $p$ 的影响)。
        
- **GCN 简化层**：$\tilde{D}^{-1/2} \tilde{W} \tilde{D}^{-1/2} \theta$。
    

---

### 💡 考前提醒 (Tips)

1. **FNO 为什么有效？** 因为很多 PDE 的解在频域是低秩的（光滑解对应低频分量）。
    
2. **为什么需要图算子？** 因为在非结构化网格或流形上，没有标准的 FFT，只能依靠图拉普拉斯特征分解。
    
3. **性能瓶颈**：FNO 瓶颈在 FFT 维数，GNO 瓶颈在特征向量求解。
    

---

**本页涵盖了从欧氏 FNO 到流形 LBO 的谱学习全路径，重点关注“如何通过谱变换将全局积分简化为局部点乘”。**