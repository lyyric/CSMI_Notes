问题 1：POD / SVD (奇异值分解)
这里我们需要对数据集 a(x), f(x) 和 u(x) 进行奇异值分解（SVD）。
在 JAX (和 NumPy) 中，SVD 将矩阵 M 分解为 U, S, Vt。
 * Vt (Right Singular Vectors)：包含空间模态（也就是 POD 模态/基向量）。
 * S (Singular Values)：奇异值，表示每个模态的能量/重要性。
 * U (Left Singular Vectors)：包含每个样本在这些模态上的投影系数（通常结合 S 使用）。
<!-- end list -->
# --- 问题 1 解答 ---

# 对 a(x) 进行 SVD
# a_x_batch_stat shape: (batch_size, n_points)
U_a, S_a, Vt_a = jnp.linalg.svd(a_x_batch_stat, full_matrices=False)

# 对 f(x) 进行 SVD
U_f, S_f, Vt_f = jnp.linalg.svd(f_x_batch_stat, full_matrices=False)

# 对 u(x) 进行 SVD
U_u, S_u, Vt_u = jnp.linalg.svd(u_batch_stat, full_matrices=False)

print("SVD 计算完成。")
print(f"a(x) 维度 -> U: {U_a.shape}, S: {S_a.shape}, Vt: {Vt_a.shape}")
print(f"u(x) 维度 -> U: {U_u.shape}, S: {S_u.shape}, Vt: {Vt_u.shape}")

问题 2：编写 MLP 类
这里我们需要使用 equinox 库编写一个多层感知机（MLP）。
该类需要接收输入维度、隐藏层维度列表和输出维度。它将构建一系列线性层（Linear Layers），并在层之间使用 ReLU 激活函数。
# --- 问题 2 解答 ---

class MLP(eqx.Module):
    layers: list

    def __init__(self, key, input_dim, hidden_dims, output_dim):
        # 构建完整的维度列表：[输入, 隐藏1, 隐藏2, ..., 输出]
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = []
        
        # 为每一层生成独立的随机 key 用于初始化权重
        keys = jr.split(key, len(dims) - 1)
        
        for i in range(len(dims) - 1):
            self.layers.append(
                eqx.nn.Linear(dims[i], dims[i+1], key=keys[i])
            )

    def __call__(self, x):
        # 对除最后一层外的所有层应用 Linear -> ReLU
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        
        # 最后一层不加激活函数（用于回归任务）
        return self.layers[-1](x)

问题 3：构建 FNO (傅里叶神经算子)
这是最核心的部分。FNO 1D 主要由两个部分组成：
 * SpectralConv1d (谱卷积层)：执行 FFT \to 在频域做线性变换（过滤高频，保留低频模式） \to IFFT。
 * FNO1d (主模型)：包含由“Lifting层”（升维）、多个“Fourier层”（谱卷积+残差连接）和“Projection层”（投影回输出维度）组成的架构。
<!-- end list -->
# --- 问题 3 解答 ---

class SpectralConv1d(eqx.Module):
    weights: jax.Array
    in_channels: int
    out_channels: int
    modes: int

    def __init__(self, key, in_channels, out_channels, modes):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # 初始化复数权重
        # Shape: (in, out, modes)
        scale = (1 / (in_channels * out_channels))
        self.weights = scale * jr.normal(key, (in_channels, out_channels, modes), dtype=jnp.complex64)

    def __call__(self, x):
        # x shape: (in_channels, x_grid)
        
        # 1. 傅里叶变换 (使用 rfft 因为输入是实数)
        x_ft = jnp.fft.rfft(x, axis=-1)
        
        # 2. 截断模态 (只保留最低的 'modes' 个频率)
        modes_to_keep = min(self.modes, x_ft.shape[-1])
        x_ft_modes = x_ft[:, :modes_to_keep]
        
        # 3. 复数乘法 (在频域混合通道)
        # einsum 用于张量收缩: 
        # 'ix' (输入频域数据), 'iox' (权重) -> 'ox' (输出频域数据)
        # i=输入通道, o=输出通道, x=频率模态
        out_ft = jnp.einsum("ix,iox->ox", x_ft_modes, self.weights[:, :, :modes_to_keep])
        
        # 4. 傅里叶逆变换 (恢复到空间域)
        # 必须指定 n=x.shape[-1] 以确保输出尺寸与输入一致
        return jnp.fft.irfft(out_ft, n=x.shape[-1], axis=-1)

class FNO1d(eqx.Module):
    lifting: eqx.nn.Conv1d
    layers: list
    projection: eqx.nn.Conv1d
    
    def __init__(self, key, in_channels, out_channels, modes, width, n_layers):
        keys = jr.split(key, n_layers + 2)
        
        # Lifting 层: 将输入通道数映射到高维特征空间 (width)
        # kernel_size=1 相当于对每个点做全连接
        self.lifting = eqx.nn.Conv1d(in_channels, width, kernel_size=1, key=keys[0])
        
        self.layers = []
        for i in range(n_layers):
            key_s, key_w = jr.split(keys[i+1])
            self.layers.append({
                'spectral': SpectralConv1d(key_s, width, width, modes), # 谱路径 (全局特征)
                'skip': eqx.nn.Conv1d(width, width, kernel_size=1, key=key_w) # Skip路径 (局部特征/线性变换)
            })
            
        # Projection 层: 将高维特征映射回输出通道 (u(x) 通常是 1 个通道)
        self.projection = eqx.nn.Conv1d(width, out_channels, kernel_size=1, key=keys[-1])

    def __call__(self, x):
        # x shape: (in_channels, grid_size)
        
        # 升维
        x = self.lifting(x)
        
        for layer in self.layers:
            # 计算谱卷积结果
            x_s = layer['spectral'](x)
            # 计算残差连接/线性变换结果
            x_w = layer['skip'](x)
            
            # 相加并激活 (GELU 通常比 ReLU 效果好)
            x = jax.nn.gelu(x_s + x_w)
            
        # 投影回输出
        x = self.projection(x)
        return x

问题 4：Loss 函数与训练步骤
最后，我们需要定义损失函数和训练的一步更新操作。
 * fno_loss_fn：使用 vmap 将模型并行应用到整个 Batch 上，计算 MSE。
 * fno_train_step：计算梯度并更新模型参数。
<!-- end list -->
# --- 问题 4 解答 ---

@eqx.filter_jit
def fno_loss_fn(model, x, y):
    """
    计算 FNO 的损失函数
    x shape: (batch, in_channels, grid)
    y shape: (batch, out_channels, grid)
    """
    # 使用 vmap 在 batch 维度 (轴 0) 上向量化应用模型
    pred = jax.vmap(model)(x)
    
    # 均方误差 (MSE)
    return jnp.mean((pred - y) ** 2)

@eqx.filter_jit
def fno_train_step(model, opt_state, x, y, optimizer):
    # 计算 Loss 和 梯度
    loss, grads = eqx.filter_value_and_grad(fno_loss_fn)(model, x, y)
    
    # 计算参数更新量
    updates, opt_state = optimizer.update(grads, opt_state, model)
    
    # 应用更新到模型
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss

下一步建议
您可以将这些代码块填入 Notebook 后，运行后续的训练循环代码。如果您在调整超参数（如 modes 或 width）时遇到显存不足或收敛问题，请随时告诉我！
