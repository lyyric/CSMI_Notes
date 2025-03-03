# 1. 引言

在许多物理问题中，问题的解用球坐标描述往往更自然；比如在医学成像（断层扫描）中，我们需要从各个方向获取图像数据来重构体内结构。因此，我们常常在球面上研究函数和信号，并利用球面谐波来展开、重构这些函数。

---

# 2. 球坐标与笛卡尔坐标的转换

在三维空间中，球坐标由三个参数表示：  
- $\rho$ 表示半径，  
- $\theta$ 表示极角（0到$\pi$），  
- $\varphi$ 表示方位角（0到$2\pi$）。

转换公式为：
$$
\begin{aligned}
x &= \rho\,\sin(\theta)\cos(\varphi),\\
y &= \rho\,\sin(\theta)\sin(\varphi),\\
z &= \rho\,\cos(\theta).
\end{aligned}
$$

对应的 Python 代码如下：

```python
def pol_to_cart(rho, theta, phi):
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return x, y, z
```

---

# 3. 在球面上绘制函数

我们可以在球面上绘制函数，展示函数的值如何分布。下面给出一个辅助类 `SpherePlot`，包含几种绘图方法：

- **scatter_color**：用散点图显示球面上每个点对应的函数值，颜色根据数值变化。
- **surface_color**：用表面图显示球面上函数的颜色变化。
- **surface_radius_from_0** 和 **surface_radius_from_1**：通过调整球面上每个点的半径，使得函数值直接体现在图形几何上（分别不做归一化和做归一化）。

```python
class SpherePlot:
    @staticmethod
    def add_labels_xyz(ax):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    @staticmethod
    def scatter_color(ax, theta, phi, value):
        SpherePlot.add_labels_xyz(ax)
        x, y, z = pol_to_cart(1, theta, phi)
        ax.scatter(x, y, z, c=value, cmap=cm.coolwarm)

    @staticmethod
    def surface_color(ax, theta_phi_theta, theta_phi_phi, theta_phi_value):
        SpherePlot.add_labels_xyz(ax)
        x, y, z = pol_to_cart(1, theta_phi_theta, theta_phi_phi)
        scmap = plt.cm.ScalarMappable(cmap=cm.coolwarm)
        ax.plot_surface(x, y, z, facecolors=scmap.to_rgba(theta_phi_value), shade=False)

    @staticmethod
    def surface_radius_from_0(ax, theta_phi_theta, theta_phi_phi, theta_phi_value, add_color=True):
        SpherePlot.add_labels_xyz(ax)
        x, y, z = pol_to_cart(theta_phi_value, theta_phi_theta, theta_phi_phi)
        if add_color:
            scmap = plt.cm.ScalarMappable(cmap=cm.coolwarm)
            ax.plot_surface(x, y, z, facecolors=scmap.to_rgba(theta_phi_value), shade=False)
        else:
            ax.plot_surface(x, y, z)

    @staticmethod
    def surface_radius_from_1(ax, theta_phi_theta, theta_phi_phi, theta_phi_value, add_color=True):
        SpherePlot.add_labels_xyz(ax)
        # 对值进行归一化，使其落在 [0,1] 区间内
        theta_phi_value = theta_phi_value - np.min(theta_phi_value)
        theta_phi_value /= np.max(theta_phi_value)
        x, y, z = pol_to_cart(1 + theta_phi_value, theta_phi_theta, theta_phi_phi)
        if add_color:
            scmap = plt.cm.ScalarMappable(cmap=cm.coolwarm)
            ax.plot_surface(x, y, z, facecolors=scmap.to_rgba(theta_phi_value), shade=False)
        else:
            ax.plot_surface(x, y, z)
```

另外，生成球面上的网格（$\theta,\varphi$ 坐标）函数如下：

```python
def mesh_grid_theta_phi(n_theta, n_phi, endpoint=True):
    theta, phi = np.meshgrid(np.linspace(0, np.pi, n_theta),
                             np.linspace(0, 2 * np.pi, n_phi, endpoint=endpoint),
                             indexing="ij")
    return theta, phi
```

我们可以定义一个函数，例如：
$$
u(\theta, \varphi) = x \, y + 0.5\, z,
$$
其中 $x,y,z$ 由 `pol_to_cart` 得到：

```python
def u_fn(theta, phi):
    rho = 1
    x, y, z = pol_to_cart(rho, theta, phi)
    return x * y + 0.5 * z
```

利用以上函数，我们可以生成网格并绘图：

```python
n_theta = 20
n_phi = 30
theta, phi = mesh_grid_theta_phi(n_theta, n_phi)
u = u_fn(theta, phi)

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

SpherePlot.scatter_color(ax1, theta, phi, u)
SpherePlot.surface_color(ax2, theta, phi, u)
SpherePlot.surface_radius_from_0(ax3, theta, phi, u)
SpherePlot.surface_radius_from_1(ax4, theta, phi, u)
```

此外，我们也可以直接用 `plot_surface` 绘制 $(\theta,\varphi,u)$ 的图形：

```python
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("θ")
ax.set_ylabel("ϕ")
ax.plot_surface(theta, phi, u, cmap=cm.coolwarm)
```

---

# 4. 球面函数连续性的条件

> **练习题 1**：  
> **问题**：一个定义在球面上的函数 $f(\theta, \varphi)$ 要在球面上连续，需要满足哪些条件？  
> **解答**：  
> - 对于 $\theta \in [0, \pi]$ 和 $\varphi \in [0, 2\pi]$，函数必须连续；  
> - 在 $\varphi$ 上要求周期性，即 $f(\theta, \varphi) = f(\theta, \varphi + 2\pi)$；  
> - 在极点处（$\theta = 0$ 和 $\theta = \pi$），由于所有不同的 $\varphi$ 实际上对应同一个点，函数在这些点必须独立于 $\varphi$（或者满足适当的匹配条件），保证连续性。

---

# 5. 球面谐波基础

调和函数是满足 Laplace 方程的函数：
$$
\Delta f(x, y, z) = 0.
$$
在球坐标下，通过变量分离法可得到形式为
$$
f(r, \theta, \varphi) = R_l(r) \, P_l^m(\cos \theta) \, e^{im\varphi},
$$
其中：
- $l \in \mathbb{N}$，对于每个 $l$， $m \in \{-l, -l+1, \dots, l\}$。
- $P_l^m$ 是关联 Legendre 函数，定义如下：
  - 当 $m = 0$ 时，$P_l^0(x) = \frac{1}{2^l l!} \frac{d^l}{dx^l} \bigl[(x^2-1)^l\bigr]$；
  - 当 $m > 0$ 时，
    $$
    P_l^m(x)= (-1)^m (1-x^2)^{m/2} \frac{d^m}{dx^m} P_l(x);
    $$
  - 当 $m < 0$ 时，可利用公式：
    $$
    P_l^{-m}(x) = (-1)^m \frac{(l-m)!}{(l+m)!} \, P_l^m(x).
    $$
- 径向部分的解为：
$$
R(r) = A\,r^l + B\,r^{-(l+1)},
$$
其中 $A, B$ 为常数。

将角度部分归一化后，我们得到球面谐波的定义：
$$
Y_{l,m}(\theta, \varphi) = K_{l,m}\, P_l^m(\cos\theta) \, e^{im\varphi},
$$
归一化系数为：
$$
K_{l,m} = \sqrt{\frac{2l+1}{4\pi} \frac{(l-|m|)!}{(l+|m|)!}}.
$$
这组函数构成了球面上所有函数的正交完备基，也就是说，任何定义在球面上的连续函数 $f(\theta, \varphi)$ 都可以展开为：
$$
f(\theta, \varphi) = \sum_{l=0}^\infty \sum_{m=-l}^{l} c_{lm}\, Y_{l,m}(\theta, \varphi),
$$
其中展开系数由
$$
c_{lm} = \int_0^{2\pi} d\varphi \int_0^\pi d\theta\, \sin\theta \, f(\theta, \varphi)\, \overline{Y_{l,m}(\theta, \varphi)}
$$
给出。注意：虽然 $Y_{l,m}$ 为复值函数，但可以通过组合得到一组实值基（例如利用 $\cos(m\varphi)$ 和 $\sin(m\varphi)$）。

---

# 6. 使用 Dipy 进行球面谐波分解

利用 [dipy](https://dipy.org/) 库，我们可以对球面上的函数进行球面谐波分解与重构。下面给出一个类 `SphericalHarmonicDecomposer` 的实现，它利用 dipy 中的 `sf_to_sh` 和 `sh_to_sf` 函数完成分解和重构。

```python
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf

class SphericalHarmonicDecomposer:
    def __init__(self, sh_order_max, full_basis, basis_type="descoteaux07"):
        self.sh_order_max = sh_order_max
        self.basis_type = basis_type
        self.full_basis = full_basis

    def compute_coef(self, theta, phi, val):
        # 构造球面采样点
        sphere = Sphere(theta=theta, phi=phi)
        self.sh_coeffs = sf_to_sh(val, sphere, sh_order_max=self.sh_order_max,
                                   basis_type=self.basis_type, full_basis=self.full_basis)

    def reconstruction(self, theta, phi):
        sphere = Sphere(theta=theta, phi=phi)
        val = sh_to_sf(self.sh_coeffs, sphere, sh_order_max=self.sh_order_max,
                       basis_type=self.basis_type, full_basis=self.full_basis)
        return val

    @staticmethod
    def fn_on_sphere(theta, phi):
        rho = 1
        x, y, z = pol_to_cart(rho, theta, phi)
        return x * y
```

使用示例：

```python
n_theta = 20
n_phi = 30
theta, phi = mesh_grid_theta_phi(n_theta, n_phi)
value = SphericalHarmonicDecomposer.fn_on_sphere(theta, phi)

# 绘制原始球面函数
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
SpherePlot.scatter_color(ax, theta, phi, value)

# 扁平化数据，计算球面谐波系数
theta_flat = theta.flatten()
phi_flat = phi.flatten()
value_flat = value.flatten()

decomposer = SphericalHarmonicDecomposer(sh_order_max=10, full_basis=True)
decomposer.compute_coef(theta_flat, phi_flat, value_flat)

# 在更细的网格上重构函数
n_theta_new, n_phi_new = 2 * n_theta, 2 * n_phi
theta_new, phi_new = mesh_grid_theta_phi(n_theta_new, n_phi_new)
theta_new_flat = theta_new.flatten()
phi_new_flat = phi_new.flatten()
value_pred = decomposer.reconstruction(theta_new_flat, phi_new_flat)

# 重构结果整形
theta_new = theta_new_flat.reshape([n_theta_new, n_phi_new])
phi_new = phi_new_flat.reshape([n_theta_new, n_phi_new])
value_pred = value_pred.reshape([n_theta_new, n_phi_new])

# 绘制重构后的结果
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
SpherePlot.scatter_color(ax, theta_new, phi_new, value_pred)
```

此外，还可以展示单一谐波的形状。比如，将某个谐波系数设为 1（其他置 0），然后重构出该单一谐波：

```python
# 设置网格更密集
n_theta, n_phi = 100, 150
theta, phi = mesh_grid_theta_phi(n_theta, n_phi)
theta_flat = theta.flatten()
phi_flat = phi.flatten()

# 查看系数形状
print(decomposer.sh_coeffs.shape)

# 将所有系数置 0，仅令某一项（例如第 50 项）为 1
decomposer.sh_coeffs = np.zeros(decomposer.sh_coeffs.shape)
decomposer.sh_coeffs[50] = 1

one_harmonic = decomposer.reconstruction(theta_flat, phi_flat)
one_harmonic = one_harmonic.reshape([n_theta, n_phi])

# 绘图展示单一谐波
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
SpherePlot.scatter_color(ax, theta, phi, one_harmonic)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
SpherePlot.surface_radius_from_0(ax, theta, phi, np.abs(one_harmonic), add_color=False)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
SpherePlot.surface_radius_from_1(ax, theta, phi, one_harmonic, add_color=False)
```

---

# 7. 球面谐波与调和函数

需要注意的是：  
- **球面谐波**是通过在 $\mathbb{R}^3$ 中求调和函数（满足 $\Delta f=0$）并进行变量分离后，在固定半径（通常为 1）上得到的函数。  
- 它们虽然称为“谐波”，但严格来说并非 $\mathbb{R}^3$ 中的调和函数，因为后者依然是三个变量的函数。

球坐标下，Laplace 算子可分解为径向部分和角度部分：
$$
\dot \Delta f = \underbrace{\frac{\partial^2 f}{\partial r^2} + \frac{2}{r}\frac{\partial f}{\partial r}}_{\Delta_r f} + \underbrace{\frac{1}{r^2}\frac{\partial^2 f}{\partial \theta^2} + \frac{1}{r^2 \tan\theta}\frac{\partial f}{\partial \theta} + \frac{1}{r^2\sin^2\theta}\frac{\partial^2 f}{\partial \varphi^2}}_{\Delta_{\theta,\varphi} f}.
$$
在单位球面（$r=1$）上，球面谐波正是角度部分的本征函数，即它们满足：
$$
\Delta_{\theta,\varphi} Y_{l,m} = \lambda \, Y_{l,m},
$$
通常本征值 $\lambda = -l(l+1)$。

---

# 8. Poisson 问题及调和函数的构造

在调和函数理论中，有两个经典问题：

### 8.1. 整个 $\mathbb{R}^3$ 上有界的调和函数

**练习题 2**：  
**问题**：在 $\mathbb{R}^3$ 上有哪些调和函数是有界的？在单位球内部有界的调和函数？在单位球外部有界的调和函数？  
**答案提示**：  
- 根据 Liouville 定理，整个 $\mathbb{R}^3$ 上有界的调和函数只有常数函数；  
- 在单位球内或单位球外有界的调和函数可以有非平凡解，其构造方法通常采用变量分离法，利用前面得到的形式 $R(r) P_l^m(\cos\theta)e^{im\varphi}$，并通过选择合适的径向函数（例如在内部取 $R(r) = A\,r^l$，在外部取 $R(r)=B\,r^{-(l+1)}$）来保证有界性。

### 8.2. 利用边界数据构造调和函数

**练习题 3**：  
**问题**：给定球面上的连续函数 $\Gamma(\theta,\varphi)$，请证明可以构造出一个调和函数 $f(r,\theta,\varphi)$，使得：
- 在单位球内部 $(r \leq 1)$ 有界，并且当 $r = 1$ 时 $f(1,\theta,\varphi)=\Gamma(\theta,\varphi)$；
- 同理，也可构造出一个在单位球外部有界的调和函数，其在 $r = 1$ 时取值为 $\Gamma$。

**构造方法**：  
利用 Poisson 积分公式可以给出明确的表达：
- 对于单位球内部，有：
  $$
  f(r,\theta,\varphi)= \frac{1-r^2}{4\pi} \int_{S^2} \frac{\Gamma(\theta',\varphi')}{(1-2r\cos\gamma + r^2)^{3/2}} \, d\Omega',
  $$
  其中 $\cos\gamma$ 表示球面上两点之间的夹角余弦，$d\Omega' = \sin\theta'\, d\theta'\, d\varphi'$ 为面积元。  
- 对于单位球外部，可通过反演变换（或类似构造）得到满足要求的解。

---

# 9. 随机点上的球面函数与均匀采样

为了验证前面的分解器不仅适用于规则网格，也适用于随机采样，我们可以随机生成球面上的点，并计算函数值：

```python
theta_rand = np.random.uniform(0, np.pi, 1000)
phi_rand = np.random.uniform(0, 2*np.pi, 1000)
value_rand = SphericalHarmonicDecomposer.fn_on_sphere(theta_rand, phi_rand)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
SpherePlot.scatter_color(ax, theta_rand, phi_rand, value_rand)
```

这说明无论采样点如何分布，只要满足球面测度（积分时需要考虑 Jacobian $\sin\theta$），我们的方法均可应用。

> [!note]
**补充说明：**  
> 在球坐标中，由 $(x,y,z) \to (r,\theta,\varphi)$ 的变换，其 Jacobian 为：
> $$
> dx\,dy\,dz = r^2 \sin\theta \, dr\, d\theta\, d\varphi.
> $$
> 这个因子在积分求面积或体积时起到了关键作用，也在计算球面上函数的正交性时体现出来。

---

# 10. 二维极坐标下的 Laplace 算子

在二维（$\mathbb{R}^2$）中，使用极坐标 $(r,\theta)$ 时，Laplace 算子写作：
$$
\Delta_{r,\theta} f = \frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial f}{\partial r}\right) + \frac{1}{r^2}\frac{\partial^2 f}{\partial \theta^2}
$$
也可以展开为：
$$
\Delta_{r,\theta} f = \frac{\partial^2 f}{\partial r^2} + \frac{1}{r}\frac{\partial f}{\partial r} + \frac{1}{r^2}\frac{\partial^2 f}{\partial \theta^2}.
$$
我们可以将其分解为径向部分和角度部分：
$$
\Delta_{r,\theta} f = \underbrace{\left(\frac{\partial^2 f}{\partial r^2} + \frac{1}{r}\frac{\partial f}{\partial r}\right)}_{\Delta_r f} + \underbrace{\left(\frac{1}{r^2}\frac{\partial^2 f}{\partial \theta^2}\right)}_{\Delta_\theta f}.
$$

**练习题 4**：  
**问题**：设 $f(r,\theta) = R(r)S(\theta)$ 且满足 $\Delta_{r,\theta} f = 0$，证明角函数 $S(\theta)$ 满足的常微分方程，并求其通解。  
**解答思路**：  
1. 将 $f(r,\theta) = R(r)S(\theta)$ 代入 Laplace 方程，利用变量分离法得到：
   $$
   \frac{r^2 R''(r) + r R'(r)}{R(r)} + \frac{S''(\theta)}{S(\theta)} = 0.
   $$
2. 令两部分均等于一个常数 $\lambda$（分离常数），则角度部分得到：
   $$
   S''(\theta) + \lambda S(\theta) = 0.
   $$
3. 由于 $S(\theta)$ 必须满足 $2\pi$ 周期性（或者在 $[0,2\pi]$ 内连续且匹配），因此 $\lambda$ 必须取形如 $m^2$ 的形式（$m \in \mathbb{Z}$），其通解为：
   $$
   S(\theta) = A\cos(m\theta) + B\sin(m\theta).
   $$
   
这也解释了为什么球面谐波变换常被看作是 Fourier 变换的推广，因为在角度部分我们实际上做了傅里叶展开。

---

# 总结

- **坐标变换**：掌握从球坐标到笛卡尔坐标的转换公式及其在积分中 Jacobian 的作用。  
- **球面函数的连续性**：要求 $\varphi$ 周期性和极点处的特殊匹配条件。  
- **球面谐波**：通过变量分离法得到的解，构成球面上正交完备的基，可以用来展开任意连续函数。  
- **Dipy 应用**：利用 dipy 库可以方便地进行球面谐波的分解和重构，并可用于各种采样（规则或随机）的情形。  
- **调和函数构造**：利用 Poisson 积分公式，我们能构造出在单位球内（或外）有界且满足边界条件的调和函数。  
- **二维极坐标**：通过变量分离法理解傅里叶变换在角度部分的作用，帮助理解球面谐波的原理。

以上内容既结合了理论推导，又配合了 Python 代码实例，帮助你全面理解球面谐波与坐标变换问题。希望这份笔记能为你在相关问题的学习和实践中提供清晰的指导。