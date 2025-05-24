# 线性弹性（Elasticité Linéaire）

作者：Christophe Prud’homme, Laurent Navoret

---

## 1. 线性弹性问题

**微分方程组**

$$
\begin{cases}
\nabla\cdot\sigma(u) + f = 0, & x\in\Omega,\\
\sigma(u) = \mu\bigl(\nabla u + \nabla u^T\bigr) + \lambda\,(\nabla\cdot u)\,I,\\
u = 0, & x\in\partial\Omega_D,\\
\sigma(u)\cdot n = h, & x\in\partial\Omega_N.
\end{cases}
$$

* $u:\Omega\to\mathbb R^d$：位移场
* $\sigma(u)$：应力张量
* $\lambda,\mu>0$：Lamé 常数
* $f$：体力
* $h$：Neumann 边界受力

---

## 2. 弹性本构与能量形式

* **小变形应变张量**
  $\displaystyle \varepsilon(u)=\tfrac12(\nabla u+\nabla u^T)$
* **平衡积分形式**
  对任意子域 $\omega\subset\Omega$：

  $$
    \int_{\partial\omega}\sigma(u)\,n \;=\;-\,\int_\omega f.
  $$

---

## 3. 超弹性材料（Saint Venant–Kirchhoff）

* 对于大变形橡胶等，线性本构不再准确。
* Saint Venant–Kirchhoff 模型：

  $$
    \varepsilon(u)=\frac12\bigl(\nabla u+\nabla u^T+(\nabla u)^T\nabla u\bigr).
  $$

---

## 4. 杨氏模量 $E$ 与泊松比 $\nu$（3D）

$$
E=\frac{\mu(3\lambda+2\mu)}{\lambda+\mu},\quad
\nu=\frac{\lambda}{2(\lambda+\mu)}.
$$

* 若在垂直方向拉伸 $\varepsilon$：

  * 纵向应力约为 $E\,\varepsilon$
  * 横向收缩应变约为 $\nu\,\varepsilon$
* $0<\nu\le0.5$，$\nu=0.5$ 对应不可压缩材料

---

## 5. 向量与张量运算

* **位移场** $u=(u_1,\dots,u_d)$
* **应变** $\varepsilon(u)=\tfrac12(\nabla u+\nabla u^T)$
* **应力-应变线性关系** $\sigma(u)=2\mu\,\varepsilon(u)+\lambda\,\mathrm{tr}(\varepsilon(u))\,I$
* **体力功** 和 **表面力功** 在弱形式中出现

---

## 6. 变分（弱）形式

定义空间

$$
V=\bigl\{v\in H^1(\Omega)^d:\;v=0\text{ on }\partial\Omega_D\bigr\}.
$$

求 $u\in V$，满足

$$
a(u,v)=\ell(v)\quad\forall v\in V,
$$

其中

$$
\begin{aligned}
a(u,v)
&=\!\tfrac12\!\int_\Omega\!\mu(\nabla u+\nabla u^T):
(\nabla v+\nabla v^T)\;+\;\lambda\!\int_\Omega\!(\nabla\cdot u)(\nabla\cdot v),\\
\ell(v)
&=\!\int_{\partial\Omega_N}\!v\!\cdot h\;-\!\int_\Omega\!f\!\cdot v.
\end{aligned}
$$

---

## 7. 存在性与唯一性

* **Korn 不等式**：若 $V$ 不含非零刚体位移，则存在 $\kappa>0$，使
  $\|v\|_{H^1}\le\kappa^{-1}\|\varepsilon(v)\|_{L^2}$
* 则双线性型 $a(\cdot,\cdot)$ 连续且 coercive，线性泛函 $\ell$ 连续
* 从而按 Lax–Milgram 定理，弱问题有唯一解

---

## 8. 有限元离散

1. **网格** $\mathcal T_h$：仿射三角/四面体网
2. **离散空间**

$$
 V_h=\{v_h\in C^0(\bar\Omega)^d:\;v_h|_K\in\mathbb P_k^d,\;v_h=0\text{ on }\partial\Omega_D\}.
$$
3. **基函数**：Lagrange 多项式

---

## 9. 收敛性

* 若精确解 $u\in H^{k+1}(\Omega)^d\cap V$，则

$$
\|u-u_h\|_{H^1}\le C\,h^k\,\|u\|_{H^{k+1}}.
$$
* P$^k$ 元在 $H^1$ 范数下获得 $O(h^k)$ 收敛阶

---

## 10. 不可压缩性失配

* 当 $\lambda/\mu\to\infty$（$\nu\to0.5$）时，coercivity 常数退化
* 导致数值病态，需特殊方法（如混合元或稳定化）

---

## 11. Feel++ 实现与示例

* 参考文档：[https://docs.feelpp.org](https://docs.feelpp.org)
* 提供 Docker 镜像和多个 CSM 案例

---
