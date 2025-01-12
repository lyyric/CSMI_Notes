**解答思路总览**  

题目要求我们证明：若一个“因果函数”(causale) $u$ 满足热方程  
$$
\partial_t u \;-\;\partial_{xx}u \;=\;0
\quad\quad(\text{Heat Equation})
$$  
或输运方程  
$$
\partial_t u \;+\;c\,\partial_x u \;=\;0
\quad (c>0)
\quad\quad(\text{Transport Equation})
$$  
并且在某个时间 $t \leq T$ 上恒为零（这正是“因果性”在此题中的体现），则 $u$ 必须是处处恒为 0。  

所谓“因果性”可简单理解为：存在一个时间 $T$ 使得
$$
\forall\, t \le T,\quad u(\cdot,\,t) \equiv 0.
$$  
我们下面分别对热方程和输运方程给出简要的证明思路。

---

## 1. 热方程情形

令 $u$ 为满足  
$$
\partial_t u \;-\;\partial_{xx}u \;=\;0
$$  
的因果函数，并且 $u(\cdot,t)=0$ 对所有 $t \leq T$ 成立。

### 1.1 能量方法 (Energy Method)

一个常用的方法是对解的 $L^2$ 范数做能量估计。记
$$
E(t) \;=\;\|u(\cdot,t)\|_{L^2(\mathbb{R})}^2
\;=\;\int_{-\infty}^{+\infty} \bigl(u(x,t)\bigr)^2\,dx.
$$

1. **计算时间导数：**  
   对热方程，有
   $$
   \partial_t u \;=\;\partial_{xx} u.
   $$  
   于是  
   $$
   \frac{d}{dt}\,\frac{1}{2} \int_{-\infty}^{\infty} \bigl(u(x,t)\bigr)^2\,dx
   \;=\;
   \int_{-\infty}^{\infty} u\,\partial_t u\;dx
   \;=\;
   \int_{-\infty}^{\infty} u\,\partial_{xx}u\;dx.
   $$
   再利用两次积分或分部积分（假设边界无穷远处 $u$ 足够快衰减，满足条件 3））可得
   $$
   \int_{-\infty}^{\infty} u\,\partial_{xx}u\;dx
   \;=\;
   -\,\int_{-\infty}^{\infty} \bigl(\partial_x u\bigr)^2\,dx
   \;\leq\;0.
   $$
   因此  
   $$
   \frac{d}{dt}\,E(t)
   \;=\;
   -\,2\,\|\partial_x u(\cdot,t)\|_{L^2}^2
   \;\le\;0.
   $$
   这说明 $E(t)$（即 $u$ 的 $L^2$ 范数的平方）是**非增的**。

2. **初值在 $t \le T$ 上为 0：**  
   已知对所有 $t \le T$，都有 $u(\cdot,t)\equiv 0$。特别地，在 $t=T$ 这一时刻，  
   $$
   E(T) \;=\;\|u(\cdot,T)\|_{L^2}^2 \;=\;0.
   $$

3. **推断 $t \ge T$ 上必为 0：**  
   由 $E(t)$ 非增且在 $t=T$ 时为 0 可知：对 $t \ge T$，$E(t)$ 只能保持在 0。换句话说，$\|u(\cdot,t)\|_{L^2}=0$，从而  
   $$
   u(\cdot,t)\equiv 0
   \quad\text{对所有 }t \ge T.
   $$

因此，综合 $t\le T$ 和 $t\ge T$ 两个区域，函数 $u$ 在整条时间轴上都恒等于 0。

---

## 2. 输运方程情形

现在令 $u$ 满足  
$$
\partial_t u + c\,\partial_x u = 0,
\quad c>0,
$$  
同样假定 $u(\cdot,t)=0$ 对所有 $t \le T$ 成立。

### 2.1 特征线方法 (Method of Characteristics)

输运方程  
$$
\partial_t u + c\,\partial_x u = 0
$$  
意味着 $u$ 沿着特征线 $\{\,x-ct=\text{常数}\}$ 上保持不变。更直观地说：给定某个点 $(x, t)$，其值由初始线（或过去某时刻）“平移”而来。

- 当 $t > T$ 时，我们追溯到过去（时间 $T$）的那个空间位置，记为 $\displaystyle x_0 = x - c\,(t - T)$。由于在 $t=T$ 的时候，$u\equiv 0$，则由特征线不变性得  
  $$
  u(x,t) \;=\; u(x_0,\,T) \;=\; 0.
  $$

由此推知，对所有 $t \ge T$，$u\equiv 0$。再加上对 $t \le T$ 的已知零解，最终得 $u\equiv 0$ 于所有时空点。

### 2.2 同样可以用能量方法

也可以仿照热方程的做法，考察 $L^2$ 范数。输运方程对应的能量守恒计算如下：
$$
\begin{aligned}
\frac{d}{dt}\,\frac{1}{2}\,\|u(\cdot,t)\|_{L^2}^2
&=\;
\int_{-\infty}^{\infty} u\,\partial_t u\,dx
\;=\;
-\;c \int_{-\infty}^{\infty} u\,\partial_x u\,dx
\;=\;
-\;\frac{c}{2} \int_{-\infty}^{\infty} \partial_x\bigl(u^2\bigr)\,dx \\
&=\;0
\quad(\text{若 }u\to 0 \text{ 当 }|x|\to\infty).
\end{aligned}
$$
因此 $\|u(\cdot,t)\|_{L^2}$ 是一个常数。由于 $u(\cdot,t)=0$ 对 $t \le T$ 恒为零，故它的 $L^2$ 范数为 0，从而在所有 $t$ 上都只能保持 0 值。

---
