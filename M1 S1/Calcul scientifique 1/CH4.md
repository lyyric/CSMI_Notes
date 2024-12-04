**Équation des ondes**  

## I Généralités

Inconnue : $u(x,t)$  
$x \in \mathbb{R}$  
$t \geq 0$  

$$
\begin{cases}
\dfrac{\partial^2 u}{\partial t^2} - c^2 \dfrac{\partial^2 u}{\partial x^2} = 0 \quad (c = 1) \\
u(x, 0) = u_0(x) \\
\dfrac{\partial u}{\partial t}(x, 0) = u_1(x)
\end{cases}
$$

$$
a = x - t \quad (x - ct)
$$
$$
b = x + t \quad (x + ct)
$$  
**Changement de variables**  

$$
x = \frac{a + b}{2}
$$  
$$
t = \frac{b - a}{2}
$$  $$
\frac{\partial}{\partial x} = \frac{\partial}{\partial a} \cdot \frac{\partial a}{\partial x} + \frac{\partial}{\partial b} \cdot \frac{\partial b}{\partial x}
$$
$$
= \frac{\partial}{\partial a} + \frac{\partial}{\partial b}
$$
$$
\frac{\partial}{\partial t} = \frac{\partial}{\partial a} \cdot \frac{\partial a}{\partial t} + \frac{\partial}{\partial b} \cdot \frac{\partial b}{\partial t}
$$
$$
= -\frac{\partial}{\partial a} + \frac{\partial}{\partial b}
$$

**Changement de fonction inconnue**  
$$
v(a, b) = u(x, t)
$$
$$
= u\left(\frac{a+b}{2}, \frac{b-a}{2}\right)
$$
$$
u(x, t) = v(x+t, x-t)
$$

$$
0 = \frac{\partial^2 u}{\partial t^2} - \frac{\partial^2 u}{\partial x^2} = 
\frac{\partial }{\partial t}\frac{\partial }{\partial t}u - \frac{\partial }{\partial x}\frac{\partial }{\partial x}u
$$
$$
= \left(-\frac{\partial}{\partial a} + \frac{\partial}{\partial b}\right)\left(-\frac{\partial}{\partial a} + \frac{\partial}{\partial b}\right) v - \left(\frac{\partial}{\partial a} + \frac{\partial}{\partial b}\right)\left(\frac{\partial}{\partial a} + \frac{\partial}{\partial b}\right) v
$$
$$
= \frac{\partial^2 v}{\partial a^2} - \frac{\partial }{\partial a} \frac{\partial }{\partial b} v - \frac{\partial }{\partial b} \frac{\partial }{\partial a} v + \frac{\partial^2 v}{\partial b^2}
-\frac{\partial^2 v}{\partial a^2} - \frac{\partial }{\partial a} \frac{\partial }{\partial b} v - \frac{\partial }{\partial b} \frac{\partial }{\partial a} v - \frac{\partial^2 v}{\partial b^2}
=0
$$

$$
\int_{-\infty}^{+\infty} f'(x) \varphi(x) \, dx = \langle f', \varphi \rangle
$$
$$
\varphi \in C^\infty \text{ à support compact}
$$
$$
\langle f', \varphi \rangle = -\langle f, \varphi' \rangle
$$
$$
= -\int_{-\infty}^{+\infty} f(x) \varphi'(x) \, dx
$$
$$
\langle L, \varphi \rangle := -\langle L, \varphi' \rangle
$$

**Théorème de Riesz**  
$H :$ espace de Hilbert  
$L : H \to \mathbb{R}$  
$L :$ linéaire continue  
$L \in H^*$  
$$
\exists ! \, u \in H, \, \forall v \in H
$$
$$
L(v) = \langle u, v \rangle
$$

$$
-4 \frac{\partial^2 v}{\partial a \partial b} = 0
$$
$$
\frac{\partial}{\partial b} \left( \frac{\partial v}{\partial a} \right) = 0
$$
Donc :  
$$
\frac{\partial v(a,b)}{\partial a} = f(a)
$$
$$
v(a,b) = F(a) + G(b)
$$
$$
u(x,t) = F(x-t) + G(x+t)
$$
2 ondes qui se propagent aux vitesses $c = \pm 1$.  


$t = 0$  
$$
u(x,0) = u_0(x)
$$
$$
u_0(x) = F(x) + G(x)
$$
$$
\frac{\partial u}{\partial t}(x,0) = \int_{-\infty}^{s} u_1(x)
$$
$$
= \int_{-\infty}^{s} (-F'(x - 0) + G'(x + 0))
$$

$u_0$ et $u_1$ sont à support compact, ainsi que $F$ et $G$.  

$$
\int_{-\infty}^x u_1(s) ds = -F(x) + G(x)
$$

$$
F(x) = \frac{1}{2} \left( u_0(x) - \int_{-\infty}^x u_1(s) \, ds \right)
$$
$$
G(x) = \frac{1}{2} \left( u_0(x) + \int_{-\infty}^x u_1(s) \, ds \right)
$$

$$
u(x,t) = F(x-t) + G(x+t)
$$
$$
= \frac{1}{2} u_0(x-t) + \frac{1}{2} u_0(x+t)
$$
$$
+ \frac{1}{2} \int_{-\infty}^{x+t} u_1(s) \, ds + \frac{1}{2} \int_{-\infty}^{x-t} u_1(s) \, ds
$$
$$
= \frac{u_0(x-t) + u_0(x+t)}{2} + \frac{1}{2} \int_{x-t}^{x+t} u_1(s) \, ds
$$  
$$
x \in [0, L]
$$
$$
\frac{\partial u}{\partial x}(0,t) = \frac{\partial u}{\partial x}(L,t) = 0
$$

**Condition de Neumann**
$$
\left( \frac{\partial u}{\partial x} = \pm \alpha u_t \right) \quad (\text{Robin})
$$

**Énergie de l'onde**
$$
\frac{\partial u}{\partial t} = u_t, \quad \frac{\partial u}{\partial x} = u_x
$$
$$
E(t) = \frac{1}{2} \int_{x=0}^L (u_t^2 + u_x^2) \, dx
$$

**Conservation de l'énergie**
$$
u_{tt} - u_{xx} = 0
$$
$$
\int_{x=0}^L (u_{tt} u_t - u_{xx} u_t) \, dx = 0
$$

$$
0 = \int_{x=0}^L \left( \frac{1}{2} \frac{\partial}{\partial t} (u_t^2) + u_x u_{xt} \right) dx - \left[ u_x u_t \right]_{x=0}^{x=L}
$$
$$
0 = \int_{x=0}^L \left( \frac{1}{2} \frac{\partial}{\partial t} (u_t^2) + u_x^2 \right) dx
$$
$$
= \frac{d}{dt} E(t)
$$

---

## II Schéma saute-mouton

(*Leap-frog*)  

**Différences finies**  

![[Pasted image 20241122090549.png]]

$N \in \mathbb{N}$ 
$$
h = \Delta x = \frac{L}{N}
$$$$
x_i = ih + \frac{h}{2}, \quad i \in \{ 0, \ldots, N-1 \}
$$

$$
u_i^n \approx u(x_i, n \tau)
$$
$$
u_1^n = u_0^n, \quad u_N^n = u_{N-1}^n
$$

$$
\frac{\partial^2 u}{\partial x^2}(x_i, t_n) \approx \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{h^2}
$$

$$
\frac{\partial^2 u}{\partial t^2}(x_i, t_n) \approx \frac{u_i^{n+1} - 2u_i^n + u_i^{n-1}}{\tau^2}
$$

$$
\frac{u_i^{n+1} - 2u_i^n + u_i^{n-1}}{\tau^2} - \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{h^2} = 0
$$

$$
\beta = \frac{\Delta t}{\Delta x} = \frac{\tau}{h}, \quad \beta^2 = \frac{c^2 \tau^2}{h^2} \quad (c = 1)
$$

$$
u_i^{n+1} = -u_i^{n-1} + u_i^n \cdot 2(1 - \beta^2) + \beta^2 \cdot (u_{i+1}^n + u_{i-1}^n)
$$

**Pour calculer $u_i^n$, on a besoin de $u_i^0$ et $u_i^1$.**  
$$
u_i^0 = u_0(x_i)
$$
$$
u_i^1 = ?
$$

$$
u(x_i, t) \approx u_i^1
$$

Développement de Taylor :  
$$
u(x_i, \tau) = u(x_i, 0) + \tau \frac{\partial u}{\partial t}(x_i, 0) + O(\tau^2)
$$

Avec $\frac{\partial u}{\partial t}(x_i, 0) = u_1(x_i)$ :  
$$
u(x_i, \tau) = u(x_i, 0) + \tau u_1(x_i)
$$

Ainsi :  
$$
u_i^1 = u_i^0 + \tau u_1(x_i)
$$

---

**Forme matricielle :**  
$$
U^n = 
\begin{pmatrix}
u_0^n \\
u_1^n \\
\vdots \\
u_{N-1}^n
\end{pmatrix}
$$

Équation :  
$$
U^{n+1} - 2U^n + U^{n-1} + \beta^2 A U^n = 0
$$

Avec :  
$$
A =
\begin{pmatrix}
-2 & 1 & 0 & \cdots & 0 \\
1 & -2 & 1 & \cdots & 0 \\
0 & 1 & -2 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & 1 \\
0 & 0 & 0 & 1 & -2
\end{pmatrix}
$$

Réarrangement :  
$$
U^{n+1} = -U^{n-1} + 2U^n + \beta^2 A U^n
$$  
ou encore :  
$$
U^{n+1} = -U^{n-1} + (2I - \beta^2 A) U^n
$$  

### Stabilité 

$$
U^n = \sum_k \alpha_k^n R_k
$$
$R_k$ : vecteurs propres de $A$  
$$
A R_k = \lambda_k R_k
$$

$$
\lambda_k = -4 \sin^2\left(\frac{k \pi h}{2}\right)
$$

On fixe $k$ :  
$$
\alpha_k^{n+1} = -\alpha_k^{n-1} + \left(2 - \beta^2 \lambda_k\right) \alpha_k^n
$$

**Condition de stabilité :**  
$$
\alpha_k^n \text{ bornée quand } n \to +\infty
$$

---

**Polynôme caractéristique :**  
$$
P(x) = x^2 + 1 - \left(2 - \beta^2 \lambda_k\right)x = 0
$$

Racine de module $< 1$ :  
**Condition de stabilité (CS) :**  
$$
\beta < 1
$$

---

$$
f_{n+1} = f_n + f_{n-1}
$$

$$
f_n = r^n
$$

$$
r^{n+1} - r^n - r^{n-1} = 0
$$

$$
r^2 - r - 1 = 0
$$

$$
r_1 \, \text{et} \, r_2
$$

$$
f_n = \lambda r_1^n + \mu r_2^n
$$

## III Schéma décentré généralisé

$$
w = 
\begin{pmatrix}
u_t \\
u_x
\end{pmatrix}
=
\begin{pmatrix}
w_1 \\
w_2
\end{pmatrix}
$$

$$
w_t = 
\begin{pmatrix}
u_{tt} \\
u_{xt}
\end{pmatrix}
\quad \text{et} \quad
w_x = 
\begin{pmatrix}
u_{tx} \\
u_{xx}
\end{pmatrix}
$$

Équation :  
$$
w_t + A w_x = 0
$$

Avec :  
$$
A =
\begin{pmatrix}
0 & -1 \\
1 & 0
\end{pmatrix}
$$

$$
(A - cI) =
\begin{pmatrix}
-c & -1 \\
-1 & -c
\end{pmatrix}
$$

$$
\det(A - cI) = c^2 - 1
$$

$$
c_1 = -1, \quad c_2 = +1
$$

Pour $c_1 = -1$ :  
$$
\begin{pmatrix}
1 & -1 \\
-1 & 1
\end{pmatrix}
\begin{pmatrix}
\alpha \\
\beta
\end{pmatrix}
= 0
$$
$$
\alpha - \beta = 0
$$

Vecteur propre associé :  
$$
r_1 = \frac{1}{\sqrt{2}}
\begin{pmatrix}
1 \\
1
\end{pmatrix}
$$

Pour $c_2 = +1$ :  

$$
\begin{pmatrix}
-1 & -1 \\
-1 & -1
\end{pmatrix}
\begin{pmatrix}
\alpha \\
\beta
\end{pmatrix}
= 0
$$
$$
\alpha + \beta = 0
$$
$$
r_2 = \frac{1}{\sqrt{2}}
\begin{pmatrix}
1 \\
-1
\end{pmatrix}
$$

---

Matrice de passage :  
$$
P = \frac{1}{\sqrt{2}}
\begin{pmatrix}
1 & 1 \\
1 & -1
\end{pmatrix}
$$

$$
P^T = P^{-1}
$$

Transformation :  
$$
P^{-1} A P =
\begin{pmatrix}
-1 & 0 \\
0 & 1
\end{pmatrix}
= D
$$

---

Nouvelles variables :  
$$
w = P v
$$

$$
P^T P v_t + P^T A P v_x = 0
$$

$$
v_t + D v_x = 0
$$

$$
v =
\begin{pmatrix}
v_1 \\
v_2
\end{pmatrix}
$$

Système :  
$$
\begin{cases}
(v_1)_t - (v_1)_x = 0 \\
(v_2)_t + (v_2)_x = 0
\end{cases}
$$

$$
v_i = \begin{pmatrix}v_1 \\ v_2\end{pmatrix}_i
$$

Équation :  
$$
u_t + c u_x = 0
$$

Avec :  
$$
c^+ = \max(c, 0), \quad c^- = \min(c, 0)
$$

Discrétisation :  
$$
\frac{u_i^{n+1} - u_i^n}{\Delta t} + c^+ \frac{u_i^n - u_{i-1}^n}{h} + c^- \frac{u_{i+1}^n - u_i^n}{h} = 0
$$ 
$$
D^+ =
\begin{pmatrix}
0 & 0 \\
0 & 1
\end{pmatrix}, \quad
D^- =
\begin{pmatrix}
-1 & 0 \\
0 & 0
\end{pmatrix}
$$

Discrétisation :  
$$
\frac{v_i^{n+1} - v_i^n}{\tau} + D^+ \frac{v_i^n - v_{i-1}^n}{h} + D^- \frac{v_{i+1}^n - v_i^n}{h} = 0
$$

---

Transformation avec $w$ :  
$$
w_i^n = P v_i^n, \quad v_i^n = P^{-1} w_i^n
$$

En termes de $w$ :  
$$
\frac{w_i^{n+1} - w_i^n}{\tau} + A^+ \frac{w_i^n - w_{i-1}^n}{h} + A^- \frac{w_{i+1}^n - w_i^n}{h} = 0
$$

Avec $A^+ = P D^+ P^{-1}$ et $A^- = P D^- P^{-1}$.  

---

Finale :  
$$
\frac{w_i^{n+1} - w_i^n}{\tau} + \frac{A^+}{h}(w_i^n - w_{i-1}^n) + \frac{A^-}{h}(w_{i+1}^n - w_i^n) = 0
$$

$$
w =
\begin{pmatrix}
u_t \\
u_x
\end{pmatrix}
$$

Représentation graphique : les conditions initiales $u_0$ et $u_1$ sont représentées.  

Formule :  
$$
u(x,t) = \int_{-\infty}^x u_x(s,t) \, ds
$$


**Propriétés de $A$** :  
- $A$ est **auto-adjoint**.  
- $\sigma(A)$, le spectre de $A$, est à déterminer.  

Forme diagonalisée :  
$$
D = P^{-1} A P
$$

Avec :  
$$
D =
\begin{pmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_n
\end{pmatrix}
$$  

Les valeurs propres $\lambda_i$ sont à déterminer en fonction de la configuration de $A$.