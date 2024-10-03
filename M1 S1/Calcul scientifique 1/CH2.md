# Équation de transport

## I) Aspects mathématiques

$u : (x, t) \rightarrow u(x, t)$

Fonction régulière de $x$ et $t$.

$x \in \mathbb{R}, \; t \geq 0$

$\partial_t u + c \partial_x u = 0$

$c \in \mathbb{R}$ constante

Condition initiale $u(x, 0) = u_0(x)$

$u_0 :$ fonction donnée.

Méthode des caractéristiques pour trouver des solutions.

![](assets/Pasted%20image%2020241003134436.png)

> [!definition]
> $\Sigma$ est une courbe caractéristique de $(x)$ si et seulement si les solutions sont constantes le long de $\Sigma$.

Supposons que $\Sigma_t$ est une courbe paramétrée par $t$,

$$
\Sigma_t = \left\{ \left( x(t), t \right), t \geq 0 \right\}
$$

Paramétrage : $t \rightarrow \begin{pmatrix}x(t)\\t\end{pmatrix} \in \Sigma'$ 

$u$ solution de $\partial_t u + c \partial_x u = 0$

$u(x(t), t) =$ constante

$$
\frac{d}{dt} u(x(t), t) = 0
$$

$$
\frac{\partial u}{\partial x}(x(t), t) x'(t) + \frac{\partial u}{\partial t}(x(t), t) \times 1 = 0
$$

![](assets/Pasted%20image%2020241003135249.png)

Si $x'(t) = c$, alors $\Sigma$ est bien une courbe caractéristique.

   $$
   x(t) = ct + x_0
   $$

$$
u(x(t), t) = u(x_0, 0) = u_0(x_0)
$$

$$
x = ct + x_0, \quad x_0 = x - ct
$$

$$
u(x, t) = u_0(x - ct)
$$

![](assets/Pasted%20image%2020241003135557.png)

Quelques propriétés :

$$
\partial_t u + c \partial_x u = 0
$$

> [!hypothesis]
>  $u_0$ est à décroissance assez rapide quand $x \rightarrow \pm \infty$. $u_0 \in L^p$ pour $\forall p \geq 1$.

### 1. Propriété de conservation

$$
M(t) = \int_{-\infty}^{+\infty} u(x, t) \, dx
$$

Proposition : $M(t)$ est constant.

$$
\int_{-\infty}^{+\infty} (\partial_t u + c \partial_x u) \, dx = 0
$$

$$
=\frac{d}{dt} \int_{-\infty}^{+\infty} u(x, t) \, dx + c \left[ u_0(x - ct) \right]_{x = -\infty}^{x = +\infty}
$$

$$
M_p(t) = \int_{-\infty}^{+\infty} |u(x, t)|^p \, dx, \quad p > 1
$$

Pour toute fonction assez régulière $f$,

$$
\partial_t f(u) + c \partial_x f(u) = 0
$$

$$

\begin{cases}
\partial_t v + c \partial_x v = 0 \\
v(x, 0) = f(u_0(x)) \\
v(x, t) = v(x-ct,0) = f(v_0(x - ct))
\end{cases}

$$

$v = f(u)$.

$$
\partial_t f(u) + c \partial_x f(u) = f'(u) (\partial_t u + c \partial_x u) = 0
$$

Corollaire : $\forall p \geq 1$,

$$
\| u(\cdot, t) \|_{L^p(\mathbb{R})} = \| u_0 \|_{L^p(\mathbb{R})}
$$

Corollaire : Unicité de la solution de l'équation de transport.

**Équation de diffusion** : Régularisation de la condition initiale si $t > 0$.

**Équation de transport** : La régularité de la condition initiale n'est ni dégradée ni améliorée.

$$
\partial_t u_i + c \partial_x u_i = 0, \quad i = 1, 2
$$

$$
u_i(x, 0) = u_{0_i}(x)
$$

$$
u = u_1 - u_2
$$

$$
\partial_t u + c \partial_x u = 0
$$

$$
u(x, 0) = 0
$$

$$
\| u(\cdot, t) \|_{L^p} = \| 0 \|_{L^p} = 0
$$

$$
\implies u_1 = u_2
$$

### 3) Conditions aux limites

Considérons $c > 0$,

$x \in [0, L], \; t \geq 0$
$$
\begin{cases}
\partial_t u + c \partial_x u = 0 \\
u(x, 0) = u_0(x) \\
u(0, t) = g(t)
\end{cases}
$$

![](assets/Pasted%20image%2020241003141918.png)

$$
g(0) = u_0(0) \; ?
$$

Si $x > ct$,

$$
u(x, t) = u_0(x - ct)
$$

Sinon,

$$
u(x, t) = g(t_0) = g\left(t - \frac{x}{c}\right)
$$

Les points $\begin{pmatrix}0 \\ t_0 \end{pmatrix}$ et $\begin{pmatrix}x \\ t \end{pmatrix}$ sont sur la même courbe caractéristique :

$$
x = ct + x_0
$$

$$
0 = ct_0 + x_0
$$

$$
t_0 = t - \frac{x}{c}
$$

Si $c < 0$, alors...

![](assets/Pasted%20image%2020241003142515.png)

## II) Méthode numérique

$h$ : pas de discrétisation.

$x_i = i h$, $i \in \mathbb{Z}$, $h = \Delta x$

$$
t^n = n \tau, \quad \tau = \Delta t
$$

$$
u_i^n \approx u(x_i, t_n)
$$

On cherche des schémas aux différences finies.

**Remarque** :

$$
\| u(\cdot, t) \|_{L^2} \leq \| u_0 \|_{L^2}
$$
$$
\|u_0^m\|_{l^2(\mathbb{Z})}^2 = l^n(u)
$$
$$
= \Delta x \sum_{i = -\infty}^{+\infty} (u_i^m)^2
$$

$$
l^n(u) \leq l^{n-1}(u)
$$

Condition de stabilité des schémas numériques.

> [!tip]
> NaN : Not a Number.

**Schémas possibles :**

$$
\frac{u_i^{n+1} - u_i^n}{\tau} + c \frac{u_{i+1}^n - u_{i-1}^n}{2h} = 0
$$

Pour $c > 0$:

![](assets/Pasted%20image%2020241003144220.png)

schéma centré en espace, explicite en temps

$$
\frac{u(x + h, t) - u(x - h, t)}{2h} = \partial_x u(x, t) + O(h^2)
$$

$$
\frac{u(x, t + \tau) - u(x, t)}{\tau} = \partial_t u(x, t) + O(\tau)
$$

$$
\frac{u(x, t) - u(x - h)}{h} = \partial_x u(x, t) + O(h)
$$

Schéma décentré à gauche :

$$
\frac{u_i^{n+1} - u_i^n}{\tau} + c \frac{u_i^n - u_{i-1}^n}{h} = 0
$$

Schéma décentré à droite :

$$
\frac{u_i^{n+1} - u_i^n}{\tau} + c \frac{u_{i+1}^n - u_i^n}{h} = 0
$$

### 4) Étude de stabilité

$I \subset \mathbb{C}, \; I^2 = -1$

$(u_i) \in l^2(\mathbb{Z})$

$$
(u_i^n) \longrightarrow (u_i^{n+1})
$$

Le schéma qui fait passer du temps $n$ au temps $n+1$ peut être considéré comme une application linéaire de $l^2(\mathbb{Z})$ à $l^2(\mathbb{Z})$.

**Stabilité** = contraction pour $\| \cdot \|_{l^2(\mathbb{Z})}$.

$(u_i) \in l^2(\mathbb{Z})$

$$
\hat{u}(\xi) := \sum_{i \in \mathbb{Z}} u_i e^{-2\pi i \xi i}
$$

$$
\hat{u} \in L^2 \left( \frac{\mathbb{R}}{\mathbb{Z}} \right) = L^2(]0, 1[)
$$

$\xi \in ]0, 1[$.

a) $\hat{}$ : Isométrie de $l^2(\mathbb{Z})$ dans $L^2([0, 1])$:

$$
\| (u_i) \|_{l^2(\mathbb{Z})} = \| \hat{u} \|_{L^2(]0, 1[)}
$$

$l^2(\mathbb{Z}) = \left\{ (u_i)_{i \in \mathbb{Z}},u_i\in \mathbb{C}, \sum_{i = -\infty}^{+\infty} |u_i|^2 < +\infty \right\}$

$$
\| (u_i) \|_{l^2(\mathbb{Z})} = \sqrt{\sum_{i = -\infty}^{+\infty} |u_i|^2}
$$

b) Formule d'inversion de Fourier :

$$
u_i = \frac{1}{\sqrt{2\pi}} \int_0^1 \hat{u}(\xi) e^{2\pi i \xi i} \, d\xi
$$

c) Opérateurs de décalage

Soit $u \in l^2(\mathbb{Z})$, l'opérateur de décalage $\tau_k: l^2(\mathbb{Z}) \to l^2(\mathbb{Z})$ est défini par :

$$
v = \tau_k u
$$

où 

$$
v_i = u_{i-k}
$$

**Isométrie de** $l^2(\mathbb{Z})$:

$$
\| \tau_k u \|_{l^2(\mathbb{Z})}^2 = \sum_i |v_i|^2 = \sum_i |u_{i-k}|^2 = \sum_j |u_{j}|^2 = \| u \|_{l^2(\mathbb{Z})}^2
$$


> [!theorem]
> La transformée de Fourier diagonalise les opérateurs de décalage.
> $$
> \widehat{(\tau_k u)}(\xi) = e^{-2I\pi k \xi} \cdot \hat{u}(\xi)
> $$

Démonstration :

$$
\widehat{(\tau_k u)}(\xi) = \sum_i (\tau_k u)_i e^{-2I\pi i \xi} = \sum_i u_{i-k} e^{-2I\pi i \xi}
$$

En posant $j = i - k$,

$$
\widehat{(\tau_k u)}(\xi) = \sum_j u_j e^{-2I\pi (j + k) \xi} = e^{-2I\pi k \xi} \sum_j u_j e^{-2I\pi j \xi} = e^{-2i\pi k \xi} \hat{u}(\xi)
$$

---

Facteur d'amplification d'un schéma :

Soit $S : (u_i^n) \to (u_i^{n+1})$,

$$
\widehat{S(u)}(\xi) = A(\xi) \hat{u}(\xi) = \widehat{S}(\hat u)(\xi)
$$ 
$A(\xi)$ : facteur d'amplification (de von Neumann).

> [!theorem]
> Le schéma est contractant dans $l^2(\mathbb{Z})$ si et seulement si :
> 
> $$
> \forall \xi \in ]0, 1], \quad |A(\xi)| \leq 1
> $$

CS :

Si $|A(\xi)| \leq 1$ pour tout $\xi$,

$$
|\widehat{S(u)}(\xi)| = |A(\xi) \hat{u}(\xi)| \leq |\hat{u}(\xi)|
$$

Donc,

$$
\int_0^1 |\widehat{S(u)}(\xi)|^2 \, d\xi \leq \int_0^1 |\hat{u}(\xi)|^2 \, d\xi = \|\hat{u}\|_{L^2}
$$

Ainsi,

$$
\|\widehat{S(u)}\|_{L^2}^2 \leq \|\hat{u}\|_{L^2}^2
$$
$\iff$
$$
\|S(u)\|_{l^2(\mathbb{Z})} \leq \|u\|_{l^2(\mathbb{Z})}
$$

Donc, le schéma est stable.

**Facteur d'amplification du schéma centré**

$$
u_i^{n+1} = u_i^n - \frac{c \tau}{h} \left( u_{i+1}^n - u_{i-1}^n \right)
$$
$$
u_0^{n+1} = u_0^n - \frac{c \tau}{h} \left(\tau_{-1} u_{0}^n - \tau_1 u_{0}^n \right)
$$

$$
\hat{u}^{n+1}(\xi) = \hat{u}^n(\xi) - \frac{c \tau}{h} \left( e^{2I \pi \xi} - e^{-2I \pi \xi} \right) \hat{u}^n(\xi)
$$
$$
\hat{u}^{n+1}(\xi) = A(\xi) \hat{u}^n(\xi)
$$

$$
A(\xi) = 1 - \frac{c \tau}{h} \left( 2I \sin(2 \pi \xi) \right)
$$

![](assets/Pasted%20image%2020241003152352.png)

Le schéma centré est toujours instable (inconditionnellement).

**Schéma décentré** : $c > 0$

$$
u_i^{n+1} = u_i^n - \frac{c \tau}{h} \left( u_i^n - u_{i-1}^n \right)
$$
$$
u^{n+1} = u^n - \frac{c \tau}{h} \left( u^n -\tau_1 u^n \right)
$$

$$
\hat{u}^{n+1}(\xi) = \hat{u}^n(\xi) - \frac{c \tau}{h} (1 - e^{-2I \pi \xi}) \hat{u}^n(\xi)
$$

$$
A(\xi) = 1 - \frac{c \tau}{h} (1 - e^{-2i \pi \xi})
$$

![](assets/Pasted%20image%2020241003152612.png)

Le schéma est stable $\iff$

$$
\frac{c \tau}{h} \leq 1
$$

Condition (CFL) :

$$
\tau \leq \frac{h}{c}
$$
($\tau\leq \frac{h^2}{2}$)

**Schéma décentré à l'aval** (instable).

Les schémas implicites et centre décentrés amont sont toujours stables.


**Stabilité** $l^\infty$ :

$$
\| u \|_{l^\infty} = \sup_i |u_i|
$$

$$
u(x, t) = u_0(x - ct)
$$

Si

$$
m \leq u_0 \leq M
$$

alors

$$
\forall x, t, \; m \leq u(x, t) \leq M
$$

Principe du maximum : $\dfrac{c\tau}{h}\leq 1$,

$$
m \leq u_i^0 \leq M
$$

Pour $c > 0$,

$$
u_i^{n+1} = \left( 1 - \frac{c \tau}{h} \right) u_i^n + \frac{c \tau}{h} u_{i-1}^n
$$

$$
m \leq u_i^n \leq \left( 1 - \frac{c \tau}{h} \right) m + \frac{c \tau}{h} M = M
$$

$$
\forall n, \forall i, \; m \leq u_i^n \leq M
$$

Principe du maximum discret :

$$
\| u_i^n \|_{l^\infty} \leq \| u_i^0 \|_{l^\infty}
$$
