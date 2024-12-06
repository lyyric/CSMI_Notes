Chapitre 5 - Méthode des éléments finis (1D)  

## I. Équation elliptique (rappel)  

On considère le problème :  
$$
\begin{cases}  
-u'' + u = f \quad \text{sur} \quad I = ]0, 1[ \\  
u(0) = u(1) = 0  
\end{cases}  
$$

avec $f \in L^2(I)$ donnée  
et $u : I \to \mathbb{R}$ inconnue.  

> [!definition]
> $$
> H^1(I) = \left\{ u \in L^2(I), \, u' \in L^2(I) \right\}  
> $$  
> (où $u'$ dérivée au sens des distributions)  
> muni du produit scalaire  
> $$
> (u, v)_{H^1} = (u, v)_{L^2} + (u', v')_{L^2}  
> $$  
> et de la norme associée  
> $$
> \| u \|_{H^1} = \sqrt{\| u \|_{L^2}^2 + \| u' \|_{L^2}^2}  
> $$ 
> est un espace de Hilbert.  
> 
> On a de plus $H^1(I) \subset C(I)$.  

> [!example]
> $$
> f(x) =  
> \begin{cases}  
> 1 - |x| & \text{si } |x| < 1 \\  
> 0 & \text{sinon}  
> \end{cases}  
> $$
> $f \in L^2([-1, 1])$.  
> ![[Pasted image 20241127081853.png]]
> $$
> f'(x) =  
> \begin{cases}  
> 1 & \text{si } x \in ]-1, 0[ \\  
> -1 & \text{si } x \in ]0, 1[ \\  
> 0 & \text{sinon}  
> \end{cases}  
> $$ 
> $f' \in L^2([-1, 1])$.  
> ![[Pasted image 20241127081917.png]]
> 
> $$
> \Rightarrow f \in H^1([-1, 1])  
> $$

> [!review]
> $$
> f(x) = g(x) \mathbf{1}_{]-\infty, a]} + h(x) \mathbf{1}_{]a, \infty[}  
> $$
> ![[Pasted image 20241127082005.png]]
> avec $g, h \in C^2$.  
> 
> $$
> f'(x) = g'(x) \mathbf{1}_{]-\infty, a]} + h'(x) \mathbf{1}_{]a, \infty[} + (h(a) - g(a)) \delta_a(x)  
> $$

> [!definition]
> On note 
> $$
> H_0^1(I) = \{ u \in H^1(I) \, | \, u(0) = u(1) = 0 \}
> $$  

1) (Inégalité de Poincaré)  
$$
\forall u \in H_0^1(I), \quad \| u \|_{L^2(I)} \leq \| u' \|_{L^2(I)}  
$$  
2) Sur $H_0^1(I)$, $\| u \|_{H^1} = \| u' \|_{L^2}$ est une norme équivalente à $\| \cdot \|_{H^1}$.  

$$
\| u' \|_{L^2}^2 \leq \| u \|_{L^2}^2 + \| u' \|_{L^2}^2 \leq 2 \| u' \|_{L^2}^2  
$$
$$
\| u' \|_{L^2}^2 = \| u \|_{H_0^1}^2  
$$
$$
\Rightarrow \| u \|_{H_0^1} \leq \| u \|_{H^1} \leq \sqrt{2} \| u \|_{H_0^1}  
$$

**Formulation variationnelle du problème** :  
Trouver $u \in H_0^1(I)$,  
$$
\int_I u' v' + \int_I uv = \int_I fv \quad \forall v \in H_0^1(I).  
$$ 
> [!theorem] Théorème de Lax-Milgram  
> $V$: espace de Hilbert  
> 
> - $a : V \times V \to \mathbb{R}$, bilinéaire continue $(\exists M > 0 \, |a(u, v)| \leq M \| u \| \| v \| \, \forall u, v \in V)$  
>   coercive $(\exists \nu > 0 \, a(u, u) \geq \nu \| u \|^2 \, \forall u \in V)$.  
> 
> - $\ell : V \to \mathbb{R}$, linéaire continue $(\exists C > 0 \, | \ell(v) | \leq C \| v \| \, \forall v \in V)$.  
> 
> Alors $\exists!\, u \in V, \, a(u, v) = \ell(v) ,\, \forall v \in V$.  

**Application** :  
$$
a(u, v) = \int_I u' v' + \int_I u v  
$$
$$
\ell(v) = \int_I f v  
$$

- $a$ bilinéaire : ok  
- $a$ continue :  
$$
|a(u, v)| = |(u', v') + (u, v)| \leq \| u \|_{H^1} \| v \|_{H^1} \leq 2 \| u \|_{H_0^1} \| v \|_{H_0^1}
$$  
- $a$ coercive :  
$$
a(u, u) = \| u \|_{H^1}^2 \geq \| u \|_{H_0^1}^2  
$$  
- $\ell$ linéaire : ok  

$\ell$ continue :  
$$
|\ell(v)| = \left| \int_I f v \right| \leq \int_I |f v| \leq \| f \|_{L^2} \| v \|_{L^2}  
$$
(par Cauchy-Schwarz)  
$$
\leq \| f \|_{L^2} \| v' \|_{L^2} \, \text{ (Poincaré) } = C \| v \|_{H_0^1}.  
$$  

$\implies$ Existence et unicité de la solution au problème variationnel (Lax-Milgram).

> [!definition]
> - Une **solution faible** est une fonction  
> $$
> u \in H_0^1(I) \, \text{vérifiant le problème variationnel.}
> $$
> 
> - Une **solution forte** est une fonction  
> $$
> u \in H_0^1(I) \cap H^2(I) \, \text{vérifiant l'équation au sens } L^2.
> $$

> [!proposition]
> La solution du problème variationnel est une solution forte. On a de plus :  
> $$
> \| u \|_{H^1}^2 \leq (1 + \sqrt{2}) \| f \|_{L^2}.
> $$

**Preuve** :  
Soit $u$ la solution du problème variationnel.  
Comme $C_c^\infty(I) \subset H_0^1(I)$, on a :  
$$
\int_I u' \psi' + u \psi = \int_I f \psi, \quad \forall \psi \in C_c^\infty(I).  
$$

On a donc, au sens des distributions :  
$$
-u'' + u = f.
$$

Donc :  
$$
u'' = u - f \in L^2(I).  
$$

Ainsi :  
$$
u \in H^2(I).  
$$ 
$$
\| u'' \|_{L^2} = \| u - f \|_{L^2} \leq \| u \|_{L^2} + \| f \|_{L^2}.
$$

On a :  
$$
\| u \|_{H^1_0}^2 \leq a(u, u) = \ell(u) \leq \| f \|_{L^2} \| u \|_{H^1_0}.
$$  
Donc :  
$$
\| u \|_{H^1_0} \leq \| f \|_{L^2} \tag{*}
$$

On en déduit :  
$$
\| u \|_{L^2} \leq \| u \|_{H^1} = \sqrt{\| u \|_{L^2}^2 + \| u' \|_{L^2}^2}   \leq \sqrt{2} \| u \|_{H^1_0} \leq^{(*)} \sqrt{2} \| f \|_{L^2} 
$$

Ainsi :  
$$
\| u'' \|_{L^2} \leq (1 + \sqrt{2}) \| f \|_{L^2}.
$$

---

**Rappel** : Distribution  
$T$: forme linéaire continue sur $C^\infty_c(I)$.  

Pour $K \subset I$ compact, $\exists C_K > 0, \, \forall p \in \mathbb{N}$,  
$$
\forall \varphi \in C^\infty_c(I), \, \sup(|\varphi^{(j)}|) \leq C_K.
$$

$$
|T(\varphi)| \leq C_K \sup_{0 \leq j \leq p} |\varphi^{(j)}|_{\infty}.
$$

**Exemple** :  
$f \in L^1_{\text{loc}}$ ($\int_K |f| < +\infty$ pour tout $K$ compact).  

$\forall \varphi, \, \operatorname{supp}(\varphi) \subset K$
$$
|T_f(\varphi)| = \left| \int f \varphi \right| \leq \| \varphi \|_{\infty} \int_K |f|.  
$$
avec $C_K = \int_K |f|$ 

$$
T_f(\varphi) = \int f \varphi \quad \text{(distribution)}.  
$$

**Dérivée au sens des distributions :**  

$T$ distribution  
$$
T'(\varphi) := -T(\varphi') \quad \forall \varphi \in C_c^\infty(\mathbb{R}).
$$  
(Toutes les distributions sont dérivables.)

**Remarque :**  
$$
f \in C^1(I) \subset L^1_{\text{loc}}.
$$

$$
T_f'(\varphi) = -T_f(\varphi') = -\int f \varphi' = \int f' \varphi - \left[f \varphi\right]_0^1 \quad \text{(intégration par parties)}.
$$
$$
= T_{f'}(\varphi).
$$

La dérivée au sens des distributions coïncide avec la dérivée classique si $f \in C^1$.  

**Exemple :**  
$\delta_a(\varphi) = \varphi(a)$  

$$
|\delta_a(\varphi)| \leq \|\varphi\|_\infty \quad (C_K = 1 \, \forall K \text{ compact}, \, p = 0).
$$

$$
\delta_a'(\varphi) = -\delta_a(\varphi') = -\varphi'(a).
$$

$$
|\delta_a'(\varphi)| \leq \|\varphi'\|_\infty \quad (C_K = 1 \, \forall K, \, p = 1).
$$  

**Rappel : Formulation variationnelle**  
$u\in C^2(I)$ 
$$
\varphi(-u'' + u) = \varphi f.
$$
$$
\Rightarrow \int_I \varphi(-u'' + u) = \int_I \varphi f.
$$
$$
\Rightarrow \int_I u' \varphi ' - [\varphi u']_0^1 + \int_I u \varphi = \int_I f \varphi.
$$

**Remarque** : Les intégrales sont bien définies dès que  
$$
u' \in L^2, \, \varphi' \in L^2 
$$
$$
u \in L^2, \, \varphi \in L^2.
$$ $$
\implies u, \varphi \in H^1(I)
$$
On cherche $u$ dans un espace plus grand : $H^1(I)$.  

---

**2) Problème approché**  

**Principe** : Remplacer $V$ (espace de Hilbert) de dimension infinie par $V_h$, de dimension finie.  

Considérons le problème variationnel en dimension finie :  
Trouver $u_h \in V_h$, tel que  
$$
a(u_h, v_h) = \ell(v_h), \quad \forall v_h \in V_h.
$$

> [!definition]
> L'approximation est dite **conforme** si  
> $$
> V_h \subset V \quad \text{et} \quad \forall v \in V, \, \text{dist}(v, V_h)_{H^1} \to 0 \quad (h \to 0),
> $$  
> où $h$ est un paramètre.  

Notons $N_h$ la dimension de $V_h$ et $\varphi_1, \dots, \varphi_{N_h}$ une base de $V_h$.  

$$
a(u_h, \varphi_j) = \ell(\varphi_j) \quad \forall v_h \in V , \forall j \in [1, N_h].
$$ par $u_h = \sum_{i=1}^{N_h} \alpha_i \varphi_i$

$$
a\left(\sum_{i=1}^{N_h} u_i \varphi_i, \varphi_j\right) = \ell(\varphi_j).
$$
$$
\forall j \in [[1, N_h]].
$$ $$
\sum_{i=1}^{N_h} a(\varphi_i, \varphi_j) u_i = \ell(\varphi_j) \quad \forall j \in [1, N_h].
$$
$$
\Leftrightarrow A_h U_h = b_h
$$

avec :  
$$
A_h = (a(\varphi_i, \varphi_j))_{i,j} \in M_{N_h}(\mathbb{R}),
$$  
$$
U_h = (u_i)_i \in \mathbb{R}^{N_h},
$$  
$$
b_h = (\ell(\varphi_j))_j \in \mathbb{R}^{N_h}.
$$

$A_h$ est définie positive.  

En effet :  
$$
(A_h U_h, U_h) = \sum_{j} \left( \sum_{i} a(\varphi_i, \varphi_j) u_i \right) u_j.
$$

Par bilinéarité :  
$$
= a\left(\sum_{i} u_i \varphi_i, \sum_{j} u_j \varphi_j \right).
$$

$$
= a(u_h, u_h) \geq \nu \| u_h \|^2 \geq 0.
$$

Si $A_h U_h = 0$, alors  
$$
\|\sum u_i \varphi _i\| =\| u_h \| = 0.
$$

Donc :  
$$
\sum_{i} u_i \varphi_i = 0 \quad \Rightarrow \quad u_i = 0 \quad \forall i \in [1, N_h].
$$
$$
\Rightarrow U_h = 0.
$$

$A_h$ est donc inversible $\Rightarrow$ existence et unicité de la solution $u_h$.  

**Proposition** (Lemme de Céa) :  
Soit $u \in V$, solution du problème, et $u_h \in V_h$, solution approchée. Alors :  
$$
\| u - u_h \|_V \leq \frac{M}{V} \operatorname{dist}(u, V_h).
$$

**Remarque** : Si $\operatorname{dist}(u, V_h) \to 0$ lorsque $h \to 0$, alors $u_h \to u$.  

**Preuve** :

On a :  
$$
a(u, v_h) = \ell(v_h), \quad \forall v_h \in V_h \subset V,
$$
$$
a(u_h, v_h) = \ell(v_h), \quad \forall v_h \in V_h.
$$

on a donc :
$$
a(u - u_h, v_h) = 0, \quad \forall v_h \in V_h.
$$  

Soit $v_h = u_h - u \in V_h$ :  
$$
v \| u - u_h \|_V^2 \leq a(u - u_h, u - u_h) = a(u - u_h, u - v_h) + a(u - u_h, v_h - u_h).
$$  

Or :  
$$
a(u - u_h, v_h - u_h) = 0,
$$  
et  
$$
a(u - u_h, u - v_h) \leq M \| u - u_h \|_V \| u - v_h \|_V.
$$  

Ainsi :  
$$
\nu \| u - u_h \|_V \leq M \| u - v_h \|_V, \quad \forall v_h \in V_h.
$$  

En prenant l'infimum sur $v_h \in V_h$, on obtient :  
$$
\| u - u_h \|_V \leq \frac{M}{\nu} \operatorname{dist}(u, V_h).
$$  
**3) Éléments finis**  

**Construction de l’espace $V_h$ basé sur un maillage**  
Subdivision de $I = [0,1]$ :  
$$
0 = x_0 < x_1 < \dots < x_{N+1} = 1.
$$

$$
I_j = [x_j, x_{j+1}]
$$  
(taille de la maille).  

On note $h_j = x_{j+1} - x_j$  
et $h = \max_{j \in [0, N]} h_j$.

---

**a) Éléments finis Lagrange $P_1$**  

On considère :  
$$
V_h = \left\{ v \in C([0,1]) \, \Big| \, v|_{I_j} \in P_1, \, \forall j \in [0, N], \, v(0) = v(1) = 0 \right\}.
$$  

$$
\Rightarrow \text{fonctions affines par morceaux.}
$$  

(Diagramme représentant des fonctions continues, affines par morceaux, correspondant à chaque intervalle $I_j$ du maillage.)  


**Propriétés** :  
**1) Les fonctions de base**  
Pour $j \in [1, N]$, on définit :  
$$
\varphi_j(x) = \frac{x - x_{j-1}}{h_{j-1}} \mathbf{1}_{I_{j-1}}(x) + \frac{x_{j+1} - x}{h_j} \mathbf{1}_{I_j}(x).
$$  

Les $\varphi_j$ forment une base de $V_h$.  
Pour tout $v \in V_h$ :  
$$
v(x) = \sum_{j=1}^N v(x_j) \varphi_j(x).
$$  

$$
\dim V_h = N.
$$  

**2) Inclusion** :  
$$
V_h \subset C([0,1]) \cap H^1([0,1]).
$$  

**Preuve** :  

**1) Famille libre**  
$$
\sum_{j=1}^N \alpha_j \varphi_j = 0 \quad \Rightarrow \quad \sum_{j=1}^N \alpha_j \varphi_j(x_i) = 0 \quad \forall i \in [1, N].
$$  
Or, $\varphi_j(x_i) = \delta_{ij}$, donc :  
$$
\alpha_i = 0 \quad \forall i \in [1, N].
$$  

---

**Famille génératrice**  
Soit $v \in V_h$. Alors $v$ et $\sum v(x_j) \varphi_j$ coïncident en tous les $x_i$ et sont affines sur chaque $I_j$. Donc elles coïncident partout :  
$$
v = \sum v(x_j) \varphi_j.
$$  

---

**2)** $\varphi_j \in L^2(I)$ : OK.  
$$
\varphi_j' = \frac{1}{h_{j-1}} \mathbf{1}_{I_{j-1}} - \frac{1}{h_j} \mathbf{1}_{I_j} \in L^2(I).
$$  

最后还没校对11.27 ！！！


---


**Problème approché**  
Trouver $u_h \in V_h$  

$$ 
a(u_h, v_h) = \ell(v_h) \quad \forall v_h \in V_h 
$$

$$ 
V_h = \left\{ v \in \mathcal{C}([0,1]) \mid v_{|[x_j, x_{j+1}]} \in \mathbb{P}_1, \ v(0) = v(1) = 0 \right\} 
$$

Base : $(\phi_j)$  
$$ 
\phi_j(x_i) = \delta_{ji} 
$$
![[Pasted image 20241204081614.png]]

**Remarque : Degré de Liberté**  

Formes linéaires :  

$$ 
\sigma : \mathcal{C}([0,1]) \to \mathbb{R} 
$$
$$ 
v \mapsto \sigma(v) 
$$

Évaluation au point $x_j$  

On a :  

$$ 
\sigma_j(\phi_i) = \phi_i(x_j) = \delta_{ji} 
$$

Se donner les degrés de liberté d’une fonction $v \in V_h$ caractérise complètement la fonction : 

$$ 
v = \sum_{j=1}^N \sigma_j(v) \phi_j 
$$

$(\phi_j)$ sont appelées fonctions de forme.

**Remarque :**  

$$ 
A_h = \left( \int_I a(\phi_i, \phi_j) \right)_{i,j} 
$$ 
$$ 
= \left( \int_I \phi_i' \phi_j' + b \phi_i \phi_j \right)_{i,j} 
$$

$$ 
b_h = \left( \ell(\phi_i) \right) = \left( \int_I b \phi_i \right)_{i} 
$$

**Calcul explicite de $A_h$**  

$$ 
\text{Supp}(\phi_i) = I_{i-1} \cup I_i 
$$
$$ 
\text{Supp}(\phi_j) = I_{j-1} \cup I_j 
$$

$$ 
\int_I \phi_i \phi_j \neq 0 \ \text{seulement si} \ \text{Supp}(\phi_i) \cap \text{Supp}(\phi_j) \neq \emptyset 
$$
$$ 
\text{Si} \ j \in \{ i-1, i, i+1 \} 
$$

$$ 
\int_I \phi^2 = \int_{I_{i-1}} \phi^2 + \int_{I_i} \phi^2 
$$
$$ 
= \int_{[x_{i-1}, x_i]} \left( \frac{x - x_{i-1}}{h} \right)^2 dx + \int_{[x_i, x_{i+1}]} \left( \frac{x_{i+1} - x}{h} \right)^2 dx 
$$
$$ 
= \left[ \frac{(x - x_{i-1})^3}{3h^2} \right]_{x_{i-1}}^{x_i} + \left[ \frac{(x_{i+1} - x)^3}{3h^2} \right]_{x_i}^{x_{i+1}} 
$$
$$ 
= \left( \frac{h^3}{3h^2} - 0 \right) + \left( 0 + \frac{h^3}{3h^2} \right) 
$$
$$ 
= \frac{h}{3} + \frac{h}{3} 
$$  
$$
\int_I \phi_i \phi_{i+1} = \int_I \phi_i \phi_{i+1}
$$

$$
= \int_{[x_i, x_{i+1}]} \left(\frac{x_{i+1} - x}{h}\right)\left(\frac{x - x_i}{h}\right) dx
$$

$$
= \int_{[x_i, x_{i+1}]} \frac{(x_{i+1} - x)(x - x_i)}{h^2} dx
$$

$$
= h \int_{[0, 1]} y (1 - y) dy
$$

$$
= h \left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^1
$$

$$
= h \left(\frac{1}{2} - \frac{1}{3}\right)
$$

$$
= \frac{h}{6}
$$

---

$$
\int_I \phi_i \phi_i' + \phi_i' \phi_i' = 0 \ \text{si} \ j \in \{i-1, i, i+1\}
$$

$$
\int_I (\phi'_i)^2 = \int_{I_{i-1}} (\phi'_i)^2 + \int_{I_i} (\phi'_i)^2
$$

$$
= \frac{1}{h_{i-1}^2} \int_{I_{i-1}} 1 + \frac{1}{h_i^2} \int_{I_i} 1
$$

$$
= \frac{h_{i-1}}{h_{i-1}^2} + \frac{h_i}{h_i^2}
$$

$$
= \frac{1}{h_{i-1}} + \frac{1}{h_i}
$$

---

$$
\int_I \phi'_i \phi'_{i+1} = \int_{I_i} \phi'_i \phi'_{i+1}
$$

$$
= \int_{I_i} \left(-\frac{1}{h_i}\right) \left(\frac{1}{h_i}\right)
$$

$$
= \left(-\frac{1}{h_i^2}\right) h_i
$$

$$
= -\frac{1}{h_i}
$$

$$
(A_h)_{i,j} =
\begin{bmatrix}
\frac{h_{i-1}}{3} + \frac{h_i}{3} & \frac{h_i}{6} & 0 & \cdots & 0 \\
\frac{h_i}{6} & \frac{h_i}{3} + \frac{h_{i+1}}{3} & \frac{h_{i+1}}{6} & \cdots & 0 \\
0 & \frac{h_{i+1}}{6} & \frac{h_{i+1}}{3} + \frac{h_{i+2}}{3} & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \frac{h_{n-1}}{3} + \frac{h_n}{3}
\end{bmatrix}
$$
$$
+
\begin{bmatrix}
\frac{1}{h_{i-1}} + \frac{1}{h_i} & -\frac{1}{h_i} & 0 & \cdots & 0 \\
-\frac{1}{h_i} & \frac{1}{h_i} + \frac{1}{h_{i+1}} & -\frac{1}{h_{i+1}} & \cdots & 0 \\
0 & -\frac{1}{h_{i+1}} & \frac{1}{h_{i+1}} + \frac{1}{h_{i+2}} & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \frac{1}{h_{n-1}} + \frac{1}{h_n}
\end{bmatrix}
$$

---

2) **Convergence**

**Rappel :** Lemme de Céa  
$$
\| u - u_h \| \leq M \inf_{v_h \in V_h} \| u - v_h \| \to 0
$$

**Prop :**  
Notant  
$$
\Pi_h : \mathcal{C}([0, 1]) \to V_h
$$  
$$
v \mapsto \sum_{i=1}^N v(x_i) \phi_i
$$  
on a pour tout $v \in H^2(I) \subset \mathcal{C}([0, 1])$.


$$
\|u - \Pi_h u\|_{L^2}^2 \leq h^2 \|u''\|_{L^2}^2
$$

$$
\|u - \Pi_h u\|_{H^1}^2 \leq h \|u''\|_{L^2}^2
$$

**Remarque :**  
$$
\text{dist}(u, V_h) \leq \|u - \Pi_h u\|_{H^1} \leq O(h^2) \to 0
$$

**Preuve :**  
Si $u \in H^1(I)$, alors :  
Pour tout $a, b$,  
$$
u(b) - u(a) = \int_{[a,b]} u'(x) dx
$$

Admis :  
$$
|u(b) - u(a)| = \left| \int_{[a,b]} u'(x) dx \right|
$$

$$
\leq \int_{[a,b]} 2x |u'(x)| dx \leq \sqrt{\int_{[a,b]} 1^2 dx} \cdot \sqrt{\int_{[a,b]} (u'(x))^2 dx}
$$

$$
\leq \sqrt{b - a} \cdot \sqrt{\int_{[a,b]} (u'(x))^2 dx}
$$

Si $u \in H^2(I) \subset \mathcal{C}^1(I)$, alors $\Pi_h u \in P_1$ sur tout intervalle $I_j$.

Donc $u - \Pi_h u \in \mathcal{C}^1 \ \forall x \in [0, 1]$.  

On a : $(u - \Pi_h u)(x_j) = 0$ et $(u - \Pi_h u)(x_{j+1}) = 0$.  

Donc, d'après le théorème de Rolle sur l'intervalle $I_j$,  
$$
\exists \xi \in ]x_j, x_{j+1}[, \quad (u - \Pi_h u)'(\xi) = 0
$$

Donc, pour tout $x \in I_j$, 

$$
|(u - \Pi_h u)'(x)| = |(u - \Pi_h u)'(x) - (u - \Pi_h u)'(\xi)|
$$

$$
\leq \sqrt{\int_{[x, \xi]} ((u - \Pi_h u)'')^2 dx} \quad (\text{d'après l'inégalité précédente appliquée à } v = (u - \Pi_h u)')
$$

$$
\leq \sqrt{\int_{I_j} ((u - \Pi_h u)'')^2 dx}.
$$

$$
\int_{I_j} ((u - \Pi_h u)')^2 dx \leq h_j^2 \int_{I_j} ((u - \Pi_h u)'')^2 dx.
$$

$$
\int_I ((u - \Pi_h u)')^2 dx = \sum_{j=0}^{N-1} \int_{I_j} ((u - \Pi_h u)')^2 dx
$$

$$
\leq \sum_{j=0}^{N-1} h_j^2 \int_{I_j} ((u - \Pi_h u)'')^2 dx
$$

$$
\leq h^2 \|u''\|_{L^2(I)}^2.
$$


$$
\|u - \Pi_h u\|_{H^1} \leq h \|u''\|_{L^2(I)}.
$$

$$
u - \Pi_h u \in \mathcal{C}^1(I), \quad (u - \Pi_h u)(x_j) = 0.
$$  

Donc, pour tout $x \in I_j$,  
$$
(u - \Pi_h u)(x) = |(u - \Pi_h u)'(x) - (u - \Pi_h u)'(\xi)|.
$$

$$
\int_I ((u - \Pi_h u)')^2 \leq h^2 \int_I ((u'')^2).
$$

$$
\|u - \Pi_h u\|_{H^1}^2 = \|u - \Pi_h u\|_{L^2}^2 + \| (u - \Pi_h u)' \|_{L^2}^2.
$$

En utilisant le même calcul que précédemment, on obtient :
$$
\|u - \Pi_h u\|_{L^2}^2 \leq h^2 \|u''\|_{L^2}^2.
$$

$$
\| (u - \Pi_h u)' \|_{L^2}^2 \leq h^2 \|u''\|_{L^2}^2.
$$

Donc :
$$
\|u - \Pi_h u\|_{H^1} \leq h \|u''\|_{L^2}.
$$

**Prop :**

Comme la solution $u \in H_0^1(I)$ appartient à $H^2(I)$, le schéma éléments finis $P_1$ converge.  

On a de plus :
$$
\|u - u_h\|_{H^1} \leq \frac{M}{\nu} h \|u''\|_{L^2}.
$$

Et aussi :
$$
\|u - u_h\|_{L^2} \leq \left(\frac{M}{\nu}\right) h^2 \|u''\|_{L^2}.
$$

---

**Preuve :**

1. **Lemme de Céa :**
$$
\text{dist}(u, V_h) \leq \|u - u_h\|_{H^1}.
$$

2. Méthode d'Aubin-Nitsche :  
En notant $e = u - u_h$, on considère le problème dual.

**Trouver $g \in H_0^1$,**  

$$
a(g, v) = (e_h, v)_{L^2} \quad \forall v \in H_0^1.
$$

---

$$
\int_I g' v' = \int_I e_h v \quad \text{pour tout } v.
$$

La solution de ce problème $g$ appartient à $H^2(I)$ (puisque $e_h \in L^2$) et de plus :  
$$
\|g\|_{H^2} \leq C \|e_h\|_{L^2}.
$$

---

$$
\|e_h\|_{L^2}^2 = (e_h, e_h)_{L^2} = a(g, e_h).
$$

$$
= a(g, e_h) - a(\Pi_h g, e_h) \quad (\Pi_h g \in V_h).
$$

$$
= a(g - \Pi_h g, e_h).
$$

---

$$
\leq M \|g - \Pi_h g\|_{H^1} \|e_h\|_{H^1}.
$$

---

$$
\|g - \Pi_h g\|_{H^1} \leq C h \|g''\|_{L^2}.
$$

---

$$
\|e_h\|_{L^2} \leq M h \|g''\|_{L^2}.
$$

---

**3. Éléments finis de Lagrange $P_2$**  

On considère :  
$$
V_h = \{ v \in \mathcal{C}([0,1]) \mid v_{|I_k} \in P_2, \forall k \in [0, N-1], v(0) = v(1) = 0 \}.
$$

Continu et polynôme de degré $2$ sur chaque sous-intervalle.

Dans chaque $I_k$, on considère les points :  
$$
x_{k,0} = x_k, \quad x_{k,1} = x_k + \frac{h_k}{2}, \quad x_{k,2} = x_{k+1},
$$
avec $h_k = x_{k+1} - x_k$, $t \in [0, h_k]$.  

On considère les polynômes élémentaires de Lagrange :  
$$
l_{k,m}(x) = \frac{\prod_{n \neq m} (x - x_{k,n})}{\prod_{n \neq m} (x_{k,m} - x_{k,n})}, \quad l_{k,m} \in P_2.
$$

On a :  
$$
l_{k,m}(x_{k,n}) = \delta_{mn}.
$$

**Propriété :** Les polynômes $l_{k,m}$ forment une base de $V_h$.

Pour $\forall j \in [1, N]$ :  
$$
\phi_j(x) =
\begin{cases}
l_{j,0}(x) \quad \text{sur } I_j, \\
l_{j-1,2}(x) \quad \text{sur } I_{j-1}, \\
0 \quad \text{ailleurs.}
\end{cases}
$$

Pour $\forall j \in [0, N], \forall m \in [1, h-1]$ :  
$$
P_{j,m}(x) =
\begin{cases}
l_{j,m}(x) \quad \text{sur } I_j, \\
0 \quad \text{ailleurs.}
\end{cases}
$$

La dimension de $V_h$ est :  
$$
\text{dim}(V_h) = N + (N+1)(h-1) = (N+1)h - 1.
$$

---

**Propriété :** Si $u \in H^4(I)$, alors :  
$$
\|u - u_h\|_{H^1} \leq C h^2 \|u\|_{H^3}.
$$

$$
\|u - u_h\|_{L^2} \leq C h^3 \|u\|_{H^4}.
$$

