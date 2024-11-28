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

最后还没校对