Chap 4 : Méthodes numériques :

La méthode est dite convergente si $J(x_k) \rightarrow \min J$

Déf. (Ordre de convergence) On dit que la méthode est d’ordre $\alpha > 0$, si $\alpha$ est le plus grand réel t.q. $\exists C > 0$ :

$$
\|x_{n+1} - \bar{x}\| \leq C \|x_n - \bar{x}\|^\alpha
\quad \text{où } \bar{x} = \arg\min J
$$

On notera $e_n = \|x_n - \bar{x}\|$.

Si $\alpha = 1$, on dit que la méthode est linéaire.  
Si $\alpha > 1$, on dit que la méthode est semi-linéaire  
Si $\alpha = 2$, on dit que la méthode est quadratique.

## 1) Algorithme sans gradient : méthode de Dichotomie :

Soit $J : \mathbb{R} \rightarrow \mathbb{R}$ et $J \in \mathcal{C}^0([a,b])$ et unimodale, c-à-d décroissante puis croissante.

![[image-8.png]]

**Algo. :**  
Données : $J \in \mathcal{C}^0([a,b])$ unimodale  
Tant que $(b-a) > \varepsilon$ faire :

- on choisit $x < y \in (a,b)$  
- Si $J(x) \leq J(y)$, alors le min. se trouve entre $a$ et $y$ : $b \leftarrow y$  
- Si $J(x) > J(y)$, $\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad$ : $a \leftarrow x$

Fin tant que :

Cet algo. définit deux suites $(a_k)$ et $(b_k)$

En pratique on choisit :  
$$
\begin{cases}
x_k = a_k + (1 - \rho)(b_k - a_k) \\
y_k = a_k + \rho(b_k - a_k)
\end{cases}
\quad \text{avec } \rho \in \left] \frac{1}{2}, 1 \right[
$$

Donc dans ce cas $b_{k+1} - a_{k+1} = \rho(b_k - a_k)$ d’où la convergence.

**Rq :** suite dorée  
On veut réutiliser la valeur du point qu’on a déjà évaluée.

$$
\text{Y : } a_{k+1} = a_k \text{ et } b_{k+1} = y_k, \text{ on veut } y_{k+1} = x_k
$$

$$
y_{k+1} = x_k \iff a_k + \rho (b_k - a_k) = a_k + (1 - \rho)(b_k - a_k)
$$

$$
\iff \rho + \rho^2 (b_k - a_k) = a_k + (1 - \rho)(y_k - a_k)
$$

$$
\iff \rho^2 + \rho - 1 = 0
$$

Les solutions sont $\rho_\pm = \frac{-1 \pm \sqrt{5}}{2}$. On a $\rho_+ \simeq 0{,}618 \in \left] \frac{1}{2}, 1 \right[$

Donc on prend :  
$$
\rho = \frac{-1 + \sqrt{5}}{2}
$$

---

## 2) Algorithme de gradient :

(Dessin d’une courbe avec points $x_{k+1}$, $x_k$, et flèche $-\nabla J$)

![[image-7.png]]

On considère $(x_n)$ de forme :  
$$
x_{n+1} = x_n + s_n d_n
$$  
où $d_n \in \mathbb{R}^n$ est la direction de descente et $s_n$ le pas de descente.

On souhaite choisir une direction de descente :  
$$
\langle \nabla J(x_n), d_n \rangle \leq 0
$$

Ainsi pour $s_n$ suffisamment petit :  
$$
J(x_{n+1}) < J(x_n)
$$

**Algo :**  
Données : $J, x_0$  
$x \leftarrow x_0$

Tant que "critère d'arrêt" faire :

- Choix de la direction de descente $d_k$  
- Choix du pas de descente $s_k$  
- $x \leftarrow x + s d$

**Fin**

**Critères d’arrêt possibles :**

- Nombre d’itérations max : $k < k_{\text{max}}$  
- Stagnation de la solution : $\| x_{n+1} - x_n \| \leq \varepsilon (1 + \| x_n \|)$  
- _______ du gradient : $\| \nabla J(x_n) \| \leq \varepsilon$  
- _______ de $J$ : $| J(x_{n+1}) - J(x_n) | \leq \varepsilon (1 + | J(x_n) |)$

En pratique, on prend $k < k_{\text{max}}$ et une stagnation.

---

**a) Algo de gradient :**  
On prend $d_k = - \nabla J(x_k)$

**Rq :** $\forall d$ t.q. $\| d \| \leq 1$, on a  
$$
| \langle \nabla J(x_n), d \rangle | \leq \| \nabla J(x_n) \| \quad \text{(Cauchy-Schwartz)}
$$

avec égalité si $d = \pm \frac{\nabla J(x_n)}{\| \nabla J(x_n) \|}$

Ainsi $d = - \frac{\nabla J(x_n)}{\| \nabla J(x_n) \|} = \frac{d_k}{\| d_k \|}$ est la direction de plus grande pente.

**(i) Pas optimal**

Méthode du gradient à pas optimal : on choisit $s_n$ t.q. $x_{n+1}$ minimise $J$ dans la direction $d_n$, c-à-d :

$$
s_n = \arg\min_{s > 0} J(x_n + s d_n)
$$

**Prop. :** Si $f$ est $\alpha$-convexe et $\mathcal{C}^1$, alors la méthode à pas optimal converge.

**Rq : (Formule quadratique)** La méthode à pas opt. est explicite.

Soit $\varphi(s) = J(x_n + s d_n)$, où  
$$
J(x) = \frac{1}{2} \langle A x, x \rangle - \langle b, x \rangle
$$

Car  
$$
0 = \varphi'(s) = \langle \nabla J(x_n + s d_n), d_n \rangle = \langle A(x_n + s d_n) - b, d_n \rangle
$$

$$
= \langle -d_n + s A d_n, d_n \rangle
$$

D’où  
$$
s_n = \frac{\| d_n \|^2}{\langle A d_n, d_n \rangle}
$$
**Rq : (cas général)** On peut par exemple trouver le pas optimal avec une méthode de Dichotomie.

---

**(ii) Pas constant :**  
On choisit $s_k = s$.

**Prop. :** Si $J$ est $\alpha$-convexe, $\mathcal{C}^1$, et $\nabla J$ $M$-Lipschitz, alors pour  
$$
s \in \left] 0, \frac{2\alpha}{M^2} \right[, \text{ la méthode converge linéairement.}
$$

**Preuve :**  
Comme $\nabla J(\bar{x}) = 0$,  
$$
x_{n+1} - \bar{x} = x_n - \bar{x} - s_k ( \nabla J(x_n) - \nabla J(\bar{x}) )
$$

$$
\Rightarrow \| x_{n+1} - \bar{x} \|^2 = \| x_n - \bar{x} \|^2 - 2 s_k \langle x_n - \bar{x}, \nabla J(x_n) \rangle + s_k^2 \| \nabla J(x_n) \|^2
$$

$$
\leq \| x_n - \bar{x} \|^2 - 2 s_k \alpha \| x_n - \bar{x} \|^2 + s_k^2 M^2 \| x_n - \bar{x} \|^2
$$

$$
= (1 - 2 s_k \alpha + s_k^2 M^2) \| x_n - \bar{x} \|^2
$$

On converge si  
$$
\beta = \sup (1 - 2 s_k \alpha + s_k^2 M^2) < 1
$$

Ce polynôme $1 - 2 s \alpha + M^2 s^2$ atteint 1 en 0 et $\frac{2 \alpha}{M^2}$.  
Donc si $s \in \left] a, b \right[$ avec $a > 0$, $b < \frac{2 \alpha}{M^2}$, on a $\beta < 1$.

**Rq : (fonctionnelle quadratique)** La méthode converge pour $s \in \left] 0, \frac{2}{\lambda_{\max}} \right[$

---

**b) Algo. du gradient conjugué :**

Soit $J : \mathbb{R}^n \rightarrow \mathbb{R}$  
$$
x \mapsto \frac{1}{2} \langle A x, x \rangle - \langle b, x \rangle
$$

On considère $x_{n+1} = x_n + s_n d_n$

avec $d_n = - \nabla J(x_n) + \beta_{n-1} d_{n-1}$ et  
$$
s_n = \frac{ \langle d_n, -\nabla J(x_n) \rangle }{ \langle d_n, A d_n \rangle }
$$

avec $\beta_{n-1} \in \mathbb{R}$ t.q. $\langle d_n, A d_{n-1} \rangle = 0$

**Prop :** Pour tout $k \leq m - 1$, $x_k$ minimise $J$ sur $\text{Vect}(d_0, \dots, d_k)$ et les directions $(d_0, \dots, d_{n})$ sont $A$-conjuguées :  
$$
\langle d_i, A d_j \rangle = 0 \quad \forall i \neq j
$$

On en déduit que $x_m = \bar{x}$

**Preuve :** Montrons par récurrence que $\nabla J(x_{n+1})$ est orthogonal à $\text{Vect}(d_0, \dots, d_n)$ et que les directions de descente $(d_0, \dots, d_n)$ sont $A$-conjuguées.

Supposons que c’est vrai au rang $k-1$. Montrons $\nabla J(x_{n+1})$ orthogonal à $\text{Vect}(d_0, \dots, d_n)$ :

$$
\nabla J(x_{n+1}) = \nabla J(x_n) + s_n A d_n
$$

Soit  
$$
\langle \nabla J(x_{n+1}), d_n \rangle = \langle \nabla J(x_n), d_n \rangle + s_n \langle A d_n, d_n \rangle = 0
$$  
(par définition de $s_n$)

et  
$$
\langle \nabla J(x_{n+1}), d_j \rangle = \langle \nabla J(x_n), d_j \rangle + s_n \langle A d_n, d_j \rangle = 0 \quad \text{par Hyp. de récurrence}
$$

Montrons que $d_{n+1}$ est $A$-conjuguée à $(d_j)_{j \leq n}$ :

$$
\langle A d_j, d_{n+1} \rangle = \langle -\nabla J(x_{n+1}), A d_j \rangle + \beta_k \langle d_k, A d_j \rangle = 0 \quad \text{par déf. de } \beta_k
$$

$j \leq n, \langle d_j, A d_j \rangle = \langle \nabla J(x_n), A d_j \rangle + \beta_k \langle d_k, A d_j \rangle$

$$
A d_j = \frac{\nabla J(x_{n+1}) - \nabla J(x_n)}{s_n} \in \text{Vect}(d_0, \dots, d_{n}) \subset \text{Vect}(d_0, \dots, d_k)
$$

Donc  
$$
\langle d_{n+1}, A d_j \rangle = 0
$$
**Rq :**  
1) On note que les gradients $\nabla J(x_n)$ ne sont pas forcément orthogonaux deux à deux, mais forment une famille orthogonale.  

---

2) **Version non-linéaire : méthode de Fletcher–Reeves**  

$$
d_n = -\nabla J(x_n) + \beta_{n-1} d_{n-1} \quad \text{avec } \beta_{n-1} = \frac{ \| \nabla J(x_{n+1}) \|^2 }{ \| \nabla J(x_n) \|^2 }
$$

---

**Prop :**  
Si $J$ est différentiable, $\alpha$-convexe, $\nabla J$ $M$-Lipschitz, alors l’algo converge pour  
$$
\beta_{n-1} = \frac{ \langle \nabla J(x_{n+1}) - \nabla J(x_n), \nabla J(x_n) \rangle }{ \| \nabla J(x_n) \|^2 } \quad \text{(Polak–Ribière)}
$$

$$ x_{n+1} = x_n + s_n d_n $$

$$ s_n = \arg\min J(x_n + s d_n) $$

$$ A x = b $$

---

## 3) Algorithme de Newton :

On rappelle qu’il est entre autres utilisé  
pour résoudre $$ F(x) = 0 $$  
avec $F : \mathbb{R}^n \to \mathbb{R}^n$, $\mathcal{C}^1$ et $DF(x)$ inversible.  

L’itérée $x_{k+1}$ est solution :  

$$ F(x_k) + DF(x_k)(x - x_k) = 0 $$

On en déduit donc :  
$$ x_{k+1} = x_k - DF(x_k)^{-1} F(x_k) $$

Dans notre problème d’optimisation, on l’applique  
à l’équation d’Euler :  

$$ \nabla J(x) = 0 $$

i.e.  
$$ \nabla J(x_n) + \nabla^2 J(x_n)(x - x_n) = 0 $$

L’itérée est donc définie par :  
$$ x_{n+1} = x_n + d_n, \quad \text{où} \quad d_n = -\nabla^2 J(x_n)^{-1} \nabla J(x_n) $$

---

**Remarque 1)** Si on remplace $J$ par son approximation quadratique :  

$$ G(x) = J(x_n) + \langle \nabla J(x_n), x - x_n \rangle + \frac{1}{2} \langle \nabla^2 J(x_n)(x - x_n), x - x_n \rangle $$

alors en supposant $\nabla^2 J(x)$ définie positive, $G$ atteint son minimum en $x_{n+1}$.

2) Si $\nabla^2 J(x_n)$ est pos., on a $\langle d_n, \nabla J(x_n) \rangle < 0,$  

alors $d_n$ est une direction de descente.

**Proposition :** Soit $J : \mathbb{R}^n \rightarrow \mathbb{R}$ une fonction $\mathcal{C}^2$  
et $\bar{x}$ solution de $\nabla J(\bar{x}) = 0$ et $\nabla^2 J(\bar{x})$ inversible.  

Alors $J \geq 0$ tq. $\forall x_0 \in \mathbb{R}^n$, $\| x_0 - \bar{x} \| \ll \varepsilon$,  
la méthode de Newton converge vers $\bar{x}$ et $\exists C > 0$,  
$$
\| x_{n+1} - \bar{x} \| \leq C \| x_n - \bar{x} \|^2
$$

**Remarque :** Avantage : la méthode est quadratique.  
Inconvénient : elle est locale. De plus, elle est  
coûteuse car il faut résoudre un syst. lin.

---

## 4) Algorithme de quasi-Newton :

Un inconvénient de la méthode de Newton  
est de connaître la Hessienne, et il faut résoudre  
un système linéaire.  

On considère une méthode plus générale :  
$$
x_{n+1} = x_n - s_n W_n \nabla J(x_n)
$$  
où $W_n$, matrice symétrique qui est une approximation  
de l’inverse de la Hessienne.  

On impose $W_n$ vérifiant :  
$$
W_n (\nabla J(x_n) - \nabla J(x_{n-1})) = x_n - x_{n-1}
$$  

Car on a :  
$$
\nabla J(x) = \nabla J(x_{n-1}) + \nabla^2 J(x_{n-1})(x - x_{n-1}) + o(\| x - x_{n-1} \|)
$$
---

La méthode BFGS consiste à choisir  
$W_n$ la matrice minimisant l’écart  
avec $W_{n-1}$ :

$$
W_n = \arg\min_{W \in \mathcal{S}_n} \| W - W_{n-1} \|_F^2
$$

$\mathcal{S}_n \leftarrow$ matr. sym.  
$W(s_n) = \delta_n$

où $\| \cdot \|_F$ la norme de Frobenius,  
$\delta_n = \nabla J(x_n) - \nabla J(x_{n-1})$  
et $s_n = x_n - x_{n-1}$

Donc :

$$
W_n = A W_{n-1} A^T + \frac{\delta_n \otimes \delta_n}{\langle \delta_n, s_n \rangle}
$$

$$
A = I_d - \frac{s_n \otimes \delta_n}{\langle \delta_n, s_n \rangle}
$$

---

**III) Algorithmes de type gradient pour l’optimisation avec contrainte :**

**1) Algorithme par pénalisation :**

On considère  
$$
\min_{x \in K} J(x)
$$  
où $K = \{ x : h(x) = 0 \text{ ou } g(x) \leq 0 \}$

Considère le problème :  
$$
\min J(x) + \frac{1}{\varepsilon} \alpha(x), \quad \varepsilon > 0
$$

où  
$$
\alpha(x) =  
\begin{cases}  
\| h(x) \|^2 & \text{si } K = \{ x : h(x) = 0 \} \\
\| [g(x)]^+ \|^2 & \text{si } K = \{ x : g(x) \leq 0 \}
\end{cases}
$$  
avec $x^+ = \max(0, x)$

---

**Proposition** Soit $J$ continue coercive sur $K$ fermé  
non vide. On suppose $\alpha$ continue, positive  
et nulle uniquement sur $K$. Alors :  

- $\forall \varepsilon > 0$, $J_\varepsilon = J + \frac{1}{\varepsilon} \alpha$ admet un minimum $x_\varepsilon$  
- $(x_\varepsilon)$ bornée  
- Toute sous-suite convergente de $(x_\varepsilon)$  
  converge vers un minimum de $J$ sur $K$

**Preuve :**  
$J$ continue coercive $\Rightarrow J$ a au moins un minimum $\bar{x}$ sur $K$ fermé.  

De la même manière, $J$ a au moins un minimum $x_\varepsilon$ de $J_\varepsilon$

---

On a  
$$
J(x_\varepsilon) \leq J(x_\varepsilon) + \frac{1}{\varepsilon} \alpha(x_\varepsilon) = J_\varepsilon(x_\varepsilon)  
\leq J_\varepsilon(\bar{x}) = J(\bar{x})
$$

Donc $(J(x_\varepsilon))_\varepsilon$ est bornée.  

Par coercivité, $(x_\varepsilon)_\varepsilon$ également.  

Soit $(x_{\varepsilon_k})$ une sous-suite convergente  
vers un $x^*$. D’après l’inégalité précédente :  
$$
J(x^*) \leq J(\bar{x})
$$  

Reste à montrer que $x^* \in K$.  

On a  
$$
\alpha(x_{\varepsilon_k}) \leq \varepsilon_k (J(\bar{x}) - J(x_{\varepsilon_k}))
$$
---

En passant à la limite, comme $J(\bar{x}) - J(x_{\varepsilon_k})$  
borné, alors $\alpha(x^*) = 0$.  
Donc $x^* \in K$.  

On en déduit que $x^*$ est bien un minimum  
de $J$ sur $K$.

---

**2) Algorithme du gradient projeté :**  

On suppose $K$ convexe fermé.  

On considère  
$$
x_{k+1} = P_K(x_k - s_k \nabla J(x_k))
$$  
où $P_K : \mathbb{R}^n \to K$ est l’opérateur de projection sur $K$

$$
P_K(x) = \arg\min_{y \in K} \| x - y \|
$$

On rappelle  
$$
P_K(x) \in K \quad \text{et} \quad \langle x - P_K(x), y - P_K(x) \rangle \leq 0 \quad \forall y \in K
$$

**Remarque :** Inconvénient : la méthode nécessite de connaître  
l’opérateur de projection. Par exemple pour $K = \prod [a_i, b_i]$  
$$
P_K(x) = \max(a_i, \min(b_i, x_i))
$$

**Proposition :** Si $J$ est $\alpha$-convexe, $\mathcal{C}^1$,  
$\nabla J$ M-Lipschitz,  

alors en prenant $s$ constant dans $\left] 0, \frac{2\alpha}{M^2} \right[$,  
la méthode du gradient projeté converge linéairement.

---

**Preuve :** $\forall x \in K$ :  
$$
\langle \bar{x} - s \nabla J(\bar{x}) - \bar{x}, x - \bar{x} \rangle = -s \langle \nabla J(\bar{x}), x - \bar{x} \rangle \leq 0  
$$  
par l’inégalité d’Ekrov.

Ainsi  
$$
\bar{x} = P_K(\bar{x} - s \nabla J(\bar{x})) \quad \forall s
$$

On a :  
$$
x_{n+1} - \bar{x} = P_K(x_n - s \nabla J(x_n)) - P_K(\bar{x} - s \nabla J(\bar{x}))
$$  
$$
\| x_{n+1} - \bar{x} \|^2 \leq \| (x_n - s \nabla J(x_n)) - (\bar{x} - s \nabla J(\bar{x})) \|^2  
$$  
par contraction de la projection sur un convexe.

---

D’où  
$$
\| x_{n+1} - \bar{x} \|^2 = \| x_n - \bar{x} \|^2 - 2s \langle x_n - \bar{x}, \nabla J(x_n) - \nabla J(\bar{x}) \rangle + s^2 \| \nabla J(x_n) - \nabla J(\bar{x}) \|^2
$$  
$$
\leq \| x_n - \bar{x} \|^2 - 2s \alpha \| x_n - \bar{x} \|^2 + s^2 M^2 \| x_n - \bar{x} \|^2
$$  
$$
= (1 - 2s \alpha + s^2 M^2) \| x_n - \bar{x} \|^2
$$

Le schéma est convergent si  
$$
1 - 2s \alpha + s^2 M^2 \leq 1 \quad \Rightarrow \quad s < \frac{2\alpha}{M^2}
$$

---

**3) Algorithme d’Uzawa :**  

(L’algo. d’Uzawa consiste à déterminer un point  
selle du Lagrangien. Les points selles du Lagrangien  
réalisent un max. dans la direction des multiplicateurs et  
un min. suivant $x$.)

---

Le Lagrangien est défini par :  
$$
\forall (x, \lambda, \mu) \in \mathcal{V} \times \mathbb{R}^p \times \mathbb{R}^q_+,  
\quad L(x, \lambda, \mu) = J(x) + \sum_{i=1}^p \lambda_i h_i(x) + \sum_{i=1}^q \mu_i g_i(x)
$$

**Problème primal :**  
$$
\min_{x \in \mathcal{V}} J(x) \quad \text{avec} \quad \mathscr{G}(x) = \max_{\lambda, \mu} L(x, \lambda, \mu)
$$

**Problème dual :**  
$$
\max_{(\lambda, \mu) \in \mathbb{R}^p \times \mathbb{R}^q_+} \mathscr{G}(\lambda, \mu)  
\quad \text{avec} \quad \mathscr{G}(\lambda, \mu) = \min_{x \in \mathcal{V}} L(x, \lambda, \mu)
$$

Les dénominations viennent du fait que le problème  
primal est exactement équivalent à minimiser :

$$
H(x) = 
\begin{cases}
J(x) & \text{si } x \in K \\
+\infty & \text{si } x \notin K
\end{cases}
$$

---

**Remarque :** La solution du problème dual est toujours plus  
petite que celle du problème primal :

$$
\max_{\lambda, \mu} \min_x L(x, \lambda, \mu) \leq \min_x \max_{\lambda, \mu} L(x, \lambda, \mu)  
= \min_x \mathscr{G}(x)
$$

---

**Théorème (dualité) :**  
$$
(\bar{x}, \bar{\lambda}, \bar{\mu}) \text{ point selle ssi } \bar{x} \text{ solution  
du problème primal et } (\bar{\lambda}, \bar{\mu}) \text{ solution du problème dual.}
$$

$$
\mathscr{G}(\bar{\lambda}, \bar{\mu}) = J(\bar{x})
$$

---

**Preuve :** Supposons que $(\bar{x}, \bar{\lambda}, \bar{\mu})$ point selle. En prenant  
le max :

$$
\max_{\lambda, \mu} L(\bar{x}, \lambda, \mu) \leq L(\bar{x}, \bar{\lambda}, \bar{\mu})
$$

La seconde inégalité implique  
$$
L(\bar{x}, \bar{\lambda}, \bar{\mu}) \leq \max_x L(x, \bar{\lambda}, \bar{\mu}) \Rightarrow  
\mathscr{G}(\bar{x}) = J(\bar{x})
$$

---

On en déduit que $\bar{x}$ est solution du problème  
primal et $\mathscr{G}(\bar{x}) = L(\bar{x}, \bar{\lambda}, \bar{\mu})$.

On a  
$$
L(\bar{x}, \bar{\lambda}, \bar{\mu}) \geq \min_x L(x, \bar{\lambda}, \bar{\mu})  
= \mathscr{G}(\bar{\lambda}, \bar{\mu})
$$

D’après la première inégalité :  
$$
L(\bar{x}, \bar{\lambda}, \bar{\mu}) \geq L(\bar{x}, \lambda, \mu) \geq \min_x L(x, \lambda, \mu) = \mathscr{G}(\lambda, \mu)
$$

On en déduit que $(\bar{\lambda}, \bar{\mu})$ solution du problème  
dual et $\mathscr{G}(\bar{\lambda}, \bar{\mu}) = L(\bar{x}, \bar{\lambda}, \bar{\mu})$.

---

Supposons que $(\bar{x}, \bar{\lambda}, \bar{\mu})$ vérifie  
$$
\mathscr{G}(\bar{\lambda}, \bar{\mu}) = \mathscr{G}(\bar{x})
$$

Alors  
$$
L(\bar{x}, \lambda, \mu) \geq \mathscr{G}(\bar{x}) = \mathscr{G}(\bar{\lambda}, \bar{\mu}) \geq L(x, \bar{\lambda}, \bar{\mu})
$$

Donc  
$$
L(\bar{x}, \bar{\lambda}, \bar{\mu}) = \mathscr{G}(\bar{x}) = \mathscr{G}(\bar{\lambda}, \bar{\mu})
$$  
car $(\bar{x}, \bar{\lambda}, \bar{\mu})$ point selle.

---

Le principe d’Uzawa consiste à résoudre  
le problème dual, c’est-à-dire à déterminer $(\bar{\lambda}, \bar{\mu})$  
puis $\bar{x}$. Pour cela, on procède de la façon  
suivante :

- Résolution du dual pour déterminer $(\bar{\lambda}, \bar{\mu})$  
- Résolution du problème sans contrainte $\min_x L(x, \bar{\lambda}, \bar{\mu})$

Les deux étapes sont en réalité réalisées en même temps :

$$
\left\{
\begin{aligned}
\lambda_{k+1} &= \lambda_k + s \nabla_\lambda \mathscr{G}(\lambda_k, \mu_k) \\
\mu_{k+1} &= P_{\mathbb{R}_+^q}(\mu_k + s \nabla_\mu \mathscr{G}(\lambda_k, \mu_k)) \\
x_{k+1} &= \arg\min_x L(x, \lambda_{k+1}, \mu_{k+1})
\end{aligned}
\right.
$$

---

$$
\partial_{\lambda_i} \mathscr{G}(\lambda, \mu) = \langle \nabla_x L(x_{\lambda,\mu}, \lambda, \mu), \partial_{\lambda_i} x_{\lambda,\mu} \rangle + \partial_{\lambda_i} L(x_{\lambda,\mu}, \lambda, \mu) = 0 + h_i(x_{\lambda,\mu})
$$

où $x_{\lambda,\mu}$ solution de $\min_x L(x, \lambda, \mu)$

De même :

$$
\partial_{\mu_i} \mathscr{G}(\lambda, \mu) = -g_i(x_{\lambda,\mu})
$$

---

**Ainsi :**

$$
\left\{
\begin{aligned}
\lambda_{k+1} &= \lambda_k + s h(x_k) \\
\mu_{k+1} &= P_{\mathbb{R}_+^q}(\mu_k + s g(x_k)) \\
x_{k+1} &= \arg\min_x L(x, \lambda_{k+1}, \mu_{k+1})
\end{aligned}
\right.
$$

---

**Théorème :** Soit $J$ $\alpha$-convexe, différentiable, $h_i$ affine,  
$g_i$ convexe, $C$-Lipschitz. Et supposons qu’il  
existe un point selle $(x, \lambda, \mu)$. Alors,  

si $s \in \left] 0, \frac{2\alpha}{C^2} \right[$, l’algorithme d’Uzawa  
converge.

