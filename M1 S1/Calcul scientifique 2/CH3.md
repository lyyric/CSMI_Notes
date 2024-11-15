Chap. 3: Méthodes itératives pour les systèmes linéaires creux

## I Rappels sur les méthodes itératives

> [!question]
> Résoudre $Ax = b$ avec $A \in M_n(\mathbb{C})$ et $b \in \mathbb{C}^n$.

Méthode : décomposition $A = M - N$

$Ax = b$  
$\Leftrightarrow (M - N)x = b$  
$\Leftrightarrow Mx = Nx + b$  

Suite itérative $(x_p)$ définie par  $Mx_{p+1} = Nx_p + b$  

> [!remark]
> $x_{p+1}$ est solution d’un système linéaire. À chaque itération, on doit résoudre un système linéaire.
> Si la suite itérative converge $x_p \to x^*$ lorsque $p \to \infty$, alors on a $Mx^* = Nx^* + b \Leftrightarrow Ax^* = b$, $x^*$ est solution du problème.

**Remarque** : si $M$ est triangulaire, $P$ itérations nécessitent $O(Pn^2)$ opérations  
($P$ systèmes linéaires à résoudre).

Si $P \ll n$, alors $Pn^2 \ll n^3$ (méthode LU)

Combien d’itérations pour que l’erreur soit faible ?

### 1) **Convergence**

> [!proposition]
> On note $G = M^{-1}N$ et $x^*$ la solution de $Ax^* = b$.
> 
> $(x_p)$ converge vers $x^*$ quelle que soit $x_0 \in \mathbb{C}^n$ $\iff$ $\rho(M^{-1}N) < 1$.

**Preuve** : On note $e_p = x_p - x^*$, erreur entre $x_p$ et $x^*$.

De  

$Mx_{p+1} = Nx_p + b$  
$Mx^* = Nx^* + b$,  

On obtient $Me_{p+1} = Ne_p$.

soit encore $e_{p+1} = G e_p$. Par récurrence, $e_p = G^p e_0$.

$\Leftarrow$ 

Supposons $\rho(G) < 1$. Il existe $\|\cdot \|_M$ norme matricielle subordonnée telle que $\| G \| < 1$. 

On a alors  
$\| e_p \|_V = \| G^p e_0 \|_V \leq \| G^p \|_M \| e_0 \|_V$,  

où $\| \cdot \|_V$ est la norme vectorielle associée à $\| \cdot \|_M$.

D’où  
$\| e_p \| \leq \| G ^p e_0 \| \leq \| G \|^p \| e_0 \|$,  

d'où $e_p \to 0$ lorsque $p \to \infty$.  
Soit $x_p \to x^*$.

$\Rightarrow$

**Par contraposée** : Supposons $\rho(G) \geq 1$ et soit $\lambda \in \mathbb{C}$, valeur propre de $G$ telle que $|\lambda| = \rho(G)$.

Soit $x_0 \in \mathbb{C}^n$ tel que $x_0 - x^* = e_0$ soit vecteur propre de $G$ associé à $\lambda$.

On a alors $e_p = G^p e_0 = \lambda^p e_0$.

D'où $\|e_p\| = |\lambda|^p |e_0| = (\rho(G))^p \|e_0\| \not\to 0$ lorsque $p \to \infty$.

D'où $x_p \not\to x^*$.

> [!remark]
> On a $\|e_p\| \leq \| G^p \| |e_0|$ grâce à la propriété suivante. ( $\| G^p \| \approx (\rho(G))^p$ )

**Propriété** : $\lim_{p \to \infty} \| G^p \|^{1/p} = \rho(G)$ pour tout $G \in M_n(\mathbb{C})$.

Prop Soit $\lambda \in \mathbb{C}$ une valeur propre de $G$ et $x$ un vecteur propre associé.

On a $Gx = \lambda x$. Donc  
$G^p x = \lambda^p x$.

Ainsi,  
$\rho(G)^{p}\|x \|=|\lambda|^p \|x \|=\|\lambda^p x \| = \| G^p x \| \leq \| G^p \| \| x \|$.

En simplifiant par $\| x \| \neq 0$, on obtient  

$\rho(G)^{p} \leq \| G^p \| \iff \rho(G) \leq \| G^p \|^{1/p}$. 

On montre que pour tout $\varepsilon > 0$, il existe $P \in \mathbb{N}$ tel que pour tout $p \geq P$,  
$\rho(G) \leq \| G^p \|^{1/p} \leq \rho(G) + \varepsilon$.

Soit $\varepsilon > 0$. On considère $G_\varepsilon = \dfrac{G}{\rho(G) + \varepsilon}$.

Les valeurs propres de $G_\varepsilon$ sont $\left\{\dfrac{\lambda}{\rho(G) + \varepsilon}\right\}$, où $\lambda$ est une valeur propre de $G$.

On a  
$$
\left| \frac{\lambda}{\rho(G) + \varepsilon} \right| \leq \frac{\rho(G)}{\rho(G) + \varepsilon} < 1.
$$ 
Donc $\rho(G_\varepsilon) < 1$.

On a $G_\varepsilon^p \to 0$ lorsque $p \to \infty$.

Comme $\exists\,\, \|\cdot\|$ une norme subordonnée $\| G_\varepsilon \| < 1$, on a donc  
$$
\| G^p \| \leq \| G_\varepsilon \|^p \to 0 \quad (p \to \infty).
$$

Donc, $\exists P \in \mathbb{N}$ tel que $\forall p \geq P$, $\| G_\varepsilon^p \| \leq 1$.

On a  
$$
\| G_\varepsilon^p \| = \left\|\frac{G^p}{(\rho(G) + \varepsilon)^p}\right\|= \frac{\| G^p \|}{(\rho(G) + \varepsilon)^p}.
$$

Donc, pour $\forall p > P$,  

$$
\| G^p \| \leq (\rho(G) + \varepsilon)^{p}.
$$
$$
\| G^p \|^{1/p} \leq \rho(G) + \varepsilon.
$$

**Remarque** : on a  
$$
\frac{\| e_p \|}{\| e_0 \|} \leq \| G^p \| \approx \rho(G)^p\leq \varepsilon \quad \text{quand } p \to \infty.
$$

Si on veut que  
$$
\frac{\| e_p \|}{\| e_0 \|} = \varepsilon \quad (10^{-6}),
$$
il suffit que
$$
\varepsilon \geq \rho(G)^p \iff \ln \varepsilon \geq p\ln(\rho(G)).
$$
$$
\iff p \leq \frac{\ln \varepsilon}{\ln (\rho(G))}
$$

Comme $\ln (\rho(G)) < 0$ (car $\rho(G) < 1$), 

le nombre d'itérations pour obtenir une précision donnée est inversement proportionnel à $\ln (\rho(G))$.

**Conclusion** : Plus $\rho(G)$ est petit, plus la convergence sera rapide !

### 2) **Jacobi, Gauss-Seidel, Relaxation**

$A = D - E - F$,  
avec $D$ diagonale, $E$ triangulaire inférieure stricte, et $F$ triangulaire supérieure stricte.

![](assets/Pasted%20image%2020240927111200.png)

Méthode de Jacobi : $M_J = D$ et $N_J = E + F$.

Méthode de Gauss-Seidel : $M_{GS} = D - E$ et $N_{GS} = F$.

Méthode de relaxation : $M_\omega = \dfrac{D}{\omega} - E$ et $N_\omega = \left( \dfrac{1}{\omega} - 1 \right)D + F$.

où $\omega$ est choisi de sorte que $\rho(M_\omega^{-1} N_\omega) = \rho(G_\omega)$ soit le plus petit possible (convergence la plus rapide possible).

> [!proposition]
> 1. Si $A$ est à diagonale dominante stricte par ligne 
>    $$
>    \left(\sum_{j \neq i} |A_{ij}| < |A_{ii}| \quad \forall i\right),
>    $$
>    alors les méthodes de Jacobi, de Gauss-Seidel et de relaxation (pour $\omega \in ]0, 1]$) convergent:
>    $$
>    \rho(G_J) < 1, \quad \rho(G_{GS}) < 1, \quad \rho(G_\omega) < 1 \quad \forall \omega \in ]0, 1].
>    $$
> 
> 2. Si $A$ est symétrique définie positive, alors les méthodes de Gauss-Seidel et de relaxation (pour $\omega \in ]0, 2[$) convergent.

### 3) **Méthode de Richardson**

On considère:

$$
M_\alpha = \frac{1}{\alpha} \text{Id} \quad \text{et} \quad N_\alpha = \frac{1}{\alpha} \text{Id} - A.
$$
On obtient:
$$
M_\alpha x_{p+1} = N_\alpha x_p + b.
$$
$$
\iff \frac{x_{p+1}}{\alpha} = \frac{x_p}{\alpha} - A x_p + b,
$$
$$
\iff x_{p+1} = x_p - \alpha (A x_p - b).
$$

> [!proposition]
> Si $A$ est symétrique définie positive et $0 < \lambda_1 \leq \dots \leq \lambda_n$ sont les valeurs propres, la méthode de Richardson converge si $\alpha \in ]0, \dfrac{2}{\lambda_n}[$.

Le taux de convergence est optimal pour 
$$
\alpha^* = \frac{2}{\lambda_1 + \lambda_n}.
$$

Dans ce cas, 
$$
\rho(G_{\alpha^*}) = \frac{\text{cond}(A) - 1}{\text{cond}(A) + 1}.
$$

**Preuve (convergence)** : 
$$
M_\alpha^{-1} N_\alpha = \alpha \left(\frac{1}{\alpha} \text{Id} - A \right) = \text{Id} - \alpha A.
$$

Les valeurs propres sont $\left\{(1 - \alpha \lambda)\right\}$, où $\lambda$ est une valeur propre de $A$.

Ainsi, 
$$
\rho(G_\alpha) = \rho(M_\alpha^{-1} N_\alpha) < 1 \iff |1 - \alpha \lambda_i| < 1 \quad \forall i \in [[1, n]].
$$
$$
\iff 0 < \alpha \lambda_i < 2 \quad \forall i.
$$
$$
\iff 0 < \alpha < \frac{2}{\lambda_i}.
$$
$$
\iff 0 < \alpha < \min \left(\frac{2}{\lambda_i}\right) = \frac{2}{\lambda_n}.
$$

Convergence optimale ?

Pour trouver le $\alpha^*$ qui minimise $\rho(G_\alpha)$, nous devons minimiser :

$$
\max \{ |1 - \alpha \lambda_i| \}.
$$

Le graphique illustre les différentes valeurs de $|1 - \alpha \lambda_i|$ en fonction de $\alpha$.

![](assets/Pasted%20image%2020240927113707.png)

Le minimum de $\alpha \mapsto \rho(G_\alpha)$ est atteint à l'intersection de $\alpha \mapsto 1 - \alpha \lambda_1$ et $\alpha \mapsto 1 - \alpha \lambda_n$. C'est-à-dire :

$$
\alpha^* \lambda_n = 1 - \alpha^* \lambda_1.
$$
$$
\iff \alpha^* \lambda_n + \alpha^* \lambda_1 = 2,
$$
$$
\iff \alpha^* = \frac{2}{\lambda_1 + \lambda_n}.
$$

De plus, 

$$
\rho(G_{\alpha^*}) = 1 - \alpha^* \lambda_1 = 1 - \frac{2 \lambda_1}{\lambda_1 + \lambda_n} = \frac{(\lambda_1 + \lambda_n) - 2 \lambda_1}{\lambda_1 + \lambda_n} = \frac{\lambda_n - \lambda_1}{\lambda_1 + \lambda_n}.
$$

Cela peut également s'écrire :

$$
\rho(G_{\alpha^*}) = \frac{\lambda_n / \lambda_1 - 1}{\lambda_n / \lambda_1 + 1}.
$$
et $\text{cond}(A) = \frac{\lambda_n}{\lambda_1}$.

Remarque

On veut estimer le nombre d'itérations $p$ pour atteindre une précision donnée :

$$
\frac{\ln \varepsilon}{\ln \rho(G)} \leq p .
$$

On a : 
$$
\rho(G) = \frac{\text{cond}(A) - 1}{\text{cond}(A) + 1} = \frac{1 - \frac{1}{\text{cond}(A)}}{1 + \frac{1}{\text{cond}(A)}}.
$$
(car $\dfrac{1}{\text{cond}(A)}$ est petit.)
$$
\approx \left( 1 - \frac{1}{\text{cond}(A)} \right) \left( 1 - \frac{1}{\text{cond}(A)} + o\left(\frac{1}{\text{cond}(A)}\right) \right).
$$
$$
\approx 1 - \frac{2}{\text{cond}(A)} + o\left(\frac{1}{\text{cond}(A)}\right).
$$

$$
\ln \rho(G) = \ln \left( 1 - \frac{2}{\text{cond}(A)} + o\left(\frac{1}{\text{cond}(A)}\right) \right).
$$
$$
\approx -\frac{2}{\text{cond}(A)} + o\left(\frac{1}{\text{cond}(A)}\right).
$$

D'après les calculs précédents, nous avons :

$$
p \geq \frac{-\ln \varepsilon}{\ln \rho(G)} \approx \frac{(-\ln \varepsilon) \, \text{cond}(A)}{2}.
$$

Cela montre que le nombre d'itérations nécessaires pour atteindre une précision donnée est proportionnel au conditionnement de $A$.

---
remark
Ensuite, dans la partie suivante, il y a un rappel pour la méthode itérative :

$$
M x_{p+1} = N x_p + b \iff M x_{p+1} = (M - A) x_p + b.
$$

(car $A = M - N$.)

$$
\iff x_{p+1} = M^{-1} (M - A) x_p + M^{-1} b.
$$
$$
\iff x_{p+1} = x_p - (M^{-1} A x_p - M^{-1} b).
$$

Une méthode itérative (quelle qu'elle soit) peut être vue comme une méthode de Richardson avec $\alpha = 1$ appliquée au problème $M^{-1} A x = M^{-1} b$. Cela correspond à résoudre le système ($A$) préconditionné par la matrice $M$.

La vitesse de convergence est donc déterminée par le conditionnement de $M^{-1} A$.

### 4) Résidu

L'itération s'écrit :
$$
x_{p+1} = x_p - \alpha (A x_p - b),
$$
où $r_p = A x_p - b$ est appelé le résidu.

Le résidu est donné par :
$$
r_p = A x_p - b = A x_p - A x^* = A (x_p - x^*) = A e_p.
$$

Ici, $e_p = x_p - x^*$ représente l'erreur à l'étape $p$.

On ne peut pas calculer l'erreur $e_p$ directement, mais on peut calculer le résidu $r_p$.

**Critères d'arrêt**

- **Critère sur le résidu** :
  $$
  \frac{\| r_p \|}{\| r_0 \|} \leq \varepsilon,
  $$
  où $\varepsilon$ est la tolérance choisie.

- **Critère sur le nombre d'itérations** :
  $$
  p \leq p_{\text{max}},
  $$
  c'est-à-dire que le nombre d'itérations ne doit pas dépasser un maximum fixé à l'avance.

Il est également judicieux de vérifier que les résidus sont de plus en plus petits au fil des itérations.

## II Méthode GMRES

### 1) Espaces de Krylov

Remarque : (méthode de Richardson)

$$
r_{p+1} = b - A x_{p+1}
$$
$$
= b - A \left( x_p + \alpha r_p \right)
$$
$$
= b - A (x_p + \alpha (b - A r_p))
$$
$$
= r_p - \alpha A r_p
$$

Soit $r_{p+1} = (Id - \alpha A) r_p$

Donc $r_p = (Id - \alpha A)^p r_0=q(A) r_0$

avec $q \in \mathbb{P}_p$. On a $r_p = \beta_0 r_0 + \beta_1 A r_0 + \beta_2 A^2 r_0 + \dots + \beta_p A^p r_0$

avec $\beta_0, \beta_1, \dots, \beta_p \in \mathbb{R}$.

> [!definition] Définition (espace de Krylov)
> 
> $$
> K_p(A, r) = \text{vect} \{r, A r, \dots, A^{p-1} r\}.
> $$
>
> $$
> = \{q(A) r, q \in \mathbb{P}_p\}.
> $$

C'est un espace vectoriel de dimension $\leq p+1$.

> [!proposition] Proposition (méthode de Richardson) 
> $r_p \in K_p(A, r_0)$ et $x_{p+1} \in x_0 + K_p(A, r_0)$.

Preuve : $x_{p+1} = x_0 + \alpha r_p \in x_0 + K_p(A, r_0)$.

> [!info] GMRES (Generalized Minimum Residual)  
> 
> À chaque itération, on choisit
> 
> $x_{p+1} \in x_0 + K_p(A, r_0)$ t.q.
> 
> $$
> \|r_{p+1}\| = \|b - A x_{p+1}\| = \min_{x \in x_0 + K_p} \|b - A x\| \quad (\text{résidu minimal, petit sous espace vectoriel})
> $$
> 
### 2) Itérations d'Arnoldi (base de $K_p$)

Étant donnée $(v_0, \dots, v_p)$ base orthonormée de $K_p$, on construit $v_{p+1}$ en orthonormalisant $(v_0, \dots, v_p, A v_p)$.

$$
v_{p+1}=\frac{A v_p - \sum_{j \leq p} (A v_p, v_j) v_j}{\|A v_p - \sum_{j \leq p} (A v_p, v_j) v_j \|}
$$

$(v_0, \dots, v_{p+1})$ est une base de $K_{p}(A, r_0)$.  

On choisit $v_0 = r_0 / \|r_0\|$.

De la formule ci-dessus, on a

$$
A v_p = \|A v_p - \sum_{j \leq p} (A v_p, v_j) v_j\| v_{p+1} + \sum_{j \leq p} (A v_p, v_j) v_j
$$
$$
= h_{p+1, p} v_{p+1} + \sum_{j \leq p} h_{j p} v_j
$$

Proposition : Si $h_{j+1, j} \neq 0$ $\forall j \leq p$, alors $(v_0, \dots, v_p)$ est une base de $K_p(A, r_0)$. Notant $V_p = \begin{bmatrix} v_0 & \dots & v_p \end{bmatrix} \in \mathcal{M}_{n, p+1}(\mathbb{R})$ et

$$
\hat{H}_p = \begin{bmatrix} h_{0,0} & h_{0,1} & \dots & h_{0,p} \\ h_{1,0} & h_{1,1} & \dots & h_{1,p} \\ 0 & h_{2,1} & \dots & h_{2,p} \\ & & \ddots & \\ & & & h_{p+1, p} \end{bmatrix} \in \mathcal{M}_{p+2, p+1}
$$

on a $A V_p = V_{p+1} \hat{H}_p$ et $V_p^T A V_p = \text{Id}_{p+1,p+2} \hat{H}_{p} = H_p$.

Exemple :

$$
A V_1 = A 
\begin{bmatrix}\begin{array}{c|c} 
 & \\
v_0 & v_1 \\ 
 &
\end{array}\end{bmatrix}
$$

$$
= \begin{bmatrix}\begin{array}{c|c} 
&\\
A v_0 & A v_1 \\
& 
\end{array}\end{bmatrix}
$$

$$
= \begin{bmatrix}\begin{array}{c|c} 
 & \\
h_{0,0} v_0 + h_{1,0} v_1 & h_{2,1} v_2 + h_{1,1} v_1 + h_{0,1} v_0 \\
&
\end{array}\end{bmatrix}
$$

$$
= \begin{bmatrix}
\begin{array}{c|c|c} 
&&\\
v_0 & v_1 & v_2\\ 
&&
\end{array}
\end{bmatrix}

\begin{bmatrix} 
h_{0,0} & h_{0,1} \\ 
h_{1,0} & h_{1,1} \\ 
0 & h_{2,1} 
\end{bmatrix}
$$
$$
= V_2 \begin{bmatrix} h_{0,0} & h_{0,1} \\ h_{1,0} & h_{1,1} \\ 0 & h_{2,1} \end{bmatrix}
$$

Remarque : 

$$
V_p^T A V_p = V_p^T V_{p+1} \hat{H}_p
$$
car
$$
V_p^T V_{p+1} = \text{Id}_{p+1, p+2}
$$

$$
V_p^T V_{p+1} = 
\begin{bmatrix} 
v_0^T \\ \vdots \\ v_p^T 
\end{bmatrix} 
\begin{bmatrix} 
v_0 & \dots & v_{p+1} 
\end{bmatrix} 
= \left( v_i^T v_j \right)_{0 \leq i \leq p+1,\, 0 \leq j \leq p+2} 
= (\delta_{ij})_{0 \leq i \leq p+1,\, 0 \leq j \leq p+2}
$$

Remarque : $V_p^T V_p = \text{Id}_{p+1}$, $V_p$ est "orthogonale".

$V_p$ conserve donc la norme :

$$
\forall y \in \mathbb{R}^{p+1} \quad \|V_p y\|_{\mathbb{R}^n}^2 = (V_p y, V_p y)_{\mathbb{R}^n} = (V_p^T V_p y, y)_{\mathbb{R}^p} = (y, y)_{\mathbb{R}^p} = \|y\|^2_{\mathbb{R}^{p+1}}
$$

$\hat{H}_p$ est de Hessenberg : $\hat{H}_{ij} = 0$ $\forall i > j + 2$.

![](assets/Pasted%20image%2020241002142705.png)

### 3) Résoudre le problème de minimisation

À l’itération $p$, on cherche

$$
x_{p+1} \in x_0 + K_p(A, r_0)
$$

ce qui revient à chercher des coefficients $y \in \mathbb{R}^{p+1}$.

$$
x_{p+1} = x_0 + (y_0 v_0 + y_1 v_1 + \dots + y_p v_p)
$$

$$
= x_0 + \begin{bmatrix} v_0 & v_1 & \dots & v_p \end{bmatrix} \begin{bmatrix} y_0 \\ y_1 \\ \vdots \\ y_p \end{bmatrix}
$$

$$
= x_0 + V_p y
$$

Le résidu associé devient :

$$
r_{p+1} = b - A x_{p+1}
$$
$$
= b - A (x_0 + V_p y)
$$
$$
= r_0 - A V_p y
$$
$$
= r_0 - V_{p+1} \hat{H}_p y
$$

$$
r_0 =\|r_0\| v_0 =  =\|r_0\| v_{p+1} e_0
$$

Donc 

$$
r_{p+1} = V_{p+1} (\|r_0\| e_0 - \hat{H}_p y)
$$

$x_{p+1}$ est défini par 

$$
\|r_{p+1}\|_{\mathbb{R}^n} = \min_{y \in \mathbb{R}^{p+1}} \|V_{p+1} (\|r_0\| e_0 - \hat{H}_p y)\|_{\mathbb{R}^n}
$$

$$
= \min_{y \in \mathbb{R}^{p+1}} \|\|r_0\| e_0 - \hat{H}_p y\|_{\mathbb{R}^{p+2}}
$$

À l'itération $p$, on doit donc résoudre un problème de minimisation en dimension $p+1$.

**Résolution par factorisation QR**

Supposons que la matrice $\hat{H}_p$ soit de rang $p+1$ (rang maximal) (c'est le cas si $h_{i+1,i} \neq 0$ $\forall i \leq p$), alors on peut factoriser $\hat{H}_p$ ainsi :

$$
\hat{H}_p = Q_p \hat{R}_p
$$

où $Q_p$ est une matrice orthogonale $\in \mathcal{M}_{p+2}(\mathbb{R})$ et $\hat{R}_p$ est une matrice triangulaire supérieure $\in \mathcal{M}_{p+2, p+1}(\mathbb{R})$.

On a alors 

$$
\|\|r_0\| e_0 - \hat{H}_p y\|^2 = \|\|r_0\| e_0 - Q_p \hat{R}_p y\|^2
$$
(puisque $Q_p$ est orthogonale)
$$
= \|Q_p^T (\|r_0\| e_0 - Q_p \hat{R}_p y)\|^2
$$
(puisque $Q_p^T Q_p=\text{Id}$)
$$
= \|\|r_0\| Q_p^T e_0 - \hat{R}_p y\|^2
$$
$$
= \left\| 
\|r_0\|
\begin{pmatrix} 
( Q_p^T e_0)_{j\leq p+1} \\ 
(Q_p^T e_0)_{p+2} 
\end{pmatrix} 
- 
\begin{pmatrix} 
R_p y \\ 
0 
\end{pmatrix} 
\right\|^2
$$
$$
= \|\|r_0\| (Q_p^T e_0)_{j\leq p+1} - R_p y\|^2 + \|r_0\|^2 {(Q^T_pe_0)_{p+2}}^2
$$
$$
= \|r_0\|^2 {(Q^T_pe_0)_{p+2}}^2
$$

On choisit $y \in \mathbb{R}^{p+1}$ solution du système linéaire

$$
R_p y = \|r_0\| \|Q_p^T e_0\|_{:-1}.
$$

**Remarque :** Factorisation QR par matrices de Givens

$$
G_{i,j}(\theta) = \begin{bmatrix} 
1 & & & & \\ 
& \cos \theta & & -\sin \theta & \\ 
& & 1 & & \\ 
& \sin \theta & & \cos \theta & \\ 
& & & & 1 
\end{bmatrix}
\in O(\mathbb{R})\quad\text{otrthogonal}.
$$

**Appliquée à un vecteur $x$ :**

$$
G_{i,j}(\theta) \begin{pmatrix} \vdots \\ x_i \\ \vdots \\ x_j \\ \vdots \end{pmatrix} = \begin{pmatrix} \vdots \\ \cos \theta \cdot x_i - \sin \theta \cdot x_j \\ \vdots \\ \sin \theta \cdot x_i + \cos \theta \cdot x_j \\ \vdots \end{pmatrix}.
$$

Effectuer une rotation d’angle $\theta$ dans le plan $(e_i, e_j)$.

Cela permet d'annuler des coefficients :

$$
G_{i,i+1}(\theta) 
\begin{pmatrix} 
\vdots \\
x_i \\ 
x_j \\
\vdots \\
\end{pmatrix} 
= 
\begin{pmatrix} 
\vdots \\
\|(x_i, x_{i+1})\| \\ 
0 \\
\vdots \\
\end{pmatrix}.
$$

en choisissant $\theta$ de manière adéquate.  ($\theta = \arctan\left(\frac{x_j}{x_i}\right)$).

À partir d’une matrice de Hessenberg, on annule un à un les coefficients sous la diagonale :

$$
\left(G_{p+1, p+2}(\theta_{p+1}) \dots G_{0,1}(\theta_0)\right) \hat{H}_p = \hat{R}_p,
$$

où 

$$
\left(G_{p+1, p+2}(\theta_{p+1}) \dots G_{0,1}(\theta_0)\right) = Q_p^T.
$$

### 4) Arrêt de l'algorithme

**Données :** $x_0 \in \mathbb{R}^m$

$$
r_0 = b - A x_0
$$
$$
v_0 = \frac{r_0}{\|r_0\|}, \quad V_0 = [v_0], \quad \hat{H}_{-1} = [\, ]
$$

Tant que (critère d'arrêt non satisfait) faire :

1. **Arnoldi** : calcul de $v_{p+1}$ à partir de $V_p$, assemble $V_{p+1}, \hat{H}_p$.
2. **Factorisation** : factorise $\hat{H}_p$, $\hat{H}_p = Q_p \hat{R}_p$.
3. **Résolution** : résout $R_p y = \|r_0\| (Q_p^T e_0)_{:-1}$.
4. Mise à jour : $x_{p+1} = x_0 + V_p y$.

Proposition :  
1) Si $h_{i+1,i} \neq 0$ $\forall j \leq p$ et $h_{p+1,p} = 0$, alors $x_p = x^*$ solution du problème.  
2) $\exists p \leq n-1$ tq $x_p = x^*$ solution du problème.

A inversible ???

Preuve :  
1) On a

$$
A v_p = \sum_{j \leq p} (A v_p, v_j) v_j.
$$

Si $h_{p+1,p} = 0$. Donc $A v_p \in K_p$. 

On en déduit que $A K_p \subseteq K_p$ (puisque $A v_j \in K_{j+1} \subseteq K_p$, $A v_p \subseteq K_p$, $\forall j \leq p-1$).

$A$ est injective et $\dim A K_p = \dim K_p$ donc $A K_p = K_p$.

On a 
$$
r_p = b - A(x_0 + v_p y) = r_0 - A v_p y.
$$

Or $r_0 \in K_p$, donc $\exists \bar{z} \in K_p$ tq $A \bar{z} = r_0$.  
$\iff \exists \bar{y} \in \mathbb{R}^{p+1}$ tq $A v_p \bar{y} = r_0$.

Donc en prenant 

$$
\bar{x} = x_0 + V_p \bar{y} \in x_0 + K_p,
$$

on obtient un résidu nul : 

$$
\bar{x} = x_p = x^*.
$$

$x_p$ est tq le rendu de norme minimale. résidu nul.

2) Si $h_{j+1, j} \neq 0$, alors $\dim K_{j+1} = \dim K_j + 1$ (car $A v_j \notin K_j$ et $K_j \subseteq K_{j+1}$).

Si $h_{j+1, j} \neq 0$ $forall p < n$, alors $\dim K_{n-1} = n$ et donc $K_{n-1} = \mathbb{R}^n$.

Donc à la $(n-1)$-ième itération, on minimise le résidu sur $\mathbb{R}^n$ :

et donc $x_{n-1} = x^*$.

**Remarque (re-start) :** Pour éviter de résoudre des problèmes triangulaires trop gros, toutes les $p$ itérations, on redémarre l'algorithme avec pour initialisation $x_p$.

**Remarque (convergence) :** À chaque itération, le résidu diminue.

$$
r_p = b - A x_p = b - A \left( x_0 + q_p(A) r_0 \right)
$$
avec $q_p(A) r_0\in K_p$ ,  $q \in \mathbb{P}_p$

$$
= r_0 - A q_p(A) r_0
$$

$$
= \hat{q}_p(A) r_0 \quad \text{avec } \hat{q}_p(x) = 1 - x q(x)
$$

avec
$$
\hat{q}_p = 1 - x q_p(x) \in \mathbb{P}_{p+1} \cap \{\hat{q}\mid\hat{q}(0) = 1\}.
$$

$$
\|r_p\| = \|\hat{q}_p(A) r_0\| = \min_{\hat{q} \in \mathbb{P}_{p+1}\cap \{\hat{q}\mid\hat{q}(0) = 1\}} \|\hat{q}(A) r_0\|
$$
$$
\leq \left(\min_{\hat{q} \in \mathbb{P}_{p+1}\cap \{\hat{q}\mid\hat{q}(0) = 1\}} \|\hat{q}(A)\| \right)\|r_0\|.
$$

Car A sym. déf. pos

$$
\|\hat{q}(A)\| = \rho(\hat{q}(A)) = \max_{\lambda_i \in \text{Vp}(A)} |\hat{q}(\lambda_i)|
$$
$$
\implies \min_{\hat{q}} \left( \max_{\lambda_i \in \text{Vp}(A)} |\hat{q}(\lambda_i)| \right).
$$
Donc
$$
\frac{\|r_{p+1}\|}{\|r_0\|} \leq \min_{\tilde{q} \in \mathbb{P}_p \cap \{\tilde{q}(0) = 1\}} \left( \max_{\lambda_i \in \text{vp}(A)} |\tilde{q}(\lambda_i)| \right)
$$

Cas A diagonalisable  
$$
A = P D P^{-1}
$$
$$
\tilde{q}(A) = \sum_{k=0}^p \alpha_k (P D P^{-1})^k = \sum_{k=0}^p \alpha_k P D^k P^{-1}
$$
$$
= P \left( \sum_{k=0}^p \alpha_k D^k \right) P^{-1}
$$
$$
= P \tilde{q}(D) P^{-1}
$$

Donc 
$$
\|\tilde{q}(A)\| = \|P \tilde{q}(D) P^{-1}\| \leq \|P\| \|\tilde{q}(D)\| \|P^{-1}\| = \text{cond}(P) \|\tilde{q}(D)\|
$$
$$
\tilde{q}(D)=\text{diag}(\tilde{q}(\lambda_1),\cdots ,\tilde{q}(\lambda_n))
$$
$$
\|\tilde{q}(D)\| = \rho (\tilde{q}(D)) = \max_{\lambda_i\text{vp de } A} |\tilde{q}(\lambda_i)|
$$
$\tilde{q}(D)$ sym.

$$
\Rightarrow \frac{\|r_{p+1}\|}{\|r_0\|} \leq \text{cond}(P) \min_{\tilde{q} \in \mathbb{P}_p \cap \{\tilde{q}(0) = 1\}} \left( \max_{\lambda_i \in \text{vp}} |\tilde{q}(\lambda_i)|\right)
$$

La convergence dépend des propriétés spectrales de $A$ :
- $\text{cond}(P)$ de position des valeurs propres;
- répartition des valeurs propres dans $\mathbb{C}$.


## III Méthode des gradients conjugués

$A \in \mathcal{M}_n(\mathbb{R})$ symétrique définie positive. $b\in \mathbb{R}^n$.

Produit scalaire associé : $$ (x, y)_A = (x, Ay) $$

Norme associée : $$ \|x\|_A = \sqrt{(x, x)_A} = \sqrt{(x, Ax)} $$

**Gradient conjugué** $(x_p)$ construit tel que $x_{p+1} \in x_0 + K_p(A, r_0)$ et

$$
\|x_{p+1} - x^*\|_A = \min_{x \in x_0 + K_p(A, r_0)} \|x - x^*\|_A
$$

Erreur minimale sur $x_0 + K_p(A, r_0)$.

$x^*$ : solution exacte du problème $Ax = b$.

> [!proposition]
> $x_{p+1} \in (x_0 + K_p(A, r_0))$ vérifie la propriété d'erreur minimale $\iff r_{p+1} \perp K_p(A, r_0)$

![](assets/Pasted%20image%2020241003103514.png)

**Preuve :**  

Soit $x \in x_0 + K_p(A, r_0)$. On note $\delta = x - x_{p+1}$. $x = x_{p+1} + \delta$. Ainsi  

$$
\|x - x^*\|_A^2 = \|(x_{p+1} - x^*) + \delta\|_A^2
$$
$$
= \|x_{p+1} - x^*\|_A^2 + 2 (x_{p+1} - x^*, \delta)_A + \|\delta\|_A^2
$$
$$
= \|x_{p+1} - x^*\|_A^2 + 2 \left( \delta, A (x_{p+1} - x^*) \right) + \|\delta\|_A^2
$$
Or, $\delta = x - x_{p+1} \in K_p(A, r_0)$, 

- **Si** $r_{p+1} \perp K_p(A, r_0)$  

alors pour tout $x\in x_0 + K_p(A, r_0)$

donc 
$$
\|x - x^*\|_A^2 = \|x_{p+1} - x^*\|_A^2 + \|\delta\|_A^2 \geq \|x_{p+1} - x^*\|_A^2
$$
donc, $x_{p+1}$ vérifie bien la propriété de minimisation.

- **Si** $r_{p+1} \not\perp K_p(A, r_0)$.

Soit $x = x_0 + K_p(A, r_0)$. Soit $\delta = x_p - x_{p+1} \in K_p$.

donc
$$
x_{p+1}+s\delta \in x_0 + K_p(A, r_0)\quad \forall s \in \mathbb{R}
$$

Ainsi,
$$
\|(x_{p+1} + s \delta) - x^*\|_A^2 = \|(x_{p+1} - x^*) + s \delta\|_A^2
$$
$$
= \|x_{p+1} - x^*\|_A^2 + 2s(x_{p+1}-x^*, \delta) + s^2 \|\delta\|_A^2
$$
$$
= \|x_{p+1} - x^*\|_A^2 - 2s(r_{p+1}, \delta) + s^2 \|\delta\|_A^2
$$
$$
=P(s)
$$

Polynôme de degré 2, minimal en $\tilde{s} = \dfrac{2(r_{p+1}, \delta)}{\|\delta\|_A^2} \neq 0$.

$$
\|(x_{p+1} + s \delta) - x^*\|_A^2 = P(\tilde{s}) < \tilde{P}(0) = \|x_{p+1} - x^*\|_A^2
$$

Donc $x_{p+1}$ ne vérifie pas la propriété de minimalité.

### 1) Construction

On cherche $x_p$ sous la forme 
$$
x_{p+1} = x_p + s_p d_p
$$

avec $s_p \in \mathbb{R}, d_p \in K_p(A, r_0)$.

On a 
$$
r_{p+1} = b - A x_{p+1} = r_p - s_p A d_p
$$

Conditions sur $s_p$ et $d_p$ pour que $(x_p)$ vérifie la propriété de minimalité :

> [!proposition]
> Supposons que les $(d_j)_{j\leq n}$ forment une base des espaces vectoriels croissants $(K_j)_{j \leq n}$. Alors,
> 
>  $x_{p+1}$ la propriété de minimalité $\iff$ $r_{p+1} \perp K_p(A, r_0)$  $\iff$ $\begin{cases}s_p = \dfrac{(r_p, d_p)}{(A d_p, d_p)},\quad j=p\\ (A d_p, d_j) = 0, \quad \forall j < p\end{cases}$.

Les $(d_j)$ doivent être $A$-conjugués, c'est-à-dire $(d_p, d_j)_A = (A d_p, d_j) = 0 \quad \forall j < p$.

**Preuve :**

Par récurrence, supposons $r_p \perp K_p$.

$$
r_{p+1} \perp K_p \iff r_{p+1} \perp d_j \quad \forall j \leq p
$$
$$
\iff 0 = (r_{p+1}, d_j) = (r_p - s_p A d_p, d_j)
$$
$$
= (r_p, d_j) - s_p (A d_p, d_j) \quad \forall j \leq p
$$
$$
\iff \begin{cases}s_p = \dfrac{(r_p, d_p)}{(A d_p, d_p)},\quad j=p\\ (A d_p, d_j) = 0, \quad \forall j < p\end{cases}
$$

Construction de $(d_j)$ $A$-conjugués tel que $(d_j)_{j\leq p}$ soit une base de $K_p$ pour tout $p \leq n$.

On la construit par récurrence. Pour construire $d_p \in K_p$, on considère

$$
r_p = b - A x_p \in K_p \quad \text{et le } A\text{-orthogonalise.}
$$
$$
d_p = r_p - \sum_{j < p} \frac{(r_p, d_j)_A}{(d_j, d_j)_A} d_j
$$
$$
= r_p - \frac{(r_p, d_{p-1})_A}{(d_{p-1}, d_{p-1})_A} d_{p-1} - \sum_{j \leq p-2} \frac{(r_p, d_j)_A}{(d_j, d_j)_A} d_j
$$

car $(r_p, d_j)_A = (r_p, A d_j)$ et $A d_j \in K_{p-1}$ car $d_j \in K_{p-2}$. 

Donc,
$$
d_p = r_p + \beta_p d_{p-1} \quad \text{avec} \quad \beta_p = \frac{(r_p, d_{p-1})_A}{(d_{p-1}, d_{p-1})_A}.
$$

$(d_j)_{j\leq p}$ est alors une famille $A$-orthogonale. C'est une famille libre à $(p+1)$ éléments dans $K_p$. C'est donc bien une base de $K_p$.

**Simplification des expressions :**

$$
r_{p+1} = r_p - s_p A d_p,
$$
$$
\iff A d_p = \frac{r_p - r_{p+1}}{s_p}.
$$

Ainsi,
$$
(r_p, d_{p-1})_A = (r_p, A d_{p-1}) = \left( r_p, \frac{r_{p-1} - r_{p}}{s_{p-1}} \right).
$$
$$
= \frac{(r_p, r_{p-1})}{s_{p-1}} - \frac{(r_p, r_{p})}{s_{p-1}}.
$$
$$
= - \frac{(r_p, r_{p})}{s_{p-1}}.
$$
$$
(d_{p-1}, d_{p-1})_A = (A d_{p-1}, d_{p-1})
$$

$$
= \left( r_{p-1} + \beta_{p-1} d_{p-2}, A d_{p-1} \right)
$$

$$
= (r_{p-1}, A d_{p-1}) + \beta_{p-1} (d_{p-2}, A d_{p-1})
$$
$(d_{p-2}, A d_{p-1})=0$ car $A$-conjuguée
$$
=\left(r_{p-1}, \frac{r_{p-1}-r_{p-2}}{s_{p-1}}\right)
$$
$$
=\frac{(r_{p-1}-r_{p-1})}{s_{p-1}}-\frac{(r_{p-1}-r_{p-2})}{s_{p-1}}
$$

Donc,
$$
\beta_p = - \frac{(r_p, r_p)}{s_{p-1}} \cdot \frac{1}{\dfrac{(r_{p-1}, r_{p-1})}{s_{p-1}}} = \frac{(r_p, r_p)}{(r_{p-1}, r_{p-1})}.
$$
$$
s_p = \frac{(r_p, d_p)}{(A d_p, d_p)}
$$

Or $(r_p, d_p)=(r_p,r_p+\beta d_{p-1})=(r_p,r_p)+\beta (r_p,d_{p-1})=(r_p,r_p)$

Donc,
$$
s_p = \frac{(r_p, r_p)}{(A d_p, d_p)}
$$

### 2) Algorithme

Donnée $x_0$,  
$r_0 = b - A x_0$, $d_0 = r_0$.  

Tant que (critère d'arrêt non satisfait) 

1. $s_p = \dfrac{(r_p, r_p)}{(A d_p, d_p)}$,
2. $x_{p+1} = x_p + s_p d_p$,
3. $r_{p+1} = r_p - s_p A d_p$.
4.  $\beta_{p+1} = \frac{(r_{p+1}, r_{p+1})}{(r_p, r_p)}$,  
5. $d_{p+1} = r_{p+1} + \beta_{p+1} d_p$.

**Fin.**

> [!remark]
> - Une seule multiplication matrice-vecteur par itération.
> - $(d_p)$ vérifie une relation de récurrence d'ordre 1, et il n'y a pas besoin de stocker tous les $(d_j)_{j\leq p}$ des itérations précédentes.

> [!proposition]
> L'algorithme converge en au plus $n$ étapes.

**Preuve :** À l'itération $n$, on numérise l'erreur sur $K_n = \mathbb{R}^n \implies x_{n-1} = x^*$.

> [!proposition] Proposition (Convergence)
> $$
> \frac{\|x_{p} - x^*\|_A}{\|x_0 - x^*\|_A} \leq 2\left( \frac{\sqrt{\text{cond}(A)} - 1}{\sqrt{\text{cond}(A)} + 1} \right)^p
> $$

**Remarque :** Rappel pour la méthode de Richardson :
$$
\rho(G_{\alpha^*})=\frac{\text{cond}(A) - 1}{\text{cond}(A) + 1}
$$
$$
\frac{\|e_p\|}{\|e_0\|} \leq \left( \frac{\text{cond}(A) - 1}{\text{cond}(A) + 1} \right)^p
$$

Quand $\text{cond}(A)$ est grand, le nombre d'itérations pour atteindre une précision donnée est proportionnel à $\text{cond}(A)$. 

Pour le gradient conjugué,  le nombre d'itérations pour atteindre une précision donnée est proportionnel à $\sqrt{\text{cond}(A)}$. 

Exemple : $\text{cond}(A) = 10,000$ et $\sqrt{\text{cond}(A)} = 100$.

## IV Préconditionnement

Soit $x^*$ la solution de $Ax = b$.  
Alors, $x^*$ est solution de :
$$
M^{-1} A x = M^{-1} b
$$
ou équivalemment,
$$
(A M^{-1}) M x = b.
$$

Remarque 
en dimension 2, $Ax = b$ $\iff$ $x$ l'intersection de deux droites.

Exemple :

![](assets/Pasted%20image%2020241003115513.png)

$$
\begin{bmatrix} 3 & -1 \\ -2 & 3 \end{bmatrix} \begin{bmatrix} x^* \\ y^* \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}.
$$

**Remarque :**  
- Le nombre d'itérations pour résoudre $Ax = b$ est proportionnel à $\text{cond}(A)$.  
- Le nombre d'itérations pour résoudre $M^{-1}Ax = M^{-1}b$ est proportionnel à $\text{cond}(M^{-1}A)$.

**Préconditionnement :** Choisir $M$ tel que $\text{cond}(M^{-1}A)$ soit petit.
- Prendre $M$ proche de $A$.
- Plus généralement, choisir $M$ tel que les valeurs propres de $M^{-1}A$ soient concentrées dans le plan complexe.

### 1) Exemples

- **Préconditionnement diagonal (Jacobi) :**  
$$
M = \text{diag}(A)
$$

- **Préconditionnement ILU (incomplet LU) :**  
  $M = L_{\text{inc}} U_{\text{inc}}$, où $L_{\text{inc}}$ et $U_{\text{inc}}$ sont les facteurs de la décomposition LU incomplète, c’est-à-dire sans ajout de coefficients non nuls en dehors de ceux de $A$.  
  (Matrice $M$ de même structure creuse que $A$).

### 2) Préconditionnement symétrique

**Remarque :** Si $A$ est symétrique définie positive, alors $M^{-1}A$ n'est plus nécessairement symétrique (sauf si $M = \text{diag}(A)$), même si $M$ est symétrique.

Soit $M = P D P^T$, une matrice symétrique définie positive.  
On considère $M^{1/2} = P D^{1/2} P^T$ (car $M^{1/2} M^{1/2} = M$).

Le problème préconditionné devient :
$$
M^{-1/2} A M^{-1/2} M^{1/2} x = M^{-1/2} b,
$$

qui est symétrique définie positive.

On souhaite $\text{cond}(M^{-\frac{1}{2}}AM^{-\frac{1}{2}})$ soit plus petit que $\text{cond}(A)$.

Exemple : Gauss-Seidel symétrique

On considère $M = D - E \quad F = E^T$

$$ 
A = \begin{pmatrix} 
&  & -E^T \\ 
 & D &  \\
-E & &
\end{pmatrix} = D - E - E^T 
$$

$$ 
x_{p+1} = x_p + M^{-1}(b - Ax_p) 
$$

$$ 
Mx_{p+1} = Nx_{p+1} + b 
$$

Gauss-Seidel

![[Pasted image 20241112215947.png]]

Symétrisation

$$ 
x_{p+\frac{1}{2}} = x_p + (D - E)^{-1}(b - Ax_p) 
$$

$$ 
x_{p+1} = x_{p+\frac{1}{2}} + (D - E^T)^{-1}(b - A x_{p+\frac{1}{2}}) 
$$
$\iff$ 

$$ 
x_{p+1} = x_p + (D - E)^{-1}(b - Ax_p) + (D - E^T)^{-1} \left( b - A \left( x_p + (D - E)^{-1}(b - Ax_p) \right) \right) 
$$

$$ 
= x_p + (D - E^T)^{-1}(D - E)^{-1}(b - Ax_p) + (D - E^T)^{-1}(D - E)(D - E)^{-1}(b - Ax_p) 
$$
$$ 
- (D - E^T)^{-1}A(D - E)^{-1}(b - Ax_p) 
$$

$$ 
= x_p + (D - E^T)^{-1} \left[ D - E^T + D - E - A \right](D - E)^{-1} (b - Ax_p) 
$$

$$ 
= x_p + (D - E^T)^{-1} D (D - E)^{-1}(b - Ax_p) 
$$
$$ 
= x_p + M^{-1}(b - Ax_p)
$$

La matrice de préconditionnement est 

$$ 
M = (D - E) D^{-1} (D - E^T) 
$$

Elle est symétrique définie positive dès que $A$ l'est. En effet, elle est symétrique (immédiat) et définie positive :

$$ 
M = (D - E)(I_d - D^{-1}E^T) 
$$
$$ 
= D - E - (D - E)D^{-1}E^T 
$$
$$ 
= D - E - E^T + ED^{-1}E^T 
$$
$$ 
= A + ED^{-1}E^T 
$$
Donc

$$ 
(x, Mx) = (x, Ax) + (x, ED^{-1}E^Tx)
$$
On a 
$$
(x, Ax) > 0
$$
$$
(x, ED^{-1}E^Tx) = (E^Tx, D^{-1}E^Tx) = \sum (D^{-1}x_i)(E^Tx_i)^2 \geq 0
$$
car
$$
D^{-1}_{ii} \geq 0
$$

## V Méthode multi-grille

"**Méthode itérative adaptée au problème considéré**"

### 1) Laplacien discret

On souhaite résoudre l'équation

$$
\begin{cases}
-u''(x) = f(x) \quad \text{sur} \quad ]0, 1[ \\
u(0) = u(1) = 0
\end{cases}
$$

On considère un maillage régulier 

$$ x_i = (ih) \quad 0 \leq i \leq N + 1 $$

avec $h = \dfrac{1}{N+1}$ pas d'espace.

Lorsque l'on utilise une méthode de différences finies ou d'éléments finis, on cherche une solution approchée 

$$ u_h \in \mathbb{R}^N $$

solution de

$$ A_h u_h = b_h $$

avec $b_h \in \mathbb{R}^N$ et

$$ A_h = \frac{1}{h^2} \begin{pmatrix} 2 & -1 & & & \\ -1 & 2 & -1 & & \\ & -1 & 2 & -1 & \\ & & -1 & 2 \end{pmatrix} \in M_N(\mathbb{R}) $$

matrice du Laplacien discret.

---

Prop : les valeurs propres de $(h^2 A_h)$ sont

$$ \lambda_h = 4 \sin^2\left(\frac{k \pi h}{2}\right) = 2 - 2\cos(k \pi h) $$

pour $\forall h \in [1, N]$

et les vecteurs propres associés sont

$$ v_h = \left( \sin\left(\frac{k \pi x_j}{i}\right) \right)_{1 \leq j \leq N} $$

pour $\forall h \in [1, N]$

### 2) Effet régularisant de Jacobi relaxé

Méthode de Jacobi relaxée :

$$ M_\omega = \frac{D}{\omega} $$

$$ N_\omega = \frac{D}{\omega} - A $$

$$ e_{p+1} = (M_\omega^{-1}N_\omega)e_p = G_\omega e_p $$

avec

$$ 
G_\omega = M_\omega^{-1}N_\omega = \omega D^{-1} \left( \frac{D}{\omega} - A \right) = I_d - \omega D^{-1}A 
$$

---

**Convergence** (dans le cas de la matrice du Laplacien discret)

$$ 
G_\omega = I_d - \omega \frac{h^2}{2} A_h \quad (car \, D = \frac{2}{h^2}I_d) 
$$

$$ 
= I_d - \frac{\omega}{2}(h^2 A_h) 
$$

Dans ce cas, la méthode de Jacobi relaxée est équivalente à la méthode de Richardson pour la matrice $(h^2 A_h)$ avec $\alpha = \dfrac{\omega}{2}$.

Le paramètre optimal pour la convergence est :

$$ \alpha^* = \frac{2}{\lambda_1 + \lambda_N} $$

Or 

$$ \lambda_1 = 4 \sin^2\left(\frac{\pi h}{2}\right) = (\pi h)^2 + o(h^2) \quad h \to 0 $$

$$ \lambda_N = 4 \sin^2\left(\frac{\pi h N}{2(N+1)}\right) = 4 \sin^2\left(\frac{\pi}{2} \frac{N}{N+1}\right) \to 4 \quad \text{lorsque} \, N \to +\infty $$

$$ \alpha^* \to \frac{2}{0 + 4} = \frac{1}{2} \quad \text{lorsque} \, h \to 0 $$

et donc 

$$ \omega^* = 2 \alpha^* = 1 $$

Utiliser la relaxation n'améliore pas la convergence quand $h \to 0$.

**Effet régularisant**

Étant donnée la décomposition de l'erreur :

$$ e_0 = \sum_{h=1}^{N} \beta_h v_h $$

dans la base associée à $(h^2 A_h)$, on a :

$$ 
e_p = G_\omega^p e_0 = \sum_{h=1}^{N} \beta_h G_\omega^p v_h 
$$

$$ 
= \sum_{h=1}^{N} \beta_h \left( 1 - \frac{\omega}{2} \lambda_h \right)^p v_h 
$$

Les composantes de l'erreur sont amorties plus ou moins vite :

$$ 
1 - \omega \frac{\lambda_h}{2} = 1 - \omega \left(1 - \cos(h \pi h)\right) 
$$
$$ 
= (1 - \omega) + \omega \cos( h \pi h ) 
$$

![[Pasted image 20241112222720.png]]

On choisit $\omega$ de sorte que les hautes fréquences de l'erreur correspondant aux termes $h \geq N / 2$ soient amorties le plus vite.

On choisit $\omega$ tel que :

$$ \max_{h \geq \frac{N}{2}} \left| 1 - \frac{\omega}{2} \lambda_h \right| $$

soit le plus petit possible.

Remarque : 

$v_j = \left( \sin(k \pi x_j) \right)_{1 \leq j \leq N}$

- $v_2$ : basse fréquence (oscille peu)
- $v_N$ : haute fréquence (oscille beaucoup)

![[Pasted image 20241112222753.png]]

### 3) Méthode à deux grilles

Principe : **Éliminer les basses fréquences** comme les hautes fréquences d'un signal plus petit.

Méthode de correction : 
Étant donnée une solution approchée $v_h$ du problème $A_h u_h = b_h$, alors $u_h = v_h + e_h$ avec $e_h$, l'erreur qui satisfait $A_h e_h = r_h$.

---

Remarque : Convergence en deux phases

![[Pasted image 20241112223133.png]]

1. Amortissement des hautes fréquences
2. Phase de convergence plus lente pour les basses fréquences

Pour déterminer une meilleure approximation de $u_h$, on doit déterminer une approximation de $e_{h}$. Pour cela, on peut résoudre un problème du type $A_{e_{h}}  = r_{h}$ sur un maillage plus grossier.

**Opérateurs de restriction et d'interpolation**

Étant donné un maillage avec pas d'espace $h$ et $2h$ :

![[Pasted image 20241112223345.png]]

On introduit :
- Opérateur de restriction : 

$$ 
R : \mathbb{R}^{N_h} \to \mathbb{R}^{N_{2h}} 
$$
$$ 
(Ru)_j = \frac{(u_{2j} + 2u_{2j+1} + u_{2j+2})}{4} 
$$ (moyenne pondérée)

- Opérateur d'interpolation :

$$ I : \mathbb{R}^{N_{2h}} \to \mathbb{R}^{N_h} $$

$$ 
(Iu)_j = \begin{cases} 
u_{\frac{j}{2}} & \text{si} \, j \, \text{est pair} \\
\frac{(u_{\frac{j-1}{2}} + u_{\frac{j+1}{2}})}{2} & \text{si} \, j \, \text{est impair} 
\end{cases} 
$$

On a 
$$
R = \frac{I^T}{2}
$$ et

$$ A_{2h} = (R A_h I) $$ (admis)

**Méthode à deux grilles**

Étant donné $u_p$ :

1. Effectuer $V_1$ itération de Jacobi relaxée pour déterminer $v_h$.
2. Restreindre le résidu sur le maillage grossier : $\tilde{r}_{2h} = R r_h$
3. Résoudre le problème $A_{2h} \tilde{e}_{2h} = \tilde{r}_{2h}$ sur le maillage grossier.
4. Interpoler l'erreur sur le maillage fin : $\tilde{e}_h = I \tilde{e}_{2h}$
5. En partant de $v_h = v_h + \tilde{e}_h$, effectuer $\frac{1}{2}$ itération de Jacobi relaxée pour obtenir $u_{p+1}$.

**Remarques** :
- L'étape 1 permet de supprimer les hautes fréquences (atténuer) dans l'erreur.
- L'étape 2-3-4 permet de restreindre le problème sur un maillage grossier.
- L'interpolation peut créer de hautes fréquences, d'où les itérations de Jacobi relaxées à l'étape 5.

Si $V_1 = V_2 = 0$, alors $v_h = u_p$, et 

$$ 
u_{p+1} = u_p + I(A_{2h})^{-1} R r_p 
$$
$$ 
= u_p + I(A_h R I)^{-1} R r_p 
$$
$$
= u_p + M^{-1}r_p
$$

**$M$ est construit en restreignant $A_h$ sur un maillage grossier.**

- Si $v_1 = v_2 = 1$ :

$$ u_{p+1} = u_p + \left[ D^{-1} + D^{-1}(I_d - A D^{-1}) + (I_d - D^{-1}A) I (R A_h I)^{-1} R (I_d - A D^{-1}) \right] r_p $$

Méthode dite en V à 1 niveau

![[Pasted image 20241112224356.png|300]]

Méthode dite en V à plusieurs niveaux

![[Pasted image 20241112224402.png|300]]

On applique la méthode récursivement à l'étape 3.

Méthode dite en W

![[Pasted image 20241112224410.png|300]]