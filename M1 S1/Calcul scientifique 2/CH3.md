Chap. 3: Méthodes itératives pour les systèmes linéaires creux

## I) Rappels sur les méthodes itératives

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