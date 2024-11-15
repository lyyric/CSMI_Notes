**Chapitre 4 : Calcul de valeurs propres et vecteurs propres pour les matrices creuses**

Problème : $A \in M_n(\mathbb{C})$. Trouver les valeurs propres $(\lambda_h)$ et les vecteurs propres $(x_h)$ :

$$
A x_h = \lambda_h x_h 
$$

**Remarque** : Pour tout $P$ inversible, $A$ et $P A P^{-1}$ ont les mêmes valeurs propres.

$A x = \lambda x$

Donc 

$$ (P^{-1} A P) y = \lambda y $$

où $y = P^{-1} x$, ce qui implique que $\lambda$ est également valeur propre de $P^{-1} A P$.

**Principe de méthode** : déterminer une suite de matrices $A_h = P_h^{-1} A P_h$ semblables à $A$, qui convergent vers une matrice dont il est facile de calculer les valeurs propres. Cela est à adapter pour les matrices de grandes tailles.

## I Conditionnement du problème

> [!proposition]
> Soit $A \in M_n(\mathbb{C})$ diagonalisable : $\exists P$ inversible tq $A = P^{-1} D P$ avec $D$ diagonale.
> 
> Soit $\delta A$ une perturbation.
> 
> Alors $\forall \mu$ vp de $A + \delta A$,
> $$
> \min_{\lambda_k \text{ vp de } A} |\mu - \lambda_k| \leq \text{cond}(P) \|\delta A\|
> $$

**Preuve:**

Soit $\mu$ vp de $A + \delta A$

Si $\mu$ est vp de $A$, alors ok.

Supposons $\mu$ non vp de $A$.

Comme $\mu$ est vp de $A + \delta A$, $\exists x \in \mathbb{C}^n$ tq
$$
\left( (A + \delta A) - \mu \, \text{Id} \right) x = 0
$$

$$
\Leftrightarrow P \left( (A + \delta A) - \mu \, \text{Id} \right) P^{-1} y = 0
$$

$$
\Leftrightarrow (P A P^{-1} + P \delta A P^{-1} - \mu \, \text{Id}) y = 0
$$

en posant $y = P^{-1} x$

$$
\Leftrightarrow (D - \mu \, \text{Id}) y + (P \delta A P^{-1}) y = 0
$$

$$
\Leftrightarrow y = - (D - \mu \, \text{Id})^{-1} (P \delta A P^{-1}) y
$$
$\implies$

$$
\| y \| \leq \| (D - \mu \, \text{Id})^{-1} \| \| P \delta A P^{-1} \| \| y \|
$$

$$
1 \leq \| (D - \mu \, \text{Id})^{-1} \| \| P \| \| \delta A \| \| P^{-1} \|
$$

$$
\frac{1}{\| (D - \mu \, \text{Id})^{-1} \|} \leq \text{cond}(P) \| \delta A \|
$$

alors,

$$
\| (D - \mu \, \text{Id})^{-1} \| = \max_k \frac{1}{|\lambda_k - \mu|}
$$

$$
\frac{1}{\min_k |\lambda_k - \mu|} \leq \text{cond}(P) \| \delta A \|
$$

## II Méthode d'Arnoldi

### 1) Vecteurs propres et valeurs propres de Ritz.

Rappel base d'Arnoldi

$(v_0, \ldots, v_p)$ base de $K_p(A; r_0)$

$A V_p = V_p \hat{H}_p$

où $V_p = [v_0, \ldots, v_p] \in M_{m,p+1}(\mathbb{R})$

Donc

$$
A V_p =
\begin{bmatrix}
\\
v_0 & v_1 & \cdots & v_{p+1}\\
&
\end{bmatrix}
\left(
\begin{bmatrix}
&H_p& \\
&0&
\end{bmatrix}
\begin{array}{c|c|c}
&H_p&\\
\hline 
\end{array}
+
\begin{bmatrix}
&0 \\
0 & &h_{p+1,p}
\end{bmatrix}
\right)
$$
$$
=
\begin{bmatrix}
\\
v_0 & v_1 & \cdots & v_{p}\\
&
\end{bmatrix}
H_p
+
h_{p+1,p}
\begin{bmatrix}
\\
v_0 & v_1 & \cdots & v_{p+1}\\
&
\end{bmatrix}
e_{p+2} e_{p+1}^T
$$
$$
= V_p H_p + h_{p+1, p} v_{p+1} e_p^T
$$

![[Pasted image 20241113082057.png]]

En particulier,

$$
V_p^T A V_p = V_p^T V_p H_p + h_{p+1,p} V_{p}^T v_{p+1} e_{p+1}^T = H_p
$$

avec $V_i^T V_j = \delta_{ij}$, $V_{p}^T v_{p+1}=0$

Remarque : $V_p V_p^T$ : projection sur $K_p(A; r_0)$

$$
P_{V_p}(x) = \sum_{i=0}^p (v_i, x) v_i = \sum_{i=0}^p v_i v_i^T x = V_p V_p^T x
$$

On projette : $V_p V_p^T A V_p V_p^T \in M_{m}(\mathbb{R})$ restriction de à $K_p(A; r_0)$

$$
V_p (V_p^T A V_p) V_p^T = V_p H_p V_p^T
$$

donc $H_p \in M_{m}(\mathbb{R})$

représente $A$ restreinte à $K_p(A; r_0)$, exprimée dans la base $(v_0, \ldots, v_p)$

> [!definition]
> Soient $(\lambda_i^{(p)})$ les valeurs propres de $H_p$ et $(y_i^{(p)})$ les vecteurs propres associés de $H_p$.

Les $(\lambda_i^{(p)})$ sont les valeurs de Ritz de $A$ et $u_i^{(p)} = V_p y_i^{(p)}$ les vecteurs de Ritz de $A$.

On a

$$
A u_i^{(p)} = \lambda_i^{(p)} u_i^{(p)} + h_{p+1, p} v_{p+1} (e_{p+1}^T y_i^{(p)})
$$

Preuve :

$$
A u_i^{(p)} = A V_p y_i^{(p)}
$$

$$
= (V_p H_p + h_{p+1, p} v_{p+1} e_{p+1}^T) y_i^{(p)}
$$

$$
= V_p H_p y_i^{(p)} + h_{p+1, p} v_{p+1} (e_{p+1}^T y_i^{(p)})
$$

$$
= \lambda_i^{(p)} u_i^{(p)} + h_{p+1, p} v_{p+1} (e_{p+1}^T y_i^{(p)})
$$

> [!remark]
> Quand $p \to +\infty$, $\lambda_i^{(p)} \to \lambda_j$ valeur propre de $A$ !

**Algorithme**

Soit $A \in M_n(\mathbb{C})$ et $b \in \mathbb{C}^m$

1. $v_0 = b / \| b \|$, $V_0 = [v_0]$, $\hat{H}_{-1} = []$

2. Tant que critère non satisfait :
   - Calculer $V_{p+1}, \hat{H}_p$ à partir de $V_p, \hat{H}_{p-1}$
   - Calculer une ou plusieurs valeurs propres de $\hat{H}_p \in M_{p+1}(\mathbb{R})$ par récurrence

$$
p + 1 \ll n
$$

Diagramme (cercle avec des vecteurs) : $x_0, Ax_0, A x_1, A x_2, \ldots$

![[Pasted image 20241113085041.png|400]]

$$
x_{k+1} = \frac{A x_k}{\| A x_k \|}=\frac{A(A^k x_0/\|A^k x_0\|)}{\|A(A^k x_0/\|A^k x_0\|)\|} = \frac{A^k x_0}{\|A^k x_0\|}
$$

---

$$
x_0 = \sum \alpha_i e_i \quad \text{(base de vecteurs propres)}
$$

$$
A x_0 = \sum \alpha_i (A e_i) = \sum \alpha_i \lambda_i e_i
$$

$$
\Rightarrow A^k x_0 = \sum \alpha_i \lambda_i^k e_i
$$

$$
= \lambda_0^k \left( \alpha_0 e_0 + \sum_{i > 0} \frac{\lambda_i}{\lambda_0}^k \alpha_i e_i \right)
$$

---

$$
\Rightarrow x_{2k} 
= \frac{A^{2k} x_0}{\| A^{2k} x_0 \|} 
= \frac{\lambda_0^{2k}}{\left|\lambda_0\right|^{2k}} \cdot \frac{\alpha_0 e_0 + \sum_{i > 0} \left( \frac{\lambda_i}{\lambda_0} \right)^{2k} \alpha_i e_i}{\left\| \alpha_0 e_0 + \sum_{i > 0} \left( \frac{\lambda_i}{\lambda_0} \right)^{2k} \alpha_i e_i \right\|} 
= 1
$$

On a $x_{2k}^T H x_{2k} \rightarrow \frac{e_0^T H e_0}{\|e_0\|} = \lambda_0$

---

**Algorithme 4 :**

1. $y_k = H^{-1} x_k \quad$ où résout sys. lin. $H y_k = x_k$

2. $x_{k+1} = y_k / \| y_k \|$

Algo :

$$
H_{(1)} = Q_{(1)} R_{(1)}
$$
$$
H_{(2)} = R_{(1)} Q_{(1)}
$$
$$
H_{(3)} = R_{(2)} Q_{(2)}
$$


$$
H_{(k+1)} = R_{(k)} Q_{(k)}
$$
$$
= Q_{(k)}^T Q_{(k)} R_{(k)} Q_{(k)}
$$
$$
= Q_{(k)}^T H_{(k)} Q_{(k)}
$$
$$
= Q_{(k)}^T Q_{(k-1)}^T \dots Q_{(1)}^T H Q_{(1)} Q_{(2)} \dots Q_{(k)}
$$
$$
= (Q_{(1)} Q_{(2)} \dots Q_{(k)})^T H Q_{(1)} Q_{(2)} \dots Q_{(k)}
$$

On a $(Q_{(1)} Q_{(2)} \dots Q_{(k)}) \in O_n(\mathbb{R})$.

$(H_{(k)})$ et $H$ ont les mêmes valeurs propres.

---

**Prop** : $H \in M_n(\mathbb{R})$ semblable avec $\lambda_1, \lambda_2, \ldots, \lambda_n$,de modules destinctes 

On suppose que $P^{-1}$ matrice de passage dans la base, diagonalisation réduction avec factorisation LU.

![[Pasted image 20241113091305.png]]

$\implies$ estimation de toutes les valeurs propres

**Remarque :**

$$
H^{(k+1)} = (Q_{(1)} R_{(1)})^{k+1}
$$

$$
= Q_{(1)} (R_{(1)} Q_{(1)} R_{(1)} \dots Q_{(1)}) R_{(1)}
$$
$$
= Q_{(1)} (R_{(1)} Q_{(1)})^k R_{(1)}
$$

$$
= Q_{(1)} (Q_{(2)}R_{(2)} )^k R_{(1)}
$$
$$
= Q_{(1)} Q_{(2)} (Q_{(2)}R_{(2)} )^{k-1} R_{(2)} R_{(1)}
$$

$$
= (Q_{(1)} \dots Q_{(k)}) (R_{(k)} \dots R_{(1)})
$$

**"Méthode de la puissance généralisée"**

**Remarque :**

- $H$ de Hessenberg  ![[Pasted image 20241113092859.png]] 
- $\implies$ facto QR avec matrice de Givens : $O((p+1)^2)$ opérations.
- $H$ de Hessenberg et $H = QR$

- Alors $Q$ et $R$ de Hessenberg.

- $R$ et $Q$ de Hessenberg impliquent $RQ$ de Hessenberg.

**Conclusion :** Si $H$ est de Hessenberg, $(H_{(k)})$ sera de Hessenberg, et chaque factorisation $QR$ est en $O((p+1)^2)$.

## III Méthode de Lanczos  
(matrices symétriques)

### 1. Méthode d'Arnoldi dans le cas symétrique.

**Proposition** : Si $A$ est symétrique, $V_p^T A V_p = H_p$ est symétrique de Hessenberg dans le cas diagonale.

$$
T_p = H_p = 
\begin{pmatrix}
\alpha_0 & \beta_0 & 0 & \dots & 0 \\
\beta_0 & \alpha_1 & \beta_1 & \dots & 0 \\
0 & \beta_1 & \alpha_2 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \beta_{p-1} \\
0 & 0 & 0 & \beta_{p-1} & \alpha_p \\
0&0&0&0&\beta_{p}
\end{pmatrix}
$$

$$
A V_p = h_{p+1, p} v_{p+1} + \sum_{j=0}^p h_{j, p} v_j
$$

$$
= \beta_p v_{p+1} + \alpha_p v_p + \beta_{p-1} v_{p-1}
$$

**Algorithme d'Arnoldi est une récurrence d'ordre 2 dans ce cas !**

**Données** : $v_0 \in \mathbb{C}^n$

$$
\beta_{-1} = 0, \quad v_{-1} = 0
$$

Pour $p$ de $0$ à $N$ :

$$
w_p = A v_p
$$
$$
\alpha_p = (w_p, v_p)
$$
$$
w_p = w_p - \alpha_p v_p - \beta_{p-1} v_{p-1}
$$
$$
\beta_p = \|w_p\|
$$
$$
v_{p+1} = w_p / \beta_p
$$

**Remarque** : Pour calculer $H_p$ à partir de $H_{p-1}$, on a besoin de calculer uniquement $v_p$ et $\beta_{p-1}$.  
→ Algo de Lanczos  
→ On ne stocke que $v_p$ et $v_{p-1}$ !!

Valeurs de Ritz = valeurs propres de $H_p = T_p$

---

Diagramme de $T$ : sa matrice tridiagonale

---

$$
\alpha_2 = \alpha_1, \quad a = \frac{a + b_1}{2} = b_2
$$

$N(|\alpha_1|) \geq i$