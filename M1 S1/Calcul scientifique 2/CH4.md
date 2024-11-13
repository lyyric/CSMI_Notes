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

## 1) Conditionnement du problème

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

