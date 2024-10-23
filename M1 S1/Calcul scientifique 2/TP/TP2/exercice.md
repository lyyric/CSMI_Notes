TP 2 : Méthodes GMRES et Gradient Conjugué

L'objectif de ce TP est de programmer les méthodes GMRES et Gradient Conjugué. Commencez par importer les modules python :

```python
import numpy as np 
import scipy as sp 
import scipy.sparse as spsp 
import scipy.sparse.linalg as spsplin
```

Pour la multiplication matricielle, utilisez la commande @ : elle est compatible avec les structures creuses de scipy.

## Partie 1. (GMRES)

1. Programmer une fonction `Arnoldi( A, V, H)` , qui à partir des matrices $A \in M_n(\mathbb{R})$, $V \in M_{n,p}(\mathbb{R})$ (contenant la base de l'espace de Krylov $K_{p-1}(A; r_0)$) et $H \in M_{p,p-1}(\mathbb{R})$, renvoie les matrices $V_p \in M_{n,p+1}(\mathbb{R})$ (contenant la base de l'espace de Krylov $K_p(A; r_0)$) et $H_p \in M_{p+1,p}(\mathbb{R})$. Nous rappelons que nous avons les relations suivantes :

$$
w_p = A v_{p-1} - \sum_{j \le p-1} \langle A v_{p-1}, v_j \rangle v_j, \quad v_p = \frac{w_p}{\|w_p\|},
$$
$$
A v_{p-1} = \|w_p\| v_p + \sum_{j \le p-1} \langle A v_{p-1}, v_j \rangle v_j = h_{p,p-1} v_p + \sum_{j \le p-1} h_{j,p-1} v_j.
$$

en notant $V = [v_0, \dots, v_{p-1}] \in M_{n,p}(\mathbb{R})$. Au vu de ces relations, commencez par calculer les $h_{j,p-1}$ et le $w_p$, puis calculer $v_p$ et $h_{p,p-1}$.
    
2. Programmer une fonction `gmres(A, b, xexact)` qui, à partir d'une matrice $A \in M_n(\mathbb{R})$, un vecteur $b \in \mathbb{R}^n$ et une solution exacte $x_{exact} \in \mathbb{R}^n$ (si disponible), renvoie la solution $x$ du système linéaire obtenue par l'algorithme GMRES, ainsi que la liste des erreurs relatives $\|x_{exact} - x_p\|/\|x_{exact}\|$, et la liste de la norme des résidus (relatifs) $\|r_p\|/\|r_0\|$. Nous rappelons ci-dessous l'algorithme :

- $x_0$ donné
- $r_0 = b - A x_0$
- $v_0 = r_0 / \|r_0\|$, $V_0 = [v_0]$, $\hat{H}_{-1} = [ ]$  
Tant que condition non satisfaite :
- Calcul de $V_{p+1}, \hat{H}_p$ à partir de $V_p, \hat{H}_{p-1}$
- $Q_p R_p = \hat{H}_p$
- $(R_p)_{0 \le i,j \le p} y = |r_0| (Q_p^T e_0)_{0 \le j \le p}$
- $x_{p+1} = x_0 + V_p y$

	Vous pourrez utiliser les fonctions `sp.linalg.qr` pour la factorisation QR et `np.linalg.solve` pour la résolution du système triangulaire.

3. Tester votre programme sur la matrice suivante :

```python
A = np.diag(2*np.ones(n)) + 0.5 * np.random.rand(n, n)/np.sqrt(n)
```

Prendre $x_0 = 0$. Afficher l’erreur et le résidu en fonction des itérations (en échelle logarithmique). On pourra calculer la solution exacte avec une fonction numpy.

4. (Facultatif) Programmer une fonction `gmres(A, b, xexact, p)` qui effectue un redémarrage de la fonction GMRES toutes les $p \in \mathbb{N}^*$ itérations.
    

## Partie 2. (Gradient Conjugué) 

5. Programmer une fonction `gradient_conjugue(A, b, xexact)` qui, à partir d’une matrice $A \in M_n(\mathbb{R})$, un vecteur $b \in \mathbb{R}^n$ et une solution exacte $x_{exact} \in \mathbb{R}^n$ (si disponible), renvoie la solution $x$ du système linéaire obtenue par l’algorithme du Gradient conjugué, ainsi que la liste des erreurs relatives, et la liste de la norme des résidus (relatifs). Nous rappelons ci-dessous l’algorithme (à gauche) :

(Gradient Conjugué)
- $x_0$ donné
- $r_0 = b - A x_0, d_0 = r_0$
Tant que condition non satisfaite :
- $s_p = \frac{(r_p, r_p)}{(A d_p, d_p)}$
- $x_{p+1} = x_p + s_p d_p$
- $r_{p+1} = r_p - s_p A d_p$
- $\beta_p = \frac{(r_{p+1}, r_{p+1})}{(r_p, r_p)}$
- $d_{p+1} = r_{p+1} + \beta_p d_p$

(Gradient Conjugué Préconditionné)
- $x_0$ donné
- $r_0 = b - A x_0,$
- $M z_0 = r_0, d_0 = z_0$
Tant que condition non satisfaite :
- $s_p = \frac{(r_p, z_p)}{(A d_p, d_p)}$
- $x_{p+1} = x_p + s_p d_p$
- $r_{p+1} = r_p - s_p A d_p$
- $M z_{p+1} = r_{p+1}$
- $\beta_p = \frac{(r_{p+1}, z_{p+1})}{(r_p, z_p)}$
- $d_{p+1} = z_{p+1} + \beta_p d_p$

6. Testez votre méthode sur la matrice suivante :

```python
B = spsp.diags([[4.]*n,[-1]*(n-1),[-1]*(n-1),[-1]*(n-d),[-1]*(n-d)], [0,1,-1,d,-d])
```

avec $n = d^2$. Afficher l’erreur et le résidu en fonction des itérations. On pourra calculer la solution exacte avec une fonction `scipy`. Comparer avec ce que vous obtenez avec la méthode GMRES ainsi que les temps de calcul avec le module `time`. Commentez. Pourquoi ces algorithmes sont-ils bien adaptés aux structures creuses ?

## Partie 3. (Préconditionnement)

7. Appliquer la méthode GMRES aux matrices
```python
C = np.diag(2+np.arange(n)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
```
Comparer les résultats avec le système préconditionné $M^{-1} C x = M^{-1} b$ où $M$ est la partie diagonale de $C$ (préconditionnement diagonal ou Jacobi). Afficher le conditionnement des matrices $C$ et $M^{-1} C$ grâce à la fonction `np.linalg.cond`. Comparer les temps de calcul. Observez les résultats pour différentes tailles de matrice $n$. Commentez.

8. Reprendre la question précédente avec

```python
D = np.diag(2*np.ones(n)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
```

Commentez. 

9. Programmer une fonction `gradient_conjugue_precond(A, b, M, xexact)` qui, à partir d’une matrice $A \in M_n(\mathbb{R})$, un vecteur $b \in \mathbb{R}^n$, une matrice de préconditionnement, et une solution exacte $x_{exact} \in \mathbb{R}^n$ (si disponible), renvoie la solution $x$ du système linéaire obtenue par l’algorithme du Gradient conjugué préconditionné, ainsi que la liste des erreurs relatives, et la liste de la norme des résidus (relatifs). L’algorithme est rappelé ci-dessus (à droite).

10. Tester votre fonction `gradient_conjugue_precond` sur la matrice $B$ (question 6), en utilisant comme matrice de préconditionnement la factorisation LU incomplète donnée par `spsplin.spilu`. Comparer les temps de calcul avec la méthode de gradient conjugué non-préconditionné. Commentez.

