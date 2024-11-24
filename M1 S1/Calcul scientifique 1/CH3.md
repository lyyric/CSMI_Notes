Laplacien 2D

$u(x)$

$$
-\frac{\partial u}{\partial x^2} = f \quad \text{sur } \Omega
$$

+ CL

## I) Équations

Fonction inconnue $u(x, y)$ régulière

$(x, y) \in \Omega = ] 0, L_x [ \times ] 0, L_y [$

$f : \mathbb{R} \rightarrow \mathbb{R}$ donnée (régulière)

On cherche une solution de :

$$
-\Delta u = f \quad \text{sur } \Omega
$$
$$
u = 0 \quad \text{sur } \partial \Omega
$$

condition aux limites de Dirichlet

Cas plus général de conditions aux limites :

$$
\frac{\partial u}{\partial n}=\nabla u \cdot n
$$
$n$ : vecteur normal sortout sur $\partial \Omega$ 
$$
\frac{\partial u}{\partial n} + \alpha u = 0 \quad \alpha \geq 0
$$

condition de Robin.

$$
\alpha = 0 \rightarrow \text{Neumann}
$$
$$
\frac{1}{\alpha} \frac{\partial u}{\partial n} + u = 0
$$
$$
\alpha \rightarrow +\infty \quad \text{on retrouve Dirichlet.}
$$

$$
\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}
$$

## II) Approximation par différences finies

Grille, maillage de $\Omega$

$$
\Delta x = \frac{L_x}{N_x}, \quad N_x \in \mathbb{N}^*
$$
$$
\Delta y = \frac{L_y}{N_y}, \quad N_y \in \mathbb{N}^*
$$

En pratique $\Delta x = \Delta y = h$

![[Pasted image 20241108083134.png|400]]

$$
(x_i, y_j) = (i \Delta x, j \Delta y)
$$
$$
\Delta x = \Delta y = h
$$

$g : \mathbb{R} \rightarrow \mathbb{R}$

$$
g''(x) = \frac{g(x + h) - 2g(x) + g(x - h)}{h^2} + \mathcal{O}(h^2)
$$

$$
u_{i,j} \simeq u(x_i, y_j) - \Delta u (x_i, y_j)
$$

$$
\simeq \frac{u_{i-1,j} + 2u_{i,j} - u_{i+1,j}}{h^2} + \frac{- u_{i,j-1} + 2u_{i,j} - u_{i,j+1}}{h^2}
$$
$$
= \frac{1}{h^2} (-u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1} + 4u_{i,j}) = f_{i,j}
$$

$$
f_{i,j} = f(x_i, y_j)
$$

$u_{i,j} = 0$ sur les bords.

$$
u_{0,j} = u_{N_x,j} = 0
$$
$$
u_{i,0} = u_{i,N_y} = 0
$$

$$
1 \leq i \leq N_x - 1
$$
$$
1 \leq j \leq N_y - 1 \quad (*)
$$

Si $(i,j)$ est à l'intérieur de la grille, $u_{i,j}$ est une inconnue. Sinon, $u_{i,j} = 0$.

$$
N = (N_x - 1)(N_y - 1)
$$

est le nombre d'inconnues.

On a aussi $N$ équations.

On est revenu à la résolution d'un système linéaire de taille $N$.

$$
AU = F
$$

$A$ matrice de taille $N \times N$.

Les valeurs de $u$ au bord sont considérées aussi comme des inconnues (fictives).

La matrice $A$ sera en fait de taille

$$
N = (N_x + 1)(N_y + 1)
$$

$A$ est une matrice creuse (sparse).

$u_{i,j}$ : 2 indices

$$
k = i + j (N_x + 1)
$$
$$
U_k = u_{i,j}
$$

$$
0 \leq i < N_x + 1
$$

$j$ : quotient de la division euclidienne de $k$ par $(N_x + 1)$

$i$ : reste

Python :

```python
j = k // (N_x + 1)
i = k % (N_x + 1)
```

Rust :

```rust
j = k / (N_x + 1)
i = k % (N_x + 1)
```

Calcul de $A$ ? de $F$ ?

Procédure d'assemblage

À quelle condition a-t-on $A_{k,l} \neq 0$ ?

Si les points $k$ et $l$ sont reliés par une arête ou si $k = l$, alors $A_{k,l} \neq 0$.

$$
k = i + j (N_x + 1)
$$
$$
l = i' + j' (N_x + 1)
$$

### Méthode possible

Tableau de connectivité

Nombre d’arêtes

$$
M_a = N_y (N_x + 1) + (N_y + 1) N_x
$$

**verticales** / **horizontales**

$$
L = [\,]
\quad (L = \text{vecteur des } (k, l))
$$

for $0 \leq i < N_x$

for $0 \leq j < N_y$

$$
k = i + j (N_x + 1)
$$
$$
l = (i + 1) + j (N_x + 1)
$$

$L.\text{push}((k, l))$

---

for $0 \leq i < N_x$

for $0 \leq j < N_y$

$$
k = i + j (N_x + 1)
$$
$$
l = i + (j + 1) (N_x + 1)
$$

$L.\text{push}((k, l))$

这里应该是代码

Assemblage de $A$ en format "Coordonnées" (COO ou more).

1 tableau de triplet

$$
\text{mat} = [( \text{val}, k, l ) \ldots]
$$

$A_{k,l} = \text{val}$

$$
\text{mat} = []
$$

for $0 \leq k < N = (N_x + 1)(N_y + 1)$

$$
\text{mat.push} \left( \left( \frac{4}{h^2}, k, k \right) \right)
$$

Termes extradiagonaux aux $i$

for $(k, l)$ in $L$

$$
\text{mat.push} \left( \left( -\frac{1}{h^2}, k, l \right) \right)
$$

$$
\text{mat.push} \left( \left( -\frac{1}{h^2}, l, k \right) \right)
$$

$$
F = [\,]
$$

for $0 \leq k < N$

$$
j = k // (N_x + 1)
$$
$$
i = k \% (N_x + 1)
$$

$$
F.\text{push}(f(x_i, y_j))
$$

sparse COO-matrix

---

Convertir mat -> matrice sparse

Résolution $AU = F$

Pour afficher $U$
-> numpy reshape

Problème : comment traiter les conditions aux limites ?

Méthode du grand pivot

On veut que $U_k = 0$ si $k$ est un indice de bord.

$$
A_{k,k} = 1e20 = 10^{20}
$$

$$
U_k = \frac{1}{A_{k,k}}(\cdots) \quad \text{avec } U \approx 10^{-20}
$$

$$
\text{bord} = []
$$

for $0 \leq i \leq N_x$

$$
k = i + 0 \cdot (N_x + 1)
$$
$$
\text{bord.push}(k)
$$

$$
k = i + N_y \cdot (N_x + 1)
$$
$$
\text{bord.push}(k)
$$

for $0 \leq j \leq N_y$

$$
k = 0 + j \cdot (N_x + 1)
$$
$$
\text{bord.push}(k)
$$

$$
k = N_x + j \cdot (N_x + 1)
$$
$$
\text{bord.push}(k)
$$

for $k$ in $\text{bord}$

$$
\text{mat.push}((10^{20}, k, k))
$$

![[Pasted image 20241108095838.png]]

![[Pasted image 20241108095844.png]]

