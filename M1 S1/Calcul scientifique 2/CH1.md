# 1. Matrices Creuses (Sparse Matrices)

## Définition

Soit $A \in M_{n,n}(\mathbb{C})$ et $p$ est le nombre de coefficients non nuls.

- **$A$ est creuse** si elle a peu de coefficients non nuls : $p \ll n^2$.
- **$A$ est pleine** si elle a beaucoup de coefficients non nuls : $p \approx n^2$.

## Remarque

Les matrices creuses apparaissent fréquemment dans les applications scientifiques et techniques, notamment lors de la discrétisation d'équations différentielles partielles.

## Exemple de Matrice Creuse

### Équation de Poisson

**Équation de Poisson**
$$
\begin{cases}
- u''(x) = f(x), \\
u(0) = 0, \quad u(1) = 0.
\end{cases}
$$

### Discrétisation de l'Intervalle $[a, b]$

On discrétise l'intervalle $[a, b]$ en $n+2$ points :

Pour tout $i \in \{0, 1, \ldots, n+1\}$ :
$$
x_i = \frac{i}{n+1},
$$
avec $\Delta x = \frac{1}{n+1}$.

### Schéma Numérique

On cherche une solution approchée $u \in \mathbb{R}^m$ vérifiant $A u = b$, où :

$$
A = \frac{1}{\Delta x^2} \begin{pmatrix} 
-2 & 1 & 0 & \cdots & 0 \\
1 & -2 & 1 & \cdots & 0 \\
0 & 1 & -2 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & 1 \\
0 & 0 & 0 & 1 & -2 
\end{pmatrix}, \quad b = \begin{pmatrix} 
f(x_1) \\
f(x_2) \\
\vdots \\
f(x_{m-1})
\end{pmatrix},
$$

avec $A \in M_{m,m}(\mathbb{R})$ une matrice creuse, et $b \in \mathbb{R}^m$.

*Remarque :* La matrice $A$ est tridiagonale, ce qui signifie que seuls les éléments sur la diagonale principale et les diagonales adjacentes sont non nuls. Cela illustre une structure creuse où le nombre de coefficients non nuls $p$ est proportionnel à $n$, plutôt qu'à $n^2$.

# 2. Normes Matricielles et Spectrales

## Norme Matricielle

### Définition du Rayon Spectral

Soit $A \in M_{n,n}(\mathbb{C})$, on appelle **rayon spectral** de $A$ la quantité :
$$
\rho(A) = \max \{ |\lambda|, \; \lambda \in \text{Sp}(A) \},
$$
où $\text{Sp}(A)$ est le spectre de $A$ (l'ensemble de ses valeurs propres).

### Propriétés des Matrices

#### Matrices Normales

1. **Si $A$ est normale**, c'est-à-dire $A^*A = AA^*$ (resp. $\bar{A}^T A = A \bar{A}^T$ pour les matrices réelles), alors $A$ est diagonalisable. Il existe donc une base orthonormée et une matrice unitaire $P \in U_n(\mathbb{C})$ (resp. $P \in O_n(\mathbb{R})$ pour les matrices réelles) telle que :
$$
A = P D P^* \quad \text{avec } D = \text{diag}(\lambda_1, \ldots, \lambda_n),
$$
où $D$ est une matrice diagonale contenant les valeurs propres de $A$.

#### Matrices Hermitiennes

2. **Si $A = A^*$**, alors $A$ est hermitienne (resp. $A = A^T$ est symétrique pour les matrices réelles). Dans ce cas, $A$ est diagonalisable dans une base orthonormée et ses valeurs propres sont réelles. Si de plus $A$ est définie positive\*, alors les valeurs propres sont strictement positives :
$$
0 < \lambda_1 \leq \lambda_2 \leq \ldots \leq \lambda_n.
$$

\* *Remarque :* Les valeurs propres $\sigma_1, \ldots, \sigma_n$ sont les valeurs singulières de $A$.

## Définition des Normes Naturelles

### Normes Compatibles avec la Multiplication

1. **Les normes naturelles sur** $M_n(\mathbb{C})$ sont des normes compatibles avec la multiplication matricielle :
$$
\forall A, B \in M_n(\mathbb{C}), \quad \|AB\| \leq \|A\| \|B\|.
$$

### Normes Subordonnées

2. **Soit** $\| \cdot \|$ **une norme sur** $\mathbb{C}^n$. On lui associe une norme dite **subordonnée** $\| \cdot \|_M$ sur $M_n(\mathbb{C})$ définie par :
$$
\|A\|_M = \sup_{x \neq 0} \frac{\|Ax\|}{\|x\|}.
$$
Elle vérifie :
$$
\forall A \in M_n(\mathbb{C}), \quad \forall x \in \mathbb{C}^n, \quad \|Ax\| \leq \|A\|_M \|x\|,
$$
et
$$
\|I\|_M = 1,
$$
où $I$ est la matrice identité.

## Propriétés des Normes et Rayons Spectraux

### Formules pour $\|A\|_\infty$, $\|A\|_1$, $\|A\|_2$

**Proposition**

Pour tout $A \in M_{n,n}(\mathbb{C})$ :

- $\|A\|_\infty = \max_{1 \leq i \leq n} \sum_{j=1}^n |a_{ij}|$

- $\|A\|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^n |a_{ij}|$

- $\|A\|_2 = \sqrt{\rho(A^* A)} = \sigma_{\max}$

## Normes de Vecteurs

Pour tout $x \in \mathbb{C}^m$ :

1. **Norme infinie** :
$$
\|x\|_\infty = \max |x_i| \quad \text{(maximum des valeurs absolues des composantes de } x)
$$

2. **Norme 1** :
$$
\|x\|_1 = \sum |x_i| \quad \text{(somme des valeurs absolues des composantes de } x)
$$

3. **Norme 2** (norme Euclidienne) :
$$
\|x\|_2 = \sqrt{\sum |x_i|^2}
$$

## Propriétés Supplémentaires

### Matrices Normales

1. **Si $A$ est normale**, alors :
$$
\|A\|_2 = \rho(A)
$$

### Matrices Hermitiennes Positives

2. **Si $A$ est hermitienne positive** :
$$
\|A\|_2 = \rho(A) = \lambda_{\max}
$$

### Matrices Unitaires

3. **Si $A$ est unitaire** :
$$
\|A\|_2 = \rho(A) = 1
$$

## Preuves des Propriétés

### Preuve pour les Matrices Normales

**Proposition**

Pour une matrice normale $A$ :
$$
\rho(A) = \inf_{\|A\|} \|A\|,
$$
où $\|A\|$ est une norme matricielle.

**Preuve**

1. Pour toute norme subordonnée $\| \cdot \|_2$ :
$$
\|A\|_2 = \sqrt{\rho(A^* A)} = \sigma_{\max}
$$

2. Pour tout $x \in \mathbb{C}^n$ :
$$
\begin{aligned}
\|Ax\|_2^2 &= (Ax, Ax) \\
&= (x, A^*Ax) \\
&= (x, P \Sigma P^* x) \\
&= (P^* x, \Sigma P^* x) \quad \text{avec } A^*A = P \Sigma P^* \quad \text{où } \Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2) \\
&= (y, \Sigma y) \quad \text{avec } y = P^* x \\
&= \left( \begin{pmatrix} y_1 \\ \vdots \\ y_n \end{pmatrix}, \begin{pmatrix} \sigma_1^2 y_1 \\ \vdots \\ \sigma_n^2 y_n \end{pmatrix} \right) \\
&= \sum_{i=1}^{n} \overline{y_i} \sigma_i^2 y_i \\
&= \sum_{i=1}^{n} \sigma_i^2 |y_i|^2 \\
&\leq \sigma_{\max}^2 \sum_{i=1}^{n} |y_i|^2 \\
&= \sigma_{\max}^2 \|y\|_2^2 \\
&= \sigma_{\max}^2 \|x\|_2^2.
\end{aligned}
$$

Pour tout $x \in \mathbb{C}^m$ :
$$
\|Ax\|_2 \leq \sigma_{\max} \|x\|_2,
$$
$$
\|A\|_2 \leq \sigma_{\max} ,
$$

On a égalité en prenant $x \in \mathbb{C}^m$ comme vecteur propre de $A^* A$ associé à $\sigma_{\max}^2$.

### Preuve pour les Matrices Hermitiennes Positives

**Propriété**

Soit $A$ une matrice hermitienne positive. Pour tout $x \in \mathbb{C}^n$ :
$$
\|A\|_2 = \rho(A)
$$

**Preuve**

1.  $x \in \mathbb{C}^n$, alors :
$$
\|Ax\|_2^2 = \|P D P^* x\|_2^2 = \|D P^* x\|_2^2,
$$
où $A = P D P^*$ avec $D = \text{diag}(\lambda_1, \ldots, \lambda_n)$, $P \in O_n(\mathbb{R})$ ou $P \in U_n(\mathbb{C})$.

2. En posant $y = P^* x$, on obtient :
$$
\|Ax\|_2^2 = \|D y\|_2^2.
$$

3. En calculant explicitement, on a :
$$
\|D y\|_2^2 = \left\| \begin{pmatrix} 
\lambda_1 y_1 \\
\vdots \\
\lambda_n y_n 
\end{pmatrix} \right\|_2^2 
= \sum_{i=1}^{n} \overline{(\lambda_i y_i)}(\lambda_i y_i)
= \sum_{i=1}^{n} \lambda_i^2 |y_i|^2.
$$

4. Si $\lambda_i \in \mathbb{R}$, alors :
$$
\|Ax\|_2^2 = \sum_{i=1}^{n} \lambda_i^2 |y_i|^2.
$$
$$
\leq \lambda_{\max}^2 \sum_{i=1}^{n} |y_i|^2 = \lambda_{\max}^2 \|y\|_2^2 = \lambda_{\max}^2 \|x\|_2^2
$$

où $y = P^* x$ avec $P$ unitaire.

# 3. Conditionnement et Sensibilité

## Définition du Conditionnement

Soit $A \in M_{n,n}(\mathbb{C})$ inversible. On appelle **conditionnement** de $A$ le réel :

$$
\text{cond}(A) = \|A\| \|A^{-1}\| \geq 1.
$$

Pour toute norme subordonnée, on a :

$$
1 = \|I\| = \|A A^{-1}\| \leq \|A\| \|A^{-1}\|.
$$

## Sensibilité des Solutions

Soit $A \in M_{n,n}(\mathbb{C})$ inversible, $b \in \mathbb{C}^n$ et $x \in \mathbb{C}^n$ la solution de $Ax = b$.

### Influence des Perturbations sur $b$

1. **Si $Ax = b$ et $A(x + \delta x) = b + \delta b$**, alors :

$$
\frac{\|\delta x\|}{\|x\|} \leq \text{cond}(A) \frac{\|\delta b\|}{\|b\|}.
$$

**Preuve :**

- À partir de $b = Ax$, on a :

  $$
  \|b\| \leq \|A\| \|x\| \implies \frac{1}{\|x\|} \leq \frac{\|A\|}{\|b\|}.
  $$

- En considérant la perturbation :

  $$
  A(x + \delta x) = b + \delta b \implies \delta x = A^{-1} \delta b \implies \|\delta x\| \leq \|A^{-1}\| \|\delta b\|.
  $$

- En combinant les deux inégalités :

  $$
  \frac{\|\delta x\|}{\|x\|} \leq \|A^{-1}\| \|\delta b\| \frac{\|A\|}{\|b\|} = \text{cond}(A) \frac{\|\delta b\|}{\|b\|}.
  $$

### Influence des Perturbations sur $A$

2. **Si $Ax = b$ et $(A + \delta A)(x + \delta x) = b$**, alors :

$$
\frac{\|\delta x\|}{\|x + \delta x\|} \leq \text{cond}(A) \frac{\|\delta A\|}{\|A\|}.
$$

**Preuve :**

- En développant l'équation perturbée :

  $$
  A(x + \delta x) + \delta A(x + \delta x) = b.
  $$

- En utilisant $Ax = b$, il vient :

  $$
  A\delta x = -\delta A(x + \delta x) \implies \delta x = -A^{-1} \delta A (x + \delta x).
  $$

- Donc :

  $$
  \|\delta x\| \leq \|A^{-1}\| \|\delta A\| \|x + \delta x\|.
  $$

- Ainsi :

  $$
  \frac{\|\delta x\|}{\|x + \delta x\|} \leq \|A^{-1}\| \|\delta A\| = \text{cond}(A) \frac{\|\delta A\|}{\|A\|}.
  $$

### Remarques sur la Propagation des Erreurs

#### Remarque 1

Si le conditionnement de $A$, noté $\text{cond}(A)$, est élevé, cela implique un mauvais contrôle de la propagation des erreurs. De petites perturbations sur $b$ ou $A$ peuvent engendrer de grandes erreurs sur la solution $x$.

#### Remarque 2

Les erreurs relatives sur les réels sont codées avec des nombres à virgule flottante en :

- **Simple précision** (32 bits)
- **Double précision** (64 bits)

**Exemple :**

Le nombre `103.000.004` est codé en base décimale comme :

- `0,10300 \times 10^9` si on garde seulement 5 chiffres significatifs.

Le nombre le plus proche de `0,10300 \times 10^9` est `0,10301 \times 10^9`, c'est-à-dire `103.001.000`.

Étant donné $x \in \mathbb{R}^n$, le nombre le plus proche $\tilde{x}$ vérifie :

$$
|x - \tilde{x}| \leq \varepsilon |x|,
$$

où $\varepsilon =$ précision machine :

- $\varepsilon \approx 10^{-7}$ (simple précision)
- $\varepsilon \approx 10^{-16}$ (double précision)

## Propositions sur le Conditionnement

### Pour Matrices Inversibles

**Proposition**

Pour $A \in M_{n,n}(\mathbb{C})$ inversible :

$$
\text{cond}(A) = \|A\|_2 \|A^{-1}\|_2 = \frac{\sigma_{\max}}{\sigma_{\min}},
$$

où $\sigma_{\max}$ et $\sigma_{\min}$ sont les valeurs singulières maximale et minimale de $A$, respectivement.

### Pour Matrices Hermitiennes Définies Positives

2. Pour $A \in M_{n,n}(\mathbb{C})$ hermitienne définie positive :

$$
\text{cond}(A) = \|A\|_2 \|A^{-1}\|_2 = \frac{\lambda_{\max}}{\lambda_{\min}},
$$

où $\lambda_{\max}$ et $\lambda_{\min}$ sont les valeurs propres maximale et minimale de $A$, respectivement.

**Remarque :**

Pour une matrice hermitienne définie positive $A$, on a :

$$
A = P D P^*, \quad A^{-1} = P D^{-1} P^*,
$$

avec $P$ unitaire et $D$ diagonale contenant les valeurs propres de $A$.

De plus :

$$
\|A^{-1}\|_2 = \rho(A^{-1}) = \frac{1}{\lambda_{\min}}.
$$

### Pour Matrices Unitaires

3. Pour $A \in M_{n,n}(\mathbb{C})$ unitaire :

$$
\text{cond}(A) = 1.
$$

**Preuve :**

- Pour une matrice unitaire $A$, on a $A^{-1} = A^*$ et $\|A\|_2 = \|A^*\|_2 = 1$, donc :

  $$
  \text{cond}(A) = \|A\|_2 \|A^{-1}\|_2 = 1 \times 1 = 1.
  $$

## Estimation de l'Erreur

### Méthode Directe de Résolution de Systèmes Linéaires

Pour résoudre l'équation $Ax = b$ où $A = BC$, on peut procéder de la manière suivante :

1. Écrire $Ax = b$ sous la forme :

   $$
   BCx = b.
   $$

2. Décomposer ce problème en deux sous-problèmes :

   $$
   \begin{cases}
   By = b, \\
   Cx = y.
   \end{cases}
   $$

   Cela signifie que nous résolvons d'abord pour $y$ en utilisant $B$, puis pour $x$ en utilisant $C$.

### Impact du Conditionnement sur l'Erreur

**Estimation de l'Erreur :**

L'erreur relative pour $x$ est donnée par :

$$
\frac{\|\delta x\|}{\|x\|} \leq \text{cond}(C) \frac{\|\delta y\|}{\|y\|} \leq \text{cond}(C) \text{cond}(B) \frac{\|\delta b\|}{\|b\|}.
$$

**Remarque :**

- On observe que :

  $$
  \text{cond}(C) \cdot \text{cond}(B) \geq \text{cond}(BC) = \text{cond}(A).
  $$

- Cela signifie qu'il peut y avoir une "perte" dans le contrôle de la propagation de l'erreur en décomposant $A$ en $B$ et $C$. En effet, le conditionnement du produit $BC$ peut être inférieur au produit des conditionnements de $B$ et $C$.

# 4. Stockage des Matrices Creuses

## Considérations sur le Calcul et le Stockage

### Remarque sur le Coût Computationnel

La multiplication matrice-vecteur nécessite en général $O(n^2)$ opérations pour calculer $Ax$.

Si la matrice est creuse avec $p$ coefficients non nuls, le calcul de $Ax$ nécessite $O(p)$ opérations.

Il est donc avantageux de stocker les coefficients non nuls de la matrice.

## Méthodes de Stockage

### Stockage par Coordonnées (Coordinate)

#### Principe et Exemple

**Stockage par coordonnées (Coordinate)**

Une matrice est représentée par 3 tableaux de taille $p$ :

- **data** : contient les coefficients non nuls.
- **row** : indices de lignes.
- **col** : indices de colonnes.

**Exemple de représentation :**

- Matrice :
  $$
  \begin{matrix}
  a & b & 0 & 0 & 0 \\
  0 & c & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & d & e \\
  0 & 0 & f & 0 & 0 \\
  \end{matrix}
  $$

- Représentation par stockage :
  - **data** : $[a, b, c, d, e, f]$
  - **row** : $[0, 1, 1, 2, 2, 3]$
  - **col** : $[0, 0, 1, 1, 2, 2]$

### Stockage Compact par Ligne (CSR)

#### Définition

Le **stockage compact par ligne** est une méthode pour stocker des matrices creuses. CSR (Compressed Sparse Row) est une méthode de stockage par ligne compressée.

- **Paramètres :**
  - $p$ : nombre d'éléments non nuls dans la matrice $A$ de taille $n \times n$.

- **Tableaux utilisés :**
  - Deux tableaux de taille $p$ :
    - **data** : valeurs non nulles.
    - **colon** : indices des colonnes.
  - Un tableau de taille $n+1$ :
    - **indptr** : indices des débuts de lignes dans le tableau **data**.

#### Exemple de Représentation

Soit la matrice $A$ :
$$
A = \begin{bmatrix}
a & b & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & d & e & 0 & 0 \\
0 & f & 0 & 0 & 0
\end{bmatrix}
$$

- **data** : $[a, b, c, d, e, f]$
- **colon** : $[0, 1, 1, 2, 3, 1]$
- **indptr** : $[0, 2, 3, 3, 3, 6]$

Les coefficients de la $i$-ème ligne sont stockés dans :
$$
\text{data}[ \text{indptr}[i] : \text{indptr}[i+1] ]
$$

**Répartition par lignes :**

- 1ère ligne : **data** $[0:2]$
- 2ème ligne : **data** $[2:3]$
- 3ème ligne : **data** $[3:3] = \varnothing$
- 4ème ligne : **data** $[3:5]$
- 5ème ligne : **data** $[5:6]$

### Stockage Bande (Band Storage)

#### Description

Les coefficients non nuls sont situés sur les diagonales de \(-r\) à \(+s\). Le stockage des diagonales se fait dans un tableau de taille \((r + s + 1) \times n\).

#### Illustration

```tikz
\begin{document} 
\begin{tikzpicture}[scale=1.2]

% 绘制矩形框代表矩阵
\draw[thick] (0,0) rectangle (5,5);

% 绘制主对角线的非零元素区域
\fill[green!30] (0,4) -- (0,5) -- (1,5) -- (5,1) -- (5,0) -- (4,0) -- cycle;

% 绘制对角线的虚线
\draw[thick, dashed] (0,5) -- (5,0);    % 主对角线
\draw[thick, dashed] (0,4) -- (4,0);    % -r 对角线
\draw[thick, dashed] (1,5) -- (5,1);    % s 对角线
\draw[thick, dashed] (0,4.5) -- (4.5,0);
\draw[thick, dashed] (0.5,5) -- (5,0.5);

% 在矩阵外标注对角线
\node at (-0.4,4) {-r};
\node at (-0.3,5.3) {0};
\node at (1,5.4) {s};

% 绘制零元素区域
\node at (0.5, 4.5) {0};
\node at (0.5, 3.5) {-r};
\node at (1.5, 4.5) {s};
\node at (1.5, 3.5) {0};
\node at (4.5, 4.5) {0};
\node at (4.5, 0.5) {0};
\node at (4.5, 1.5) {s};
\node at (3.5, 0.5) {-r};
\node at (1.5, 0.5) {0};
\node at (0.5, 0.5) {0};

% 添加矩阵名称和说明
\node at (-1.5, 2.5) {$A=$};

\end{tikzpicture}
\end{document} 
```

### Stockage Ligne de Ciel (Skyline Storage)

#### Méthode de Stockage

Le **stockage ligne de ciel (Skyline)** est une méthode de stockage pour les matrices creuses où les coefficients non nuls se trouvent principalement autour de la diagonale.

```tikz
\begin{document} 
\begin{tikzpicture}

% 绘制矩形框代表矩阵
\draw[thick] (0,0) rectangle (11,4.5);
\draw[thick] (0,4) rectangle (7,4.5);
\draw[thick] (0,3.5) rectangle (8,4);
\draw[thick] (0,3) rectangle (9,3.5);
\draw[thick] (0,2.5) rectangle (10,3);
\draw[thick] (0,2) rectangle (11,2.5);
\draw[thick] (1,1.5) rectangle (11,2);
\draw[thick] (2,1) rectangle (11,1.5);
\draw[thick] (3,0.5) rectangle (11,1);
\draw[thick] (4,0) rectangle (11,0.5);

% 绘制零元素区域
\node at (-0.5, 4.25) {-r};
\node at (-0.5, 2.25) {0};
\node at (-0.5, 0.25) {s};
\node at (0.5, 0.5) {0};
\node at (10, 4) {0};

\end{tikzpicture}
\end{document} 
```

#### Exemple et Illustration

**Exemple de matrice :**

```tikz
\begin{document}
\begin{tikzpicture}

% 绘制矩阵的边框
\draw[thick] (0,0) rectangle (6,6);

% 绘制星号 (*)
\node at (0.5,5.5) {*};
\node at (1.5,5.5) {*};
\node at (0.5,4.5) {*};
\node at (3.5,4.5) {*};
\node at (2.5,3.5) {*};
\node at (5.5,3.5) {*};
\node at (1.5,2.5) {*};
\node at (3.5,2.5) {*};
\node at (4.5,1.5) {*};
\node at (2.5,0.5) {*};
\node at (5.5,0.5) {*};

% 绘制零 (0)
\node at (2.5,5.5) {0};
\node at (3.5,5.5) {0};
\node at (4.5,5.5) {0};
\node at (5.5,5.5) {0};

\node at (1.5,4.5) {0};
\node at (2.5,4.5) {0};
\node at (4.5,4.5) {0};
\node at (5.5,4.5) {0};

\node at (0.5,3.5) {0};
\node at (1.5,3.5) {0};
\node at (3.5,3.5) {0};
\node at (4.5,3.5) {0};

\node at (0.5,2.5) {0};
\node at (2.5,2.5) {0};
\node at (4.5,2.5) {0};
\node at (5.5,2.5) {0};

\node at (0.5,1.5) {0};
\node at (1.5,1.5) {0};
\node at (2.5,1.5) {0};
\node at (3.5,1.5) {0};
\node at (5.5,1.5) {0};

\node at (0.5,0.5) {0};
\node at (1.5,0.5) {0};
\node at (3.5,0.5) {0};
\node at (4.5,0.5) {0};

% 用红色画出“skyline”的轮廓
\draw[red, rounded corners] (0,5.5) -- (0.5,6) -- (6,0.5) -- (5.5,0) -- cycle;
\draw[red, rounded corners] (1.1,5.4) -- (1.1,5.9) -- (1.9,5.9) -- (1.9,4.6);
\draw[red, rounded corners] (3.1,3.4) -- (3.1,4.9) -- (3.9,4.9) -- (3.9,2.6);
\draw[red, rounded corners] (5.1,1.4) -- (5.1,3.9) -- (5.9,3.9) -- (5.9,0.6);

\end{tikzpicture}
\end{document}

```

**Méthode de stockage :**

1. **Tableau de taille $n$ :**
   - **diag** : coefficients de la diagonale de la matrice.

2. **Tableau supplémentaire :**
   - **sup** : tableau contenant les éléments au-dessus de la diagonale.

**Indices et stockage des sous-colonnes :**

- **indptr** : indices des débuts de sous-colonnes dans **sup**
  $$
  [0, 1, 1, 3, 3, 6]
  $$

- La sous-colonne $i$ est stockée dans **sup** entre les indices :
  $$
  \text{sup}[\text{indptr}[i] : \text{indptr}[i+1]]
  $$

**Illustration :**

```tikz
\begin{document}
\begin{tikzpicture}[scale=0.8]

% Contour de la matrice
\draw[thick] (0,0) rectangle (6,6);

% Diagonale
\draw[thick] (0,6) -- (6,0);

% Barres verticales pour les sous-colonnes
\foreach \x/\y in {0.5/5.5, 1/4.5, 1.5/3.5, 2/2.5, 2.5/1.5, 3/0.5} {
    \draw[thick] (\x,6) -- (\x,\y);
    \draw[thick] (\x-0.5,\y) -- (\x+0.5,\y);
}

% Barres verticales pour les colonnes de l'autre côté de la diagonale
\foreach \x/\y in {4/5.5, 4.5/4.5, 5/3.5, 5.5/2.5} {
    \draw[thick] (\x,0) -- (\x,\y);
    \draw[thick] (\x-0.5,\y) -- (\x+0.5,\y);
}

\end{tikzpicture}
\end{document}
```

# 5. Renumérotation des Matrices

## Objectif de la Renumérotation

**Changer la numérotation des lignes et des colonnes** permet d'améliorer le stockage bande ou ligne de ciel. L'objectif est de **concentrer les coefficients non nuls autour de la diagonale**, ce qui facilite le stockage et les calculs pour les matrices creuses.

## Permutation des Lignes et Colonnes

### Exemple Simple

Considérons une matrice $A$ :

$$
A = \begin{bmatrix} 
a & b \\ 
c & d \\ 
\end{bmatrix}
$$

En permutant les lignes et les colonnes, on obtient la matrice $B$ :

$$
B = \begin{bmatrix} 
d & c \\ 
b & a \\ 
\end{bmatrix}
$$

### Définition de la Permutation $\pi$

Plus généralement :

- Soit $\pi : [[0,n]] \to [[0,n]]$ une permutation des indices de lignes et de colonnes.

- On définit la matrice permutée $B$ par :

$$
  B_{ij} = A_{\pi(i), \pi(j)} = \left( P_{\pi} A P_{\pi}^T \right)_{ij}
$$

  où $P_{\pi}$ est la **matrice de permutation** associée à $\pi$.

### Matrice de Permutation $P_{\pi}$

La matrice de permutation $P_{\pi}$ est définie par :

$$
\left( P_{\pi} \right)_{i,j} = \delta_{\pi(i), j}
$$

**Exemple :**

Pour la permutation $\pi$ définie par :

$$
\pi = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix},
$$

la matrice de permutation correspondante est :

$$
P_{\pi} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}.
$$

**Objectif :** Concentrer les coefficients non nuls autour de la diagonale.

## Graphe Associé à une Matrice

Pour construire une permutation $\pi$ qui améliore la localisation des coefficients non nuls, on considère le **graphe associé à la matrice $A$** :

- **Nœuds :** $V = \{0, 1, \ldots, n-1\}$
- **Arêtes orientées :** $E = \{ (i, j) \ | \ A_{ij} \neq 0 \}$

Si $A$ est symétrique, les arêtes sont non orientées.

**Exemple :**

Considérons la matrice suivante :

```tikz
\begin{document}
\begin{tikzpicture}
% Dessiner la grille de la matrice
\draw[thick] (0, 0) rectangle (4, 4);

% Indices des lignes et colonnes
\node at (-0.5, 3.5) {0};
\node at (-0.5, 2.5) {1};
\node at (-0.5, 1.5) {2};
\node at (-0.5, 0.5) {3};

\node at (0.5, 4.5) {0};
\node at (1.5, 4.5) {1};
\node at (2.5, 4.5) {2};
\node at (3.5, 4.5) {3};

% Remplir la matrice avec les valeurs
\node at (0.5, 3.5) {2};
\node at (1.5, 3.5) {0};
\node at (2.5, 3.5) {0};
\node at (3.5, 3.5) {-1};

\node at (0.5, 2.5) {0};
\node at (1.5, 2.5) {2};
\node at (2.5, 2.5) {-1};
\node at (3.5, 2.5) {0};

\node at (0.5, 1.5) {0};
\node at (1.5, 1.5) {-1};
\node at (2.5, 1.5) {2};
\node at (3.5, 1.5) {1};

\node at (0.5, 0.5) {-1};
\node at (1.5, 0.5) {0};
\node at (2.5, 0.5) {-1};
\node at (3.5, 0.5) {2};

\end{tikzpicture}
\end{document}
```

Le graphe associé est :

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[row sep=large, column sep=large, cells={nodes={circle, draw}}, scale=1.5]
0 \arrow[loop, distance=2em, in=305, out=235] \arrow[rrr, bend left, shift left=1, leftrightarrow] & 1 \arrow[loop, distance=2em, in=305, out=235] \arrow[r, bend left, leftrightarrow] & 2 \arrow[loop, distance=2em, in=305, out=235] \arrow[r, bend left, leftrightarrow] & 3 \arrow[loop, distance=2em, in=305, out=235] \\
& S_2 \arrow[u, no head]  & S_1 \arrow[u, no head] \arrow[llu, no head] & S_0 \arrow[u, no head]                       
\end{tikzcd}
\end{document}
```

## Renumérotation par Niveaux de Distance

### Parcours en Largeur du Graphe

1. **Départ :**

   - On choisit un nœud initial $k$.
   - L'ensemble $S_0 = \{ k \}$ représente le niveau 0.

2. **Détermination des Niveaux :**

   - **Niveau 1 ($S_1$) :** Nœuds à une distance 1 de $k$ (voisins de $k$) non inclus dans les niveaux précédents.
   - **Niveau 2 ($S_2$) :** Nœuds à une distance 2 de $k$, c'est-à-dire les voisins des nœuds de $S_1$ non inclus dans les niveaux précédents.
   - On continue ainsi jusqu'à couvrir tous les nœuds du graphe.

3. **Construction de la Permutation $\pi$:**

   - La permutation $\pi$ est obtenue en listant les nœuds selon l'ordre des niveaux $S_0, S_1, S_2, \ldots$.

### Illustration du Parcours

```latex
\begin{tikzpicture}
% Nœud initial
\node[circle, draw] (k) at (0,0) {$k$};
\node at (0,-0.5) {$S_0$};

% Niveau 1
\node[circle, draw] (S1a) at (2,1) {};
\node[circle, draw] (S1b) at (2,-1) {};
\node at (2,1.5) {$S_1$};

% Niveau 2
\node[circle, draw] (S2a) at (4,1.5) {};
\node[circle, draw] (S2b) at (4,0) {};
\node[circle, draw] (S2c) at (4,-1.5) {};
\node at (4,2) {$S_2$};

% Arêtes
\draw (k) -- (S1a);
\draw (k) -- (S1b);
\draw (S1a) -- (S2a);
\draw (S1a) -- (S2b);
\draw (S1b) -- (S2b);
\draw (S1b) -- (S2c);

\end{tikzpicture}
```

### Exemple d'Application

**Exemple :**

- Choisissons $S_0 = \{3\}$.
- Déterminons les niveaux :

  - $S_1 = \{0, 2\}$ (voisins de 3)
  - $S_2 = \{1\}$ (voisins de 0 et 2 non inclus dans $S_0 \cup S_1$)

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[row sep=large, column sep=large, cells={nodes={circle, draw}}, scale=1.5]
0 \arrow[loop, distance=2em, in=305, out=235] \arrow[rrr, bend left, shift left=1, leftrightarrow] & 1 \arrow[loop, distance=2em, in=305, out=235] \arrow[r, bend left, leftrightarrow] & 2 \arrow[loop, distance=2em, in=305, out=235] \arrow[r, bend left, leftrightarrow] & 3 \arrow[loop, distance=2em, in=305, out=235] \\
S_0 \arrow[u, no head] & S_3 \arrow[u, no head] & S_2 \arrow[u, no head] & S_1 \arrow[u, no head]                  
\end{tikzcd}
\end{document}
```

**Permutation résultante :**

$$
\pi = \begin{pmatrix} 0 & 1 & 2 & 3 \\ 3 & 0 & 2 & 1 \end{pmatrix}
$$

**Matrice après renumérotation :**

```tikz
\begin{document}
\begin{tikzpicture}
    % Dimensions de la grille
	\draw[thick] (-0.5, -0.5) rectangle (3.5, 3.5);

    % Les éléments de la matrice
    \node at (0,3) {2};
    \node at (1,3) {-1};
    \node at (2,3) {-1};
    \node at (3,3) {0};

    \node at (0,2) {-1};
    \node at (1,2) {2};
    \node at (2,2) {0};
    \node at (3,2) {0};

    \node at (0,1) {-1};
    \node at (1,1) {0};
    \node at (2,1) {2};
    \node at (3,1) {-1};

    \node at (0,0) {0};
    \node at (1,0) {0};
    \node at (2,0) {-1};
    \node at (3,0) {2};

    % Les indices de ligne et de colonne
    \node at (-1,3) {3};
    \node at (-1,2) {0};
    \node at (-1,1) {2};
    \node at (-1,0) {1};

    \node at (0,4) {3};
    \node at (1,4) {0};
    \node at (2,4) {2};
    \node at (3,4) {1};

\end{tikzpicture}
\end{document}
```

La matrice renumérotée est **tridiagonale**, ce qui facilite son stockage et son traitement.

## Algorithme de Cuthill-McKee (CMK)

### Principe de l'Algorithme

L'**algorithme de Cuthill-McKee** est une méthode pour renuméroter les nœuds d'un graphe afin de réduire la bande de la matrice associée.

- **Dans chaque niveau $S_j$, les nœuds sont ordonnés par degré croissant.**
  - **Degré d'un nœud :** Nombre d'arêtes reliées à ce nœud.

### Propriété de la Renumérotation

**Proposition :**

L'ensemble des nœuds $V$ peut être décomposé en une union disjointe de niveaux :

$$
V = \bigcup_{k=0}^{\ell} S_k
$$

Chaque ensemble $S_k$ sépare deux sous-graphes disjoints :

- $\bigcup_{j=0}^{k-1} S_j$
- $\bigcup_{j=k+1}^{\ell} S_j$

Il n'y a pas d'arêtes reliant ces deux sous-graphes à travers $S_k$.

**Remarque :**

La matrice renumérotée présente une structure **tridiagonale par blocs**. Cela signifie que les blocs diagonaux principaux sont non nuls, tandis que les autres blocs sont nuls ou négligeables.

### Illustration de la Structure de la Matrice

```tikz
\begin{document}

\begin{tikzpicture}

% Grille principale
\draw[step=1cm] (0,0) grid (5,5);

\node[red] at (-0.5,0.5) {$S_1$};
% Lignes des ensembles S1, S2, S3, S4
\node[red] at (-0.5,0.5) {$S_1$};
\node[red] at (-0.5,1.5) {$S_2$};
\node[red] at (-0.5,2.5) {$S_3$};
\node[red] at (-0.5,3.5) {$S_4$};
\node at (-0.5,4.5) {$k$};

% Colonnes des ensembles S1, S2, S3, S4
\node at (0.5,5.5) {$k$};
\node[red] at (1.5,5.5) {$S_1$};
\node[red] at (2.5,5.5) {$S_2$};
\node[red] at (3.5,5.5) {$S_3$};
\node[red] at (4.5,5.5) {$S_4$};

% Remplissage des blocs
\fill[blue!30] (0,4) rectangle (1,5);
\fill[blue!30] (0,3) rectangle (1,4);
\fill[blue!30] (1,2) rectangle (2,3);
\fill[blue!30] (1,3) rectangle (2,4);
\fill[blue!30] (1,4) rectangle (2,5);
\fill[blue!30] (2,1) rectangle (3,2);
\fill[blue!30] (2,2) rectangle (3,3);
\fill[blue!30] (2,3) rectangle (3,4);
\fill[blue!30] (3,0) rectangle (4,1);
\fill[blue!30] (3,1) rectangle (4,2);
\fill[blue!30] (3,2) rectangle (4,3);
\fill[blue!30] (4,0) rectangle (5,1);
\fill[blue!30] (4,1) rectangle (5,2);

\end{tikzpicture}

\end{document}

```
### Choix du Nœud Initial $k$

Pour optimiser la renumérotation :

- **Choisir $k$ de manière à ce que les ensembles de niveaux $S_k$ soient les plus petits possibles.**
- **Stratégie :** Maximiser la distance entre les nœuds du graphe pour augmenter le nombre de niveaux.

**Exemple :**

- Niveaux déterminés :

  - $S_0 = \{0\}$
  - $S_1 = \{3\}$
  - $S_2 = \{2\}$
  - $S_3 = \{1\}$

- Permutation correspondante :

$$
  \pi = \begin{pmatrix} 0 & 1 & 2 & 3 \\ 0 & 3 & 2 & 1 \end{pmatrix}
$$

- Matrice renumérotée :

```tikz
\begin{document}
\begin{tikzpicture}
    % Dimensions de la grille
	\draw[thick] (-0.5, -0.5) rectangle (3.5, 3.5);

    % Les éléments de la matrice
    \node at (0,3) {2};
    \node at (1,3) {-1};
    \node at (2,3) {0};
    \node at (3,3) {0};

    \node at (0,2) {-1};
    \node at (1,2) {2};
    \node at (2,2) {-1};
    \node at (3,2) {0};

    \node at (0,1) {0};
    \node at (1,1) {-1};
    \node at (2,1) {2};
    \node at (3,1) {-1};

    \node at (0,0) {0};
    \node at (1,0) {0};
    \node at (2,0) {-1};
    \node at (3,0) {2};

    % Les indices de ligne et de colonne
    \node at (-1,3) {0};
    \node at (-1,2) {3};
    \node at (-1,1) {2};
    \node at (-1,0) {1};

    \node at (0,4) {0};
    \node at (1,4) {3};
    \node at (2,4) {2};
    \node at (3,4) {1};

\end{tikzpicture}
\end{document}
```

  La matrice est **tridiagonale** avec des ensembles de niveaux de taille 1.
