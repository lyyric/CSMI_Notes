
> [!def]
> Soit $A \in M_{n,n}(\mathbb{C})$ et $p$ est le nombre de coefficients non nuls.
> - $A$ est creuse si elle a peu de coefficients non nuls. $p \ll n^2$
> - $A$ est pleine si elle beaucoup de coefficients non nuls. $p \approx n^2$

> [!rmk]
> Exemple de matrice creuse :
> - Résolution de l'équation de Poisson par différences finies.
>
> **Équation de Poisson**
> $$
> \begin{cases}
> -u''(x) = f(x) \\
> u(0) = 0, \; u(1) = 0
> \end{cases}
> $$
> Discrétisation de $[0, 1]$
> Pour tout $i \in [[0, n+1]]$ :
> $$
> x_i = \frac{i}{n+1}
> $$
> avec $\Delta x = \frac{1}{n+1}$.
>
> $$
> u''(x_i) = \frac{u(x_{i+1}) - 2u(x_i) + u(x_{i-1})}{(\Delta x)^2}
> $$
>
> Solution approchée $u \in \mathbb{R}^m$ vérifiant $A u = b$, avec :
> $$
> A = \frac{1}{\Delta x^2} \begin{pmatrix} 
> 2 & -1 & 0 & \cdots & 0 \\
> -1 & 2 & -1 & \cdots & 0 \\
> 0 & -1 & 2 & \cdots & 0 \\
> \vdots & \vdots & \vdots & \ddots & -1 \\
> 0 & 0 & 0 & -1 & 2 
> \end{pmatrix}, \quad b = \begin{pmatrix} 
> f(x_1) \\
> f(x_2) \\
> \vdots \\
> f(x_{m-1})
> \end{pmatrix}
> $$
> où $A \in M_{m,m}(\mathbb{R})$ est une matrice creuse, et $b \in \mathbb{R}^m$.

1) Norme Matricielle

> [!def] 
> Soit $A \in M_{n,n}(\mathbb{C})$, on appelle rayon spectral de $A$ (la quantité) :
> $$
> \rho(A) = \max \{ |\lambda|, \; \lambda \in \text{Sp}(A) \}
> $$

> [!rmk]
> $A^*=\bar A ^T$ 

> [!prp]
> Soit $A \in M_{n,n}(\mathbb{C})$ (resp. $M_{n,n}(\mathbb{R})$) :
> 
> 1. **Si $A$ est normale**, c'est-à-dire $A^*A = AA^*$ (resp. $\bar{A}^T A = A \bar{A}^T$), alors $A$ est diagonalisable, donc il existe une base orthonormée $\exists P \in U_n(\mathbb{C})$ , telle que :
> $$
>    A = PDP^* \quad \text{avec } D = \text{diag}(\lambda_1, \ldots, \lambda_n)
> $$
> (resp. $\exists P \in O_n(\mathbb{R})$, telle que : $A = PDP^T$)
> 
> 2. **Si $A = A^*$**, $A$ est hermitienne (resp. $A = A^T$ symétrique) et $A$ est diagonalisable dans une base orthonormée et a des valeurs propres réelles. Si $A$ est définie positive*, alors les valeurs propres sont strictement positives :
> $$
>    0 < \lambda_1 \leq \lambda_2 \leq \ldots \leq \lambda_n.
>   $$
> 
>    Les valeurs propres $0 \leq \sigma_1 \leq \ldots \leq \sigma_m$ sont les valeurs singulières de $A$.

Définition
1. **Les normes naturelles sur** $M_n(\mathbb{C})$ :
   - Ce sont des normes compatibles avec la multiplication :
$$
     \forall A, B \in M_n(\mathbb{C}), \; \|AB\| \leq \|A\| \|B\|
$$

2. **Soit** $\| \cdot \|_V$ **une norme sur** $\mathbb{C}^n$. On lui associe une norme dite subordonnée $\| \cdot \|_M$ sur $M_n(\mathbb{C})$
$$
   \|A\|_M = \sup_{x \neq 0} \frac{\|Ax\|_V}{\|x\|_V}
$$
   Elle vérifie :
$$
   \forall A \in M_n(\mathbb{C}), \; \forall x \in \mathbb{C}^n, \; \|Ax\|_V \leq \|A\|_M \|x\|_V
$$
   et 
$$
   \|I\|_M = 1
$$

Proposition
Pour tout $A \in M_{n,n}(\mathbb{C})$ :
- $$\|A\|_\infty = \max_{1 \leq i \leq n} \sum_{j=1}^n |a_{ij}|$$
- $$\|A\|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^n |a_{ij}|$$
- $$\|A\|_2 = \sqrt{\rho(A^* A)} = \sigma_{\max}$$
mq

Pour tout $x \in \mathbb{C}^m$ :
1. Norme infinie :
$$
   \|x\|_\infty = \max |x_i|
$$
2. Norme 1 :
$$
   \|x\|_1 = \sum |x_i| 
$$
3. Norme 2 (norme Euclidienne) :
$$
   \|x\|_2 = \sqrt{\sum |x_i|^2}
$$
 
 Prop
1. Si $A$ est normale, alors :
   $$
   \|A\|_2 = \rho(A)
   $$

2. Si $A$ est hermitienne positive :
   $$
   \|A\|_2 = \rho(A) = \lambda_{\max}
   $$

3. Si $A$ est unitaire :
   $$
   \|A\|_2 = \rho(A) = 1
   $$

Prop
Pour une matrice normale $A$ :
$$
\rho(A) = \inf_{\|\|} \|A\|
$$
où $\|A\|$ est une norme matricielle.

Preuve
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
           &= (P^* x, \Sigma P^* x) \quad \text{avec } A^*A = P \Sigma P^* \quad \text{où } \Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2)\\
           &= (y, \Sigma y) \quad \text{avec } y = P^* x \\
           &= \left( \begin{pmatrix} y_1 \\ \vdots \\ y_n \end{pmatrix}, \begin{pmatrix} \sigma_1^2 y_1 \\ \vdots \\ \sigma_n^2 y_n \end{pmatrix} \right) \\
           &= \sum_{i=1}^{n} \overline{y_i} \sigma_i^2 y_i \\
           &= \sum_{i=1}^{n} \sigma_i^2 |y_i|^2 \\
           &\leq \sigma_{\max}^2 \sum_{i=1}^{n} |y_i|^2 \\
           &= \sigma_{\max}^2 \|y\|_2^2 \\
           &= \sigma_{\max}^2 \|x\|_2^2.
\end{aligned}
$$
$$
\iff x \in \mathbb{C}^m,\,\|Ax\|_2 \leq \sigma_{\max} \|x\|_2,
$$
$$
\iff \|A\|_2 \leq \sigma_{\max} ,
$$

On a égalité en prenant $x \in \mathbb{C}^m$ comme vecteur propre de $A^* A$ associé à $\sigma_{\max}^2$ 

 Prop
- Soit $A$ une matrice hermitienne positive. Pour tout $x \in \mathbb{C}^n$ :
$$
  \|A\|_2 = \rho(A)
$$

Preuve
1.  $x \in \mathbb{C}^n$, alors :
   $$
   \|Ax\|_2^2 = \|P D P^* x\|_2^2 = \|D P^* x\|_2^2,
   $$
   où $A = PDP^*$ avec $D = \text{diag}(\lambda_1, \ldots, \lambda_n)$, $P \in O_n(\mathbb{R})$ ou $P \in U_n(\mathbb{C})$.

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
\leq \lambda_{\max} \sum_{i=1}^{m} |y_i|^2=\lambda_{\max} \|y\|_2^2=\lambda_{\max} \|x\|_2^2
$$

où $y = P^* x$ avec $P$ unitaire.

 2. Conditionnement

**Définition :** Soit $A \in M_{n,n}(\mathbb{C})$ inversible. On appelle **conditionnement** de $A$ le réel :

$$
\text{cond}(A) = \|A\| \|A^{-1}\| \geq 1.
$$

Pour toute norme subordonnée, on a :

$$
1 = \|I\| = \|A A^{-1}\| \leq \|A\| \|A^{-1}\|.
$$

 3. Sensibilité des Solutions

**Définition :** Soit $A \in M_{n,n}(\mathbb{C})$ inversible, $b \in \mathbb{C}^n$ et $x \in \mathbb{C}^n$ la solution de $Ax = b$.
1. Si $Ax = b$ et $A(x + \delta x) = (b + \delta b)$, alors :
$$
   \frac{\|\delta x\|}{\|x\|} \leq \text{cond}(A) \frac{\|\delta b\|}{\|b\|}.
$$

2. Si $Ax = b$ et $(A + \delta A)(x + \delta x) = b$, alors :
$$
   \frac{\|\delta x\|}{\|x + \delta x\|} \leq \text{cond}(A) \frac{\|\delta A\|}{\|A\|}.
$$

Remarque 1
Si le conditionnement de $A$, noté $\text{cond}(A)$, est élevé, cela implique un mauvais contrôle de la propagation des erreurs.

Remarque 2
Les erreurs relatives sur les réels sont codées avec des nombres à virgule flottante en :

- **Simple précision** (32 bits)
- **Double précision** (64 bits)

Exemple

Le nombre `103.000.004` est codé en base décimale comme :

- `0.10300 \times 10^9` si on garde seulement 5 chiffres significatifs.

Le nombre le plus proche de `0.10300 \times 10^9` est `0.10301 \times 10^9`, c'est-à-dire `103.001.000`.

Étant donné $x \in \mathbb{R}^n$, le nombre le plus proche $\tilde{x}$ vérifie :
$$
|x - \tilde{x}| \leq \varepsilon |x|,
$$
où $\varepsilon = \text{précision machine}$ :
- $\varepsilon \approx 10^{-7}$ (simple précision)
- $\varepsilon \approx 10^{-16}$ (double précision)

 Preuve

1. **Cas 1 :**

   Si $Ax = b$ et $A(x + \delta x) = b + \delta b$, alors :
$$
   b = Ax \implies \|b\| \leq \|A\| \|x\| \implies \frac{1}{\|x\|}\leq \frac{\|A\|}{\|b\|}
$$
   et
   $$
   A(x + \delta x) = b + \delta b \implies \delta x = A^{-1} \delta b \implies \|\delta x\| \leq \|A^{-1}\| \|\delta b\|.
   $$
  En multipliant les deux inégalités :
$$
   \frac{\|\delta x\|}{\|x\|} \leq \|A\|\|A^{-1}\| \frac{\|\delta b\|}{\|b\|}.
$$

2. **Cas 2 :**

   Pour les perturbations dans $A$ :
$$
   A(x + \delta x) + \delta A(x + \delta x) = b.
$$
   
   En développant :
$$
   A\delta x= -\delta A(x + \delta x) \implies \delta x = -A^{-1} \delta A (x + \delta x).
$$
   Donc :
$$
   \|\delta x\| \leq \|A^{-1}\| \|\delta A\| \|x + \delta x\| \leq \|A\|\|A^{-1}\| \frac{\|\delta A\|}{\|A\|}\|x + \delta x\|.
$$

   Cela conduit à :
   $$
   \frac{\|\delta x\|}{\|x + \delta x\|} \leq \text{cond}(A) \frac{\|\delta A\|}{\|A\|}.
   $$
 Proposition

1. Pour $A \in M_{n,n}(\mathbb{C})$ inversible :
   $$
   \text{cond}(A) = \|A\|_2 \|A^{-1}\|_2 = \frac{\sigma_{\max}}{\sigma_{\min}}.
   $$

2. Pour $A \in M_{n,n}(\mathbb{C})$ hermitienne définie positive :
$$
   \text{cond}(A) =\|A\|_2\|A^{-1}\|_2 = \frac{\lambda_{\max}}{\lambda_{\min}}.
   $$

3. Pour $A \in M_{n,n}(\mathbb{C})$ unitaire :
$$
   \text{cond}(A) = 1.
$$

Remarque

Pour une matrice hermitienne définie positive $A$, on a :
$$
A = PDP^*, \quad A^{-1} = (P^{-1})^* D^{-1} P^{-1},
$$
avec $P$ unitaire et $D$ diagonal.

Si $A$ est une matrice hermitienne définie positive, alors :
$$
\|A^{-1}\|_2 = \rho(A^{-1}) = \frac{1}{\lambda_{\max}}.
$$

Proposition

Pour toutes matrices $B, C \in M_{n,n}(\mathbb{C})$ :
$$
\text{cond}(BC) \leq \text{cond}(B) \cdot \text{cond}(C).
$$
$$
\|BC\|\|BC^{-1}\| \leq \|B\|\|B^{-1}\| \cdot \|C\|\|C^{-1}\|.
$$

Remarque : Méthode Directe de Résolution de Systèmes Linéaires

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

Estimation de l'Erreur

L'erreur relative pour $x$ est donnée par :
$$
\frac{\|\delta x\|}{\|x\|} \leq \text{cond}(C) \frac{\|\delta y\|}{\|y\|}\leq \text{cond}(C)\text{cond}(B) \frac{\|\delta b\|}{\|b\|}.
$$

$$
\text{cond}(C) \cdot \text{cond}(B) \geq \text{cond}(BC)=\text{cond}(A).
$$

"Perte" dans le contrôle de la propagation de l'erreur.

 3) Matrices Creuses

Remarque
La multiplication matrice-vecteur nécessite en général $O(n^2)$ opérations pour calculer $Ax$.

Si la matrice est creuse avec $p$ coefficients non nuls, le calcul de $Ax$ nécessite $O(p)$ opérations.

Il est donc avantageux de stocker les coefficients non nuls de la matrice.

 a) Stockage

**Stockage par coordonnées (Coordinate)**

Une matrice est représentée par 3 tableaux de taille $p$ :

- **data** : contient les coefficients non nuls
- **row** : indices de lignes
- **col** : indices de colonnes

Exemple de représentation :

- Matrice :
$$
  \begin{matrix}
  a & b & 0 & 0 &0 \\
  0 & c & 0 & 0 &0 \\
  0 & 0 & 0 & 0 &0 \\
  0 & 0 & 0 & d &e \\
  0 & 0 & f & 0 &0 \\
  \end{matrix}
$$

- Représentation par stockage :
  - **data** : $[a, b, c, d, e, f]$
  - **row** : $[0, 1, 1, 2, 2, 3]$
  - **col** : $[0, 0, 1, 1, 2, 2]$

Stockage Compact par Ligne (CSR - Compressed Sparse Row)

1. **Définition :**  
   - Le stockage compact par ligne est une méthode pour stocker des matrices creuses (sparse matrices).  
   - CSR (Compressed Sparse Row) est une méthode de stockage par ligne compressée.

2. **Paramètres :**  
   - $p =$ nombre d'éléments non nuls dans la matrice $A$ de taille $n \times n$.

3. **Tableaux utilisés :**  
   - 2 tableaux de taille $p$ :
     - **data** : valeurs non nulles.
     - **colon** : indices des colonnes.
   - 1 tableau de taille $n$ (ou $\leq p$) :
     - **indptr** : indices des débuts de lignes dans le tableau **data**.

4. **Exemple :**  
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

   - **data** = $[a, b, c, d, e, f]$  
   - **colon** = $[0, 1, 1, 2, 3, 1]$  
   - **indptr** = $[0, 2, 3, 3, 3, 6]$  

1. **Les coefficients de la i-ème ligne sont stockés dans** :  
$$ \text{data}[ \text{indptr}[i] : \text{indptr}[i+1] ] $$

2. **Répartition par lignes :**  
   - 1ère ligne : **data** $[0:2]$  
   - 2ème ligne : **data** $[2:3]$  
   - 3ème ligne : **data** $[3:3] = \varnothing$  
   - 4ème ligne : **data** $[3:5]$  
   - 5ème ligne : **data** $[5:6]$  

3. **Stockage Bande (Band Storage) :**  
   - Les coefficients nuls sont sous les diagonales de -r à +s.  
   - Le stockage des diagonales se fait dans un tableau de taille $(r + s + 1) \times m$.
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

Stockage Ligne de Ciel (Skyline Storage)

Le **stockage ligne de ciel (Skyline)** est une méthode de stockage pour les matrices creuses (sparse matrices) où les coefficients non nuls se trouvent principalement autour de la diagonale.

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

Méthode de Stockage

1. **Tableau de taille n :**  
   - **diag :** coefficients de la diagonale de la matrice.

2. **Tableau supplémentaire :**  
   - **sup :** tableau contenant les sous-colonnes supérieures non nulles.

Exemple de Stockage

Soit la matrice suivante :
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

- **diag** contient les éléments sur la diagonale principale.
- **sup** contient les éléments des colonnes au-dessus de la diagonale qui sont non nuls :
$$
  [*, 0, *, 0, 0, *, *]
$$
Voici la transcription du texte de l'image fournie :

Indices et Stockage des Sous-Colonnes

- **indptr** : indices des débuts de sous-colonnes dans **sup**  
$$
[0, 1, 1, 3, 3, 6]
$$

- La sous-colonne $i$ est stockée dans **sup** entre les indices :  
$$
\text{sup}[\text{indptr}[i] : \text{indptr}[i+1]]
$$

Illustration

L'illustration en bas de l'image montre une représentation graphique des sous-colonnes dans une structure de matrice où les valeurs non nulles sont stockées sous forme compressée. La ligne diagonale divisant la matrice illustre les zones où les valeurs sont effectivement présentes (représentées par des barres verticales) contre les zones vides ou nulles.

```tikz
\begin{document}
\begin{tikzpicture}[scale=0.8]

% Dessin du contour de la matrice
\draw[thick] (0,0) rectangle (6,6);

% Tracer la diagonale
\draw[thick] (0,6) -- (6,0);

% Dessin des barres pour représenter les sous-colonnes compressées
\foreach \x/\y in {0.5/5.5, 1/4.5, 1.5/3.5, 2/2.5, 2.5/1.5, 3/0.5} {
    \draw[thick] (\x,6) -- (\x,\y);
    \draw[thick] (\x-0.5,\y) -- (\x+0.5,\y);
}

% Dessin des barres pour représenter les colonnes compressées de l'autre côté de la diagonale
\foreach \x/\y in {4/5.5, 4.5/4.5, 5/3.5, 5.5/2.5} {
    \draw[thick] (\x,0) -- (\x,\y);
    \draw[thick] (\x-0.5,\y) -- (\x+0.5,\y);
}

\end{tikzpicture}
\end{document}

```

 2) Renumérotation

**Changer la numérotation des lignes et des colonnes** permet d'améliorer le stockage bande ou ligne de ciel.

Exemple :
$$
A = \begin{bmatrix} 
a & b \\ 
c & d \\ 
\end{bmatrix} 
\quad \xrightarrow{\text{Permutation}} \quad 
B = \begin{bmatrix} 
d & c \\ 
b & a \\ 
\end{bmatrix}
$$

**Plus généralement :**

- Soit $\pi : [[0,n]] \to [[0,n]]$ une permutation.

- Définir $B_{ij} = A_{\pi(i), \pi(j)} = (P_{\pi} A P_{\pi}^T)_{ij}$

où $P_{\pi}$ est la **matrice de permutation** associée à $\pi$. 
$$
(P_{\pi})_{i,j}=\delta _{\pi(i),j}
$$
$$
\pi = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix},
\quad \quad \quad
P_{\pi} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}.
$$

**Objectif :** Concentrer les coefficients non nuls autour de la diagonale.

Construction de $\pi$ ?

Pour construire $\pi$, on considère le **graphe associé à la matrice $A$** :

- **Nœuds :** $V = \{0, 1, \ldots, m-1\}$

- **Arêtes orientées :** $E = \{ (i,j) \ | \ A_{ij} \neq 0 \}$

Les arêtes non orientées si $A$ est symétrique.
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

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[row sep=large, column sep=large, cells={nodes={circle, draw}}, scale=1.5]
0 \arrow[loop, distance=2em, in=305, out=235] \arrow[rrr, bend left, shift left=1, leftrightarrow] & 1 \arrow[loop, distance=2em, in=305, out=235] \arrow[r, bend left, leftrightarrow] & 2 \arrow[loop, distance=2em, in=305, out=235] \arrow[r, bend left, leftrightarrow] & 3 \arrow[loop, distance=2em, in=305, out=235] \\
& S_2 \arrow[u, no head]  & S_1 \arrow[u, no head] \arrow[llu, no head] & S_0 \arrow[u, no head]                       
\end{tikzcd}
\end{document}
```
Renumérotation par Niveaux de Distance

1. **Parcours du graphe en largeur** :
   - À partir d'un nœud initial, $S_0 = \{ k \}$, on détermine l'ensemble des nœuds qui sont à une distance 1 dans le graphe : $S_1$ (voisins de $k$).

2. **Détermination des Niveaux** :
   - On détermine ensuite les voisins des nœuds de $S_1$ (non présents dans $S_0 \cup S_1$), qui sont à une distance 2 de $k$ : $S_2$, etc.

3. **Construction de la Permutation $\pi$** :
   - La permutation $\pi$ est obtenue en lisant les nœuds suivant l'ordre de parcours.
```tikz
\begin{document}
\begin{tikzpicture}
    % Dessiner les ensembles S1, S2, Sl
    \node[red] at (1.5, 1) {$S_1$};
    \node[red] at (2.5, 1) {$S_2$};
    \node[red] at (3.5, 1) {$S_l$};
    \node at (0.5, 1.5) {$k$};
    \draw[red] (1.2, 1.2) rectangle (1.8, 1.8);
    \draw[red] (2.2, 1.2) rectangle (2.8, 1.8);
    \draw[red] (3.2, 1.2) rectangle (3.8, 1.8);
    
    \node at (0.5, 2.5) {0};
    \node at (1.5, 2.5) {1};
    \node at (2.5, 2.5) {$\dots$};
    \node at (3.5, 2.5) {$(n-1)$};

    % Flèches et texte
    \node at (-0.5, 2) {$\pi_1 = \left( \right.$};
    \node at (4.5, 2) {$\left. \right)$};
    
\end{tikzpicture}
\end{document}
```
Exemple :
- $S_0 = \{3\}$
- $S_1 = \{0, 2\}$
- $S_2 = \{1\}$

Permutation résultante : $\pi = \left(\begin{matrix} 0 & 1 & 2 & 3 \\ 3 & 0 & 2 & 1 \end{matrix}\right)$

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

L'image contient deux sections principales :

1. **Proposition**:
   - Elle décrit une décomposition en ensembles de niveaux. La formule présente est :
$$
   V = \bigcup_{k=0}^{\ell} S_k
$$
   où chaque ensemble $S_k$ sépare deux unions disjointes de sous-ensembles $S_j$, avec $0 \leq j < k$ et $k+1 \leq j \leq \ell$.
   - $S_k$ sépare $\bigcup_{j=0}^{k-1} S_j$ et $\bigcup_{j=k+1}^{\ell} S_j$ en deux graphes indépendants, sans arêtes en commun.
   - Une note précise que $S_1$ sépare $S_0$ et $S_2$.

2. **Remarque (Rem)**:
   - La matrice renumérotée est qualifiée de "tridiagonale par blocs". Cela signifie que la matrice a une structure où les blocs diagonaux principaux sont non nuls, tandis que les autres blocs sont nuls ou non significatifs.
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

Comment choisir $k$ ? $k$ tel que les ensembles de niveaux soient les plus petits possibles, donc là où il y a le plus d'ensemble de niveaux.

- On cherche à maximiser la distance entre les nœuds du graphe pour déterminer une numérotation ou une organisation optimale.
- **Exemple donné :**
  - $S_0 = \{0\}$, $S_1 = \{3\}$, $S_2 = \{2\}$, $S_3 = \{1\}$ : 
```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[row sep=large, column sep=large, cells={nodes={circle, draw}}, scale=1.5]
0 \arrow[loop, distance=2em, in=305, out=235] \arrow[rrr, bend left, shift left=1, leftrightarrow] & 1 \arrow[loop, distance=2em, in=305, out=235] \arrow[r, bend left, leftrightarrow] & 2 \arrow[loop, distance=2em, in=305, out=235] \arrow[r, bend left, leftrightarrow] & 3 \arrow[loop, distance=2em, in=305, out=235] \\
S_0 \arrow[u, no head] & S_3 \arrow[u, no head] & S_2 \arrow[u, no head] & S_1 \arrow[u, no head]                  
\end{tikzcd}
\end{document}
```
$\pi = \left(\begin{matrix} 0 & 1 & 2 & 3 \\ 0 & 3 & 2 & 1 \end{matrix}\right)$
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
Tridiagonale, ensemble de niveaux de taille 1.

**L'algorithme est appelé algo de Cuthill-McKee (CMK)**

- (Dans chaque ensemble de niveaux $S_j$, les nœuds sont ordonnés par degré croissant.)
- **Degré d'un nœud** : Nombre d'arêtes reliées à ce nœud.
- Exemple illustratif : Un nœud de degré $\deg = 4$.
