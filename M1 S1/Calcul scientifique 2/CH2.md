Chapitre 2 : Résolution des systèmes linéaires creux par factorisation $LU$

**Factorisation $LU$ :**

Soit une matrice $A$ telle que $A = LU$, où :

- $L$ est une matrice triangulaire inférieure avec des 1 sur la diagonale (lower),
- $U$ est une matrice triangulaire supérieure (upper).

Résolution: $Ax = b \iff LUx = b$

$Ly = b$

$Ux = y$

1) déterminer la <font color="#2DC26B">facto</font> **LU**

2) résolution du système $Ly = b$

3) résolution du système $Ux = y$

Rem: résolution des systèmes triangulaires

![](assets/Pasted%20image%2020240917224200.png)

```tikz
\begin{document}
\begin{tikzpicture}
    % 绘制三角矩阵
    \draw[thick] (0,0) -- (2,0) -- (2,2) -- (0,2) -- cycle;
    \fill[gray!30] (0,0) -- (2,0) -- (0,2) -- cycle; % 阴影部分

    % 标记行和列
    \node at (-0.3, 1) {p};
    \node at (1, -0.3) {p};
    
    % 绘制矩阵元素
    \draw[thick] (1.5,0.5) rectangle (2,1);
    \fill[green!50] (1.5,0.5) rectangle (2,1); % 绿色部分
    
    % 绘制列向量x
    \draw[thick] (3,0) -- (3,2);
    \draw[thick] (3.5,0) -- (3.5,2);
    \draw[thick] (3,1) -- (3.5,1);
    \fill[green!50] (3,0) rectangle (3.5,0.5); % 绿色部分
    \node at (3.75,1) {$x$};
    \node at (3.75,0.25) {$x_p$};

    % 绘制等号
    \node at (4,1) {$=$};
    
    % 绘制列向量b
    \draw[thick] (4.5,0) -- (4.5,2);
    \draw[thick] (5,0) -- (5,2);
    \draw[thick] (4.5,1) -- (5,1);
    \fill[red!30] (4.5,0) rectangle (5,0.5); % 红色部分
    \node at (5.25,1) {$b$};
    \node at (5.25,0.25) {$b_p$};

    % 添加箭头
    \draw[->, thick] (5.5,0.25) -- (3.75,0.25);

\end{tikzpicture}
\end{document}
```

$$
x_{n-1} = \frac{b_{n-1}}{A_{n-1,n-1}}
$$

$$
\forall p \in \{m-2, \ldots, 1\}
$$

$$
A_{p,p} x_p + \sum_{j>p} A_{p,j} x_j = b_p
$$

$$
\Rightarrow x_p = \frac{b_p - \sum_{j>p} A_{p,j} x_j}{A_{p,p}}
$$

algorithme de remontée: $O(m^2)$ opérations (pas plus couteux que la multiplication matriciel)

I) Factorisation $LU$

1) Opérations sur les lignes

Déf. :
* ajout à la ligne $Li$ d'un multiple de la ligne $Lj$ :

$L_i \leftarrow L_i + \mu_{i,j}L_j$

$\Rightarrow$ revient à multiplier à gauche par 

![](assets/Pasted%20image%2020240917224413.png)

* ajout aux lignes $L_i$, $i > j$ de multiples de la ligne $L_j$: 

$L_i \leftarrow L_i + (\mu_{j})_iL_j$ 

$\forall i > j$

![](assets/Pasted%20image%2020240917224615.png)

On a $E_j(\mu_{j})^{-1} = E_j(-\mu_{j})$

(ici $\mu_j$ vecteur de taille $(n-j+1)$)

$$
\mu_j = \begin{pmatrix}
(\mu_{j})_{j+1} \\
\vdots \\
(\mu_{j})_{n-1}
\end{pmatrix}
$$
**Prop**:
ne modifie pas

![](assets/Pasted%20image%2020240917225101.png)

si $\ell < j$

**Preuve**:

$E_{\ell}[\mu_{\ell}][E_{j}[\mu_{ij}] A] = E_{\ell}[\mu_{\ell}]\tilde{A}=\tilde{\tilde{A}}$ 

$E_{j}(\mu_{i})$
$$
\hat{L}_i = 
\begin{cases} 
L_i + (\mu_{\ell j}) L_j & \forall i > j \\
L_i & \forall i \leq j 
\end{cases}
$$
$E_{\ell}(\mu_{\ell})$
$$
\tilde{\tilde{L_i}} =
\begin{cases}
\tilde{L_i} + (\mu_{\ell}) \tilde{L_{\ell}} = L_i + (\mu_{\ell})L_{\ell} & \forall i > j \\
\tilde{L_i} + (\mu_{\ell})_i \tilde{L_{\ell}} =L_i + (\mu_{\ell})_i  L_{\ell} & \forall i > j > \ell \\
\tilde{L_i}=L_i & \forall i \leq \ell
\end{cases}
$$
$$
= 
\begin{cases}
L_i + (\mu_{j})_i L_j + (\mu_{\ell})_i L_{\ell} & \forall i > j \\
L_i + (\mu_{\ell})_i L_{\ell} & \forall i > j > \ell \\
L_i & \forall i \leq \ell 
\end{cases}
$$

2) Algo du pivot de Gauss p - ième itération

![](assets/Pasted%20image%2020240917225600.png)

On suppose que $A^{(p)}_{p,p}$ est non nul (pivot)

On effectue 

$L_j \leftarrow L_j - (\mu_{p})_j L_p \ \ \forall j > p$

avec $(\mu_{p})_j = \dfrac{A_{j,p}^{(p)}}{A_{p,p}^{(p)}}$

$E_p(-\mu_p) A^{(p)} =$ ![](assets/Pasted%20image%2020240917231616.png) $= A^{(p+1)}$

Après $(n-1)$ itérations, $A^{(n-1)}$ est triangulaire supérieure.

3) Factorisation

$$
E_{n-2}(-\mu_{n-2}) \ldots E_1(-\mu_1) E_0(-\mu_0) A = A^{(n-1)} = U
$$
$$
\iff A = (E_{n-2}(-\mu_{n-2}) \dots E_0(-\mu_0))^{-1} U
$$
$$
= (E_0(-\mu_0)^{-1} \dots E_{n-2}(-\mu_{n-2})^{-1}) U
$$
$$
= (E_0(\mu_0) \ldots E_{n-2}(\mu_{n-2})) U
$$
$$
= LU
$$
**Révision**

$$(E_{n-2} \cdots E_0) A = U$$

$E_{j} = E_{j}(-\mu_{j}) =$  ![[assets/Pasted image 20240918081927.png]]

$$A = (E_{n-2} \cdots E_0)^{-1} U$$

$$= E_0^{-1} \cdots E_{m-2}^{-1} U$$

$$= LU$$

**Prop:** $L$ triang inf avec des 1 sur la diagonale et

$$L_{ij} = \left\{
\begin{array}{ll}
0 & \text{pour } i < j \quad (\text{triang sup}) \\
1 & \text{si } i = j \\
(\mu_{j})_{i-j} & \text{pour } i > j \quad (\text{triang inf})
\end{array}
\right.$$

![[assets/Pasted image 20240918082002.png]]

**Preuve:** 

$$L = E_0(-\mu_0)^{-1} \cdots E_{n-2}(-\mu_{n-2})^{-1}$$
$$= E_0 \cdots E_{n-2}(\mu_{n-2})$$

et on utilise la propriété

$$E_i(\mu_i^j) E_j(\mu_i^j) =$$
![[assets/Pasted image 20240918082247.png]]
si $j > i$

**Rappel:**

$$(\mu_j)_i = \frac{A_{ij}^{(j)}}{A_{jj}^{(j)}}$$

---

**Def:** Les mineurs principaux de $A$ sont les déterminants de sous matrices principales $A_{:p,:p}$, $\forall p \in [[1,m]]$.

$A=$ 
![[assets/Pasted image 20240918082559.png]]

**Rappel:** pour obtenir la facto $LU$, on a supposé que 

$$A_{p,p}^{(p)} \neq 0. \quad \text{(pivot de l'algorithme)}$$

---

**Prop:** Soit $A$ inversible.

Existence de la facto $LU$ $\iff$ tous les mineurs principaux sont non nuls

Le cas échéant, il y a unicité.

---

**Preuve:**

Supposons qu'on ait réalisé l'algo jusqu'à l'étape $p$.

On a

![[assets/Pasted image 20240918083532.png]]

Pour continuer l'algo, il faut que $A_{p,p}^{(p)}$ soit non nul.

Or, la sous-matrice $A_{:p+1, : p+1}^{(p)}$ a pour déterminant celui de $A_{:p+1, : p+1}.$

---

Car on a effectué sur la matrice $A$ des opérations sur les lignes qui ne modifient pas le déterminant.

Donc,

$$0 \neq \det A_{:p+1, : p+1}^{(p)} = \det A_{:p+1, :p+1}^{(p+1)} = * * \cdots ** = * * \cdots * A_{pp}^{(p)}.$$

On a bien $A_{pp}^{(p)} \neq 0.$ 

On peut donc continuer l'algorithme.

$$A = LU =$$
![[assets/Pasted image 20240918083551.png]]
![](assets/Pasted%20image%2020240918083900.png)
$A$ étant inversible, les éléments diagonaux de $L$ et de $U$ sont non nuls, donc les éléments diagonaux de $L_p$ et $U_p$ sont non nuls. Donc 

$$\det A_{:p,:p} = \det (L_p U_p) = \det L_p \det U_p \neq 0.$$

Vrai pour tout $p$.

**Unicité:**

$$A = LU = \tilde{L} \tilde{U}$$

alors comme $L, U$ et $\tilde{L}, \tilde{U}$ sont inversibles, on a 

$$\tilde{L}^{-1} L = \tilde{U} \tilde{U}^{-1} = \text{Id}$$

*(triang inf avec des 1 sur la diag (à montrer), triang sup (à montrer))*

---

4) Algorithme

**Factorisation sur place**

![](assets/Pasted%20image%2020240918084928.png)

Stockage de $L$ et $U$ sur le même espace mémoire que $A$.

---

$A \in M_n(\mathbb{R})$ inversible

Pour tout $p$ de $0$ à $n-2$

Pour $j$ de $p + 1$ à $n - 1$

$$
A_{j,p} \leftarrow \frac{A_{jp}}{A_{pp}} =(\mu_p)_j = L_{p,j}
$$

$$
A_{j, p+1:} \leftarrow A_{j, p+1:} - A_{j,p} A_{p, p+1:}
$$

---

**Coût:**

À l'étape 0:

$$
(n-1) \text{ divisions}
$$
$$
+ (n-1)^2 \text{ multiplications}
$$
$$
+ (n-1)^2 \text{ soustractions}
$$
![](assets/Pasted%20image%2020240918085001.png)

---

À l'étape 1:

$$
(n-2) \text{ divisions}
$$
$$
+ (n-2)^2 \text{ multiplications}
$$
$$
+ (n-2)^2 \text{ soustractions}
$$
![](assets/Pasted%20image%2020240918085023.png)

---

À l'étape $n-2$:

$$
1 \text{ division} \\
+ 1 \text{ multiplication} \\
+ 1 \text{ soustraction}
$$
![](assets/Pasted%20image%2020240918085334.png)

---

Au total, on a

$$
\sum_{p=1}^{n-1} p + 2p^2 = \frac{(n-1)n}{2} + 2 \left[ \frac{(n-1)n(2n-1)}{6} \right]
$$

$$
\sim \frac{2n^3}{3}, \quad \text{quand } n \to +\infty
$$

Opérations.

**Bilan:** pour résoudre un système linéaire,

$$
O(n^3) \quad \text{(facto LU)} + 2O(n^2) \quad \text{(résoudre 2 systèmes triangulaires)}
$$
Here is the recognized text from the image, formatted according to your instructions:

---

$A_{j, p+1:}$

![](assets/Pasted%20image%2020240918085741.png)

---

II) **Propriétés**

1) Effets des erreurs d'arrondi

Exemple:

$$
A = \begin{pmatrix} 10^{-20} & 1 \\ 1 & 1 \end{pmatrix}
$$
$$
= \begin{pmatrix} 1 & 0 \\ 10^{20} & 1 \end{pmatrix} \begin{pmatrix} 10^{-20} & 1 \\ 0 & 1 - 10^{20} \end{pmatrix}
$$
$$
 = LU
$$
$$
\approx \begin{pmatrix} 1 & 0 \\ 10^{20} & 1 \end{pmatrix} \begin{pmatrix} 10^{-20} & 1 \\ 0 & -10^{20} \end{pmatrix}
$$
$$
=\tilde{L}\tilde{U}
$$
$$
= \begin{pmatrix} 10^{-20} & 1 \\ 1 & 0 \end{pmatrix} = \tilde{A}
$$

**Prop:** $A = LU$. La factorisation obtenue en virgule flottante

$$
\tilde{L} \tilde{U} = A + \Delta A
$$

 vérifie

$$
\|\delta A\| \leq \varepsilon \|L\| \|U\|
$$

avec $\varepsilon$ précision machine.

**Pb:** $\|L\|$ et $\|U\|$ peuvent être arbitrairement grandes comparées à $\|A\|$.

**Pivot partiel:** On choisit, à l'étape $p$, comme pivot, le plus grand élément en valeur absolue de la sous-colonne $A_{p+1:, p}^{(p)}$.

On le note $A_{r_p, p}^{(p)}$.

![](assets/Pasted%20image%2020240918091018.png)

On applique une permutation des lignes $p$ et $r_p$

![](assets/Pasted%20image%2020240918091350.png)

Ensuite, on applique les opérations sur les lignes $\forall j > p$

$$
\tilde{L}_j^{(p)} \leftarrow L_j^{(p)} - \frac{\tilde{A}_{j,p}^{(p)}}{\tilde{A}_{p,p}^{(p)}} L_p^{(p)}
$$
$$
\frac{\tilde{A}_{j,p}^{(p)}}{\tilde{A}_{p,p}^{(p)}}\leq 1
$$

On obtient donc

$$
(E_{n-2} P_{n-2} \cdots E_1 P_1 E_0 P_0) A = U
$$

(permutations marquées)

Notons $P = (P_{n-2} \cdots P_0)$

$$
[( E_{n-2} P_{n-2} \cdots P_1 E_0 P_0)P^{-1}] P A = U
$$

$$
= (E_{n-2}' \cdots E_0')
$$

où $E'_p = E_p (-(\mu_{p})_{\pi_p})$ avec $\pi_p = P_{n-2} \cdots P_p$

![](assets/Pasted%20image%2020240918092122.png)

Here is the recognized text from the image, formatted according to your instructions:

---

**Rem:**

$$
(E_{n-2} P_{n-2} \cdots E_0 P_0) P^{-1}
$$
$$
= (E_{n-2} P_{n-2} \cdots E_0 P_0) (P_0^{-1} \cdots P_{n-2}^{-1})
$$
$$
= E_{n-2} P_{n-2} \cdots  E_0 (P_0^{-1} \cdots P_{n-2}^{-1})
$$
$$
= E_{n-2} P_{n-2} \cdots E_1 P_1 (P_1^{-1} \cdots P_{n-2}^{-1}) (P_{n-2} \cdots P_1)E_0 (P_1^{-1} \cdots P_{n-2}^{-1})
$$
$$
= E_{n-2} P_{n-2} \cdots E_2 P_2 (P_2^{-1} \cdots P_{n-2}^{-1}) (P_{n-2} \cdots P_2)E_1 (P_2^{-1} \cdots P_{n-2}^{-1})E_0'
$$
$$
\cdots
$$
$$
E_i' = (P_{n-2} \cdots P_i) E_i (P_i^{-1} \cdots P_{n-2}^{-1})
$$

*(ensemble des permutations opérées sur ligne $\Delta_i$)*

$$
(P_{n-2} \cdots P_1)E_0 (P_1^{-1} \cdots P_{n-2}^{-1})= E_0'
$$

On obtient une facto de la forme 

$$
PA = LU
$$

avec $$L = (E_{n-2}' \cdots E_0')^{-1}$$

**Rem:** Tous les coeffs de $L$ sont plus petits que $1$.

**Prop:** Soit $PA = LU$. La factorisation en virgule flottante

$$
\tilde{L} \tilde{U} = \tilde{P}A + \delta A
$$

vérifie

$$
\|\delta A\| \leq \varepsilon \rho \|A\|
$$

avec $\rho \leq 2^{n-1}$

**Rem:** en pratique, $\rho \approx \sqrt{n}$

---

2) **Remplissage (fill-in) et structure creuse**

À la $p$-ème étape

$$
\forall i > p, \quad \forall j > p, \quad A_{ij} \leftarrow A_{ij} - \frac{A_{i,p}}{A_{p,p}}A_{p,j}
$$

$A_{ij}$ peut devenir non nul (même s'il était nul auparavant) dès que $A_{i,p} \neq 0$ et $A_{p,j} \neq 0$.

**Exemple :** cas extrême.

![](assets/Pasted%20image%2020240918094349.png)

**Prop :** Soit $A$ une matrice bande de largeur de bande $2r+1$.

$L$ et $U$ sont alors des matrices bande de largeur de bande $2r+1$.

*(préservation de la structure bande)*

```tikz
\begin{document}
\tikzset{every picture/.style={line width=0.75pt}} %set default line width to 0.75pt        
\begin{tikzpicture}[x=0.75pt,y=0.75pt,yscale=-1,xscale=1]
%uncomment if require: \path (0,288); %set diagram left start at 0, and has height of 288

%Shape: Polygon [id:ds37052969878994024] 
\draw  [fill={rgb, 255:red, 0; green, 255; blue, 101 }  ,fill opacity=1 ] (373,177.17) -- (373,224) -- (328,224.17) -- (219,115.17) -- (219,70) -- (266,70.17) -- cycle ;
%Shape: Square [id:dp09260878052964294] 
\draw   (219,70) -- (373,70) -- (373,224) -- (219,224) -- cycle ;
%Straight Lines [id:da2510168688440275] 
\draw    (219,62) -- (244.99,62.09) -- (266,62.17) ;
%Straight Lines [id:da028672380248759843] 
\draw    (212,71) -- (212,115.17) ;

% Text Node
\draw (196,84.4) node [anchor=north west][inner sep=0.75pt]    {$r$};
% Text Node
\draw (236,41.4) node [anchor=north west][inner sep=0.75pt]    {$r$};
% Text Node
\draw (342,86.4) node [anchor=north west][inner sep=0.75pt]    {$0$};
% Text Node
\draw (238,192.4) node [anchor=north west][inner sep=0.75pt]    {$0$};
\end{tikzpicture}
\end{document}
```

Here is the recognized text from the image, formatted according to your instructions:

---

**Preuve :** Par récurrence sur la taille de $A$.

$n = r + 1$ ok.

$n \implies n + 1$

$$
A = LU=
\begin{pmatrix} 
1 & 0 \\ 
\frac{x}{A_{0,0}} & \tilde{L}
\end{pmatrix}
\begin{pmatrix} 
A_{0,0} & y^{T} \\ 
0 & \tilde{U}
\end{pmatrix}
$$

$$
= \begin{pmatrix} A_{0,0} & y^T \\ x & \frac{x y^T}{A_{0,0}} + \tilde{L} \tilde{U} \end{pmatrix}
$$

On a donc

$$
A_{1:,1:} = \frac{x y^T}{A_{0,0}} + \tilde{L} \tilde{U}
$$

ou encore $A_{1:,1:} - \dfrac{xy^T}{A_{1,0}} = \tilde{L} \tilde{U}$

$\tilde{L} \tilde{U}$ est la facto $LU$ de $A_{1:,1:} - \dfrac{xy^T}{A_{1,0}}$.

On $A_{1:,1:}$, matrice bande de largeur de bande $2r+1$.

et $\dfrac{xy^T}{A_{0,0}}$ aussi, car 
$$
\left(\frac{xy^{T}}{A_{0,0}}\right)_{i,j} = \frac{x_iy_i}{A_{0,0}}=
\begin{cases}
0,\quad\text{si}\,i>j\geq r,\text{triang inf}\quad\text{car}\, x_i=0,\,\forall i\geq r+1\\
0,\quad\text{si}\,j>i\geq r,\text{triang sup}\quad\text{car}\, y_j=0,\,\forall j\geq r+1
\end{cases}
$$

Elles sont de tailles $n^2$. Donc on peut appliquer l'hypothèse de récurrence.

Here is the recognized text from the images, formatted according to your instructions:

---

**Prop:** La factorisation $LU$ conserve la structure ligne de ciel.

**Preuve:**

$$
\forall i > p, \quad \forall j > p, \quad A_{ij} \leftarrow A_{ij} - \frac{A_{i,p} }{A_{p,p}}A_{p,j}
$$

Si $A_{i,p} \neq 0$ et $A_{p,j} \neq 0$, alors $A_{i,j}$ est déjà stocké dans la structure ligne de ciel. En effet, si $i > j > p$ (triang inf), $A_{i,j}$ est stocké dans la sous-ligne $A_{i,p:j}$.

![](assets/Pasted%20image%2020240918214841.png)

Si $j > i > p$ (triang sup), $A_{ij}$ est stocké dans la sous-colonne $A_{p:i,j}$.

![](assets/Pasted%20image%2020240918214916.png)

**Rem:** Renuméroter les lignes et colonnes permet de réduire les stockages bande ou lignes de ciel et permet d’avoir une factorisation LU plus creuse.

Pour la factorisation LU, il vaut mieux utiliser l’algo Cuthill McKee inverse (prendre la permutation inverse).
