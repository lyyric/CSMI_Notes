# Amélioration des méthodes numériques grâce à l’apprentissage  

## I) Rappels sur les méthodes de Galerkin

### 1) Équation elliptique  

**Problème de Poisson**  
On considère $\Omega \subset \mathbb{R}^d$ et $f \in L^2(\Omega)$.  

**Problème de Poisson :** Trouver $u \in H^2(\Omega)$ tel que  

$$
\begin{cases}  
-\Delta u = f \quad \text{sur } \Omega \\  
u = 0 \quad \text{sur } \partial \Omega  
\end{cases}
$$

(**formulation forte**)

$$
(-\Delta u, v)_{L^2} = (f, v)_{L^2} \quad \forall v \in H_0^1
$$

$$
\int_{\Omega} -\Delta u \, v = \int_{\Omega} f v \quad \forall v \in H_0^1(\Omega)
$$

$$
\int_{\Omega} (-\nabla \cdot \nabla u) v + \int_{\Omega} \nabla u \cdot \nabla v = \int_{\Omega} f v
$$

$$
-\int_{\partial \Omega} (\mathbf{n} \cdot \nabla u) v \underbrace{= 0}_{\text{car } v = 0 \text{ sur } \partial \Omega} + \int_{\Omega} \nabla u \cdot \nabla v = \int_{\Omega} f v
$$

---

Trouver $u \in H_0^1(\Omega)$ tel que  

$$
\int_{\Omega} \nabla u \cdot \nabla v = \int_{\Omega} f v \quad \forall v \in H_0^1(\Omega)
$$

(**formulation faible**)  

---

$u$ solution d'après le théorème de Lax-Milgram.  

De plus, $(u, v) \mapsto \int_{\Omega} \nabla u \cdot \nabla v$ est **symétrique**.


Le pb est équivalent à  

$$
\arg \min_{u \in H_0^1} \left( \frac{1}{2} \int_{\Omega} |\nabla u|^2 - \int_{\Omega} f u \right)
$$

(**formulation énergétique**)

**Formalime** : on considère des problèmes de la forme  

#### (forme forte)  
Trouver $u \in H$,  
$$ A u = f $$  
$A = -\Delta$  

#### (forme variationnelle)  
Trouver $u \in V$,  
$$ a(u,v) = l(v) \quad \forall v \in V $$  
$a$ symétrique  

#### (forme énergétique)  
$$ u = \arg\min_{u \in V} \left[ \frac{1}{2} a(u,u) - l(u) \right] $$  

$V, H$ espace de Hilbert  

On suppose que les problèmes sont bien posés  
$\quad ( a$ bilinéaire, continu, coercif et symétrique $)$  
$\quad ( l$ linéaire, continu $)$  

### 2) Approximation

**Espace d’approximation** $V_h$ **CV** : sous-espace vectoriel de dimension $J$.  

Notant $(\phi_j)_{j=1, \dots, J}$ une base de $V_h$,  

$$
V_h = \left\{ \sum_{j=1}^{J} \theta_j \phi_j(x) \quad \text{avec} \quad \theta \in \mathbb{R}^J \right\}.
$$

(Diagramme illustrant l’inclusion de $V_h$ dans $V$, avec $u$ et son approximation $u_h$.)  


**Projection orthogonale sur** $V_h$  

Soit $u \in V$, le **projeté orthogonal** de $u$ sur $V_h$ est l’unique $P(u) \in V_h$ vérifiant  

$$
\| u - P(u) \|_V^2 = \min_{\nu \in V_h} \| u - \nu \|_V^2
$$

Il est caractérisé par :  

$$
( u - P(u), v - P(u) )_V = 0 \quad \forall v \in V_h
$$

$$
( u - P(u), v )_V = 0 \quad \forall v \in V_h
$$

**Preuve** : Pour tout $u \in V$,  

$$
P(u) = \sum_{j=1}^{J} \theta_j \phi_j
$$

avec $\theta \in \mathbb{R}^J$ solution de $M\theta = b$,  

où  

$$
M = \left( (\phi_i, \phi_j) \right)_{i,j}
$$

et  

$$
b = \left( (\phi_i, u) \right)_i
$$

**Remarque** : Si $(\phi_j)$ sont orthonormés, alors $M = \text{Id}$,  

et donc  

$$
\theta = \left( (\phi_i, u) \right)_i
$$

et  

$$
P(u) = \sum_{j=1}^{J} (\phi_j, u) \phi_j
$$

On retrouve la formule de projection habituelle dans une base orthonormée.  

**Remarque** : $M$ est appelée **matrice de masse**.  

**Preuve** :  

$$
(u - P(u), v) = 0 \quad \forall v \in V_h
$$

$$
\Leftrightarrow (u - P(u), \phi_i) = 0 \quad \forall i \in \{1, \dots, J\}
$$

$$
\Leftrightarrow \left( u - \sum_{j=1}^{J} \theta_j \phi_j, \phi_i \right) = 0 \quad \forall i
$$

$$
\Leftrightarrow (u, \phi_i) - \sum_{j=1}^{J} (\phi_j, \phi_i) \theta_j = 0 \quad \forall i \in \{1, \dots, J\}
$$

$$
\Leftrightarrow M \theta = b.
$$

**Interpolation** :  

$$
V \subset C(\Omega).
$$

Soit $u \in V$ et $(x_i)_{i=1, \dots, J}$ des points de $\Omega$ donnés.  

L’interpolé de $u$ aux points $(x_i)$ est l’unique

$$
I(u) \in V_h \quad \forall g
$$

$$
I(u)(x_i) = u(x_i) \quad \forall i \in \{1, \dots, J\}
$$

**Preuve** :  

$$
\text{Pour tout } u \in V,
$$

$$
I(u) = \sum_{j=1}^{J} \theta_j \phi_j
$$

avec $\theta \in \mathbb{R}^J$ vérifiant  

$$
V \theta = c
$$

où  

$$
V = \left( \phi_j(x_i) \right)_{i,j}, \quad c = (u(x_i))_i.
$$

**Preuve** :  

$$
I(u)(x_i) = u(x_i) \quad \forall i \in \{1, \dots, J\}
$$

$$
\Leftrightarrow \sum_{j=1}^{J} \theta_j \phi_j(x_i) = u(x_i) \quad \forall i \in \{1, \dots, J\}
$$

$$
\Leftrightarrow V \theta = c.
$$

**Remarque** :  

Si $(\phi_j)$ vérifie la propriété d'interpolation  

$$
\phi_j(x_i) = \delta_{ij} =  
\begin{cases} 
1 \text{ si } i = j, \\ 
0 \text{ sinon},
\end{cases}
$$

alors $V = \text{Id}$,  

$$
\theta = (u(x_i))
$$

et  

$$
I(u) = \sum_{j=1}^{J} u(x_j) \phi_j.
$$

**Remarque** :  

$$
\forall u \in V, \quad \| u - P(u) \|_V \leq \| u - I(u) \|_V
$$

*plus facile à estimer avec des développements de Taylor !*  

### 3) Méthodes de Galerkin

**Méthode de Galerkin** : on remplace $V$ par $V_h$.  

#### **Forme variationnelle**  

$$
\text{Trouver } u_h \in V_h \quad \text{tel que} \quad a(u_h, v_h) = l(v_h) \quad \forall v_h \in V_h
$$

#### **Forme énergétique**  

$$
u_h = \arg\min_{u \in V_h} \left[ \frac{1}{2} a(u,u) - l(u) \right]
$$

$$
\Leftrightarrow \quad \text{cas où } a \text{ est symétrique}
$$

**Preuve** :  

La solution du problème est donnée par  

$$
u_h = \sum_{j=1}^{J} \theta_j \phi_j
$$

avec $\theta \in \mathbb{R}^J$ solution de  

$$
A \theta = d
$$

avec  

$$
A = \left( a(\phi_i, \phi_j) \right)_{i,j}, \quad d = \left( l(\phi_i) \right)_i.
$$

**Preuve** : exercice (ou voir cours Calcul Scientifique 2).  

**Remarque** : Le lemme de Céa montre que  

$$
\| u - u_h \|_V \leq \frac{M}{\nu} \| u - P(u) \|_V \leq \frac{M}{\nu} \| u - I(u) \|_V
$$

(erreur)  

$$
\rightarrow 0 \quad \text{quand} \quad h \rightarrow 0.
$$

**Méthode de collocation**  

$$
H \subset C(\Omega) \quad \text{et} \quad (x_i)_{1 \leq i \leq J} \quad J \text{ points de } \Omega
$$

#### **Forme forte**  

$$
\text{Trouver } u_h \in H_h \quad \text{tel que} \quad (A u_h)(x_i) = f(x_i) \quad \forall i \in \{1, \dots, J\}
$$

C'est équivalent au système linéaire  

$$
A \left( \sum_{j=1}^{J} \theta_j \phi_j \right)(x_i) = f(x_i) \quad \forall i \in \{1, \dots, J\}
$$

$$
\Leftrightarrow \sum_{j=1}^{J} \theta_j A(\phi_j)(x_i) = f(x_i) \quad \forall i \in \{1, \dots, J\}
$$

**A linéaire**

**Méthode de Galerkin moindres carrés**  

#### **Forme forte (bis)**  

$$
u_h = \arg\min_{u \in H_h} \| A(u) - f \|_{L^2}^2
$$

**Proposition** :  

La solution du problème de Galerkin moindres carrés est donnée par  

$$
u_h = \sum_{j=1}^{J} \theta_j \phi_j
$$

avec $\theta \in \mathbb{R}^J$ vérifiant  

$$
A_{LS} \theta = d_{LS}
$$

où  

$$
A_{LS} = \left( (A(\phi_i), A(\phi_j))_{L^2} \right)_{i,j}
$$

$$
d_{LS} = \left( (A(\phi_i), f)_{L^2} \right)_i
$$

**"LS = least squares"**  

**Preuve (formelle)**  

$$
J(u) = \| A(u) - f \|_{L^2}^2
$$

$$
= \int_{\Omega} (A(u) - f)^2
$$

$J$ est continue car composée d'application continue (*si $A$ est continue*).  

$J$ est différentiable  

$$
J(u+v) = \int_{\Omega} (A u + A v - f)^2
$$

$$
= \int_{\Omega} (A u - f)^2 + (A v)^2 + 2(A u - f) A v
$$

$$
= J(u) + L(u)(v) + R(v)
$$

avec  

$$
L(u)(v) = \int_{\Omega} 2(A u - f) A v
$$

$$
R(v) = \int_{\Omega} (A v)^2
$$
Si $u_h$ est un minimum de $J$ sur $H_h$, alors  

$$
L(u_h)(v) = 0 \quad \forall v \in H_h
$$

$$
\Leftrightarrow \int_{\Omega} A_h(u_h) A_h(v) = \int_{\Omega} f A_h(v) \quad \forall v \in H_h
$$

$$
\Leftrightarrow \int_{\Omega} A_h(u_h) A_h(\phi_i) = \int_{\Omega} f A_h(\phi_i) \quad \forall i \in \{1, \dots, J\}
$$

$$
\Leftrightarrow \sum_{j=1}^{J} \left( \int_{\Omega} (A \phi_j) (A \phi_i) \right) \theta_j = \int_{\Omega} f (A \phi_i) \quad \forall i \in \{1, \dots, J\}
$$

$$
\Leftrightarrow u_h = \sum_{j=1}^{J} \theta_j \phi_j
$$

**Remarque** : $A$ étant linéaire  

$$
\Rightarrow \text{le problème revient à résoudre un système linéaire}
$$

car $J(u)$ est quadratique et $L(u_h)$ est linéaire.

### 4) Exemples d’espace d’approximation

#### **Approximation par éléments finis (2D)**  

- **Maillage** : partitionnement de $\Omega$ en polygones  
  (triangle, quadrilatère)  

- **Éléments finis** $P_k$ :  

$$
V_h = \{ u \in C(\Omega) \mid u_{|K} \in P_k \quad \forall K \text{ cellule du maillage} \}
$$

Fonctions continues, polynomiales par morceaux.

![[image-14.png|257x257]]

- $(\phi_i)$ base avec propriété d’interpolation  

$$
\phi_i(x_j) = \delta_{ij}
$$

$$
u_h(x) = \sum_{j=1}^{J} u_h(x_j) \phi_j(x)
$$

#### **Approximation spectrale (2D)**  

- $\Omega = [0, L_1] \times [0, L_2]$  

-  
$$
V_h = \left\{ \sum_{k_1, k_2 = 1}^{K} \theta_j \sin \left( \frac{k_1 \pi}{L_1} x_1 \right) \sin \left( \frac{k_2 \pi}{L_2} x_2 \right), \quad \theta \in \mathbb{R}^{K^2} \right\}
$$

où  

$$
j = k + K (l - 1)
$$

Fonctions trigonométriques de degré $\leq K$, nulles au bord.  

- $(\phi_i)$ orthogonaux.  
- **Remarque** : pas besoin de maillage.  

#### **Approximation par base radiale**  

- Soit $(x_i)$ : $J$ points de $\Omega$.  
-  

$$
V_h = \left\{ \sum_{j=1}^{J} \theta_j \varphi(\| x - x_j \|), \quad \theta \in \mathbb{R}^J \right\}
$$

avec, par exemple,  

$$
\varphi(r) = \exp \left( - \frac{r^2}{\varepsilon^2} \right)
$$

- **Pas besoin de maillage**.
![[image-13.png|294x294]]

## II) Méthode neuronale pour les équations stationnaires

**Idée** : prendre comme espace d’approximation  

$$
M_h = \left\{ NN_{\theta}(x), \quad \theta \in \mathbb{R}^p \right\}
$$

avec $NN_{\theta}$ un réseau de neurones avec une architecture fixée.  

$$
\mathbb{R}^d \to \mathbb{R}
$$

$$
x \mapsto NN_{\theta}(x)
$$

- $NN_{\theta_1} + NN_{\theta_2} \neq NN_{\theta_1 + \theta_2}$ s’il y a des activations non linéaires.  

$$
\Rightarrow M_h \text{ n’est pas un espace vectoriel}
$$

$$
\Rightarrow M_h \text{ est une sous-variété : "surface"}
$$
![[image-12.png|335x335]]

On remplace $V$ par $M_h$ :  

### **Deep-Ritz**  

$$
u_h = \arg\min_{u \in M_h} \left[ \frac{1}{2} a(u,u) - l(u) \right]
$$

### **Physics-Informed Neural Network (PINN)**  

$$
u_h = \arg\min_{u \in M_h} \| A(u) - f \|^2
$$

$M_h$ n’étant pas un espace vectoriel :  

- Le problème n’est plus linéaire par rapport aux paramètres $\theta$. *(même si $A$ est linéaire)*  
- **Optimisation non linéaire**.

### **1) Évaluation : intégration numérique**  

**Rappel : méthode de Galerkin**  

$$
A = \left( a(\phi_i, \phi_j) \right) = \left( \int_{\Omega} \nabla \phi_i (x) \cdot \nabla \phi_j (x) \, dx \right)
$$

*(problème de Poisson)*  

$$
d = \left( l(\phi_i) \right) = \left( \int_{\Omega} f(x) \phi_i (x) \, dx \right)
$$

- **Calcul des intégrales** : analytiquement  
- ou **méthode de quadrature** *(règles de Gauss)*  

**Pour les méthodes neuronales : méthode de Monte-Carlo**  

### **(Deep-Ritz)**  

$$
\frac{1}{2} a(u,u) - l(u) = \frac{1}{2} \int_{\Omega} \|\nabla u (x) \|^2 - f(x) u(x) \, dx
$$

*(problème de Poisson)*  

$$
\approx \frac{1}{N} \sum_{i=1}^{N} \|\nabla u (x_i) \|^2 - f(x_i) u(x_i)
$$

### **(PINN)**  

$$
\| A(u) - f \|_{L^2}^2 = \int_{\Omega} (-\Delta u (x) - f(x))^2 \, dx
$$

$$
\approx \frac{1}{N} \sum_{i=1}^{N} (-\Delta u (x_i) - f(x_i))^2
$$

Avec $(x_i)$ tirés aléatoirement uniformément sur $\Omega$.  

- **Méthode ne nécessitant pas de maillage**  
- **Domaine complexe**  

![[image-11.png|373x373]]

Pour cela, on utilise deux méthodes :  

- Une application continue bijective d’inverse continue *(homéomorphisme)*  

$$
\phi : \Omega_{\text{ref}} \to \Omega
$$

avec $\Omega_{\text{ref}}$ domaine de référence simple (typiquement $\Omega_{\text{ref}} = [0,1] \times [0,1]$).

Il suffit alors d’échantillonner dans le domaine de référence.  

**Exemple** :  

$$
\Omega = D(0, R)
$$

$$
\phi : [0,R] \times [0,2\pi[ \to \Omega
$$

$$
(r, \theta) \mapsto (r \cos \theta, r \sin \theta)
$$

$$
\int_{\Omega} R(x) \,dx = \int_{\Omega_{\text{ref}}} R(y) \left| \det \text{Jac} \, \phi(y) \right| dy
$$

$$
\phi(\Omega_{\text{ref}}) = \Omega, \quad x = \phi(y), \quad dx = \left| \det \text{Jac} \, \phi(y) \right| dy
$$

- Une fonction contour de niveau *(level set)* de la frontière du domaine.

$$
\begin{cases} 
\phi(x) = 0 & \text{si } x \in \partial \Omega \\
\phi(x) < 0 & \text{si } x \in \Omega \\
\phi(x) > 0 & \text{si } x \notin \Omega
\end{cases}
$$

![[image-10.png|339x339]]

On échantillonne dans un rectangle contenant $\Omega$, puis on ne garde que les points tels que $\phi(x) < 0$.  

**Exemple** :  

$$
\Omega = D(0,R)
$$

$$
\phi(x) = \| x \| - R
$$

*(fonction distance signée)*

![[image-9.png|412x275]]

(Deep Ritz)  
$$
\frac{1}{2} a(u,u) - \ell(u)
= \int_{\Omega} \left( \frac{1}{2} \|\nabla u\|^2 - f u \right) 
\approx \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{2} \|\nabla u(x_i)\|^2 - f(x_i) u(x_i) \right)
$$

$(x_i)$ tiré aléatoirement dans $\Omega$ uniformément

→ pas besoin de maillage

Échantillonnage préférentiel (importance sampling)

On note $r(x)$ l’intégrande  
$$
\left\{
\begin{aligned}
r(x) &= \frac{1}{2} \|\nabla u(x)\|^2 - f(x) u(x) \\
r(x) &= \| - \Delta u(x) - f(x) \|^2
\end{aligned}
\right.
$$

$$
\int_{\Omega} r(x) \, dx = \int_{\Omega} \frac{r(x)}{p(x)} p(x) \, dx = \mathbb{E}_p \left[ \frac{r(x)}{p(x)} \right]
\quad \text{avec } x \sim p
$$

où $p(x)$ est une densité de probabilité.  
En appliquant la méthode de Monte Carlo :

$$
\int_{\Omega} r(x) \, dx \approx \frac{1}{N} \sum_{i=1}^{N} \frac{r(x_i)}{p(x_i)}
$$

avec $(x_i)$ tiré aléatoirement suivant $p$.

L’erreur est donnée par le thm central limité :

$$
\sqrt{N} \left( \int_{\Omega} r(x) \, dx - \frac{1}{N} \sum_{i=1}^{N} \frac{r(X_i)}{p(X_i)} \right)
\underset{N \to \infty}{\sim} \mathcal{N}(0, \sigma^2)
$$

avec $\sigma^2$ variance :

$$
\sigma^2 = \mathbb{V}_p \left[ \frac{r(X_1)}{p(X_1)} \right]
= \mathbb{E}_p \left[ \left( \frac{r(X_1)}{p(X_1)} - \mathbb{E}_p \left[ \frac{r(X_2)}{p(X_2)} \right] \right)^2 \right]
$$

Pour minimiser l’erreur, il faut que la variance soit la plus petite possible.  
Pour cela, on souhaite que

$$
\frac{r(X_i)}{p(X_i)} \approx \text{constante} \quad \Rightarrow \quad p(x) \approx \frac{r(x)}{\int_{\Omega} r(x) \, dx}
\quad \text{↑ probabilité}
$$

Plus de points là où $r(x)$ est grand  
*(l’erreur est grande)*

Méthode RAR (Residual Adaptive Refinement)

On tire des points **uniformément**, et on ne garde que les $N$ points de valeurs $r(x)$ les **plus grandes**.

Autre méthode

On construit une approximation $p_{\theta}(x)$ de

$$
\frac{r(x)}{\int_{\Omega} r(x) \, dx}
$$

par une **méthode générative** (voir chapitre suivant).

> $p_{\theta}(x)$ est un **réseau de neurones**.

Remarque

En pratique, on remplace :
$$
\int_{\Omega} \frac{r(x)}{p(x)} \, p_{\theta}(x) \, dx
\quad \Longrightarrow \quad
\int_{\Omega} r(x) \, p_{\theta}(x) \, dx
$$
Et on approxime :
$$
\int_{\Omega} r(x) \, p_{\theta}(x) \, dx
\approx \frac{1}{N} \sum_{i=1}^{N} r(x_i)
\quad \text{où } x_i \sim p_{\theta}
$$
### 2) Optimisation

On souhaite optimiser

$$
\mathcal{L}(\theta) = \int_{\Omega} \left( \frac{1}{2} \|\nabla u_{\theta}(x)\|^2 - f(x) u_{\theta}(x) \right) dx
$$

$$
\approx \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{2} \|\nabla u_{\theta}(x_i)\|^2 - f(x_i) u_{\theta}(x_i) \right) = \widehat{\mathcal{L}}(\theta)
$$

où $u_{\theta} = \mathrm{NN}_{\theta}$, réseau de neurones de paramètres $\theta$.

Problème d’optimisation non linéaire :

Méthodes de type gradient (Adam)  
de type Newton (L-BFGS)

**Rem** : les points $(x_i)$ peuvent être modifiés à chaque itération de l'étape de gradient.  
*(notamment pour l’échantillonnage préférentiel)*

---

### 3) Inclure les conditions aux limites

Pour les méthodes de Galerkin (Galerkin-moindre carré) :

- **Méthode forte** : toutes les fonctions de $V_h$ satisfont les conditions aux limites.
- **Méthode de pénalisation**

On minimise  
$$
\left( \frac{1}{2} a(u,u) - \ell(u) \right) + \frac{1}{\varepsilon} \int_{\partial \Omega} \|B(u)\|^2
$$

où $B(u) = 0$ est la condition aux limites.  
Avec $\varepsilon > 0$ **petit**.

---

Même technique pour les méthodes neuronales :

**Méthode de pénalisation** : on considère

$$
\mathcal{L}(\theta) + \lambda_b \mathcal{L}_b(\theta) =
\left( \int_{\Omega} \left( \frac{1}{2} \|\nabla u_{\theta}(x)\|^2 - f(x) u_{\theta}(x) \right) dx \right)^2
+ \lambda_b \int_{\partial \Omega} \left( u_{\theta}(x) - g(x) \right)^2 dx
$$

pour imposer la condition aux limites $u_{\theta}(x) = g(x)$ sur le bord.

En discrétisant les intégrales :

$$
\widehat{\mathcal{L}}(\theta) + \lambda_b \widehat{\mathcal{L}}_b(\theta) =
\left( \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{2} \|\nabla u_{\theta}(x_i)\|^2 - f(x_i) u_{\theta}(x_i) \right) \right)^2
+ \lambda_b \left( \frac{1}{N_b} \sum_{i=1}^{N_b} \left( u_{\theta}(x_i^b) - g(x_i^b) \right)^2 \right)
$$

avec $(x_i)$ uniformément  sur $\Omega$,  *(ou avec échantillonnage préférentiel)*.
$(x_i^b)$ uniformément sur $\partial \Omega$.

Avec $\lambda_b > 0$, paramètre permettant d’équilibrer les fonctions coût.

**Rem.** : comment choisir $\lambda_b$ ? À chaque étape de gradient,  
il faut que $\widehat{\mathcal{L}}$ et $\widehat{\mathcal{L}}_b$ soient optimisées.  
Pour cela, on choisit à l’étape $k$ de l’algorithme :

$$
\lambda_b^k = \frac{\|\nabla_{\theta} \widehat{\mathcal{L}}(\theta^k)\|}{\|\nabla_{\theta} \widehat{\mathcal{L}}_b(\theta^k)\|}
$$
$$
\theta^{k+1} = \theta^k - \beta \left( \nabla_{\theta} \widehat{\mathcal{L}}(\theta^k) + \lambda_b \nabla_{\theta} \widehat{\mathcal{L}}_b(\theta^k) \right)
\Rightarrow \|\nabla_{\theta} \widehat{\mathcal{L}}(\theta^k)\| \approx \lambda_b \|\nabla_{\theta} \widehat{\mathcal{L}}_b(\theta^k)\|
$$
ou avec méthode relaxée :
$$
\lambda_b^k = (1 - \alpha) \lambda_b^{k-1} + \alpha \frac{\|\nabla_{\theta} \widehat{\mathcal{L}}(\theta^k)\|}{\|\nabla_{\theta} \widehat{\mathcal{L}}_b(\theta^k)\|}
\quad \text{avec } \alpha \in [0, 1]
$$

**Méthode forte** : pour imposer que $u_{\theta}(x)$ soit égal à $g(x)$  
sur le bord, on considère

$$
\mathcal{M}_{h} = \left\{ u_{\theta}(x) | g(x) + \phi(x) \mathrm{NN}_{\theta}(x), \quad \theta \in \mathbb{R}^p \right\}
$$

avec $\phi(x)$ une fonction courbe de niveau de $\partial \Omega$ *(level-set)*

**Inconvénient** : il faut connaître $\phi(x)$ ou la construire.

**Rem.** : pour le disque, si on choisit $\phi(x) = \|x\| - R = \sqrt{x^2 + y^2} - R$  
  
elle est continue mais pas dérivable en 0  
$\Rightarrow$ problème dans l’apprentissage.

Même remarque dans le cas général : la distance signée au bord $d_{\Omega}(x)$ n’est pas différentiable partout.

On peut régulariser :

$$
\phi_{\varepsilon}(x) =
\begin{cases}
\phi(x) & \text{pour } |x| \geq \varepsilon \\
\text{polynôme de degré 2} & \text{pour } |x| < \varepsilon \\
\text{avec raccord } \mathcal{C}^1
\end{cases}
$$

(Diagramme de $d_{\varepsilon}(x)$ avec coin adouci entre $-\varepsilon$ et $\varepsilon$)

---

Pour le disque, on peut aussi prendre  
$$
\widetilde{\phi}(x) = \|x\|^2 - R^2
$$

### 4) Bilan

|                            | **classique**                                                    | **neuronal**                                                      |
| -------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------- |
| **espace d’approximation** | espace vectoriel                                                 | réseau de neurones *(sous-variété)*                               |
| **méthode de quadrature**  | méthode de Gauss  méthode déterministe  *(maillage)*             | méthode de Monte-Carlo  *(pas de maillage)*                       |
| **optimisation**           | *(exacts)* résolution de l’équation d’Euler                      | méthode de gradients                                              |
| **conditions aux bords**   | méthode de pénalisation ($\varepsilon$ petit)  méthode forte | méthode de pénalisation *(avec $\varepsilon$)*  méthode forte |
| **convergence**            | preuve de l’ordre de convergence                                 | ?                                                                 |

**Plusieurs sources d’erreurs** :  
- erreur de l’espace d’approximation *(difficile)*  
- erreur de la méthode de quadrature *(OK)*  
- erreur de l’algorithme d’optimisation *(difficile car $\mathcal{L}(\theta)$ non convexe)*

**Approximation des réseaux de neurones**  
sensible à la dimension.

**Problèmes paramétriques** : équation de Poisson paramétrique

$$
\begin{cases}
- \Delta u(x; \mu) = f(x; \mu) & \text{sur } \Omega \\
u(x; \mu) = g(x; \mu) & \text{sur } \partial \Omega
\end{cases}
$$

Où $\mu \in D_{\mu} \subset \mathbb{R}^m$ désigne les paramètres du problème.

On introduit l’espace d’approximation  
$$
\mathcal{M}_h = \left\{ u_{\theta}(x; \mu)\,|\,\mathrm{NN}_{\theta}(x; \mu), \quad \theta \in \mathbb{R}^p \right\}
$$

et on utilise la méthode Deep Ritz ou PINN.

---

**Rem.** : $(x, \mu) \in \mathbb{R}^{d + m}$  
Pour une méthode classique, il faudrait un maillage de $\Omega \times D_{\mu}$.  
Si on veut $n$ points par direction, on a $n^{d + m} = N$ points en tout.

$$
\Rightarrow \text{“modèle de réduction d’ordre”}
\quad \text{(voir cours de M2)}
$$
![[image-15.png|293x293]]

### 5) Autre méthode

**“Extreme Machine Learning” – “random features”**

$$
\mathcal{M}_h = \left\{ \sum_{i=1}^{J} \theta_i \, \sigma(w_i^T x + b_i), \quad \theta \in \mathbb{R}^J \right\}
\quad \text{avec } \sigma(w_i^T x + b_i) = \varphi_i(x)
$$

avec $(w_i, b_i)$ paramètres du réseau de neurones fixés *(aléatoirement)*

$\mathcal{M}_h$ = réseau de neurones à deux couches avec une couche linéaire en sortie,  
dont on ne fait varier **que les paramètres de la deuxième couche**

$$
\Rightarrow \mathcal{M}_h = \underline{\text{espace vectoriel}}
$$



$\Rightarrow$ résolution comme dans le cas classique  
*(avec méthode de Monte Carlo pour le calcul des intégrales)*

---

## III) Méthode neuronale pour les équations d’évolution

On considère l’équation de diffusion :

$$
\begin{cases}
\partial_t u - \Delta u = 0 & \text{sur } \mathbb{R}_+ \times \Omega \\
u(t, x) = g(t, x) & \text{pour } (t,x) \in \mathbb{R}_+ \times \partial \Omega \\
u(0, x) = u_{\text{init}}(x) & \text{pour } x \in \Omega
\end{cases}
$$

que l’on réécrit de manière formelle :

$$
\begin{cases}
\partial_t u - \mathcal{T}(u) = 0 & \text{sur } \mathbb{R}_+ \times \Omega \\
B(u) = 0 & \text{sur } \mathbb{R}_+ \times \partial \Omega \\
I(u) = 0 & \text{sur } \Omega
\end{cases}
$$

### 1) Méthode espace-temps : **PINN**

On considère l’espace d’approximation :

$$
\mathcal{M}_h = \left\{ \mathrm{NN}_{\theta}(t, x), \quad \theta \in \mathbb{R}^p \right\}
$$

> “Le temps est considéré comme une variable d’espace.”

*(avec schéma d’un cylindre représentant $\Omega \times [0, T]$, $x_1, x_2$ et $t$)*

![[imageeeee.png|208x208]]

Avec les conditions aux limites traitées de manière pénalisée, on considère :

$$
\mathcal{L}(\theta) + \lambda_b \mathcal{L}_b(\theta) + \lambda_i \mathcal{L}_i(\theta)
$$

$$
= \left( \int_{[0,T] \times \Omega} \|\partial_t u - \mathcal{A}(u)\|^2 \right)
+ \lambda_b \left( \int_{[0,T] \times \partial \Omega} \|B(u)\|^2 \right)
+ \lambda_i \left( \int_{\Omega} \|\mathcal{I}(u)\|^2 \right)
$$

*(avec $\|B(u)\|^2$ contenant $(u - g)^2$，et $\|\mathcal{I}(u)\|^2$ contenant $(u(0, \cdot) - u_{\text{init}})^2$)*

Avec $\lambda_b, \lambda_i$ paramètres pour équilibrer les fonctions coût.


---

$$
\begin{cases}
\partial_t \mu - A \mu = 0 \\
B(\mu) = 0 \\
I(\mu) = 0
\end{cases}
$$

*pas de condition*

![[image-16.png|477x318]]

**Causalité** : l’apprentissage a tendance à se faire plus rapidement pour les temps grands que pour les temps petits.

**Première solution** : on découpe $[0,T]$ en sous-intervalles $[t_i, t_{i+1}]$ avec $t_i = i \Delta t$ et $\Delta t > 0$.

![[ChatGPT Image Apr 3, 2025, 11_35_13 AM.png|318x318]]

On apprend successivement sur les différents intervalles.

$$
\forall j \in [0, N-1] \quad \mathcal{L}^j(\theta) + \lambda_b \mathcal{L}_b^j(\theta) + \lambda_i \mathcal{L}_i^j(\theta)
$$

$$
= \left( \int_{[t_j, t_{j+1}] \times \Omega} \| \partial_t u - A u \|^2 \right)
+ \lambda_b \left( \int_{[t_j, t_{j+1}] \times \partial \Omega} \| B(u) \|^2 \right)
+ \lambda_i \left( \int_{\Omega} \| I(u) \|^2 \right) \delta_{j,0}
$$

**et on boucle !!**

**Deuxième méthode** : on pondère les contributions des différents intervalles dans la fonction coût.

$$
\mathcal{L}(\theta) \rightarrow \tilde{\mathcal{L}}(\theta) = \sum_{j=1}^{N-1} w_j \left( \int_{[t_j, t_{j+1}] \times \Omega} \| \partial_t u - A u \|^2 \right)
$$

où le poids $w_j$ dépend de la qualité de l’apprentissage sur les intervalles précédents :

$$
w_j = \exp \left( - \varepsilon \left( \int_{[0, t_j] \times \Omega} \| \partial_t u - A u \|^2 \right) \right)
$$

*erreur sur $[0, t_j]$*

$$
\Rightarrow w_j \text{ devient grand quand l’erreur avant } t_j \text{ devient petite}
$$

**Appendice** : **Noyau neuronal tangent**  
*(Neural Tangent Kernel)*  
outil pour analyser la vitesse d’apprentissage.

On considère le problème de régression

$$
\mathcal{L}(\theta) = \frac{1}{2} \sum_{i=1}^{n} \left( f_\theta(x_i) - y_i \right)^2
$$

avec $(x_i, y_i) \in \mathbb{R}^2$ et $f_\theta = \text{NN}_\theta$ est un réseau de neurone.

Algorithme de type gradient :

$$
\theta_{k+1} = \theta_k - \eta \nabla \mathcal{L}(\theta_k)
$$

L’équivalent continu s’écrit  
$$
\theta'(t) = -\nabla \mathcal{L}(\theta(t))
$$  
(dynamique de "gradient flow").

**Dynamique de** $\mathcal{L}(\theta(t))$ :

$$
\frac{d}{dt} \left[ \mathcal{L}(\theta(t)) \right] = \sum_{i=1}^{p} \partial_{\theta_i} \mathcal{L}(\theta(t)) \, \theta'_i(t)
$$
$$
= \left( \nabla \mathcal{L}(\theta(t)), \theta'(t) \right)
$$
$$
= - \left( \mathcal{L}(\theta(t)), \nabla \mathcal{L}(\theta(t)) \right)
$$
$$
= - \left\| \nabla \mathcal{L}(\theta(t)) \right\|^2
$$

Donc $\mathcal{L}(\theta(t))$ est décroissante.  

**Quelle est la vitesse de décroissance ?**

On calcule le gradient de la fonction de coût :

$$
\nabla_\theta \mathcal{L}(\theta) = \nabla_\theta \left[ \frac{1}{2} \sum_{i=1}^{m} (f_\theta(x_i) - y_i)^2 \right]
= \sum_{i=1}^{m} (f_\theta(x_i) - y_i) \nabla_\theta f_\theta(x_i)
$$

Alors, son carré norme :

$$
\| \nabla_\theta \mathcal{L}(\theta) \|^2
= \left\langle \sum_{i=1}^{m} (f_\theta(x_i) - y_i) \nabla_\theta f_\theta(x_i), \sum_{j=1}^{m} (f_\theta(x_j) - y_j) \nabla_\theta f_\theta(x_j) \right\rangle
$$
$$
= \sum_{i,j=1}^{m} (f_\theta(x_i) - y_i)(f_\theta(x_j) - y_j) \langle \nabla_\theta f_\theta(x_i), \nabla_\theta f_\theta(x_j) \rangle
= (r, K_\theta r)
$$

où :
- $r = (f_\theta(x_i) - y_i)_i \in \mathbb{R}^n$
- $K_\theta = \left( \langle \nabla_\theta f_\theta(x_i), \nabla_\theta f_\theta(x_j) \rangle \right)_{i,j} \in \mathcal{M}_m(\mathbb{R})$

$K_\theta$ est **symétrique positive**, donc :

$$
(r, K_\theta r) \geq (\lambda_1)_\theta \| r \|^2
$$

où $(\lambda_1)_\theta$ est la plus petite valeur propre de $K_\theta$.

Nous avons donc

$$
\frac{d}{dt} \mathcal{L}(\theta(t))
= - \| \nabla_\theta \mathcal{L}(\theta(t)) \|^2
= - \langle r_{\theta(t)}, K_{\theta(t)} r_{\theta(t)} \rangle
\leq - (\lambda_1)_{\theta(t)} \| r_{\theta(t)} \|^2
$$
$$
= - (\lambda_1)_{\theta(t)} \sum_{i=1}^{n} (f_\theta(x_i) - y_i)^2
= -2 (\lambda_1)_{\theta(t)} \mathcal{L}(\theta(t))
$$

ce qui implique :  
$$
\mathcal{L}(\theta(t)) \leq \mathcal{L}(\theta(0)) \exp\left(- \int_0^t (\lambda_1)_{\theta(s)} \, ds \right)
$$

**lemme de Gronwall**  
En effet, on a  
$$
\exp\left(\int_0^t (\lambda_1)_{\theta(s)} \, ds\right) \left[ \frac{d}{dt} \mathcal{L}(\theta(t)) + (\lambda_1)_{\theta(t)} \mathcal{L}(\theta(t)) \right] \leq 0
$$

$$
= \frac{d}{dt} \left[ \exp\left(\int_0^t (\lambda_1)_{\theta(s)} \, ds\right) \mathcal{L}(\theta(s)) \right]
$$

$$
\Rightarrow \exp\left(\int_0^t (\lambda_1)_{\theta(s)} \, ds\right) \mathcal{L}(\theta(t)) \leq \exp\left(\int_0^0 (\lambda_1)_{\theta(s)} \, ds\right) \mathcal{L}(\theta(0))
= 1 \cdot \mathcal{L}(\theta(0))
$$

$$
\Rightarrow \mathcal{L}(\theta(t)) \leq \exp\left(- \int_0^t (\lambda_1)_{\theta(s)} \, ds\right) \mathcal{L}(\theta(0))
$$

**Rem.** : Si $(\lambda_1)_{\theta(t)} \simeq \lambda_1$ constante alors  
$$
\mathcal{L}(\theta(t)) \leq \mathcal{L}(\theta(0)) \exp(-\lambda_1 t)
$$

décroissance exponentielle avec taux $\lambda_1$.

Matrice $K_\theta = \left( \langle \nabla_\theta f_\theta(x_i), \nabla_\theta f_\theta(x_j) \rangle \right) \in \mathcal{M}_n(\mathbb{R})$  
est appelée **noyau tangent**

---

### 2) Méthode PINN direct

Méthode de résolution **Galerkin classique**

On cherche une solution approchée telle que  
$$
\forall t \geq 0 \quad u(t, \cdot) \in V_h = \left\{ \sum_{j=1}^{r} \theta_j \phi_j,\ \theta \in \mathbb{R}^r \right\}
$$

Donc  
$$
u(t,x) = \sum_{j=1}^{r} \theta_j(t) \, \phi_j(x)
$$

On réalise ensuite une discrétisation en temps :  
on cherche $u^n = \sum_{j=1}^{r} \theta_j^n \, \phi_j$ solution approchée au temps $t^n$ satisfaisant :

$$
\left( \frac{u^{n+1} - u^n}{\Delta t} - A u^n , \tau \right) = 0 \quad \forall \tau \in V_h
$$

Soit encore :

$$
(u^{n+1} - u^n + \Delta t A u^n, \tau) = 0 \quad \forall \tau \in V_h
$$
$$
\Leftrightarrow \left( \sum_j \theta_j^{n+1} \phi_j , \phi_i \right)
- \left( \sum_j \theta_j^n \phi_j , \phi_i \right)
- \Delta t \left( \sum_j \theta_j^n A \phi_j , \phi_i \right)
= 0 \quad \forall i \in \{1, \dots, r\}
$$
$$
\Leftrightarrow \sum_{j=1}^{r} (\phi_j, \phi_i) \, \theta_j^{n+1}
- \sum_{j=1}^{r} (\phi_j, \phi_i) \, \theta_j^n
- \Delta t \sum_{j=1}^{r} (A \phi_j, \phi_i) \, \theta_j^n = 0
\quad \forall i \in \{1, \dots, r\}
$$

$$
\Leftrightarrow \quad M \theta^{n+1} - M \theta^n - \Delta t A \theta^n = 0
$$

avec $M = \left( (\phi_i, \phi_j) \right)$ matrice de masse,  
et $A = \left( (A \phi_j, \phi_i) \right)$

---

**Rem.** : $u^{n+1}$ est la projection **orthogonale** de $u^n + \Delta t A u^n$ sur $V_h$

$$
u^{n+1} = \mathop{\arg\min}_{u \in V_h} \left\| u - \left( u^n + \Delta t A u^n \right) \right\|^2
$$

**PINN direct.** Même stratégie en remplaçant $V_h$ par  
$$
\mathcal{M}_h = \left\{ \text{NN}_\theta(x),\ \theta \in \mathbb{R}^p \right\}
$$

La solution approchée est donc de la forme  
$$
u(t,x) = \text{NN}_{\theta(t)}(x)
$$

On discrétise en temps : on cherche  
$$
u^n = \text{NN}_{\theta^n},\ \text{solution approchée au temps } t^n
$$

On définit $\theta^{n+1}$ ainsi :  
$$
\theta^{n+1} = \arg\min_{\theta \in \mathbb{R}^p} \left\| \text{NN}_\theta - \left( u^n + \Delta t A u^n \right) \right\|_{L^2}^2
$$
$$
= \arg\min_{\theta \in \mathbb{R}^p} \int_{\Omega} \left( \text{NN}_\theta(x) - \left( \hat{u}^n(x) + \Delta t \, A \hat{u}^n(x) \right) \right)^2 dx
$$

où l’intégrale est calculée par une méthode de Monte Carlo.

La donnée initiale est définie par :

$$
\theta^0 = \arg\min_{\theta \in \mathbb{R}^p} \left\| I(\text{NN}_\theta) \right\|^2
$$
$$
= \arg\min_{\theta \in \mathbb{R}^p} \int_{\Omega} \left\| \text{NN}_\theta(x) - u_{\text{ini}}(x) \right\|^2 dx
$$

**Rem.** : on effectue un apprentissage à chaque itération en temps.  
Pour accélérer l’optimisation, on initialise les paramètres par les paramètres obtenus au temps précédent.

---

### 3) Méthode Neural Galerkin

- **PINN direct** : discrétise en temps puis optimisation  
- **Neural Galerkin** : optimiser puis discrétiser en temps

*(Schéma avec trajectoire réelle $x(t)$, trajectoire approchée $\tilde{x}(t) = U(\theta(t))$, espace d'approximation $\mathcal{M}$, dynamique projetée)*

![[image-17.png|493x329]]

**Projection d’une dynamique (dim finie)**

$$
x'(t) = F(x(t))
$$

On veut regarder cette dynamique seulement sur  
$$
\mathcal{M} = \left\{ U(\theta),\ \theta \in \mathbb{R}^p \right\}
\quad : \text{on souhaite déterminer une solution approchée de la forme } \tilde{x}(t) = U(\theta(t))
$$
$$
\tilde{x}(t) \in \mathcal{M} \qquad \tilde{x}'(t) = P_{T_{\tilde{x}(t)} \mathcal{M}} F(\tilde{x}(t))
$$

où $T_{\tilde{x}(t)} \mathcal{M}$ est l’espace tangent à $\mathcal{M}$.

On a  
$$
T_{U(\theta)} \mathcal{M} = \left\{ \sum_{i=1}^p \partial_{\theta_i} U(\theta) \, s_i,\ s \in \mathbb{R}^p \right\}
= \left\{ U'(\theta) s,\ s \in \mathbb{R}^p \right\}
$$

avec  
$$
U'(\theta) = \left[ \partial_{\theta_1} U(\theta), \dots, \partial_{\theta_p} U(\theta) \right] \quad \text{matrice Jacobienne de } U
$$

*(Schéma en bas à droite : vecteurs tangents à $U(\theta)$ dans $\mathcal{M}$)

![[image-18.png|427x268]]

On a donc *(par déf. de la projection)* :  
$$
(\tilde{x}'(t) - F(\tilde{x}(t)),\ s) = 0 \quad \forall s \in T_{\tilde{x}(t)} \mathcal{M}
$$

$$
\Leftrightarrow\ (\tilde{x}(t) = U(\theta(t)))\ \Rightarrow\ \left( (U(\theta(t)))' - F(U(\theta(t))),\ U'(\theta(t)) s \right) = 0 \quad \forall s \in \mathbb{R}^p
$$

$$
\Leftrightarrow\ \left( U'(\theta(t)) \theta'(t) - F(U(\theta(t))),\ U'(\theta(t)) s \right) = 0
$$

$$
\Leftrightarrow\ \left( U'(\theta(t))^T \left( U'(\theta(t)) \theta'(t) - F(U(\theta(t))) \right),\ s \right) = 0 \quad \forall s \in \mathbb{R}^p
$$

$$
\Leftrightarrow\ \boxed{ \left( U'(\theta(t))^T U'(\theta(t)) \right) \theta'(t) = U'(\theta(t))^T F(U(\theta(t))) }
$$

**En dimension infinie** : $x(t) \in L^2(\Omega)$

et $\mathcal{M} \subset L^2(\Omega)$ de dimension finie.

$$
U(\theta) \in L^2(\Omega)
$$

$$
U'(\theta) = \left[ \partial_{\theta_1} U(\theta), \dots, \partial_{\theta_p} U(\theta) \right] \in \left( L^2(\Omega) \right)^p
$$

**Projection en norme $L^2$** :

$$
\int_{\Omega} \left( U'(\theta(t)) \theta'(t) - F(U(\theta(t))) \right) \cdot U'(\theta(t))\, s \, dx = 0
$$

$$
\Leftrightarrow \left( \int_{\Omega} U'(\theta(t))^T U'(\theta(t)) \, dx \right) \theta'(t)
= \int_{\Omega} U'(\theta(t))^T F(U(\theta(t))) \, dx
$$

**Dans le cas où le modèle est linéaire** :

$$
U(\theta) = \sum_{j=1}^{J} \theta_j \, \phi_j
$$

$$
U'(\theta) = \left[ \phi_1, \dots, \phi_J \right]
$$

$$
U'(\theta)^\top U'(\theta) =
\begin{bmatrix}
\phi_1 \\
\vdots \\
\phi_J
\end{bmatrix}
\begin{bmatrix}
\phi_1 & \cdots & \phi_J
\end{bmatrix}
=
\begin{bmatrix}
\phi_1^2 & \cdots & \phi_1 \phi_J \\
\vdots & \ddots & \vdots \\
\phi_J \phi_1 & \cdots & \phi_J^2
\end{bmatrix}
= \left( \langle \phi_i, \phi_j \rangle \right)
$$

On obtient  
$$
\left( \int_{\Omega} \phi_i \cdot \phi_j \right) \theta_j'(t) = \left( \int_{\Omega} \phi_i \cdot F\left( \sum_j \theta_j(t) \phi_j \right) \right)
\quad \text{(sommation sur } j)
$$

$$
\underbrace{\left( \int_{\Omega} \phi_i \cdot \phi_j \right)}_{= M}
$$

C’est la version continue de la méthode de Galerkin classique.

---

**Méthode Neural Galerkin** : on discrétise en temps  
**La formulation générale** :

$$
\left( \int_{\Omega} U'(\theta_n)^T U'(\theta_n) \right)
\left( \frac{\theta^{n+1} - \theta^n}{\Delta t} \right)
= \left( \int_{\Omega} U'(\theta_n)^T F(U(\theta_n)) \right)
$$

Avec  
$$
U(\theta) = \text{NN}_\theta
\quad\text{et}\quad
U'(\theta) = \left( \nabla_\theta \text{NN}_\theta \right)^\top
$$

$$
\left( \int_\Omega U'(\theta)^\top U'(\theta) \right)
= \left( \int_\Omega \left( \nabla_\theta \text{NN}_\theta(x) \right) \left( \nabla_\theta \text{NN}_\theta(x) \right)^\top dx \right)
\in \mathcal{M}_p(\mathbb{R})
$$

$$
\Rightarrow \text{système linéaire à résoudre à chaque itération}
$$

