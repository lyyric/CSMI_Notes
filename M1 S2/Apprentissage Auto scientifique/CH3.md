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

![[image-3.png|180x179]]

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

![[image-4.png|208x214]]

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

![[image-5.png]]

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

![[image-6.png]]

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
![[image-7.png]]

On échantillonne dans un rectangle contenant $\Omega$, puis on ne garde que les points tels que $\phi(x) < 0$.  

**Exemple** :  

$$
\Omega = D(0,R)
$$

$$
\phi(x) = \| x \| - R
$$

*(fonction distance signée)*

![[image-8.png]]
