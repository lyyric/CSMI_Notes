Chapitre 2 : Apprentissage des dynamiques temporelles

**Question** : Étant données des observations $x_1, \dots, x_n$ à différents temps $t_1, \dots, t_n$,  
on veut prédire $x_{n+1}$ au temps $t_{n+1}$.  

## I) Apprentissage par réseaux récurrents

**Première approche** : Chercher une expression du type  

$$
\tilde{x}_{n+1} = f_{\theta} (x_n)
$$

avec un réseau de neurones.  

**Hypothèse** : Dynamique sans mémoire (Markovienne).  

Pour garder en mémoire la trajectoire, on ajoute une variable latente $h_n$ :

$$
(\tilde{x}_{n+1}, h_{n+1}) = f_{\theta} (x_n, h_n)
$$

(information sur l'historique)

**Principe d'apprentissage** : On applique le réseau à toute une trajectoire et on minimise :

$$
J(\theta) = \sum_{i=2}^{m+1} (\tilde{x}_i - x_i)^2
$$

![[Pasted image 20250207085307.png]]

**→ Apprend à faire un décalage.**  


**Exemple de cellules de réseaux récurrents :**  

- **Cellule d’Elman**  

$$
h_{n+1} = \sigma \left( W_h x_n + U_h h_n + b_h \right)
$$

$$
\tilde{x}_{n+1} = \sigma \left( W_x x_n + U_x h_{n+1} + b_x \right)
$$

(paramètres $\theta$, $f_{\theta} (x, h)$)

- **Cellules LSTM, GRU**  (G : Garde) 

"oubli du passé proche"  
pour éviter des problèmes d’apprentissage.  

## **II) Apprentissage de champs de vecteurs**  

**Observation issue de la physique, de la biologie, quantitative, dynamique régulière**  

**Stratégie** :  

1) **Apprentissage d’un modèle** : on cherche un champ de vecteurs  

$$
f_{\theta} : \mathbb{R}^d \to \mathbb{R}^d
$$

tel que l’équation différentielle  

$$
x'(t) = f_{\theta} (x(t))
$$

explique le mieux les données.  

2) **Utilisation d’un schéma numérique** pour résoudre l’équation différentielle  
(par exemple, schéma d’Euler, Runge-Kutta).  

### **1) Principe**  

Étant données des observations $x_1, \dots, x_m$ aux temps $t_1, t_2, \dots, t_m$, on suppose  
que l'on connaît les vitesses $v_1, \dots, v_m$ en ces mêmes temps.  

On cherche $f_{\theta} (x)$ qui rende minimal  

$$
J(\theta) = \sum_{i=1}^{m} \| v_i - f_{\theta} (x_i) \|^2
$$
![[Pasted image 20250207091318.png|400]]

Sous l’hypothèse que $f_{\theta} (x)$ est $C^1$ et donc localement Lipschitzienne, l’équation différentielle  

$$
x'(t) = f_{\theta} (x(t))
$$

admet des solutions localement pour toute donnée initiale (**théorème de Cauchy-Lipschitz**).  

$\implies$ **Fonction d’approximation $C^1$ en réseaux de neurones.**  

**Remarque** : Si on n’a pas accès aux vecteurs vitesses,  
alors on suppose les données sont issues de trajectoires  
en temps suffisamment proches  

$$
t_i = t_1 + (i-1) \Delta t \quad \forall i \in [[ 1, m+1 ]]
$$

On fait l’estimation par **différence finie**  

$$
v_i \approx \frac{x_{i+1} - x_i}{\Delta t} \quad \forall i \in [[ 1, m ]]
$$

![[Pasted image 20250207092323.png|400]]

**La fonction coût se réécrit** :  

$$
J(\theta) = \sum_{i=1}^{m} \left( \frac{x_{i+1} - x_i}{\Delta t} - f_{\theta} (x_i) \right)^2
$$

$$
= \sum_{i=1}^{m} \frac{1}{\Delta t^2} \left( x_{i+1} - \left( x_i + \Delta t f_{\theta} (x_i) \right) \right)^2
$$

$$
= \sum \Phi_{\Delta t}^{E} (x_i)
$$

Où  

$$
\Phi_{\Delta t}^{E} (x) = x + \Delta t f_{\theta} (x)
$$

est le flot discret associé au schéma d’Euler.  

- **Apprentissage** sur plusieurs trajectoires de manière à couvrir le plan de phase.  
- **Champ de vecteurs** valide uniquement dans la région délimitée par les trajectoires.  

### **2) Modèles symboliques**  

**But** : obtenir des modèles avec une expression analytique (**formule**).  

**Définition** : La méthode **SINDy** (*Sparse Identification of Nonlinear Dynamics*)  
consiste à chercher $f_{\theta} (x)$ comme une combinaison linéaire de  
fonctions données $g_1(x), \dots, g_p(x)$ :  
$$
g_i : \mathbb{R}^d \to \mathbb{R}^d
$$
$$
f_{\theta} (x) = \sum_{i=1}^{p} \theta_i g_i (x) = \Theta \cdot g(x)
$$
avec les paramètres $\Theta \in \mathbb{R}^p$, de sorte à minimiser  
$$
J(\theta) = \sum_{i=1}^{m} \left\| v_i - \sum_{i=1}^{p} \theta_i g_i (x) \right\|^2 + \lambda \| \Theta \|_1
$$
avec $\lambda > 0$ et  
$$
\| \Theta \|_1 = \sum_{i=1}^{p} | \theta_i |
$$

**Remarque** (*régularisation $L^1$ ou LASSO - Least Absolute Shrinkage and Selection Operator*)  
Elle a pour effet de sélectionner des vecteurs $\theta$ **parcimonieux** (*peu de coefficients non nuls*).  

Considérons le problème de minimisation :  

$$
J(\theta) = \frac{1}{2} (\gamma - \theta)^2 + \lambda |\theta|
$$

$$
J(\theta) =
\begin{cases}
\frac{1}{2} (\gamma - \theta)^2 + \lambda \theta, & \text{si } \theta \geq 0 \\
\frac{1}{2} (\gamma - \theta)^2 - \lambda \theta, & \text{si } \theta < 0
\end{cases}
$$

Sa dérivée (*en $\theta \neq 0$*) :  

$$
J'(\theta) =
\begin{cases}
(\theta - \gamma) + \lambda, & \text{si } \theta > 0 \\
(\theta - \gamma) - \lambda, & \text{si } \theta < 0
\end{cases}
$$

Dans le cas où $|\gamma| < \lambda$, alors  

$$
-\lambda < \gamma < \lambda
$$

$\Rightarrow$ $J'(\theta) > 0$ pour tout $\theta > 0$  

$\Rightarrow$ $J'(\theta) < 0$ pour tout $\theta < 0$  

$\Rightarrow$ donc le minimum de $J$ est atteint en $\theta^* = 0$.  

Dans le cas où $\gamma > \lambda$, donc $\gamma - \lambda > 0$  
$\Rightarrow$ le minimum est atteint en $\theta^* = \gamma - \lambda$.  

![[Pasted image 20250207095224.png|400]]

Dans le cas où $\gamma < -\lambda$, donc $\gamma + \lambda < 0$  
$\Rightarrow$ le minimum est atteint en $\theta^* = \gamma + \lambda$.  

![[Pasted image 20250207095239.png|400]]

Donc le minimum est :  

$$
\theta^*_{\lambda} (\gamma) =
\begin{cases}
\gamma + \lambda & \text{si } \gamma < -\lambda \\
0 & \text{si } |\gamma| < \lambda \\
\gamma - \lambda & \text{si } \gamma > \lambda
\end{cases}
$$

![[Pasted image 20250207095301.png|300]]

**Sans pénalisation** ($\lambda = 0$), le minimum est atteint en  

$$
\theta^*_0 (\gamma) = \gamma
$$

Quand $|\theta^*_0 (\gamma)| \leq \lambda$  
$\Rightarrow$ $\theta^*_{\lambda} (\gamma) = 0$ (**sélection**)  

Quand $|\theta^*_0 (\gamma)| > \lambda$  
$\Rightarrow$ $|\theta^*_{\lambda} (\gamma)| = |\gamma| - \lambda$ (**shrinkage**)  

**Remarque** : **Algorithme STLSQ** (*Soft-thresholding - Least Square*)  

- **Problème** : $J$ n’est pas dérivable en $\theta = 0$  
  $\Rightarrow$ algorithme de gradient à adapter  

- **Alternative** : à chaque itération, on effectue une étape de méthode de gradient  
  avec **pénalisation ridge**, puis on met à zéro les paramètres plus petits que $\lambda$.  

re:

$x_1, \dots, x_m$  
$v_1, \dots, v_m$  

**But :** déterminer $f_\theta$ tel que $x'(t) = f_\theta(x(t))$  

$$
\rightarrow \min \sum_{i=1}^{m} \| v_i - f_\theta(x_i) \|^2
$$
$$
\rightarrow f_\theta : \text{réseau de neurones}
$$

**SINDy :**  
$$
f_\theta(x) = \sum_{h=1}^{n} \theta_h g_h(x)
$$

$$
\{ x, x^2, x^3, \sin(2x), \dots, \exp(-x) \}.
$$

**Remarque :** (*régression symbolique*).  

On considère l’arbre des expressions construites de  
fonctions élémentaires ($x, \sin x, \exp x, \dots$) et des  
opérateurs binaires ($+, \times$, min, max, …).  

$$
\theta_1 x^2 + \theta_2 \sin(\theta_3 x + \theta_4 \exp(-x))
$$

```txt
	+
θ1 / \ θ2
  *   sin
 / \    \
x   x    +
	  θ₃/ \ θ₄
	  x   exp(-x)
```

**Méthode d’optimisation de type algorithme génétique ou renforcement.**  

## III) Stabilité en temps  

**Question :** une fois $f_\theta(x)$ appris, est-ce que la dynamique $x'(t) = f_\theta(x(t))$ a des bonnes propriétés en temps long quand on utilise le schéma numérique ?  

### 1) Préserver des structures  

**Systèmes hamiltoniens :** $x(t) \in \mathbb{R}^{2d}$ satisfait un système hamiltonien si  

$$
x'(t) = J \nabla H(x(t))
$$

avec $H : \mathbb{R}^{2d} \to \mathbb{R}$ (Hamiltonien) et  

$$
J =
\begin{pmatrix}
0 & \text{Id} \\
-\text{Id} & 0
\end{pmatrix}
$$

**Remarque :** $J$ est anti-symétrique ($J^T = -J$) et $J^2 = -\text{Id}$ ($J^{-1} = -J = J^T$).  

Notant  
$$
x(t) =
\begin{pmatrix}
q(t) \\
p(t)
\end{pmatrix}
\in \mathbb{R}^{2d}
$$
avec $q(t), p(t) \in \mathbb{R}^d$, on a  
$$
H(x) = H(q, p)
$$

et donc  
$$
\nabla H(x) =
\begin{pmatrix}
\nabla_q H(q, p) \\
\nabla_p H(q, p)
\end{pmatrix}.
$$

Le système se réécrit :  
$$
\begin{cases}
q'(t) = \nabla_p H(q(t), p(t)) \\
p'(t) = -\nabla_q H(q(t), p(t))
\end{cases}
$$

**Exemple :** ($d = 1$)  

$$
H(q, p) = \frac{p^2}{2} + V(q)
$$

On a donc  

$$
\nabla H(q, p) =
\begin{pmatrix}
V'(q) \\
p
\end{pmatrix}
$$

Et donc  

$$
\begin{cases}
q'(t) = p(t) \\
p'(t) = -V'(q(t))
\end{cases}
$$

$\implies$ Équations d'une particule soumise à une force  
dérivant du potentiel $V$.  

$q$ : position  
$p$ : impulsion ($= m v$)  

$$
H(q, p) = \frac{p^2}{2} + V(q)
$$

- **énergie cinétique** (sous $\frac{p^2}{2}$)
- **énergie potentielle** (sous $V(q)$)

**Définition :** On appelle **flot** d'une équation différentielle l'application  

$$
\Phi_t : x(0) \in \mathbb{R}^{2d} \to x(t) \in \mathbb{R}^{2d}
$$

solution au champ $f$.  

![[image.png]]

Cette application vérifie :  

$$
\Phi_0 = \text{Id}
$$
$$
\Phi_t \circ \Phi_s = \Phi_{t+s} \quad \forall t, s > 0
$$

(*propriété de semi-groupe*).  

**Propriété :** Le flot $\Phi_t$ d'une équation hamiltonienne préserve :  

- **Le Hamiltonien** :  
  $$
  H(\Phi_t(x_0)) = H(x_0) \quad \forall t
  $$  
- **La structure symplectique**, i.e.  
  $$
  \nabla \Phi_t(x_0)^T J \nabla \Phi_t(x_0) = J
  $$
- **Le volume** :  
  $$
  \text{vol} (\Phi_t(A)) = \text{vol} (A) \quad \forall t \geq 0.
  $$

**Preuve :** Hamiltonien  

$$
\frac{d}{dt} \left[ H(\Phi_t(x_0)) \right] = \frac{d}{dt} \left[ H(x(t)) \right]
$$

$$
= \langle \nabla H(x(t)), x'(t) \rangle
$$

$$
= \langle \nabla H(x(t)), J \nabla H(x(t)) \rangle
$$

$$
\equiv 0 \quad \text{car } J \text{ antisymétrique}
$$

$$
\left[ (a, J a) = (J a, a) = -(T a, a) = -(a, T a) \right]
$$

$$
= 0 \quad \forall a
$$

**• Structure symplectique**  

On a $\Phi_t(x_0) = x(t)$, donc  

$$
\frac{d}{dt} \Phi_t(x_0) = J \nabla H(\Phi_t(x_0))
$$

D’où  
$$
\frac{d}{dt} \nabla \Phi_t(x_0) = \nabla \frac{d}{dt} \Phi_t(x_0)
$$
$$
= J \nabla^2 H(\Phi_t(x_0)) \nabla \Phi_t(x_0)
$$

On calcule ensuite  

$$
\frac{d}{dt} \left[ \nabla \Phi_t(x_0)^T J \nabla \Phi_t(x_0) \right]
$$
$$
= \frac{d}{dt} \nabla \Phi_t(x_0)^T J \nabla \Phi_t(x_0) + \nabla \Phi_t(x_0)^T J \frac{d}{dt} \nabla \Phi_t(x_0)
$$

$$
= \nabla \Phi_t(x_0)^T \nabla^2 H(x(t)) J^T J \nabla \Phi_t(x_0)
+ \nabla \Phi_t(x_0)^T J J \nabla^2 H(x(t)) \nabla \Phi_t(x_0)
$$
$$
= 0
$$

**Donc**  
$$
\nabla \Phi_t(x_0)^T J \nabla \Phi_t(x_0) = \nabla \Phi_0(x_0)^T J \nabla \Phi_0(x_0)
$$
$$
= \text{Id} J \text{Id}
$$
$$
= J
$$

Encadré :  
$$
\Phi_0(x) = x
$$
$$
\nabla \Phi_0(x) = \text{Id}
$$

**• Volume**  

$A \subset \mathbb{R}^{2d}$  
$$
\text{vol}(\Phi_t(A))
$$
$$
= \int_{\Phi_t(A)} dy
$$
$$
= \int_A |\det \nabla \Phi_t(x)| dx
$$
par
$$
y = \Phi_t(x)
$$
$$
dy = |\det \nabla \Phi_t(x)| dx
$$

![[image-1.png]]

Or  
$$
\det(J) = \det (\nabla \Phi_t(x)^T J \nabla \Phi_t(x))
$$

$$
\det(J) = \det(\nabla \Phi_t(x))^2 \det(J)
$$
$$
\Rightarrow \det(\nabla \Phi_t(x))^2 = 1
$$
$$
\Rightarrow |\det \nabla \Phi_t(x)| = 1
$$

On en déduit  

$$
\text{vol}(\Phi_t(A)) = \int_{\Phi_t(A)} dx = \text{vol}(A).
$$

**Remarque :** Inversement, il est possible de montrer que tout $t \mapsto \Phi_t$ avec $\Phi_t$  
symplectique contient localement le flot d’un système Hamiltonien.  

### **Schémas symplectiques**  

**Propriété :** Le schéma numérique suivant est symplectique :  

$$
q_{n+1} = q_n + \Delta t \nabla_p H(q_{n+1}, p_n)
$$

$$
p_{n+1} = p_n - \Delta t \nabla_q H(q_{n+1}, p_n)
$$

(*Schéma semi-implicite*), c’est-à-dire  

$$
(q_{n+1}, p_{n+1}) = \Phi_{\Delta t} (q_n, p_n)
$$

est une transformation symplectique.  

**Remarque :**  

$$
H(q, p) = H_1(p) + H_2(q) \quad \text{(Hamiltonien séparable)}
$$

$$
\nabla_p H(q, p) = \nabla H_1(p)
$$
$$
\nabla_q H(q, p) = \nabla H_2(q)
$$

$$
\Rightarrow q_{n+1} = q_n + \Delta t \nabla H_1(p_n)
$$
$$
p_{n+1} = p_n - \Delta t \nabla H_2(q_{n+1})
$$

Le schéma a une résolution explicite.  

**Preuve :** **Calcul de** $\nabla \Phi_{\Delta t}$  

$$
\nabla \Phi_{\Delta t} =
\begin{pmatrix}
\dfrac{\partial q_{n+1}}{\partial q_n} & \dfrac{\partial q_{n+1}}{\partial p_n} \\
\dfrac{\partial p_{n+1}}{\partial q_n} & \dfrac{\partial p_{n+1}}{\partial p_n}
\end{pmatrix}
$$

**Pour simplifier,**  
induis le cas séparable  

$$
\frac{\partial q_{n+1}}{\partial q_n} = 1
$$

$$
\frac{\partial q_{n+1}}{\partial p_n} = \Delta t \nabla^2 H_1(p_n)
$$

$$
\frac{\partial p_{n+1}}{\partial q_n} = -\Delta t \nabla^2 H_2(q_{n+1}) \frac{\partial q_{n+1}}{\partial q_n}
$$

$$
\frac{\partial p_{n+1}}{\partial p_n} = 1 - \Delta t \nabla^2 H_2(q_{n+1}) \frac{\partial q_{n+1}}{\partial p_n}
$$

Donc  

$$
\nabla \Phi_{\Delta t}(x) =
\begin{pmatrix}
1 & \Delta t \nabla^2 H_1(p_n) \\
-\Delta t \nabla^2 H_2(q_{n+1}) & 1 - \Delta t^2 \nabla^2 H_2(q_{n+1}) \nabla^2 H_1(p_n)
\end{pmatrix}
$$

On montre que  

$$
\nabla \Phi_{\Delta t}(x)^T J \nabla \Phi_{\Delta t}(x)
$$
$$
=
\begin{pmatrix}
1 & -\Delta t \nabla^2 H_2 \\
\Delta t \nabla^2 H_1 & 1 - \Delta t^2 \nabla^2 H_2 \nabla^2 H_1
\end{pmatrix}
\begin{pmatrix}
-\Delta t \nabla^2 H_2 & 1 - \Delta t^2 \nabla^2 H_2 \nabla^2 H_1 \\
-1 & -\Delta t \nabla^2 H_1
\end{pmatrix}
$$
$$
= J
$$

**Remarque :** les schémas symplectiques préservent le **volume**  
(*comme dans le cas continu*).  

![[image-2.png]]

**Préserve le Hamiltonien ?**  

Cas particulier ($d = 1$) :  
$$
H(q, p) = \frac{q^2}{2} + \frac{p^2}{2}
$$

(*Oscillateur harmonique*)  

$$
q_{n+1} = q_n + \Delta t p_n
$$
$$
p_{n+1} = p_n - \Delta t q_{n+1}
$$
et  
$$
H(q_{n+1}, p_{n+1}) = \frac{1}{2} \left[ (q_n + \Delta t p_n)^2 + (p_n - \Delta t q_{n+1})^2 \right]
$$
$$
= \frac{1}{2} \left[ q_n^2 + 2 \Delta t q_n p_n + \Delta t^2 p_n^2 + p_n^2 - 2 \Delta t p_n q_{n+1} + \Delta t^2 q_{n+1}^2 \right]
$$
$$
= \frac{1}{2} (q_n^2 + p_n^2) + \Delta t p_n (q_n - q_{n+1}) + \frac{\Delta t^2}{2} q_{n+1}^2 + \frac{\Delta t^2}{2} p_{n}^2
$$
$$
= H(q_n, p_n) - \Delta t^2 p_n^2 + \frac{\Delta t^2}{2} q_{n+1}^2 + \frac{\Delta t^2}{2} p_{n}^2
$$
$$
= \frac{1}{2} H(q_n, p_n) + \frac{\Delta t^2}{2} (q_{n+1}^2 - p_n^2)
$$
On  
$$
q_{n+1} p_{n+1} = q_{n+1} (p_n - \Delta t q_{n+1})
$$
$$
= q_{n+1} p_n - \Delta t q_{n+1}^2
$$
$$
= (q_n + \Delta t p_n) p_n - \Delta t q_{n+1}^2
$$
$$
= q_n p_n + \Delta t p_n^2 - \Delta t q_{n+1}^2
$$

Donc  
$$
H(q_{n+1}, p_{n+1}) = H(q_n, p_n) + \frac{\Delta t^2}{2} (q_n p_n - q_{n+1} p_{n+1})
$$

Soit  
$$
H(q_{n+1}, p_{n+1}) + \frac{\Delta t^2}{2} q_{n+1} p_{n+1} = H(q_n, p_n) + \frac{\Delta t^2}{2} q_n p_n = \tilde{H}(q_n, p_n)
$$

- **Préservation d’un Hamiltonien modifié !**  
- **Garantie une stabilité en temps long.**  

### **Hamiltonian Neural Network (HNN)**  

- Détermine le champ de vecteurs sous la forme  

$$
f_\theta (x) = J \nabla H_\theta (x)
$$

  avec $H_\theta : \mathbb{R}^{2d} \to \mathbb{R}$, réseau de neurones.  

- Utilisation d’un schéma symplectique pour la simulation de  

$$
x'(t) = J \nabla H_\theta (x)
$$
