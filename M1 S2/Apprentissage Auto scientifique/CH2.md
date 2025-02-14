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
