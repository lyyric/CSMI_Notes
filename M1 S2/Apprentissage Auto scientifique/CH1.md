# Chapitre 1 : Introduction  

Apprentissage : apprendre un modèle paramétrique pour expliquer des données, à l’aide d’outils d’optimisation.  

## I Régression linéaire  

Considérant des données $(x_1, y_1), \dots, (x_n, y_n) \in X \times Y$, variables d’entrée/sortie (input/output), on cherche un modèle de la forme $y = f(x)$ qui explique au mieux les données dans le but de faire de nouvelles prédictions.  

$f$ : fonction de régression  
$(x_i, y_i)$ : données d’entraînement (training data)  

$X \subset \mathbb{R}^d, Y \subset \mathbb{R}$  

### 1) Modèles paramétriques  

On choisit une famille de modèles paramétriques $f_\theta(x)$ avec $\theta \in \Theta \subset \mathbb{R}^p$ ($p$ paramètres).  

Régression linéaire :  
$f_\theta(x) = \langle w, x \rangle_{\mathbb{R}^d} + b = \langle \theta, \bar{x} \rangle$  

avec $w \in \mathbb{R}^d$ et $b \in \mathbb{R}$. Ici  
$\theta = \begin{pmatrix} w \\ b \end{pmatrix} \in \mathbb{R}^{d+1}$  
et $\bar{x} = \begin{pmatrix} x \\ 1 \end{pmatrix} \in \mathbb{R}^{d+1}$.  

$\rightarrow$ modèle linéaire en $x$ et en $\theta$.  

Remarque : Une famille de modèles plus grande est donnée par  
$f_\theta(x) = \langle \theta, \phi(x) \rangle$  

avec $\phi(x) \in \mathbb{R}^p$, vecteurs des caractéristiques (features) associées à $x$.  

Régression linéaire : $\phi(x) = \begin{pmatrix} x \\ 1 \end{pmatrix}$, avec $p = d+1$.  

Régression polynomiale (ordre 2) :  
$$
\phi(x) = \begin{pmatrix}  
x_1 \\  
\vdots \\
x_d \\  
x_1^2 \\  
x_1 x_2 \\  
\vdots \\
x_d^2 \\  
1  
\end{pmatrix} \in \mathbb{R}^p
$$
avec $p = 1 + d + \frac{d(d+1)}{2}$.  

### 2) Modèle optimal  

On choisit $\theta \in \Theta$ de sorte à minimiser l’erreur quadratique moyenne ("Mean Square Error", MSE) :  
$$
\hat{\theta} = \underset{\theta \in \Theta}{\mathrm{argmin}} \frac{1}{n} \sum_{i=1}^n (y_i - f_\theta(x_i))^2 = \underset{\theta \in \Theta}{\mathrm{argmin}} \, J(\theta)
$$

$J$ est appelée fonction perte ("loss function").  

Formulation mathématique :  $(x_i, y_i) \in \mathbb{R}^d \times \mathbb{R}$ 
$$
J(\theta) = \frac{1}{n} \lVert (y_i - f_\theta(x_i))_{i=1}^n \rVert_{\mathbb{R}^n}^2
$$
$$
= \frac{1}{n} \left\|
\begin{pmatrix}  
y_1 \\  
\vdots \\  
y_n  
\end{pmatrix}  
-  
\begin{pmatrix}  
f_\theta(x_1) \\  
\vdots \\  
f_\theta(x_n)  
\end{pmatrix}  
\right\|_{\mathbb{R}^n}^2
$$
$$
= \frac{1}{n} \left\|  
\begin{pmatrix}  
y_1 \\  
\vdots \\  
y_n  
\end{pmatrix}  
-  
\begin{pmatrix}  
w_1 x_1^{(1)} + \dots + w_d x_1^{(d)} + b \\  
\vdots \\  
w_1 x_n^{(1)} + \dots + w_d x_n^{(d)} + b  
\end{pmatrix}  
\right\|_{\mathbb{R}^n}^2
$$
$$
= \frac{1}{n} \left\|  
\begin{pmatrix}  
y_1 \\  
\vdots \\  
y_n  
\end{pmatrix}  
-  
\begin{pmatrix}  
x_1^{(1)} & \dots & x_1^{(d)} & 1 \\  
\vdots & \ddots & \vdots & \vdots \\  
x_n^{(1)} & \dots & x_n^{(d)} & 1  
\end{pmatrix}  
\begin{pmatrix}  
w_1 \\  
\vdots \\  
w_d \\  
b  
\end{pmatrix}  
\right\|_{\mathbb{R}^n}^2
$$
$$
= \frac{1}{n} \lVert Y - A \theta \rVert_{\mathbb{R}^n}^2
$$

Diagramme : distance entre le modèle et les données en $x_i$ (erreur verticale).  
$f_\theta(x) = \langle w, x \rangle + b$.  
![[Pasted image 20250123221421.png]]

> [!proposition]
> - $J$ admet toujours un minimum.  
> - $\hat{\theta}$ est un minimiseur de $J$ ssi :  
> $$
> \nabla J(\hat{\theta}) = 0
> $$
> $$
> (A^T A) \hat{\theta} = A^T Y
> $$
> - Si $A$ est de rang $p$, alors $A^T A$ est inversible et il y a un unique minimum donné par
> $$
> \hat{\theta} = (A^T A)^{-1} A^T Y
> $$
> - Si $A$ est de rang $< p$, alors il y a une infinité de minimisateurs.  

**Remarque :**  
$$
A =  
\begin{pmatrix}  
x_1^T & 1 \\  
\vdots & \vdots \\  
x_m^T & 1  
\end{pmatrix}  
\in M_{n,p}(\mathbb{R})  
$$
avec $p = n + 1$.  

Dans ce cas, la matrice $A$ est au maximum de rang $n \leq p$.  
$\rightarrow$ Infinité de minimisateurs.  

**Preuve partielle :** *Voir cours d’optimisation.*  

En développant :  
$$
J(\theta, h) = \frac{1}{n} \lVert Y - A(\theta + h) \rVert^2  
$$
$$
=\frac{1}{n} \lVert Y - A \theta - A h \rVert^2
$$
$$
= \frac{1}{n} \left[ \lVert Y - A \theta \rVert^2 - 2 \langle Y - A \theta, A h \rangle + \lVert A h \rVert^2 \right]
$$
$$
= \frac{1}{n} \left[ J(\theta) + 2 \langle A^T (A \theta - Y), h \rangle + \frac{1}{2}\cdot 2 \langle A^T A h,  h \rangle \right]
$$
$$
= \frac{1}{n} \left[ J(\theta) + 2 \langle \nabla J(\theta), h \rangle + \frac{1}{2} \langle \nabla² J(\theta) , h \rangle \right]
$$

**On obtient** $\nabla J(\theta)$ et $\nabla^2 J(\theta)$ par identification.  

L’équivalence entre $\nabla J(\hat{\theta}) = 0$ et $\hat{\theta}$ minimum découle du caractère convexe de $J$.  

#### Interprétation probabiliste

Les données $(x_i, y_i)_{i=1}^n$ sont des variables aléatoires indépendantes et identiquement distribuées. On cherche à estimer la loi de $y$ sachant $x$, de densité notée $p(y|x)$.  

Sous forme paramétrique $p_\theta(y|x)$.  

Par indépendance :  
$$
p_\theta(y_1, \dots, y_n | x_1, \dots, x_n) = \prod_{i=1}^n p_\theta(y_i | x_i)
$$

**Vraisemblance (Likelihood)**  
On se donne une famille de lois de probabilité.  

$y | x \sim \mathcal{N}(f_\theta(x), \sigma^2)$  

- Moyenne (espérance) : $f_\theta(x)$  
- Variance : $\sigma^2$  

$$
p_\theta(y_1, \dots, y_n | x_1, \dots, x_n) = \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(y_i - f_\theta(x_i))^2}{2 \sigma^2}\right)
$$

On cherche $\theta$ qui maximise cette quantité, c’est-à-dire telle que la probabilité conditionnelle d’observer les données soit la plus grande.  

Maximiser la vraisemblance $\iff$ Maximiser la log-vraisemblance ($\ln$).  

$$
\ln p_\theta(y_1, \dots, y_n | x_1, \dots, x_n) = \sum_{i=1}^n - \ln \left(\frac{1}{\sqrt{2 \pi \sigma^2}}\right) - \frac{1}{2 \sigma^2} (y_i - f_\theta(x_i))^2
$$
$$
= -\frac{n}{2} \ln(2 \pi \sigma^2) - \frac{1}{2 \sigma^2} \sum_{i=1}^n (y_i - f_\theta(x_i))^2
$$

Maximiser la log-vraisemblance revient à minimiser :
$$
\frac{1}{2 \sigma^2} \sum_{i=1}^n (y_i - f_\theta(x_i))^2
$$

Donc, maximiser la log-vraisemblance revient à résoudre le problème de régression linéaire avec l'erreur quadratique moyenne, sous l'hypothèse d'une loi normale.

**Remarque :** La log-vraisemblance peut s’interpréter comme une distance entre $p_\theta(y|x)$ et la loi $p(y|x)$ associée aux données (divergence de Kullback-Leibler).

-> Minimiser la distance $D_{KL}(p_\theta(y|x), p(y|x))$ .

**Remarque :**  
Si $f_\theta(x) = \langle \theta, \phi(x) \rangle$ avec $\phi(x) \in \mathbb{R}^p$, alors le problème de régression s'écrit :  
$$
J(\theta) = \frac{1}{n} \lVert (y_i - \langle \theta, \phi(x_i) \rangle)_{i=1}^n \rVert^2
$$
$$
= \frac{1}{n} \left\|  
\begin{pmatrix}  
y_1 \\  
\vdots \\  
y_n  
\end{pmatrix}  
-  
\begin{pmatrix}  
\phi(x_1)^T \\  
\vdots \\  
\phi(x_n)^T  
\end{pmatrix}  
\begin{pmatrix}  
\theta_1 \\  
\vdots \\  
\theta_p  
\end{pmatrix}  
\right\|^2
$$

avec 
$$
A =  
\begin{pmatrix}  
\phi(x_1)^T \\  
\vdots \\  
\phi(x_n)^T  
\end{pmatrix}  
\in M_{n,p}(\mathbb{R})
$$

### 3) Résolution

#### Différentes méthodes :

- Résoudre l’équation $A^T A \hat{\theta} = A^T Y$ grâce à la décomposition QR ou SVD (Singular Value Decomposition).  
- Méthode de gradient : on construit une suite $(\theta^{(k)})$ :  
$$
\theta^{(k+1)} = \theta^{(k)} - \eta \nabla J(\theta^{(k)})
$$

avec $\eta > 0$, pas de descente.  

Si $\theta^{(k)} \to \theta^*$, alors $\theta^* = \theta^* - \nabla J(\theta^*) \implies \nabla J(\theta^*) = 0 \implies \theta^*\ \text{est un minimum}$  

**Méthode de gradient stochastique, par paquet (batch)**  

$$
J(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(y_i, \hat{y}_i) \quad \text{avec} \quad \ell(y, \hat{y}) = (y - \hat{y})^2
$$

On optimise seulement sur un sous-ensemble d’indices, différent à chaque itération :  
$$
\theta^{(k+1)} = \theta^{(k)} - \eta \nabla_\theta \left( \frac{1}{|I|} \sum_{i \in I} \ell(y_i, f_\theta(x_i)) \right)
$$

Avec $I \subset \{1, \dots, n\}$ de cardinal fixé.  

### 4) Sur-apprentissage 

Lorsque le nombre de paramètres $p$ est plus grand que $n$ ($p > n$) :  
- Il y a une infinité de minimiseurs. Le problème est sur-paramétré.  

**Propriété :**  
Si $A$ est de rang $n$ ($p > n$) , alors les minimiseurs $\theta^*$ vérifient le problème d’interpolation :  
$$
f_\theta(x_i) = y_i \quad \forall i \in \{1, \dots, n\}.
$$

On passe exactement par les observations :  
- Très bon modèle sur les observations, mais très mauvaises prédictions.  

$\rightarrow$ Modèle très oscillant.  

![[Pasted image 20250123225229.png]]

$$
f_\theta(x) = \langle \theta, \phi(x) \rangle = \sum_{k=0}^{p-1} \theta_k x^k
$$

$$
\phi(x) = \begin{pmatrix} 1 \\ x \\ x^2 \\ \vdots \\ x^{p-1} \end{pmatrix}
$$

**Preuve :**  

$$
A = \begin{pmatrix} A_{n,n} & A_{n,p-n} \end{pmatrix}
$$

Quitte à permuter les colonnes, on peut supposer $A_{n,n}$ inversible.

On prend :  
$$
\theta^* = \begin{pmatrix} (A_{n,n})^{-1} Y \\ 0 \end{pmatrix} \in \mathbb{R}^p
$$

Alors :  
$$
A \theta^* = Y
$$

Le problème est résolu avec un vecteur $\hat{\theta}$ construit de cette manière.

donc
$$
J(\theta^*) = \frac{1}{n} \lVert Y - A \theta^* \rVert^2 = 0 \quad \implies \quad f_{\theta^*}(x_i) = y_i, \, \forall i \in \{1, \dots, n\}.
$$

**Remarque : arrêt précoce (early stopping)**  
Dans une méthode de gradient, on arrête l’algorithme lorsque l’erreur sur les données test commence à augmenter, même si l’erreur sur les données d’entraînement continue de diminuer.

(Graphique illustrant l’erreur sur les données d’entraînement et de test, montrant où arrêter.)

![[Pasted image 20250123225408.png]]

### 5) Méthodes de pénalisation

**Pénalisation ridge :** On considère le problème :  
$$
J(\theta) = \frac{1}{n} \lVert Y - A \theta \rVert^2 + \lambda \lVert \theta \rVert^2
$$  
où $\lambda > 0$ contrôle la pénalisation (les paramètres ne doivent pas être trop grands).

**Prop :**  
Cette fonction admet un unique minimiseur $\theta^* \in \mathbb{R}^p$, caractérisé par :  
$$
\nabla J(\theta^*) = 0 \implies \theta^* = (A^T A + \lambda I_p)^{-1} A^T Y
$$  
**Interprétation probabiliste : régression bayésienne**  

On choisit une loi de probabilité dite *a priori* sur les paramètres et veut estimer une loi *a posteriori*.  

Pour cela, on utilise la formule de Bayes :  
$$
p(\theta | x, y) = \frac{p(y | x, \theta) \, p(\theta | x)}{p(y | x)}
$$

Avec :  
$$
p(y | x, \theta) = \prod_{i} p(y_i | x_i, \theta)
$$
(car $\theta$ et $x$ sont indépendants).  

La formule de Bayes :  
$$
p(A | B) = \frac{p(B | A) \, p(A)}{p(B)}
$$

si
$$
\theta \sim \mathcal{N}(0, \alpha^2 I_d) \quad \text{(loi a priori, vecteur Gaussien)}  
$$

- Moyenne nulle  
- Variance $\alpha^2 I_d$  

$$
y_i | (x_i, \theta) \sim \mathcal{N}(f_\theta(x_i), \sigma^2)  
$$

Alors :  
$$
\theta | (x, y) \sim \mathcal{N}(\hat{\theta}, \Sigma)
$$
Avec :  
$$
\hat{\theta} = \left(A^T A + \frac{\sigma^2}{\alpha^2} I_d \right)^{-1} A^T Y  
$$
$$
\Sigma = \sigma^2 \left(A^T A + \frac{\sigma^2}{\alpha^2} I_d \right)^{-1}
$$

**Remarque :**  
L’espérance $\hat{\theta}$ est la solution du problème de pénalisation ridge.  

