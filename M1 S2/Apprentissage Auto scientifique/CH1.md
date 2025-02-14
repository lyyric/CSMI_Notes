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


Continuez à penser à pénalisation : 
$$
J(\theta) = \frac{1}{m} \| Y - A\theta \|^2 + \lambda \| \theta \|^2
$$
$$
\hat{\theta} = \arg\min_{\theta} J(\theta) = \left( \frac{1}{m} A^T A + 2\lambda I d \right)^{-1} A^T Y
$$

**Interprétation probabiliste**

$$
p(\theta) \sim \mathcal{N}(0, \alpha^2 I d)
$$
$$
p(\theta | y_1, \dots, y_m, x_1, \dots, x_m) = \frac{p(y_1, \dots, y_m | \theta) p(\theta)}{p(y_1, \dots, y_m)}
$$

**Remarque (principe du maximum de vraisemblance)**

$$
\hat{\theta} = \arg\max_{\theta} \ln p(\theta | y_1, \dots, y_n, x_1, \dots, x_n)
$$
$$
= \arg\max_{\theta} \ln p(y_1, \dots, y_n | x_1, \dots, x_n, \theta) + \ln p(\theta) - \ln p(y_1, \dots, y_n | x_1, \dots, x_n)
$$
$$
= \arg\max_{\theta} -\frac{n}{2} \ln (2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - f(x_i, \theta))^2
$$
$$
+ \ln\left[ \frac{1}{(2\pi\alpha^2)^{p/2}} \exp \left( -\frac{1}{2\alpha^2} \|\theta\|^2 \right)\right]
$$


$$
\hat{\theta} = \arg\max_{\theta} -\frac{n}{2} \ln (2\pi\sigma^2)
$$
$$
-\frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - f_{\theta}(x_i))^2
$$

$$
-\frac{p}{2} \ln (2\pi\alpha^2)
$$

$$
-\frac{1}{2\alpha^2} \|\theta\|^2
$$

$\implies$ même problème qu’initialement.

### 6) Régression à noyaux

$$
J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (\theta, \phi(x_i)))^2 + \lambda \|\theta\|^2
$$

**On montre**  
$$
\hat{\theta} \in \text{Vect}(\phi(x_i)) \subset \mathbb{R}^p
$$
On note
$$
\hat{\theta} = \sum_{i=1}^{n} \alpha_i \phi(x_i), \quad \alpha_i \in \mathbb{R}
$$
On a alors
$$
f_{\hat{\theta}}(x) = \langle \hat{\theta}, \phi(x) \rangle
$$
$$
= \sum_{i=1}^{n} \alpha_i \langle \phi(x_i), \phi(x) \rangle
$$

En restreignant le problème à $\text{Vect}(\phi(x_i))$, on a
$$
J(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \sum_{j=1}^{n} \alpha_j \langle \phi(x_j), \phi(x_i) \rangle \right)^2
$$
$$
+ \lambda \left\| \sum_{j=1}^{m} \alpha_j \phi(x_j) \right\|^2
$$
Avec
$$
\theta = \sum_{j=1}^{n} \alpha_j \phi(x_j)
$$

**Problème de régression à noyau**  

$$
\widetilde{J} (\alpha) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \sum_{j=1}^{n} \langle \phi(x_j), \phi(x_i) \rangle \alpha_j \right)^2
$$
$$
+ \lambda \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \langle \phi(x_i), \phi(x_j) \rangle \alpha_j
$$
$$
= \frac{1}{n} \| Y - K\alpha \|^2 + \lambda (\alpha, K\alpha)
$$
Avec  
$$
K = \left( \langle \phi(x_i), \phi(x_j) \rangle \right)_{1 \leq i, j \leq n} \in M_n(\mathbb{R})
$$

**Solution du problème :**  

$$
\hat{\alpha} = \frac{1}{n} (K^T K + \lambda K)^{-1} K^T Y
$$

$$
\hat{\theta} = \sum_{i=1}^{m} \hat{\alpha}_i \phi(x_i)
$$
$$
= A^T \hat{\alpha}
$$

**Remarque :**  

Pour définir le problème, il suffit de connaître  
$$
\langle \phi(x), \phi(y) \rangle = k(x, y)
$$

$\implies$ pas besoin de connaître $\phi(x)$. Il suffit de connaître $k(x, y)$.  (kernel trick)

**Remarque :**  

- $k$ est un noyau défini positif $\iff$ 
$(k(x_i, x_j))_{1 \leq i, j \leq n}$ est symétrique et définie positive $\forall (x_i)_{i=1}^{n} \in \mathcal{X}$, $\forall n \in \mathbb{N}^*$

- (Théorème d'Aronszajn) $k$ noyau défini positif  
$\iff \exists H$, espace de Hilbert, $\exists \phi : \mathcal{X} \to H$ , tel que  
$$
\langle \phi(x), \phi(y) \rangle = k(x, y)
$$
- 
$$
f_{\hat{\theta}}(x) = \sum_{j=1}^{n} \hat{\alpha}_j \langle \phi(x_j), \phi(x) \rangle
$$
$$
= \sum_{j=1}^{n} \hat{\alpha}_j k(x_j, x)
$$
$$
\in \text{Vect} \left( k(x_i, \cdot) \right)
$$


**Exemple :**  

1. **Noyau gaussien (RBF Kernel)**  

$$
k(x, y) = \exp \left( -\frac{\| x - y \|^2}{2\sigma^2} \right)
$$
$$
\Rightarrow f_{\hat{\theta}}(x) = \sum_{j=1}^{m} \hat{\alpha}_j \exp \left( -\frac{\| x - x_j \|^2}{2\sigma^2} \right)
$$

2. **Noyau linéaire (Produit scalaire classique)**  

$$
k(x_i, y_j) = \langle x_i, y_j \rangle_{\mathbb{R}^d}
$$
$$
\implies f_{\hat{\theta}}(x) = \sum_{j=1}^{n} \hat{\alpha}_j \langle x_j, y \rangle
$$
$$
= \left\langle \sum_{j=1}^{n} \hat{\alpha}_j x_j, y \right\rangle
$$
$$
= \langle \hat{\theta}, y \rangle
$$

**(Régression linéaire)**

## II) Réseaux de neurones

**Définition :**  
Une couche d’un **réseau de neurones** :
$$
l_i(z) = \sigma (A_i z + b_i)
$$
est la composée d’une fonction affine avec  
$$
A_i \in M_{d_i, d_{i-1}} (\mathbb{R}) \quad \text{et} \quad b_i \in \mathbb{R}^{d_i}
$$
et d’une **fonction non linéaire**, fonction d’activation $\sigma : \mathbb{R} \to \mathbb{R}$, appliquée composante par composante.

Un réseau de neurones est la composée de $p$ couches :
$$
f_{\theta}(x) = l_p \circ \dots \circ l_1(x)
$$
avec
$$
\theta = (A_1, b_1, \dots, A_p, b_p)
$$

- **$p$** : profondeur du réseau (**depth**)  
- **$d_i$** : largeur de couches (**width**)

**Remarque : Fonctions d’activation**  

- **ReLU** :  
$$
\text{ReLU}(x) = \max(0, x) = x_+ \quad \in \mathbb{R}_+
$$

- **Softplus** :  
$$
\text{softplus}(x) = \ln(1 + e^{-x}) \quad \in \mathbb{R}_+
$$

- **Tanh** :  
$$
\tanh(x) \in [-1,1]
$$

- **Sigmoïde** :  
$$
\sigma(x) = \frac{1}{1 + e^{-x}} \quad \in [0,1]
$$

---

**解析：**

这一部分介绍了神经网络的 **参数表示** 和 **激活函数**：
- **参数 $\theta$**：包含所有层的权重矩阵 $A_i$ 和偏置向量 $b_i$。
- **深度（Depth）$p$**：神经网络的层数。
- **宽度（Width）$d_i$**：每一层的神经元数量。

**常见的激活函数**：
1. **ReLU（整流线性单元）**：
   - 作用：抑制负值，保留正值。
   - 优点：计算简单，不易饱和，有效缓解梯度消失问题。
   - 常用于 **隐藏层**。

2. **Softplus**：
   - 平滑版本的 ReLU，适用于某些特殊情况。
   - 计算比 ReLU 更复杂，较少使用。

3. **Tanh（双曲正切函数）**：
   - 归一化到 $[-1,1]$，适用于中心化数据。
   - 比 Sigmoid 更对称，梯度更大，收敛更快。

4. **Sigmoïde（S 型函数）**：
   - 归一化到 $[0,1]$，常用于 **二分类问题的输出层**。
   - 缺点：容易出现梯度消失。

这些激活函数赋予神经网络 **非线性能力**，是深度学习的关键组成部分。


**Remarque (rétropropagation) :** Calculer le gradient de la fonction  

$$
J(\theta) = \frac{1}{n} \sum_{i=1}^{m} (y_i - f_{\theta}(x_i))^2
$$

On a  

$$
\nabla_{\theta} J(\theta) = \frac{2}{n} \sum_{i=1}^{m} (y_i - f_{\theta}(x)) \nabla_{\theta} f_{\theta}(x_i)
$$

$\nabla_{\theta} f_{\theta}(x_i)$ **Gradient du réseau de neurones par rapport à** $\theta$


**Rappel :**  

$$
f_{\theta}(x) = l_p \circ l_{p-1} \circ \dots \circ l_1 (x)
$$

**Notant**  

$$
z_1 = l_1(x), \quad z_2 = l_2 \circ l_1 (x), \quad z_k = l_k \circ \dots \circ l_1 (x),
$$

$$
\text{Jac } f_{\theta}(x) = \text{Jac } l_p (z_{p-1}) \text{Jac } l_{p-1} (z_{p-2}) \dots \text{Jac } l_2 (z_1) \text{Jac } l_1 (x)
$$

Du fait que  

$$
\text{Jac }(g \circ h)(x) = \text{Jac } g (h(x)) \text{Jac } h(x)
$$

D'où  

$$
\text{Jac } f_{\theta}(x)^{T} = \text{Jac } l_p (z_{p-1})^{T} \text{Jac } l_{p-1} (z_{p-2})^{T} \dots \text{Jac } l_2 (z_1)^{T} \text{Jac } l_1 (x)^{T}
$$

$$
= \left( \begin{array}{c}
\partial_{\theta_2} f_{\theta}(x) \\
1 \\
\partial_{\theta_1} f_{\theta}(x)
\end{array} \right)
$$
$$
f_{\theta}(x) = f(\theta, x)
$$
$$
= f(\theta^{(1)}, \theta^{(2)}, \theta^{(p)}, x)
$$

**Calcul de droite à gauche ou de gauche à droite**  

Si $f_{\theta}(x) \in \mathbb{R}$, alors $l_p (z_{p-1}) \in \mathbb{R}$  

$$
\Rightarrow \quad l_{p-1} (z_{p-1})
$$

$$
\Rightarrow \quad \text{Jac } l_p (z_{p-1})
$$

Si calcul de gauche à droite, les bulles de matrices sont  
$$
\theta^{(1)} \times z_1
$$
$$
\theta^{(1)} \times z_2
$$
$$
\theta^{(1)} \times z_3 \dots
$$

**Si calcul de droite à gauche**  

$$
z_{p-1} \times 1
$$

$$
\text{puis} \quad z_p \times 1
$$

$$
\text{puis} \quad z_{p-2} \times 1
$$

(plus intéressant)

**Algorithme de rétropropagation**  

- Calcul de $z_1, z_2$  
- $z_i = l_i (z_{i-1})$  
- Calcul du gradient de droite à gauche :  

$$
{(\text{Jac } l_p)} ^{T}
$$
$$
\text{puis} \quad (\text{Jac } l_{p-1})^{T} (\text{Jac } l_p)^{T}
$$
$$
\text{puis} \quad \dots
$$

