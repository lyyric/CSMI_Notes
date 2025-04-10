# Chapitre 1 : Introduction

L’objectif de l’apprentissage supervisé est d’estimer un modèle paramétrique qui « explique » les données observées et qui permettra de faire de nouvelles prédictions. On se focalise ici sur le cas de la régression, c’est-à-dire l’estimation d’une fonction $f$ reliant des variables d’entrée $x$ à des variables de sortie $y$.

---

## I. Régression Linéaire

### 1. Présentation du Problème

Soit un jeu de données d’entraînement  
$$
(x_1, y_1), \dots, (x_n, y_n) \in X \times Y,
$$
avec $X \subset \mathbb{R}^d$ et $Y \subset \mathbb{R}$. L’objectif est de trouver un modèle $f$ de la forme
$$
y = f(x)
$$
qui permette de prédire de nouvelles valeurs de $y$ à partir de $x$.

La qualité du modèle est évaluée à l’aide d’une **fonction perte**. Dans le cas de la régression, on utilise souvent l’erreur quadratique moyenne (MSE – Mean Square Error) :
$$
J(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - f_\theta(x_i) \right)^2,
$$
où $\theta$ représente les paramètres du modèle.

---

### 2. Modèles Paramétriques

On choisit une famille de modèles paramétriques $f_\theta(x)$ avec
$$
\theta \in \Theta \subset \mathbb{R}^p.
$$

#### 2.1 Régression Linéaire Simple

Le modèle linéaire classique est défini par
$$
f_\theta(x) = \langle w, x \rangle_{\mathbb{R}^d} + b,
$$
avec $w \in \mathbb{R}^d$ et $b \in \mathbb{R}$. On introduit alors le vecteur de paramètres étendu :
$$
\theta = \begin{pmatrix} w \\ b \end{pmatrix} \in \mathbb{R}^{d+1}
$$
et la représentation augmentée de $x$ :
$$
\bar{x} = \begin{pmatrix} x \\ 1 \end{pmatrix} \in \mathbb{R}^{d+1},
$$
de sorte que
$$
f_\theta(x) = \langle \theta, \bar{x} \rangle.
$$
Ce modèle est linéaire à la fois par rapport à $x$ et aux paramètres $\theta$.

#### 2.2 Extensions par les Fonctions de Caractéristiques

On peut généraliser ce modèle en introduisant une transformation $\phi$ qui associe à chaque $x$ un vecteur de caractéristiques dans $\mathbb{R}^p$ :
$$
f_\theta(x) = \langle \theta, \phi(x) \rangle.
$$
- **Régression linéaire :** $\phi(x) = \begin{pmatrix} x \\ 1 \end{pmatrix}$ avec $p = d+1$.
- **Régression polynomiale (ordre 2) :**  
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
  \end{pmatrix} \in \mathbb{R}^p,
  $$
  avec $p = 1 + d + \frac{d(d+1)}{2}$.

---

### 3. Formulation du Problème d’Optimisation

On cherche à minimiser la fonction perte
$$
\hat{\theta} = \arg\min_{\theta \in \Theta} J(\theta) = \arg\min_{\theta \in \Theta} \frac{1}{n} \lVert Y - A\theta \rVert^2,
$$
où
$$
Y = \begin{pmatrix} y_1 \\ \vdots \\ y_n \end{pmatrix} \quad \text{et} \quad A = \begin{pmatrix}
\phi(x_1)^T \\
\vdots \\
\phi(x_n)^T
\end{pmatrix} \in M_{n,p}(\mathbb{R}).
$$

#### 3.1 Calcul du Minimum par l’Équation Normale

La fonction $J(\theta)$ est convexe et différentiable. Pour un minimum, il faut que le gradient soit nul :
$$
\nabla J(\hat{\theta}) = 0.
$$
Calculons le gradient. En écrivant explicitement :
$$
J(\theta) = \frac{1}{n} \lVert Y - A\theta \rVert^2 = \frac{1}{n} (Y - A\theta)^T (Y - A\theta),
$$
la dérivation par rapport à $\theta$ donne :
$$
\nabla J(\theta) = -\frac{2}{n} A^T (Y - A\theta).
$$
Ainsi, la condition d’optimalité devient
$$
A^T(Y - A\hat{\theta}) = 0 \quad \Longrightarrow \quad A^T A \hat{\theta} = A^T Y.
$$

- **Si $A$ est de rang $p$ :** $A^T A$ est inversible et la solution unique est
  $$
  \hat{\theta} = (A^T A)^{-1} A^T Y.
  $$
- **Si $A$ est de rang inférieur à $p$ :** il existe une infinité de solutions minimisant $J$.

#### 3.2 Développement par Déplacement (Preuve par Taylor)

Soit $h$ un vecteur de perturbation. On écrit
$$
J(\theta + h) = \frac{1}{n} \lVert Y - A(\theta + h) \rVert^2 = \frac{1}{n} \lVert (Y - A\theta) - Ah \rVert^2.
$$
En développant :
$$
J(\theta + h) = \frac{1}{n} \left[ \lVert Y - A\theta \rVert^2 - 2 \langle Y - A\theta, Ah \rangle + \lVert Ah \rVert^2 \right].
$$
On identifie alors :
- Le terme linéaire en $h$ est $-\frac{2}{n} \langle A^T(Y - A\theta), h \rangle$,
- Le terme quadratique est $\frac{1}{n} \langle A^T A\, h, h \rangle$.

La condition nécessaire pour que $\theta$ soit minimum (premier ordre) est donc
$$
A^T(Y - A\theta) = 0.
$$

---

### 4. Interprétation Probabiliste

Une autre approche consiste à interpréter la régression linéaire dans un cadre probabiliste.

#### 4.1 Modèle Gaussien

On suppose que, pour chaque $i$,
$$
y_i = f_\theta(x_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2),
$$
ce qui revient à poser
$$
y_i \mid x_i \sim \mathcal{N}(f_\theta(x_i), \sigma^2).
$$

La vraisemblance des observations (en supposant l’indépendance) est
$$
p_\theta(y_1, \dots, y_n \mid x_1, \dots, x_n) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right).
$$
La log-vraisemblance est alors
$$
\ln p_\theta(y_1, \dots, y_n \mid x_1, \dots, x_n) = -\frac{n}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n}(y_i - f_\theta(x_i))^2.
$$
Maximiser la log-vraisemblance revient donc à minimiser
$$
\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2,
$$
ce qui est exactement le problème de régression linéaire en erreur quadratique.

#### 4.2 Maximum de Vraisemblance et Interprétation Bayésienne

Si l’on impose une loi a priori sur les paramètres, par exemple
$$
\theta \sim \mathcal{N}(0, \alpha^2 I_p),
$$
alors, par le théorème de Bayes,
$$
p(\theta \mid x, y) = \frac{p(y \mid x, \theta) \, p(\theta)}{p(y \mid x)}.
$$
En maximisant la log-vraisemblance a posteriori (MAP), on obtient un problème de minimisation qui correspond à la régression pénalisée (voir section suivante).

---

### 5. Méthodes de Résolution

#### 5.1 Équation Normale et Décompositions Numériques

La solution par l’équation normale,
$$
\hat{\theta} = (A^T A)^{-1} A^T Y,
$$
peut être obtenue numériquement à l’aide de décompositions telles que la décomposition QR ou la SVD (Singular Value Decomposition).

#### 5.2 Méthode de Gradient

Une approche itérative consiste à mettre à jour $\theta$ par descente de gradient :
$$
\theta^{(k+1)} = \theta^{(k)} - \eta \nabla J(\theta^{(k)}),
$$
où $\eta > 0$ est le pas d’apprentissage. Si la suite converge vers $\theta^*$, alors
$$
\nabla J(\theta^*) = 0,
$$
et $\theta^*$ est un minimum de $J$.

##### Gradient Stochastique par Batch

Pour de grands jeux de données, il est courant d’utiliser une version par lot (mini-batch) :
$$
\theta^{(k+1)} = \theta^{(k)} - \eta \nabla_\theta \left( \frac{1}{|I|} \sum_{i \in I} \ell(y_i, f_\theta(x_i)) \right),
$$
avec $I \subset \{1, \dots, n\}$ un sous-ensemble d’indices.

---

### 6. Sur-Apprentissage (Overfitting)

Lorsque le nombre de paramètres $p$ est supérieur au nombre d’exemples $n$ (c.-à-d. $p > n$), le système est sur-paramétré. Dans ce cas, il existe une infinité de minimisateurs qui interpellent parfaitement les données d’entraînement, c’est-à-dire :
$$
f_\theta(x_i) = y_i \quad \text{pour tout } i \in \{1, \dots, n\}.
$$
Cependant, ce modèle « passe » exactement par les observations et peut être très oscillant, menant à de mauvaises performances en généralisation.

Une technique pour pallier ce problème est l’**arrêt précoce (early stopping)** dans les méthodes de gradient : on arrête l’algorithme dès que l’erreur sur un jeu de test commence à augmenter.

---

### 7. Méthodes de Pénalisation

Pour éviter que les paramètres ne deviennent trop grands et pour améliorer la généralisation, on ajoute un terme de pénalisation dans la fonction perte.

#### 7.1 Pénalisation Ridge

On considère la fonction objectif pénalisée :
$$
J(\theta) = \frac{1}{n} \lVert Y - A\theta \rVert^2 + \lambda \lVert \theta \rVert^2,
$$
avec $\lambda > 0$ qui contrôle la force de la pénalisation.

**Démonstration de l’unicité de la solution :**

La fonction $J$ est strictement convexe car le terme quadratique $\lVert \theta \rVert^2$ est strictement convexe. Le gradient s’annule lorsque :
$$
-\frac{2}{n} A^T (Y - A\theta) + 2\lambda \theta = 0,
$$
ce qui s’écrit :
$$
A^T A \theta + n\lambda \theta = A^T Y.
$$
Ainsi,
$$
\theta^* = \left( A^T A + n\lambda I_p \right)^{-1} A^T Y.
$$
*(On trouve parfois des notations différentes, par exemple en écrivant la fonction perte avec un facteur $\frac{1}{m}$ et un terme $2\lambda$, ce qui conduit à une écriture équivalente.)*

#### 7.2 Interprétation Bayésienne

Si l’on impose une loi a priori
$$
\theta \sim \mathcal{N}(0, \alpha^2 I_p),
$$
et en supposant toujours
$$
y_i \mid x_i, \theta \sim \mathcal{N}(f_\theta(x_i), \sigma^2),
$$
le maximum de vraisemblance a posteriori (MAP) conduit à minimiser :
$$
-\ln p(y \mid x, \theta) - \ln p(\theta),
$$
ce qui équivaut, à des constantes près, à minimiser :
$$
\frac{1}{2\sigma^2} \sum_{i=1}^{n} \left(y_i - f_\theta(x_i)\right)^2 + \frac{1}{2\alpha^2} \lVert \theta \rVert^2.
$$
En posant $\lambda = \frac{\sigma^2}{\alpha^2}$ (ou des constantes multiplicatives proches), on retrouve exactement le problème de la régression ridge.

---

### 8. Régression à Noyaux (Kernel Regression)

La régression à noyaux permet de traiter des problèmes non linéaires sans avoir à connaître explicitement la transformation $\phi(x)$. On part du problème pénalisé :
$$
J(\theta) = \frac{1}{n} \sum_{i=1}^{n}\left( y_i - \langle \theta, \phi(x_i) \rangle \right)^2 + \lambda \lVert \theta \rVert^2.
$$
On montre que le minimiseur $\hat{\theta}$ peut être écrit comme combinaison linéaire des vecteurs $\phi(x_i)$ :
$$
\hat{\theta} = \sum_{i=1}^{n} \alpha_i \phi(x_i).
$$
La fonction prédite s’exprime alors par
$$
f_{\hat{\theta}}(x) = \langle \hat{\theta}, \phi(x) \rangle = \sum_{i=1}^{n} \alpha_i \langle \phi(x_i), \phi(x) \rangle.
$$
En introduisant le noyau défini par
$$
k(x, x') = \langle \phi(x), \phi(x') \rangle,
$$
le problème s’exprime uniquement en termes de $k$. En définissant la matrice de noyaux
$$
K = \left( k(x_i, x_j) \right)_{1 \leq i,j \leq n},
$$
le problème de minimisation devient
$$
\widetilde{J}(\alpha) = \frac{1}{n} \lVert Y - K\alpha \rVert^2 + \lambda\, \alpha^T K\alpha.
$$
La solution optimale est donnée par
$$
\hat{\alpha} = \frac{1}{n}\left( K^T K + \lambda K \right)^{-1} K^T Y,
$$
et la fonction prédite est
$$
f_{\hat{\theta}}(x) = \sum_{j=1}^{n} \hat{\alpha}_j \, k(x_j, x).
$$

#### Exemples de Noyaux

- **Noyau gaussien (RBF Kernel) :**
  $$
  k(x, y) = \exp\left(-\frac{\| x - y \|^2}{2\sigma^2}\right).
  $$
- **Noyau linéaire :**
  $$
  k(x, y) = \langle x, y \rangle.
  $$


Voici une version réorganisée et détaillée de vos notes sur les réseaux de neurones, avec une présentation claire en français, une structuration par sections et des démonstrations détaillées pour le calcul des gradients par rétropropagation.

---

# II. Réseaux de Neurones

Les réseaux de neurones artificiels sont des modèles paramétriques composés de plusieurs couches qui appliquent à la fois des transformations affines et des fonctions non linéaires (activations). Ils permettent de modéliser des relations complexes entre les données d’entrée et de sortie.

---

## 1. Définition et Architecture

### 1.1. La Couche d’un Réseau de Neurones

Une **couche** d’un réseau de neurones est définie par la formule suivante :
$$
l_i(z) = \sigma(A_i\, z + b_i),
$$
où :
- $A_i \in M_{d_i,\, d_{i-1}}(\mathbb{R})$ est la matrice des poids,
- $b_i \in \mathbb{R}^{d_i}$ est le vecteur de biais,
- $\sigma : \mathbb{R} \to \mathbb{R}$ est la fonction d’activation, appliquée de manière composante.

### 1.2. Réseau de Neurones Complet

Un réseau de neurones de profondeur $p$ est la composition de $p$ couches :
$$
f_{\theta}(x) = l_p \circ l_{p-1} \circ \cdots \circ l_1(x),
$$
avec l’ensemble des paramètres :
$$
\theta = \bigl( A_1, b_1, A_2, b_2, \dots, A_p, b_p \bigr).
$$

**Terminologie :**
- **Profondeur (depth) $p$** : nombre de couches dans le réseau.
- **Largeur (width) $d_i$** : nombre de neurones (unités) de la $i$-ème couche.

---

## 2. Fonctions d’Activation

Les fonctions d’activation introduisent la non-linéarité indispensable pour que le réseau de neurones puisse modéliser des fonctions complexes. Voici quelques fonctions d’activation couramment utilisées :

### 2.1. ReLU (Rectified Linear Unit)

$$
\operatorname{ReLU}(x) = \max(0, x) = x_+ \quad \text{avec} \quad \operatorname{ReLU}(x) \in \mathbb{R}_+.
$$
- **Avantages :** Calcul simple, évite la saturation sur la partie positive, et aide à atténuer le problème de gradient qui disparaît.

### 2.2. Softplus

$$
\operatorname{softplus}(x) = \ln\bigl(1 + e^{x}\bigr) \quad \text{(ou parfois } \ln(1+e^{-x}) \text{ selon la convention)},
$$
- **Remarque :** Fonction lissée qui peut être vue comme une version « douce » de ReLU.

### 2.3. Tanh (Hyperbolic Tangent)

$$
\tanh(x) \in [-1,1].
$$
- **Avantages :** Fonction symétrique centrée sur zéro, ce qui peut favoriser une convergence plus rapide lors de l’entraînement.

### 2.4. Sigmoïde

$$
\sigma(x) = \frac{1}{1 + e^{-x}} \quad \text{avec } \sigma(x) \in [0,1].
$$
- **Usage typique :** Souvent utilisée dans la couche de sortie pour des problèmes de classification binaire.
- **Inconvénient :** Tendance à la saturation, ce qui peut entraîner le problème de gradient qui disparaît.

---

## 3. Rétropropagation : Calcul des Gradients

### 3.1. Formulation de la Fonction Perte

Considérons la fonction perte (erreur quadratique moyenne) pour un ensemble de $m$ exemples d’entraînement :
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \Bigl( y_i - f_{\theta}(x_i) \Bigr)^2.
$$

### 3.2. Calcul du Gradient par Rapport aux Paramètres

Pour mettre à jour les paramètres du réseau lors de l’entraînement (par exemple avec une descente de gradient), il faut calculer le gradient de $J(\theta)$ par rapport à $\theta$. On a :
$$
\nabla_{\theta} J(\theta) = \frac{2}{m} \sum_{i=1}^{m} \Bigl( y_i - f_{\theta}(x_i) \Bigr) \nabla_{\theta} f_{\theta}(x_i).
$$
La difficulté principale réside dans le calcul de $\nabla_{\theta} f_{\theta}(x_i)$, c’est-à-dire le gradient de la fonction de sortie du réseau par rapport à l’ensemble des paramètres.

### 3.3. Rétropropagation et Composition de Fonctions

Le réseau de neurones étant une composition de $p$ couches, on note :
$$
f_{\theta}(x) = l_p \circ l_{p-1} \circ \cdots \circ l_1 (x).
$$
On définit pour $k = 1, \dots, p$ :
$$
z_1 = l_1(x), \quad z_2 = l_2(z_1), \quad \dots, \quad z_p = l_p(z_{p-1}) = f_{\theta}(x).
$$

#### Règle de la Chaîne pour les Jacobiennes

Si l’on considère la dérivation d’une composition de fonctions, on a :
$$
\text{Jac}(g \circ h)(x) = \text{Jac}\, g \bigl(h(x)\bigr) \cdot \text{Jac}\, h(x).
$$
Pour le réseau complet, la jacobienne de $f_{\theta}$ s’écrit :
$$
\text{Jac } f_{\theta}(x) = \text{Jac } l_p(z_{p-1}) \cdot \text{Jac } l_{p-1}(z_{p-2}) \cdots \text{Jac } l_1(x).
$$
Il est parfois pratique de considérer la transposée de la jacobienne pour le calcul des dérivées par rapport aux paramètres :
$$
\text{Jac } f_{\theta}(x)^{T} = \text{Jac } l_p(z_{p-1})^{T} \cdot \text{Jac } l_{p-1}(z_{p-2})^{T} \cdots \text{Jac } l_1(x)^{T}.
$$

### 3.4. Algorithme de Rétropropagation

La rétropropagation (backpropagation) consiste à calculer le gradient de la fonction perte par rapport aux paramètres en procédant de la couche de sortie vers la couche d’entrée (calcul de droite à gauche).

**Étapes :**

1. **Propagation avant (Forward Pass) :**  
   - Calculer successivement $z_1, z_2, \dots, z_p$ en appliquant pour chaque couche :
     $$
     z_1 = l_1(x), \quad z_k = l_k(z_{k-1}) \quad (k = 2, \dots, p).
     $$

2. **Propagation arrière (Backward Pass) :**  
   - À partir du gradient de la perte par rapport à la sortie $f_{\theta}(x) = z_p$, calculer le gradient par rapport aux paramètres de la dernière couche.
   - Remonter couche par couche en utilisant la règle de la chaîne :
     $$
     \delta^p = \nabla_{z_p} J, \quad \delta^{k} = \bigl( \text{Jac } l_{k+1}(z_{k}) \bigr)^{T} \delta^{k+1} \quad \text{pour } k = p-1, \dots, 1.
     $$
   - À chaque couche, utiliser $\delta^k$ pour calculer les gradients par rapport aux paramètres $A_k$ et $b_k$.

3. **Mise à Jour des Paramètres :**  
   - Avec les gradients calculés, mettre à jour les paramètres $\theta = (A_1, b_1, \dots, A_p, b_p)$ selon une règle de descente (par exemple, la descente de gradient stochastique).

---

## 4. Remarques et Conseils Pratiques

- **Calcul "de gauche à droite" vs "de droite à gauche" :**  
  Lors de la propagation avant, le calcul s’effectue de l’entrée vers la sortie (de gauche à droite). Pour la rétropropagation, le calcul se fait de la sortie vers l’entrée (de droite à gauche), ce qui permet de réutiliser les gradients intermédiaires et de minimiser le coût en calcul.

- **Dimensions des Matrices :**  
  Dans la propagation, il est important de veiller aux dimensions :  
  - Les matrices $A_i$ transforment un vecteur de dimension $d_{i-1}$ en un vecteur de dimension $d_i$.  
  - Lors du calcul des jacobiennes et de leurs transposées, la compatibilité des dimensions garantit la cohérence du produit matriciel.

- **Fonctions d’Activation et Saturation :**  
  Le choix de la fonction d’activation a une influence majeure sur le comportement du réseau (vitesse de convergence, risque de saturation des gradients, etc.). Par exemple, la fonction ReLU est souvent privilégiée dans les couches cachées, alors que la sigmoïde est couramment utilisée en sortie pour des problèmes de classification binaire.
