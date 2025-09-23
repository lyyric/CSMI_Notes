# Chapitre 1 : Validité et estimation

## I. Lois unidimensionnelles

### I.1 Loi de variable aléatoire

Définition d’une loi de variable aléatoire unidimensionnelle.

---

### I.2 Centre et dispersion

* **Espérance (centre)** :

  $$
  E(X) = \int x f(x)\, dx
  $$

* **Médiane** : valeur qui partage la distribution en deux parties égales.

* **Variance et écart-type (dispersion)** :

$$
\mathrm{Var}(X) = E\!\big[(X - E(X))^2\big],
\quad 
\sigma_X = \sqrt{\mathrm{Var}(X)}
$$

* **Quantiles** :
  Le quantile d’ordre $p$ est la valeur $q$ telle que

$$
P(X \leq q) = p
$$

  *(représentation : boîte à moustaches avec quartiles)*

---

## I.3 Estimation

### I.3.1 En l’absence de données observées

* **Entropie d’une densité $f$** :

$$
H(f) = - \int f(x) \log f(x)\, dx
$$

* Principe du maximum d’entropie, sous contraintes :

$$
\int g_j(x) f(x)\, dx = C_j, \quad j = 1,\dots,J
$$

* Problème d’optimisation :

$$
\max_f H(f) 
\quad \text{sous contraintes } \int f(x)\, dx = 1, 
\quad \int g_j(x) f(x)\, dx = C_j
$$

* **Exemples** :

  * Si $a \leq x \leq b$, pas d’autre contrainte :

$$
X \sim \mathcal{U}([a,b])
$$
  * Si $X \geq 0$ et $E(X) = \mu$ :

$$
X \sim \mathrm{Exp}\!\left(\tfrac{1}{\mu}\right)
$$

---

### I.3.2 Estimation non paramétrique

* **Sans lissage** : fonction de répartition empirique

  $$
  \hat{F}(x) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}_{\{X_i \leq x\}}
  $$

* **Avec lissage** : estimation par noyau

  $$
  \hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^n K\!\left(\frac{x - X_i}{h}\right)
  $$

  où $K$ est une densité (exemple : loi uniforme sur $[-0.5,0.5]$).

---

### I.3.3 Estimation paramétrique

On suppose une forme connue pour la loi de $X$ (ex : normale, Poisson).
Il s’agit d’estimer les **paramètres**.

Deux méthodes principales :

#### a) Méthode des moments

* On calcule les moments théoriques ($E(X), E(X^2),\dots$) en fonction des paramètres.
* On les remplace par les moments empiriques :

  $$
  \hat{m}_k = \frac{1}{n} \sum_{i=1}^n X_i^k
  $$

**Exemple : Loi Beta $\mathrm{Beta}(a,b)$**

$$
E(X) = \frac{a}{a+b}, 
\quad 
\mathrm{Var}(X) = \frac{ab}{(a+b)^2 (a+b+1)}
$$

On obtient :

$$
\bar{X} \approx \frac{\hat{a}}{\hat{a} + \hat{b}}
$$

---

#### b) Méthode du maximum de vraisemblance (EMV)

* Pour une observation $x$ :

  $$
  l_x(\theta) = f_\theta(x)
  $$

* Pour un échantillon :

  $$
  L(x_1,\dots,x_n)(\theta) = \prod_{i=1}^n f_\theta(x_i)
  $$

* Log-vraisemblance :

  $$
  \ell(x_1,\dots,x_n)(\theta) = \log L(x_1,\dots,x_n)(\theta)
  $$

* Estimateur du maximum de vraisemblance :

  $$
  \hat{\theta} = \arg\max_\theta L(\theta) 
  \quad \Leftrightarrow \quad 
  \arg\max_\theta \ell(\theta)
  $$

**Exemple : Loi de Poisson $\mathcal{P}(\lambda)$**

$$
L(\lambda) = \prod_{i=1}^n \frac{\lambda^{x_i}}{x_i!} e^{-\lambda}
$$

$$
\ell(\lambda) = \sum_{i=1}^n (x_i \log \lambda - \lambda) + \text{cte}
$$

$$
\frac{\partial \ell}{\partial \lambda} = 0 
\quad \Rightarrow \quad 
\hat{\lambda} = \bar{X}
$$

---

## I.4 Intervalle de confiance

* Définition :
  Un IC d’indice $1-\alpha$ est un intervalle aléatoire $I_X$ tel que :

  $$
  P(\theta \in I_X) = 1 - \alpha
  $$

⚠️ Remarque : $\theta$ est **fixe**, c’est l’intervalle $I_X$ qui est aléatoire.

---

### Qualité d’une méthode

Critère : **taux de couverture**, i.e. la proportion des IC qui contiennent bien $\theta$.

👉 Trois méthodes principales pour construire des IC.

---

### I.4.1 Étude théorique de la loi de $\hat{\theta}$

* Cas où l’on connaît une fonction pivot dont la loi est connue.

**Exemple :** $X \sim \mathcal{N}(\mu, \sigma^2)$, avec $\sigma$ connue.

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0,1)
$$

Donc pour $q_{1-\alpha/2}$ le quantile d’ordre $1-\alpha/2$ de $\mathcal{N}(0,1)$ :

$$
P\!\left( -q_{1-\alpha/2} \leq \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \leq q_{1-\alpha/2} \right) = 1-\alpha
$$

Ainsi, l’IC pour $\mu$ est :

$$
\mu \in \Big[ \, \bar{X}_n - q_{1-\alpha/2}\tfrac{\sigma}{\sqrt{n}}, \; 
\bar{X}_n + q_{1-\alpha/2}\tfrac{\sigma}{\sqrt{n}} \, \Big]
$$

* ✅ Avantage : pas d’approximation.
* ❌ Inconvénient : nécessite une fonction pivot connue.

