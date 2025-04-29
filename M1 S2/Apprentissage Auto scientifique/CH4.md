Chap 4 - Méthodes génératives

Objectif : construire une densité de probabilité $p(x)$ associée à des observations  

pour pouvoir échantillonner de nouveaux points.

## I) Mélange de Gaussiennes (Gaussian mixture)

Développé pour faire du partitionnement de données.

Étant donnés des points $(x_i)$, on veut les classer dans $K$ sous-groupes.

### 1) Modèle de densité

On cherche une **densité** de probabilité sous la forme :
$$
p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{G}_{\theta_k}(x)
$$

où les paramètres sont :

- les poids $\pi_k \geq 0$ vérifiant $\sum_{k=1}^{K} \pi_k = 1$ *(combinaison convexe)*  
- les paramètres $\theta_k = (\mu_k, \Sigma_k) \in \mathbb{R}^d \times \mathcal{S}_d^+(\mathbb{R})$  
  *(symétrique positive)*

de chaque gaussienne

$$
\mathcal{G}_{\theta_k}(x) = \frac{1}{(2\pi)^{d/2} \, \sqrt{\det \Sigma_k}} \exp\left( -\frac{1}{2} (x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) \right)
$$

$K = 3$ , $d = 2$
![[image-19.png|515x329]]

1D
![[image-20.png|518x359]]

**Modélisation probabiliste**  
On considère $(X, Z)$ un couple de variables aléatoires où $X \in \mathbb{R}^d$ désigne la donnée et $Z \in \{1, \dots, K\}$ le sous-groupe associé, dont la loi est définie :

- la loi de $Z$ est $(\pi_1, \dots, \pi_K) \quad \left( \mathbb{P}(Z = k) = \pi_k \right)$  
- la loi de $X \mid Z = k$ est normale de densité $\mathcal{G}_{\theta_k}(x)$

on note $p(x \mid k) = \mathcal{G}_{\theta_k}(x)$

$$
\mathbb{P}\bigl(X \in [x_i, x_i + dx] \mid Z = k\bigr) = \mathcal{G}_{\theta_k}(x)\,dx + o(dx)= p(x \mid k)\,dx + o(dx)
$$

Ainsi la loi jointe de $(X, Z)$ est “densité”

$$
p(x, k) = p(x \mid k)\,p(k) = \mathcal{G}_{\theta_k}(x)\,\pi_k
$$

$$
\begin{align}
\mathbb{P}\bigl(X \in [x_i, x_i + dx],\, Z = k\bigr) &= \mathbb{P}\bigl(X \in [x_i, x_i + dx] \mid Z = k\bigr)\,\mathbb{P}(Z = k)   \\
&= \bigl(\mathcal{G}_{\theta_k}(x)\,dx\bigr)\,\pi_k = p(x, k)\,dx + o(dx)
\end{align}
$$
et $X$ est de densité

$$
p(x) = \sum_{k=1}^{K} p(x, k) = \sum_{k=1}^{K} \pi_k\, \mathcal{G}_{\theta_k}(x)
$$

C’est bien la densité souhaitée.

$Z$ est appelée *variable latente* ou *variable cachée*

$Z \mid X = x$ est de densité

$$
p(j \mid x) = \frac{p(j, x)}{p(x)} = \frac{\mathcal{G}_{\theta_j}(x)\, \pi_j}{\sum\limits_{k=1}^{K} \pi_k\, \mathcal{G}_{\theta_k}(x)}
$$
En pratique : si les paramètres $(\pi_k,\theta_k)$ sont connus, on peut déterminer le groupe auquel appartient un point $x\in\mathbb{R}^d$ à l’aide des $p(j\mid x)$ :

![[image-21.png|452x229]]
$$
p(1\mid x) > p(2\mid x) > p(3\mid x)
$$
$$
x \in \mathbb{R}^d \longrightarrow k \in \{1, \ldots, K\} 
$$

**ENCODER**  
*(encodage)*

$x$ appartient au sous-groupe  
$$
k = \arg\max_{j\in\{1,\dots,K\}} p(j\mid x).
$$
Pour générer une nouvelle donnée dans le groupe $k$ (proche de $x$), on tire un point selon la densité $p(x \mid Z = k)$ :

$$
k \in \{1, \ldots, K\} \longrightarrow x \in \mathbb{R}^d
$$

**DECODER**  
*(décodage)*

L’Encoder compresse l’information.

---

### 2) Algorithme EM

Étant donnés $(x_1, \ldots, x_n)$ indépendante, on ajuste les paramètres $(\pi_k, \theta_k)$ de sorte à maximiser la log vraisemblance :

$$
\ln\left( p(x_1, \ldots, x_n) \right) = \ln\left( \prod_{i=1}^n p(x_i) \right)
= \sum_{i=1}^n \ln p(x_i)
= \sum_{i=1}^n \ln \left( \sum_{k=1}^K \pi_k\, \mathcal{G}_{\theta_k}(x_i) \right)
$$


**Algorithme EM (Expectation-Maximization)**

On considère la log-vraisemblance complète :
$$
\ln\left( p((x_1, z_1), \ldots, (x_n, z_n)) \right)
= \ln\left( \prod_{i=1}^n p(x_i, z_i) \right)
$$
$$
= \sum_{i=1}^n \ln \, p(x_i, z_i)
= \sum_{i=1}^n \ln \left( \pi_{z_i} \, \mathcal{G}_{\theta_{z_i}}(x_i) \right)
= \sum_{i=1}^n \left( \ln(\pi_{z_i}) + \ln(\mathcal{G}_{\theta_{z_i}}(x_i)) \right)
$$

On a affecté chaque point à un groupe.  
Plus généralement, on considère :
$$
\mathbb{E}_{Z \mid X, (\pi, \theta)} \left[ \ln \, p\left((x_1, z_1), \dots, (x_n, z_n)\right) \right]
= \sum_{i=1}^n \mathbb{E}_{Z_i \mid X, (\pi, \theta)} \left[ \ln(\pi_{Z_i}) + \ln(\mathcal{G}_{\theta_{Z_i}}(x_i)) \right]
$$
$$
\sum_{i=1}^n \sum_{k=1}^K \bigl(\ln\pi_k + \ln\mathcal{G}_{\theta_k}(x_i)\bigr)\,\mathbb{P}\bigl(Z=k \mid X=x_i\bigr)
= Q\bigl(\pi,\theta \mid \pi',\theta'\bigr)
$$
$$
\mathbb{P}\bigl(Z=k \mid X=x_i\bigr) = \dfrac{\pi'\mathcal{G}_{\theta'_k}(x_i)}{\sum_{j=1}^k \pi'\mathcal{G}_{\theta'_k}(x_j)}
$$

Cette quantité minore la log-vraisemblance :
$$
\ln\bigl(p(x_1,\dots,x_n)_{(\pi,\theta)}\bigr)
= \sum_{i=1}^n \ln\!\Bigl(\sum_{k=1}^K \pi_k\,\mathcal{G}_{\theta_k}(x_i)\Bigr)
$$
$$
\ge \sum_{i=1}^n \sum_{k=1}^K p\bigl(k\mid x_i\bigr)_{(\pi',\theta')}
\;\ln\!\Biggl(\frac{\pi_k\,\mathcal{G}_{\theta_k}(x_i)}
{p\bigl(k\mid x_i\bigr)_{(\pi',\theta')}}\Biggr)
$$
($\ln$ est concave et $\sum_{k=1}^K p(k\mid x_i)_{(\pi',\theta')}=1$.)

$$
\ge \sum_{i=1}^{m}\sum_{k=1}^{K} p(k\mid x_i)_{(\pi',\theta')}\,\ln\bigl(\pi_k\,\mathcal{G}_{\theta_k}(x_i)\bigr)
- \sum_{i=1}^{N}\sum_{k=1}^{K} p(k\mid x_i)_{(\pi',\theta')}\,\ln\bigl(p(k\mid x_i)_{(\pi',\theta')}\bigr)
$$
$$
= Q(\pi,\theta \mid \pi',\theta') - \sum_{i=1}^N H\bigl(p(\,\cdot\mid x_i)_{(\pi',\theta')}\bigr)
$$

entropie  
ne dépend pas de $(\pi,\theta)$

donc maximiser $Q(\pi,\theta \mid \pi',\theta')$ permet d’augmenter la log-vraisemblance

**Algorithme** : on construit une suite $\left( \pi_k^{(m)}, \theta_k^{(m)} \right)_k$  
avec : à chaque itération

1) étape **E** (*expectation*) : calcul des  
$$
p\left(h \mid x_i \right)_{(\pi^{(m-1)}, \theta^{(m-1)})} = \left( \frac{\pi_h^{(m-1)} \, \mathcal{G}_{\theta_h^{(m-1)}}(x_i)}{\sum_{j=1}^{K} \pi_j^{(m-1)} \, \mathcal{G}_{\theta_j^{(m-1)}}(x_i)} \right)
$$

2) étape **M** (*maximization*) :  
$$
\pi^{(m)}, \theta^{(m)} = \arg\max_{(\pi, \theta)} Q(\pi, \theta \mid \pi^{(m-1)}, \theta^{(m-1)})
$$

**Rem.** : *similaire à l’algorithme des K-moyennes* :  
on affecte des probabilités d’appartenance à chaque groupe **(E)**,  
puis on met à jour les paramètres des gaussiennes *(et des groupes)* **(M)**.

**Rem.** : l’étape **M** peut être résolue explicitement (1D)

$$
Q(\pi, \theta \mid \pi', \theta') = \sum_{i=1}^n \sum_{k=1}^K \left( \ln \pi_k + \ln \mathcal{G}_{\theta_k}(x_i) \right) \, \underbrace{p(k \mid x_i)_{(\pi', \theta')}}_{= \, T_{ki}}
$$
$$
= \sum_{i=1}^n \sum_{k=1}^K  \left( \ln \pi_k - \ln (\sqrt{2\pi}) - \ln \sigma_k - \frac{(x_i - \mu_k)^2}{2 \sigma_k^2} \right) \, T_{ki}
$$

En 1D, la densité gaussienne est :
$$
\mathcal{G}_{\theta_k}(x) = \frac{1}{\sqrt{2\pi} \, \sigma_k} \exp\left( -\frac{(x - \mu_k)^2}{2 \sigma_k^2} \right)
$$
avec $\theta_k = (\mu_k, \sigma_k)$

$$
\sum_{j=1}^K\pi_j=1,\quad 
h(\pi)=\sum_{j=1}^K\pi_j-1,\quad 
\partial_{\pi_j}h(\pi)=1
$$
Lagrangien + contrainte :
$$
\lambda\,\partial_{\pi_j}h+\partial_{\pi_j}Q
=\sum_{i=1}^n\frac{T_{ji}}{\pi_j}+\lambda=0
\tag{1}
$$
$$
\partial_{\sigma_j}Q
=\sum_{i=1}^n\Bigl(-\frac1{\sigma_j}+\frac{(x_i-\mu_j)^2}{\sigma_j^3}\Bigr)\,T_{ji}=0
\tag{2}
$$
$$
\partial_{\mu_j}Q
=\sum_{i=1}^n\frac{x_i-\mu_j}{\sigma_j^2}\,T_{ji}=0
\tag{3}
$$
D’où, pour chaque $j$ :
$$
\mu_j
=\frac{\sum_{i=1}^n x_i\,T_{ji}}{\sum_{i=1}^nT_{ji}},
\qquad
\sigma_j^2
=\frac{\sum_{i=1}^n(x_i-\mu_j)^2\,T_{ji}}{\sum_{i=1}^nT_{ji}},
\qquad
\pi_j
=\frac{1}{n}\sum_{i=1}^nT_{ji},
\qquad 
\lambda = n.
$$

on admet que ce point critique est un max.

## II) Auto-encodeur variationnel

Objectif : apprendre une densité de probabilité $p_\theta(x)$  
à partir d’observations $(x_i)_{i=1...n}$

→ modèle à variable latente : description riche  
→ variable latente : $z \in \mathbb{R}^q$ de loi **fixée** $p(z)$

$$
p_\theta(x) = \int_{\mathbb{R}^q} p_\theta(x, z) \, dz
$$
$$
= \int_{\mathbb{R}^q} p_\theta(x \mid z) \, p(z) \, dz
$$
On veut apprendre $p_\theta(x \mid z)$.

Si on veut appliquer une méthode de type EM, il faudrait  
pouvoir calculer  
$$
p_\theta(z \mid x) = \frac{p_\theta(x, z)}{p_\theta(x)}
$$

(mélange Gaussien)  
$$
= \frac{\pi_k \, G_{\theta_k}(x)}{\sum_{j=1}^{K} \pi_j \, G_{\theta_j}(x)}
$$

![[Unsaved Image 1.jpg|273x273]]

$$
p_\theta(x) = \frac{p_\theta(x, z)}{\int_{\mathbb{R}^q} p_\theta(x \mid z) \, p(z) \, dz}
\quad \leftarrow \text{quantité difficile à calculer.}
$$

Au lieu de calculer $p_\theta(z \mid x)$, on va en calculer une approximation  
$$
q_\phi(z \mid x) \approx p_\theta(z \mid x)
$$
$$
\quad \uparrow \text{approximation de l'encodage}
$$

**Paramétrisation des densités de probabilité :**

- **variable latente** :  
  $p(z) = \mathcal{G}_{0, \mathrm{Id}}(z)$ (loi normale centrée réduite)  
  $$
  = \frac{1}{(2\pi)^{\ell/2}} \exp\left(-\frac{\|z\|^2}{2}\right)
  $$

- **décodage** :  
  $p_\theta(x \mid z) = \mathcal{G}_{\mu_\theta(z), \, \mathrm{diag}(\sigma_\theta^2(z))}(x)$  
  avec  
  $$
  \mu_\theta(z) \in \mathbb{R}^d, \quad \sigma_\theta^2(z) \in \mathbb{R}, \quad = \mathrm{NN}_\theta(z) \in \mathbb{R}^{2d}
  $$

- **encodage** :  
  $q_\phi(z \mid x) = \mathcal{G}_{\mu_\phi(x), \, \mathrm{diag}(\sigma_\phi^2(x))}(z)$  
  avec  
  $$
  \mu_\phi(x), \, \sigma_\phi(x) \in \mathbb{R}^q, \quad = \mathrm{NN}_\phi(x) \in \mathbb{R}^{2q}
  $$
Ajustement des paramètres  
Étant donné $(x_1, \ldots, x^N)$,  
on souhaite maximiser la log-vraisemblance :

$$
\theta = \arg\max_\theta \log \, p_\theta(x_1, \ldots, x^N)
$$
$$
= \arg\max_\theta \log \prod_{i=1}^N p_\theta(x_i)
$$
$$
= \arg\max_\theta \sum_{i=1}^N \log \, p_\theta(x_i)
$$

On fait apparaître la variable latente :

$$
\log \, p_\theta(x_i) = \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \, p_\theta(x_i) \right]
$$
$$
= \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \left( \frac{p_\theta(x_i \mid z) \, p(z)}{p(z)} \right) \right]
$$
$$
= \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \left( \frac{p_\theta(x_i \mid z)}{q_\phi(z \mid x_i)} \cdot \frac{q_\phi(z \mid x_i)}{p(z)} \right) \right]
$$
$$
= \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \left( \frac{p_\theta(x_i \mid z)}{q_\phi(z \mid x_i)} \right) \right]
+ \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \left( \frac{q_\phi(z \mid x_i)}{p(z)} \right) \right]
$$
$$
= \mathrm{ELBO}_{x_i}(\theta, \phi) + \mathrm{KL} \left( q_\phi(z \mid x), \, p(z) \right)
$$

**Prop.** Étant donné deux densités de probabilité $q(z), \, p(z)$  
on définit  
$$
\mathrm{KL}(q, p) = \mathbb{E}_q \left[ \log \left( \frac{q}{p} \right) \right] = \mathbb{E}_q \left[ -\log \left( \frac{p}{q} \right) \right]
$$

appelée **distance de Kullback-Leibler** (attention : ce n’est pas une distance) ; elle vérifie :  
$$
\mathrm{KL}(q, p) > 0
\quad \text{et} \quad
\mathrm{KL}(q, p) = 0 \ \text{ssi} \ q = p
$$
**Preuve :**  
$$
\mathbb{E}_q \left[ -\log \left( \frac{p}{q} \right) \right]
= \int -\log \left( \frac{p(z)}{q(z)} \right) q(z) \, dz
$$
$$
\geq -\log \left( \int \frac{p(z)}{q(z)} q(z) \, dz \right)
= -\log(1) = 0
\quad \text{car } p(z) \text{ densité de probabilité}
$$

$$
\text{(inégalité de Jensen)}
\quad \text{et } -\log \text{ convexe}
$$

$$
\frac{1}{M} \sum_{j=1}^{M} -\log(a_j)
\geq -\log \left( \frac{1}{M} \sum_{j=1}^{M} a_j \right)
\quad \text{(moyenne de } -\log \geq \log \text{ de la moyenne)}
$$

**Le cas d’égalité** est obtenu lorsque l’intégrande est constante  
($-\log$ strictement convexe)

$$
- \log \left( \frac{p(z)}{q(z)} \right) = \text{conste}
\Rightarrow \frac{p(z)}{q(z)} = \text{conste}
\Rightarrow p(z) = q(z)
\quad \text{car } p, q \text{ densités de probabilité.}
$$

Nous avons donc :  
$$
\mathrm{ELBO}_{x_i}(\theta, \phi) = \log \, p_\theta(x_i) - \mathrm{KL} \left( q_\phi(z \mid x_i), \, p(z) \right)
$$
$$
\leq \log \, p_\theta(x_i)
$$

**ELBO** = *Evidence Lower BOund*  
= Borne inférieure variationnelle.

On maximise l’**ELBO** pour maximiser la vraisemblance, sous la forme :

$$
\mathrm{ELBO}_{x_i}(\theta, \phi) = \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \left( \frac{p_\theta(x_i, z)}{q_\phi(z \mid x_i)} \right) \right]
$$

$$
p_\theta(z \mid x_i) = \frac{p_\theta(z, x_i)}{p_\theta(x_i)}
$$

$$
\log \, p_\theta(x_i)
= \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \, p_\theta(x_i) \right]
= \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \left( \frac{p_\theta(x_i, z)}{p_\theta(z \mid x_i)} \right) \right]
$$

$$
= \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \left( \frac{p_\theta(x_i, z)}{q_\phi(z \mid x_i)} \cdot \frac{q_\phi(z \mid x_i)}{p_\theta(z \mid x_i)} \right) \right]
$$

$$
= \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \left( \frac{p_\theta(x_i, z)}{q_\phi(z \mid x_i)} \right) \right]
+ \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \left( \frac{q_\phi(z \mid x_i)}{p_\theta(z \mid x_i)} \right) \right]
$$
$$
= \mathrm{KL}\left(q_\phi(z \mid x_i), \, p_\theta(z \mid x_i)\right)
$$

---

**Méthode de gradient**

$$
\mathrm{ELBO}_{x_i}(\theta, \phi) =
\mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \, p_\theta(x_i, z) \right]
- \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \, q_\phi(z \mid x_i) \right]
$$

---

**Calcul du gradient par rapport à** $\phi$ :

$$
\nabla_\phi \, \mathrm{ELBO}_{x_i}(\phi) =
\mathbb{E}_{z \mid x_i \sim q_\phi} \left[
\nabla_\phi \log \, p_\theta(x_i, z)
\right]
$$

$$
= \log \left( p_\theta(x_i \mid z) \, p(z) \right)
- \log \, p_\theta(x_i \mid z) + \log \, p(z)
$$

$$
= \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \nabla_\theta \log \left( p_\theta(x_i \mid z) \right) \right]
\approx \frac{1}{M} \sum_{j=1}^{M} \nabla_\theta \log \left( p_\theta(x_i \mid z_j) \right)
$$
avec $z_j$ tiré aléatoirement suivant $q_\phi(z \mid x_i)$

---

**Calcul du gradient par rapport à $\phi$** : difficulté car l’espérance dépend aussi de $q_\phi$

→ on remarque que  
$$
z \mid x_i = \mu_\phi(x_i) + \sigma_\phi(x_i) \cdot \varepsilon
\quad \text{avec } \varepsilon \sim \mathcal{N}(0, \mathrm{Id})
$$

$$
\nabla_\phi \, \mathrm{ELBO}_{x_i}(\theta, \phi) =
\mathbb{E}_{\varepsilon \sim \mathcal{N}(0, \mathrm{Id})} \left[
\nabla_\phi \log \left(
p_\theta(x_i \mid \mu_\phi(x_i) + \sigma_\phi(x_i) \cdot \varepsilon)
\right)
\right]
$$
$$
- \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, \mathrm{Id})} \left[
\nabla_\phi \log \left(
q_\phi\left( \mu_\phi(x_i) + \sigma_\phi(x_i) \cdot \varepsilon \mid x_i \right)
\right)
\right]
$$
$$
\approx \frac{1}{M} \sum_{j=1}^{M}
\nabla_\phi \log \left( p_\theta(x_i \mid \mu_\phi(x_i) + \sigma_\phi(x_i) \cdot \varepsilon_j) \right)
- \nabla_\phi \log \left( q_\phi(\mu_\phi(x_i) + \sigma_\phi(x_i) \cdot \varepsilon_j \mid x_i) \right)
$$

**"astuce de reparamétrisation"**

---

**Expression dans le cas Gaussien**

$$
\mathrm{ELBO}_{x_i}(\theta, \phi) = \mathbb{E}_{z \mid x_i \sim q_\phi} \left[
\log \left( \frac{p_\theta(x_i, z)}{q_\phi(z \mid x_i)} \right)
\right]
$$

$$
= \mathbb{E}_{z \mid x_i \sim q_\phi} \left[
\log \left( \frac{p_\theta(x_i \mid z) \, p(z)}{q_\phi(z \mid x_i)} \right)
\right]
$$
$$
= \mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \, p_\theta(x_i \mid z) \right]
- \mathrm{KL} \left( q_\phi(z \mid x_i), \, p(z) \right)
$$

On a  
$$
\mathbb{E}_{z \mid x_i \sim q_\phi} \left[ \log \, p_\theta(x_i \mid z) \right]
= \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,1)} \left[
\log \, p_\theta(x_i \mid \mu_\phi(x_i) + \sigma_\phi(x_i) \cdot \varepsilon)
\right]
$$

$$
= \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,1)} \left[
- \frac{\left\| x_i - \mu_\phi(x_i) + \sigma_\phi(x_i) \cdot \varepsilon \right\|^2}{2 \, \sigma_\theta(x_i)^2}
\right]
$$

$$
\Rightarrow p_\theta(x \mid z) = \frac{1}{(2\pi \, \sigma^2)^{d/2}} \exp \left(
-\frac{\|x - \mu\|^2}{2\sigma^2}
\right)
\Rightarrow -\frac{d}{2} \log (2\pi \, \sigma_\theta^2(x_i))
$$

$$
\Rightarrow -\frac{d}{2} \log (2\pi \, \sigma_\theta^2(x_i)) + \frac{1}{M} \sum_{j=1}^{M}
\left[
- \frac{\left\| x_i - \mu_\phi(x_i) + \sigma_\phi(x_i) \cdot \varepsilon_j \right\|^2}
{2 \, \sigma_\theta(x_i)^2}
\right]
$$

$$
\text{et } \mathrm{KL} \left( q_\phi(z \mid x_i), \, p(z) \right)
= \frac{1}{2} \left( \sigma_\phi^2(x) \, d - d + \| \mu_\phi(x) \|^2 - 2d \log \, \sigma_\phi(x) \right)
$$

$$
q_\phi = \mathcal{G}_{\mu_\phi, \, \mathrm{diag} \, \sigma_\phi^2}
\quad ; \quad p(z) = \mathcal{G}_{0, \mathrm{Id}}
$$

---

**Conclusion :**

$$
\mathrm{ELBO}_{x_i}(\theta, \phi) \approx
\frac{1}{M} \sum_{j=1}^{M}
\left[
- \frac{\left\| x_i - \left( \mu_\phi(x_i) + \sigma_\phi(x_i) \cdot \varepsilon_j \right) \right\|^2}
{2 \, \sigma_\theta(x_i)^2}
- \frac{d}{2} \log \left( 2\pi \, \sigma_\theta^2(x_i) \right)
\right]
$$

$$
- \frac{1}{2} \left( \sigma_\phi^2(x) \, d - d + \| \mu_\phi(x) \|^2 \right)
- 2d \log \, \sigma_\phi(x)
$$
