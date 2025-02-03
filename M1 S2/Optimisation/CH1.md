# Chapitre 1 : Introduction  

## I Exemples de problèmes d'optimisation  

### Problème du voyageur de commerce  

Un représentant doit visiter $N$ villes en un temps minimal, en passant une seule fois par ville.  

$t_{ij} = \text{temps de parcours entre la ville } i \text{ et la ville } j$  

$$
\min_{\sigma \in \Sigma_N} \sum_{i=1}^{N-1} t_{\sigma(i)\sigma(i+1)}
$$

$\text{card}(\Sigma_N) = N!$  

→ optimisation discrète difficile ($\text{NP-difficile}$)  

### Problème de transport optimal  

Transport de minerais entre $N$ mines et $M$ usines  

**Données** :  
$C_{ij}$ : coût de transport entre la mine $i$ et l’usine $j$ (par unité de minerai)  
$P_{ij}$ : quantité de minerai transportée entre la mine $i$ et l’usine $j$  

**Minimise le coût de transport**  (Problème de Kantorovitch)
$$
\min_{P \in M_{N,M}(\mathbb{R})} \sum_{i=1}^N \sum_{j=1}^M C_{ij} P_{ij}
$$

**Sous contraintes** :  
$$
\sum_{j=1}^M P_{ij} = a_i \quad \text{quantité de minerai produite par la mine } i
$$
$$
\sum_{i=1}^N P_{ij} = b_j \quad \text{quantité de minerai nécessaire à l’usine } j
$$

→ optimisation discrète sous contrainte  

### Analyse de données : réseaux de neurones  

Données $(x_i, y_i)_{i=1}^N \in \mathbb{R}^n \times \mathbb{R}^m$  

On cherche une fonction $f_\theta(x)$ qui approche au mieux les données : $f_\theta(x) \simeq y$  

$f_\theta$ est une fonction paramétrée $\theta \in \mathbb{R}^p$  

On détermine $\Theta$ de sorte à ce que soit minimal (écart quadratique moyen) :  
$$
\frac{1}{N} \sum_{i=1}^N \|y_i - f_\theta(x_i)\|^2
$$

**Régression linéaire** :  
$$
f_\theta(x) = Wx + b \quad \text{avec } W \in M_{m,n}(\mathbb{R}), b \in \mathbb{R}^m
$$  
$$
\theta = (W, b) \in \mathbb{R}^{m \times n + m}
$$
Réseaux de neurones  
$$
f_\theta(x) = g_k \circ \dots \circ g_1(x)
$$  
avec  
$$
g_i(x) = \sigma(W_i x + b_i)
$$  

$\sigma$ : fonction non linéaire appliquée composante par composante  

→ problème d’optimisation continu  

### Problème de Dido : type isopérimétrique  

Déterminer le domaine $\Omega$ le plus grand possible à périmètre fixé  
$$
\max_{\Omega \subset \mathbb{R}^2, \, \text{Pér}(\Omega) = l} \text{ Aire}(\Omega)
$$

Hypothèses : côté rivière, les bords sont fixés (en $a$ et $b$)  
Frontière est le graphe d’une fonction  

$y : [a, b] \to \mathbb{R}$  

$$
\max_{y \in V} \int_a^b y(x) \, dx
$$

avec  
$$
V = \left\{ y \in \mathcal{L}^2((a, b), \mathbb{R}) \, \middle| \, \int_a^b \sqrt{1 + y'(x)^2} \, dx = l \right\}
$$

→ optimisation en dimension infinie

### Minimisation d'énergie  

**Inconnue** : $u(x)$, déplacement vertical d’une peau de tambour  

**Donnée** : $f(x)$, force normale exercée sur le tambour (verticale)  

$$
\min_{u \in H_0^1(\Omega)} \int_\Omega \left( \|\nabla u(x)\|^2 - f(x) u(x) \right) dx
$$
$$
\text{Énergie élastique} - \text{travail de la force } f
$$

Avec  
$$
H_0^1(\Omega) = \{ u \in L^2(\Omega) \mid u' \in L^2(\Omega) \text{ et } u_{|\partial \Omega} = 0 \}
$$  
(espace de Sobolev)  

## II Vocabulaire  

$V$ : espace vectoriel normé, $K \subset V$, $J : K \to \mathbb{R}$  

**Problème de minimisation** :  
$$
\min_{x \in K} J(x)
$$
$J$ : fonction coût, critère, objectif  

- $K = V$ : problème sans contrainte  
- $K \subsetneq V$ : problème avec contrainte  

> [!definition]
> - $J$ admet un **minimum global** $\bar{x} \in K$, si $\forall x \in K, J(x) \geq J(\bar{x})$  
> 
> On note :  
> $$
> \inf_K J = \min_K J \quad \text{et} \quad \bar{x} = \arg\min_{x \in K} J(x)
> $$  
> 
> $\bar{x}$ est appelé minimum de $J$ sur $K$.  
> 
> - $J$ admet un **minimum local** $\bar{x} \in K$, si $\exists \varepsilon > 0$, $\forall x \in K \cap B(\bar{x}, \varepsilon), J(x) \geq J(\bar{x})$ 

> [!remark]
> $J$ admet un minimum si et seulement si $(-J)$ admet un maximum en $\bar{x}$.  

**Infimum et suite minorante** : problème plus général.  

Pas toujours de minimum :  
$$
\inf_{x \in V} J(x)
$$  
Exemple : $x \mapsto e^x$ n’a pas de minimum sur $\mathbb{R}$.  

C'est-à-dire trouver l'infimum de $\{J(x) \mid x \in K\} \subset \mathbb{R}$.  

→ Toujours une solution notée $\inf J \in [-\infty, +\infty]$.  

**Prop.** Tout sous-ensemble $A \subset \mathbb{R}$ admet une borne inférieure (infimum = plus grand des minorants) dans $[- \infty, +\infty[$.  

Cet infimum $a = \inf A$ est caractérisé par :  
- $a$ est un minorant de $A$, et  
- $\exists (y_n)$ suite dans $A$, $y_n \to a$.  

**Déf.** Il existe toujours une suite $(x_n)$ de $K$, appelée suite minorante, tq  
$$
J(x_n) = y_n \quad \longrightarrow \quad \bar{J} = \inf_{x \in K} J(x)
$$

**Rem.** Appliquons la caractérisation à $\{J(x) \mid x \in K\}$ :  
$$
\exists (y_n) \quad \text{suite dans } \{J(x) \mid x \in K\}, \quad y_n \to \bar{J}
$$
$$
\Rightarrow \exists (x_n), \quad J(x_n) \to \bar{J}
$$

**Question générale :**  
- $(x_n)$ converge-t-elle ? (existence et unicité)  
- Comment construire explicitement $(x_n)$ ? (méthodes numériques)  

## III Calcul différentiel  

### 1) Différentiabilité  

> [!definition]
> $E, F$ espace vectoriel normé réel, $U \subset E$ ouvert et $f : U \to F$.  
> 
> 1. $f$ est différentiable en $x \in U$, s’il existe une application linéaire continue $Df(x) \in \mathcal{L}(E, F)$ telle que :  
> $$
> f(x + h) = f(x) + Df(x)(h) + o(\|h\|) \quad \text{quand } \|h\| \to 0.
> $$
> 2. $f$ est $\mathcal{C}^1$ sur $U$ si $f$ est différentiable en tout $x \in U$ et  
> $$
> Df : x \in U \mapsto Df(x) \in \mathcal{L}(E, F) \text{ continue.}
> $$
nn             m
**Rappel** : $o(\|h\|) = \|h\| \epsilon(\|h\|)$ avec $\epsilon(\|h\|) = \frac{o(\|h\|)}{\|h\|} \to 0 \quad \text{quand } \|h\| \to 0.$  

Si $f$ est différentiable en $x \in U$, on peut définir les dérivées directionnelles suivant un vecteur $h \in E$ fixé :  

$$
\frac{\partial f}{\partial h}(x) = \lim_{t \to 0} \frac{f(x + t h) - f(x)}{t}
$$
$$
= \lim_{t \to 0} \frac{Df(x)(t h) + o(t \|h\|)}{t}
$$
$$
= \lim_{t \to 0} Df(x)(h) + \|h\| \frac{o(t)}{t}
$$
$$
= Df(x)(h)
$$  
$f \in \mathcal{C}^1 \Rightarrow f$ différentiable  
$f$ différentiable $\Rightarrow f \in \mathcal{C}^0$ (continue)  

$f$ différentiable $\Rightarrow f$ admet des dérivées directionnelles dans toutes les directions  

**Exemple :**  
$$
f(x, y) = 
\begin{cases} 
\frac{x^3}{x^2 + y} & \text{si } x \neq -y \\ 
0 & \text{si } x = -y 
\end{cases}
$$  

$f$ n’est pas continue en $0$, mais admet des dérivées directionnelles dans toutes les directions.  

En coordonnées polaires :  
$$
\begin{cases} 
x = r \cos\theta \\ 
y = r \sin\theta 
\end{cases}
$$
$$
f(x, y) = \frac{r^3 \cos^3\theta}{r (\cos\theta + \sin\theta)} = r^2 \frac{\cos^3\theta}{\cos\theta + \sin\theta}
$$
Pour tout $r > 0$, on a  
$$
\frac{\cos^3\theta}{\cos\theta + \sin\theta} \to +\infty
$$  
donc $f$ n’est pas bornée au voisinage de $(0, 0)$.  
Elle ne peut pas être continue.  

Soit $h = (h_1, h_2) \in \mathbb{R}^2$. On suppose que $h_1 \neq -h_2$.  
$$
\frac{f(th_1, th_2) - f(0, 0)}{t} = \frac{1}{t} \left( \frac{(t h_1)^3}{t h_1 + t h_2} - 0 \right)
$$
$$
= t \left( \frac{h_1^3}{h_1 + h_2} \right) \to 0 \quad \text{quand } t \to 0.
$$

Donc $\frac{\partial f}{\partial h}(0, 0)$ existe.  

Si $h_1 = -h_2$,  
$$
\frac{f(th_1, th_2) - f(0, 0)}{t} = \frac{0 - 0}{t} = 0.
$$
$$
\forall t > 0, \quad \text{donc } \frac{\partial f}{\partial h}(0, 0) \text{ existe.}
$$

**Rmq.** $f$ est deux fois différentiable en $x \in U$ si $f$ est différentiable sur un voisinage de $x$ et $Df : E \to \mathcal{L}(E, F)$ est différentiable en $x$.  

La différentielle seconde :  
$$
D^2f(x) \in \mathcal{L}(E, \mathcal{L}(E, F)) \simeq \mathcal{L}(E \times E, F).
$$

On note :  
$$
D^2f(x)(h, k) = D^2f(x)(h)(k), \text{ et on a : }
$$
$$
Df(x + h) = Df(x) + D^2f(x)(h) + o(\|h\|).
$$

**Prop. (Taylor-Young)** : Si $f$ est deux fois différentiable, alors :  
$$
f(x + h) = f(x) + Df(x)(h) + \frac{1}{2} D^2f(x)(h, h) + o(\|h\|^2) \quad \text{quand } \|h\| \to 0.
$$  
**Rappel** :  
1. $l : E \to F$ linéaire et continue si $\exists M > 0$, $\|l(x)\|_F \leq M \|x\|_E \quad \forall x \in E$.  
2. $b : E \times E \to F$ bilinéaire et continue si $\exists M > 0$, $\|b(x, y)\|_F \leq M \|x\|_E \|y\|_E \quad \forall x, y \in E$.  

### 2) Fonctions à valeurs réelles

Soit $V = \mathbb{R}^m$ ou un espace de Hilbert, muni d’un produit scalaire $\langle \cdot, \cdot \rangle$.  
Soit $f : V \to \mathbb{R}$.  

**Déf.**  
Si $f$ est différentiable en $x$,  
$$
Df(x) \in \mathcal{L}(V, \mathbb{R}) \simeq V',
$$  
le théorème de Riesz assure qu’il existe $z \in V$ tel que  
$$
\forall y \in V, \, Df(x)(y) = \langle z, y \rangle.
$$  
$z$ est appelé **gradient** de $f$ en $x$, noté $z = \nabla f(x)$.  

**Exemple :**  
$$
f(x) = \|x\|^2
$$

Pour tout $h \in V$,  
$$
f(x + h) = \|x + h\|^2 = \|x\|^2 + 2 \langle x, h \rangle + \|h\|^2
$$
$$
= f(x) + L(h) + R(h),
$$  
où $L(h) = 2 \langle x, h \rangle$ est linéaire et continue, car  
$$
|L(h)| = |2 \langle x, h \rangle| \leq 2 \|x\| \|h\| \quad \text{(Cauchy-Schwarz)}.
$$

$R(h) = \|h\|^2$ vérifie $\frac{R(h)}{\|h\|} = \|h\| \to 0 \quad \text{quand } \|h\| \to 0.$  

Donc $f$ est différentiable en $x \in V$ et  
$$
Df(x)(h) = L(h) = 2 \langle x, h \rangle.
$$

Ainsi, $\nabla f(x) = 2x$.  

**Rmq.** On peut aussi introduire la Hessienne :  
$$
Hf(x) = \nabla^2 f(x) \in \mathcal{L}(V, V)
$$
$$
D^2f(x)(h, h) = \langle \nabla^2 f(x) h, h \rangle.
$$

**Rmq. (Formule de Taylor-Young)**  
Si $f$ est deux fois différentiable en $x \in V$, alors :  
$$
f(x + h) = f(x) + \langle \nabla f(x), h \rangle + \frac{1}{2} \langle \nabla^2 f(x) h, h \rangle + o(\|h\|^2).
$$

#### Cas particulier : dimension finie
$V = \mathbb{R}^n$  

**Déf. (Dérivées partielles)**  
1. $f$ admet une dérivée partielle d’indice $i \in \{1, \dots, n\}$ si $f$ admet une dérivée directionnelle suivant $e_i$. On note $\partial_i f(x)$ cette dérivée.  
2. $f$ est $\mathcal{C}^k$ si toutes les dérivées partielles existent jusqu’à l’ordre $k$ et sont continues.  

**Rmq.** Si $f$ est différentiable en $x$, alors :  
$$
Df(x)(h) = \sum_{i=1}^n \partial_i f(x) h_i = \langle \nabla f(x), h \rangle,
$$
avec  
$$
\nabla f(x) = (\partial_i f(x))_{1 \leq i \leq n} \in \mathbb{R}^n.
$$

Si $f$ est deux fois différentiable en $x$, alors :  
$$
D^2f(x)(h, h) = \sum_{i=1}^n \sum_{j=1}^n \partial_{ij}^2 f(x) h_i h_j,
$$
avec  
$$
\nabla^2 f(x) = (\partial_{ij}^2 f(x))_{1 \leq i, j \leq n}.
$$
Ainsi :  
$$
D^2f(x)(h, h) = \langle \nabla^2 f(x) h, h \rangle.
$$

**Prop (Taylor-Intégral)**  

$f : U \subseteq E \to F$ et soit $[x, x+h]$ un intervalle inclus dans $U$.  

- Si $f \in \mathcal{C}^1$,  
$$ 
f(x+h) = f(x) + \int_0^1 Df(x + t h)(h) \, dt 
$$

- Si $f \in \mathcal{C}^2$,  
$$ 
f(x+h) = f(x) + Df(x)(h) + \int_0^1 (1-t) D^2f(x + t h)(h, h) \, dt 
$$

**Prop (Taylor-Lagrange)**  

$f : U \subseteq E \to \mathbb{R}$ différentiable  
et $[x, x+h] \subset U$. Alors $\exists s \in ]0, 1[$,  
$$ 
f(x+h) = f(x) + Df(x + s h)(h) 
$$

- Si $f \in \mathcal{C}^1$ et deux fois différentiable :  
  $\exists s \in ]0, 1[$,  
$$ 
f(x+h) = f(x) + Df(x)(h) + \frac{1}{2} D^2f(x + s h)(h, h) 
$$

### 3)  Analyse convexe 

Pour simplifier les notations, $V = \mathbb{R}^n$ ou Hilbert.  
Toutes les définitions et propositions s'étendent à $V$ evn.  

**Déf :** $K \subset V$ est convexe si $\forall x, y \in K$,  
$\forall \theta \in [0, 1], \theta x + (1-\theta) y \in K$.  

*(Schéma d'une courbe convexe $K$ avec deux points $x, y$ et le segment entre eux inclus dans $K$.)*  

![[Pasted image 20250127134829.png]]

**Déf :**  
$f : K \subset V \to \mathbb{R}$ avec $K$ convexe non vide.  
$f$ est convexe si $\forall x, y \in K$, $\forall \theta \in [0, 1]$,  
$$ 
f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta) f(y) 
$$

$f$ est strictement convexe si l'inégalité est stricte dès que $x \neq y$ et $\theta \in ]0, 1[$.  

*"f est en dessous de ses cordes"*  

![[Pasted image 20250127135325.png|400]]

**Remarque (régularité)**  

En dimension finie, $f$ convexe 
$\implies$ $f$ localement Lipschitzienne (Lipschitzienne sur tout compact).  
$\implies$ (Théorème de Rademacher) $f$ est différentiable p.p.  

$\implies$ $f$ est continue sur $\mathring{K}$.  

**Exemples :**  

- $f : \mathbb{R} \to \mathbb{R}$  
  $x \mapsto x^2$  
  *(continue sur $\mathbb{R}$)*  

![[Pasted image 20250127140737.png|300]]

- $f : [0, 1] \subset \mathbb{R} \to \mathbb{R}$  
$$
x \mapsto  
\begin{cases}  
0 & \text{si } x \in ]0, 1[, \\  
1 & \text{si } x = 0 \text{ ou } x = 1.  
\end{cases}  
$$  
  *(continue sur $\mathring{[0, 1]} = ]0, 1[$)*  

![[Pasted image 20250127140757.png|300]]

En dimension infinie, $f$ convexe $\not\Rightarrow$ $f$ continue sur $\mathring{K}$.  
Il existe des formes linéaires (fonctions convexes) non continues.  

**Remarque :** $f$ est convexe si l'épigraphe de $f$,  
$$ \{ (x, y) \in K \times \mathbb{R} \mid y \geq f(x) \} \subset K \times \mathbb{R}, $$
est convexe.  

![[Pasted image 20250127141005.png|300]]

$f$ convexe $\Rightarrow$ les sous-ensembles de niveaux de $f$,  
$$ \{ x \in K \mid f(x) \leq \alpha \}, $$
sont convexes.  

![[Pasted image 20250127141022.png|300]]

Contre-exemple 

![[Pasted image 20250127141115.png|300]]

un graphe non convexe et ses ensembles de niveaux sont convexe (egaux soit à $\varnothing, \mathbb{R}_-,\mathbb{R}$)

**Prop (Caractérisation de la convexité pour $f$ différentiable)**  

$f : U \subset V \to \mathbb{R}$ différentiable et $K \subset U$ convexe non vide.  

On a les équivalences entre :  

(i) $f$ est convexe sur $K$,  

(ii) $\forall x, y \in K$,  
$$ 
f(y) \geq f(x) + (\nabla f(x), y - x), 
$$
(iii) $\forall x, y \in K$,  
$$ 
(\nabla f(y) - \nabla f(x), y - x) \geq 0. 
$$

*Les mêmes équivalences valent pour le cas strictement convexe en remplaçant $\geq$ par $>$ et en supposant $x \neq y$.*  

**Prop (Caractérisation pour $f \in \mathcal{C}^1$ deux fois différentiable)**  

Avec les mêmes notations, $f \in \mathcal{C}^1$ deux fois différentiable.  

On a l'équivalence :  

(i) $f$ convexe sur $K$,  
(iv) $\forall x \in K, \nabla^2 f(x)$ est (semi-définie) positive,  c'est-à-dire $\forall h \in V, (\nabla^2 f(x)h, h) \geq 0$.  

Dans le cas strictement convexe, on a $(i) \Rightarrow (iv)$ , $(\nabla^2 f(x))$ définie positive, mais la réciproque est fausse $(i) \not\Rightarrow (iv)$.  

**Remarque :**  
- (ii) $f$ est au-dessus de son hyperplan tangent.  

![[Pasted image 20250127142625.png|500]]

- (iii) $\nabla f(x)$ est monotone.  

En dimension 1 :  
$$
f'(y) - f'(x)(y - x) \geq 0 \quad \forall x, y \implies f' \text{ est croissante.} 
$$

- (iv) En dimension 1 :  
$$ 
f''(x) \geq 0. 
$$

**Preuve :** (cas convexe)  

$(i) \implies (ii)$  
$\forall x, y \in K, \forall \theta \in ]0, 1[$,  
$$ 
f(\theta x + (1-\theta) y) \leq \theta f(x) + (1-\theta) f(y). 
$$
On a :  
$$ 
f(y + \theta (x-y)) - f(y) \leq \theta (f(x) - f(y)). 
$$

En divisant par $\theta$, on obtient :  
$$ 
\frac{f(y + \theta (x-y)) - f(y)}{\theta} \leq f(x) - f(y). 
$$

En faisant tendre $\theta \to 0^+$, on a :  
$$ 
\frac{\partial f}{\partial (x-y)}(y) = (\nabla f(y), x-y). 
$$

Donc $\forall x, y \in K$,  
$$ 
(\nabla f(y), x-y) + f(y) \leq f(x). 
$$

$(ii) \implies (iii)$  
$\forall x, y \in K$,  
$$ 
f(y) \geq f(x) + (\nabla f(x), y-x), 
$$
$$ 
f(x) \geq f(y) + (\nabla f(y), x-y). 
$$

En sommant, on obtient :  
$$ 
f(x) + f(y) \geq f(x) + f(y) + (\nabla f(x) - \nabla f(y), y-x), 
$$
$$ 
0 \geq (\nabla f(x) - \nabla f(y), y-x), 
$$
$$ 
0 \leq (\nabla f(y) - \nabla f(x), y-x). 
$$

$(iii) \implies (ii)$  
On définit :  
$$ 
\phi(t) = f(x + t(y-x)), \quad \phi : [0, 1] \to \mathbb{R}. 
$$
$\phi$ est dérivable car composée de fonctions différentiables.  
$$ 
\phi'(t) = Df(x + t(y-x))(y-x) = (\nabla f(x + t(y-x)), y-x). 
$$

$$ 
\phi'(t) - \phi'(0) = (\nabla f(x + t(y-x)), y-x) - (\nabla f(x), y-x), 
$$
$$ 
= \frac{1}{t} ( \nabla f(x + t(y-x)) - \nabla f(x), t(y-x) ) \geq 0 \quad \text{(d'après (ii))}. 
$$
$\forall t \in [0, 1]$,  

on applique le théorème des accroissements finis :  
$\exists s \in ]0, 1[$,  
$$ 
\phi(1) - \phi(0) = \phi'(s) \geq \phi'(0),
$$  
$$ 
\implies f(y) - f(x) \geq (\nabla f(x), y-x). 
$$

$(ii) \implies (i)$  
On a $\forall x, y \in K, \forall \theta \in [0, 1]$,  
$$ 
f(y) \geq f(\theta x + (1-\theta)y) + (\nabla f(\theta x + (1-\theta)y), y - (\theta x + (1-\theta)y)), 
$$
$$ 
f(x) \geq f(\theta x + (1-\theta)y) + (\nabla f(\theta x + (1-\theta)y), x - (\theta x + (1-\theta)y)). 
$$

On multiplie la première équation par $(1-\theta)$ et la seconde par $\theta$, puis on somme.  

$f \in \mathcal{C}^2$, $f$ deux fois différentiable.  

$(ii) \implies (iv)$  
Soit $x \in K$ et $h \in V$. D’après la formule de Taylor à l’ordre 2 :  
$$
0\leq f(x + th) - f(x) - (\nabla f(x), th) = \frac{t^2}{2} (\nabla^2 f(x)h, h) + o(\|th\|²) \quad (t \to 0).
$$

Donc $\forall t > 0$,  
$$ 
0 \leq \frac{1}{2} (\nabla^2 f(x)h, h) + o(1) \quad (t \to 0). 
$$

En divisant par $t^2 > 0$ et en faisant tendre $t \to 0$, on obtient :  
$$ 
0 \leq \frac{1}{2} (\nabla^2 f(x)h, h). 
$$

$(iv) \implies (ii)$  
Soit $x, y \in K$. D’après la formule de Taylor-Lagrange à l’ordre 2 :  
$$
f(y) = f(x) + (\nabla f(x), y - x) + \frac{1}{2} (\nabla^2 f(x + s(y - x))(y - x), y - x),
$$
avec $s \in [0, 1]$.  

Donc :  
$$
f(y) - f(x) - (\nabla f(x), y - x) \geq 0.
$$

**Remarque :** $(i) \implies (ii)$  
Dans le cas strictement convexe, on montre que les pentes sont strictement croissantes :  
$$ 0 < \theta < \omega < 1, $$  
$$ \frac{f(x + \theta(y - x)) - f(x)}{\theta} < \frac{f(x + \omega(y - x)) - f(x)}{\omega} < \frac{f(y) - f(x)}{1}. $$  
![[Pasted image 20250127150641.png]]

**Remarque : (fonction quadratique)**  
$f : \mathbb{R}^n \to \mathbb{R},$  
$$ f(x) = \frac{1}{2}(Ax, x) - (b, x), $$  
avec $A \in S_n(\mathbb{R})$ et $b \in \mathbb{R}^n$.  $f$ est-elle convexe ?  

- $f$ est polynomiale dans $\ell^2$,  
$$ 
f(x) = \frac{1}{2} \sum_{i,j=1}^n A_{ij}x_i x_j - \sum_{i=1}^n b_i x_i. 
$$

**Calcul des $\nabla f$ et $\nabla^2 f$ :**  

$$
f(x + h) = \frac{1}{2}(A(x+h), x+h) - (b, x+h),
$$
$$
= \frac{1}{2} \left[ (Ax, x) + (Ax, h) + (Ah, x) + (Ah, h) \right] - \left[ (b, x) + (b, h) \right],
$$
$$
= f(x) + (Ax, h) - (b, h) + \frac{1}{2}(Ah, h),
$$
$$
= f(x) + (Ax-b, h) + \frac{1}{2}(Ah, h).
$$

Par identification :  
$$
\nabla f(x) = Ax - b, \quad \nabla^2 f(x) = A.
$$

Pour $n = 1$ :  
$$
f(x) = \frac{1}{2}ax^2 - bx, \quad f'(x) = ax - b, \quad f''(x) = a.
$$

**Conclusion :**  
$f$ est convexe 
$\iff \forall x \in \mathbb{R}^n, \nabla^2 f(x) = A$ est (semi-définie) positive,  
$\iff A$ est (semi-définie) positive.  

$f$ strictement convexe :  $\iff$ $A$ définie positive (d'après (iii)(iv))

$$
\forall x \neq y, \quad (\nabla f(x) - \nabla f(y), x-y) > 0,
$$
$$
\iff (Ax - b) - (Ay - b), x-y > 0,
$$
$$
\iff (A(x-y), x-y) > 0.
$$

**Définition :**  
$f : K \subset V \to \mathbb{R}$ avec $K$ convexe non vide.  
$f$ est dite $\alpha$-convexe si :  
$$
\forall x, y \in K, \forall \theta \in [0, 1],
$$
$$
f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y) - \frac{\alpha}{2} \theta (1-\theta)\|x-y\|^2 ,
$$
$\alpha > 0$ .

![[Pasted image 20250127152312.png|400]]
la fonction $\alpha$-convexe avec son écart quantifié par $\theta f(x) + (1-\theta)f(y) -\frac{\alpha}{2} \theta (1-\theta) \|x-y\|^2$, un polynôme de degré 2 en $\theta$.

**Remarque :**  
$f \alpha$-convexe $\implies f$ est strictement convexe.  
*La $\alpha$-convexité « quantifie » la stricte convexité.*  

**Proposition :**  
$f$ est $\alpha$-convexe si et seulement si  
$$
g(x) = f(x) - \frac{\alpha}{2} \|x\|^2 \quad \text{est convexe.}
$$

**Preuve :**  
$$
\theta g(x) + (1-\theta) g(y) = \theta f(x) + (1-\theta) f(y) - \frac{\alpha}{2} \left( \theta \|x\|^2 + (1-\theta) \|y\|^2 \right),
$$
$$
g(\theta x + (1-\theta)y) = f(\theta x + (1-\theta)y) - \frac{\alpha}{2} \| \theta x + (1-\theta)y \|^2.
$$

Donc :  
$$
\theta g(x) + (1-\theta) g(y) - g(\theta x + (1-\theta)y)  
= \theta f(x) + (1-\theta) f(y) - f(\theta x + (1-\theta)y) - \frac{\alpha}{2}C,
$$  
avec :  
$$
C = \theta (1-\theta) \|x\|^2 + (1-\theta)(1-\theta) \|y\|^2 - 2\theta (1-\theta) (x, y),
$$
$$
= \theta (1-\theta) \left[ \|x\|^2 + \|y\|^2 - 2(x, y) \right],
$$
$$
= \theta (1-\theta) \|x - y\|^2.
$$

**Proposition :**  
Si $f$ est différentiable,  
$f : U \subset V \to \mathbb{R}$ différentiable,  
$K \subset U$ convexe non vide.  

On a les équivalences :  
(i) $f$ est $\alpha$-convexe sur $K$,  
(ii) $\forall x, y \in K, \, f(y) \geq f(x) + \langle \nabla f(x), y-x \rangle + \alpha \|x-y\|^2,$  
(iii) $\forall x, y \in K, \, (\nabla f(y) - \nabla f(x), y-x) \geq \alpha \|y-x\|^2,$  

Si $f$ est deux fois différentiable :  
(iv) $\forall x \in K, \, \forall h \in V, \, (\nabla^2 f(x)h, h) \geq \alpha \|h\|^2.$  

**Preuve :**  
Découle des caractérisations sur $g$ convexe, sachant que  
$$
\nabla g(x) = \nabla f(x) - \alpha x.
$$

**Exemple :** (fonction quadratique dans $\mathbb{R}^n$)  
$A$ symétrique définie positive de valeurs propres $0 < \lambda_1 \leq \ldots \leq \lambda_n$.  
Alors  
$$
(Ah, h) \geq \lambda_1 \|h\|^2.
$$

$f$ strictement convexe $\iff A$ définie positive $\iff f$ est $\lambda_1$-convexe.  

**Preuve de (Exemple)**  

Soit $A = P D P^T$ avec $P \in O_n(\mathbb{R})$ et $D = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$.  

$$
(Ah, h) = (PDP^T h, h) = (DP^T h, P^T h),
$$
$$
= \sum_{i=1}^n \lambda_i \left( (P^T h)_i \right)^2,
$$
$$
\geq \lambda_1 \sum_{i=1}^n \left( (P^T h)_i \right)^2,
$$
$$
= \lambda_1 \|P^T h\|^2.
$$
Comme $P \in O_n(\mathbb{R})$,  
$$
\|P^T h\|^2 = \|h\|^2.
$$
Donc :  
$$
(Ah, h) \geq \lambda_1 \|h\|^2.
$$

