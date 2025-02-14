# Chapitre 2 : Existence et unicité

## I) Existence d’un minimum  

**Théorème (Weierstrass)**  
Soit $J : K \subset V \to \mathbb{R}$ Continue sur $K$ compact, alors $J$ admet un minimum.

**Remarque :** $V = \mathbb{R}^n$ ou espace de Hilbert.

**Preuve :**  
Soit $(x_m)$, suite à valeurs dans $K$, **suite minimisante** :  
$$
\lim_{m \to \infty} J(x_m) = \inf_{x \in K} J(x).
$$
$K$ étant compact, on peut extraire une sous-suite **convergente** :  
$$
(x_{m_n}) \quad \text{tel que} \quad x_{m_n} \to x^* \in K.
$$
Par **continuité** de $J$ :  
$$
J(x_{m_n}) \to J(x^*).
$$
Or $(x_{m_n})$ est une sous-suite de $(x_m)$, donc $J(x_{m_n})$ converge aussi vers $\inf J$. Par **unicité de la limite** :  
$$
J(x^*) = \inf_{x \in K} J(x) = \min_{x \in K} J(x).
$$
(**L'infimum est atteint.**)


**Définition :**  
1) $f$ est **continue** en $x_0 \in K$ si  
$$
\lim_{x \to x_0} f(x) = f(x_0),
$$
i.e.  
$$
(x_n \to x_0) \Rightarrow f(x_n) \to f(x_0) \quad \text{(quand $n \to \infty$)}.
$$

2) $f$ est **semi-continue inférieure** (**S.C.I.**) en $x_0, x \in K$ à  
$$
\liminf_{x \to x_0} f(x) \geq f(x_0),
$$
i.e.  
$$
(x_n \to x_0) \Rightarrow \left( \liminf_{n \to \infty} f(x_n) \geq f(x_0) \right).
$$

**Remarque :** *(limite inférieure d’une suite)*  

Soit $(u_m)$ une suite :  
$$
\liminf_{n \to \infty} u_n = \lim_{n \to \infty} \left( \inf_{k \geq n} u_k \right).
$$
La limite inférieure existe toujours et prend des valeurs dans $[-\infty, +\infty]$.  

Exemple : $u_n = (-1)^n$  
$$
\inf_{k \geq n} u_k = (-1),
$$
donc  
$$
\liminf_{n \to \infty} u_n = -1.
$$

> [!example]
>  (fonction S.C.I.)
> 
> **Cas 1** : $\mathbb{1}_{[0,1]}$  
> - $x_n = \frac{1}{n} \to 0$  
>   $$
>   \liminf f(x_n) = 1 = f(0)
>   $$  
> - $x_n = -\frac{1}{n} \to 0$  
>   $$
>   \liminf f(x_n) = 0 \leq f(0) = 1
>   $$  
>   $f$ **n’est pas** S.C.I.  
> 
> **Cas 2** : $\mathbb{1}_{]0,1]}$  
> - $x_n \to 0$  
>   $$
>   \liminf f(x_n) \geq 0 = f(0) \quad (\forall n, f(x_n) > 0).
>   $$  
>   $f$ **est** S.C.I.

**Prop.**  
$f$ est **S.C.I.** si et seulement si les sous-ensembles de niveaux  
$$
\{ x \in V \mid f(x) \leq \alpha \}
$$
sont **fermés** pour tout $\alpha \in \mathbb{R}$.  

*(Voir TD)*  

**Exemple (suite) :**  
$f = \mathbb{1}_{]0,1[}$ est **S.C.I.**  

Car  
$$
\{ x \in \mathbb{R} \mid f(x) \leq \alpha \} =
\begin{cases}
\emptyset, & \text{si } \alpha < 0, \\
]-\infty,0] \cup [1,+\infty[, & \text{si } \alpha \in [0,1[, \\
\mathbb{R}, & \text{si } \alpha \geq 1.
\end{cases}
$$
Ces ensembles sont **fermés**.


**Théorème (Weierstrass bis) :**  
Soit $J : K \subset V \to \mathbb{R}$ **semi-continue inférieure**.  
Si $K$ est **compact**, alors $J$ admet un **minimum** sur $K$.  

**Preuve :**  
Soit $(x_m)$ une **suite minimisante** :  
$$
J(x_m) \to \inf_K J.
$$
$K$ étant **compact**, $(x_m)$ admet une sous-suite **convergente** dans $K$ :  
$$
x_{m_n} \to x^* \in K.
$$
$J$ étant **S.C.I.** :  
$$
\liminf J(x_{m_n}) \geq J(x^*).
$$
Or $J(x_{m_n})$ converge vers $\inf_K J$, donc  
$$
\liminf J(x_{m_n}) = \lim J(x_{m_n}) = \inf_K J.
$$
D'où :  
$$
\inf_K J \geq J(x^*),
$$
donc  
$$
\inf_K J = J(x^*), \quad x^* \in K.
$$

1) **Sur un compact :**  
   **Théorème de Weierstrass.**  

2) **Sur un fermé (dimension finie).**  

**Définition :**  
$J : K \subset V \to \mathbb{R}$ est dit **coercif** (ou infini à l’infini) sur $K$ fermé si  
$$
\forall (x_m) \text{ suite dans } K, \quad \| x_m \| \to \infty \Rightarrow J(x_m) \to \infty.
$$

**Théorème :**  
Soit $J : K \subset V \to \mathbb{R}$ **continue** et **coercive** sur $K$ **fermé**.  
Alors $J$ admet un **minimum** sur $K$.

**Preuve :**  
Soit $(x_m)$ une suite **minimisante** :  
$$
J(x_m) \to \inf_K J.
$$
Cas 1 : $(x_m)$ est **bornée**.  
Dans ce cas, on peut extraire une sous-suite convergente et conclure comme dans le théorème de **Weierstrass**.  

Cas 2 : $(x_m)$ **n’est pas bornée**.  
- Alors il existe une **sous-suite** $(x_{m_n})$ telle que  
$$
\|x_{m_n}\| \to \infty.
$$  
- Comme $J$ est **coercive**, on aurait  
$$
J(x_{m_n}) \to \infty,
$$  
  ce qui contredit  
$$
J(x_{m_n}) \to \inf_K J \in \mathbb{R}.
$$

Donc, il **existe** $M > 0$ tel que  
$$
x_m \in B(0, M) \quad \forall m \in \mathbb{N}.
$$
- Or, $B(0, M) \cap K$ est **fermé et borné**.  
- **En dimension finie**, cela implique que cet ensemble est **compact**.  

Ainsi, $(x_m)$ admet une **sous-suite convergente**, et on conclut **comme dans le théorème de Weierstrass**.

**Exemple :**  
Existence d’un minimum pour  
$$
J(x, y) = x^4 + y^4 - x^2
$$
sur  
$$
K = \{ (x, y) \in \mathbb{R}^2 \mid x + y \leq 24 \}.
$$

![[Pasted image 20250203143146.png]]

### **1) $J$ est continue :**  
$J$ est un **polynôme**, donc **continue**.

### **2) $K$ est fermé :**  
On peut écrire  
$$
K = g^{-1} (]-\infty, 24])
$$
avec  
$$
g : (x, y) \in \mathbb{R}^2 \mapsto x + y.
$$
- $g$ est **continue**.  
- L’image réciproque d’un **fermé** par une fonction continue est **fermée**, donc $K$ est **fermé**.

### **3) $J$ est coercive :**  
On cherche une fonction  
$$
\varphi : \mathbb{R}_+ \to \mathbb{R}
$$
telle que  
$$
J(x, y) \geq \varphi(\| (x, y) \|) \quad \text{et} \quad \lim_{t \to \infty} \varphi(t) = +\infty.
$$
Cela garantit que $J(x, y) \to \infty$ lorsque $\| (x, y) \| \to \infty$, ce qui prouve que $J$ est **coercive**.  

![[Pasted image 20250203143208.png]]

Conclusion :
$J$ est **continue** et **coercive** sur $K$ fermé.  
D’après le **théorème précédent**, $J$ admet un **minimum** sur $K$.

**Inégalité de Young :**  
Pour tout $\mu, \nu \in \mathbb{R}$,  
$$
|\mu \nu| \leq \frac{\mu^2 + \nu^2}{2}.
$$
Démonstration :  
$$
(\mu + \nu)^2 = \mu^2 + \nu^2 + 2\mu\nu \geq 0.
$$
$$
(\mu - \nu)^2 = \mu^2 + \nu^2 - 2\mu\nu \geq 0.
$$
En ajoutant ces deux inégalités, on obtient :  
$$
\mu^2 + \nu^2 \geq 2 |\mu\nu|.
$$

Application avec $\mu = x^2$ et $\nu = 1$ :  
$$
x^2 \leq \frac{x^4 + 1}{2}.
$$
Donc :  
$$
x^4 \geq 2x^2 - 1.
$$

**Application à $J(x,y)$ :**  
$$
J(x,y) = x^4 + y^4 - x^2.
$$
Utilisant l'inégalité précédente :  
$$
x^4 \geq 2x^2 - 1, \quad y^4 \geq 2y^2 - 1.
$$
Donc :  
$$
J(x,y) \geq (2x^2 - 1) + (2y^2 - 1) - x^2.
$$
$$
J(x,y) \geq x^2 + y^2 - 2.
$$
$$
J(x,y) \geq \| (x,y) \|^2 - 2 = \varphi(\| (x,y) \|).
$$

Avec  
$$
\varphi(t) = t^2 - 2.
$$
Et comme  
$$
\lim_{t \to \infty} \varphi(t) = +\infty,
$$
cela prouve que $J$ est **coercive**.

**Proposition :**  
Soit $K \subset V$ **convexe fermé** et  
$$
J : K \to \mathbb{R}
$$
**$\alpha$-convexe** $\Rightarrow$ $J$ est **coercive** et **continue**.  


**Preuve (dans le cas où $J$ est différentiable) :**  

Soit $x_0 \in K$. Alors pour tout $x \in K$, on a l'inégalité de convexité quadratique :  
$$
J(x) \geq J(x_0) + \langle \nabla J(x_0), x - x_0 \rangle + \frac{\alpha}{2} \| x - x_0 \|^2.
$$
En utilisant l'inégalité de Cauchy-Schwarz :  
$$
\langle a, b \rangle \geq - \| a \| \| b \|,
$$
on obtient :  
$$
J(x) \geq J(x_0) - \| \nabla J(x_0) \| \| x - x_0 \| + \frac{\alpha}{2} \| x - x_0 \|^2.
$$
avec  
$$
g(t) = \frac{\alpha}{2} t^2 - \| \nabla J(x_0) \| t + J(x_0),
$$
on a  
$$
g(t) \to +\infty \quad \text{quand } t \to +\infty, \quad (\alpha > 0).
$$

Comme  
$$
\| x \| - \| x_0 \| \leq \| x - x_0 \|,
$$
si $x_n \to \infty$, alors $\| x_n - x_0 \| \to \infty$ et donc  
$$
g(\| x_n - x_0 \|) \to +\infty.
$$
Par conséquent,  
$$
J(x_n) \to +\infty.
$$

**Exemple (en dimension infinie) :**  

On considère  
$$
J : \ell^2(\mathbb{N}) \to \mathbb{R}
$$
défini par  
$$
J(x) = \left( \| x \|_{\ell^2}^2 - 1 \right)^2 + \sum_{m=0}^{+\infty} \frac{x_m^2}{n+1}.
$$

Avec  
$$
\ell^2(\mathbb{N}) = \left\{ (x_n) \text{ suite à valeurs réelles } \bigg| \sum_{n=0}^{+\infty} x_n^2 < +\infty \right\},
$$
muni du **produit scalaire**  
$$
\langle x, y \rangle_{\ell^2} = \sum_{n=0}^{+\infty} x_n y_n.
$$
La **norme** associée est  
$$
\| x \|_{\ell^2} = \sqrt{\sum_{n=0}^{+\infty} x_n^2}.
$$
C'est un **espace de Hilbert**.

**$J$ est continue :**  

- $J_1(x) = (\| x \|^2 - 1)^2$ est **continue**, car la fonction  
  $$
  x \mapsto \| x \|
  $$  
  est continue, grâce à l’inégalité  
  $$
  |\| x \| - \| y \|| \leq \| x - y \|.
  $$  
  Ainsi, $J_1(x)$ est la **composition de fonctions continues**.  

- $J_2(x) = \sum_{m=0}^{+\infty} \frac{x_m^2}{n+1}$ est également continue.  

### **Preuve de la continuité de $J$ :**  

Pour tout $x, y \in B(0, M)$, on a :  
$$
| J(x) - J(y) | = \left| \sum_{m=0}^{+\infty} \frac{x_m^2 - y_m^2}{n+1} \right|
= \left| \sum_{m=0}^{+\infty} \frac{(x_m + y_m)(x_m - y_m)}{n+1} \right|.
$$

Utilisant l'inégalité triangulaire :  
$$
\sum_{m=0}^{+\infty} \left| \frac{(x_m + y_m)(x_m - y_m)}{n+1} \right|
\leq \sum_{m=0}^{+\infty} \frac{|x_m + y_m| |x_m - y_m|}{n+1}.
$$

Puisque  
$$
| x_m + y_m | \leq 2M \quad \text{(car $\| x \|, \| y \| \leq M$)},
$$
on obtient :  
$$
\sum_{m=0}^{+\infty} \frac{|x_m + y_m| |x_m - y_m|}{n+1} \leq 2M \sum_{m=0}^{+\infty} \frac{|x_m - y_m|}{n+1}.
$$
Ainsi, $J$ est bien **continue**.

---

### **Infimum de $J$ et suite minimisante**  

- $\inf J = 0$ car pour tout $x \in \ell^2(\mathbb{N})$, $J(x) \geq 0$.  
- On construit une **suite minimisante** :  
  $$
  x^{(h)} = (0, 0, \dots, 0, 1, 0, \dots),
  $$
  où l’unique $1$ est à la position $h$, soit $x^{(h)} = (\delta_{m h})_{m \in \mathbb{N}}$.  
- Cette suite appartient bien à $\ell^2(\mathbb{N})$ et  
  $$
  \| x^{(h)} \|_{\ell^2} = 1.
  $$
- On calcule :  
  $$
  J(x^{(h)}) = J_1(x^{(h)}) + J_2(x^{(h)}).
  $$
  $$
  J_1(x^{(h)}) = 0, \quad J_2(x^{(h)}) = \frac{1}{h+1}.
  $$
  $$
  J(x^{(h)}) = \frac{1}{h+1} \to 0 \quad \text{quand } h \to \infty.
  $$
- Ainsi, $\inf J = 0$ est atteint asymptotiquement.

**$J$ est coercive**  

On a montré que  
$$
J(x) \geq J_1(x) = (\| x \|^2 - 1)^2.
$$
Définissons  
$$
\varphi(t) = (t^2 - 1)^2.
$$
Alors  
$$
\lim_{t \to \infty} \varphi(t) = +\infty.
$$
Donc $J$ est **coercive**.

**$\ell^2(\mathbb{N})$ est fermé, mais l’infimum de $J$ n’est pas atteint**  

On sait que  
$$
\inf J = 0.
$$
Supposons qu’il existe $x$ tel que $J(x) = 0$. Alors  
$$
J_1(x) + J_2(x) = 0.
$$
Comme $J_1(x) \geq 0$ et $J_2(x) \geq 0$, cela implique que  
$$
J_1(x) = 0 \quad \text{et} \quad J_2(x) = 0.
$$
Or,  
$$
J_1(x) = 0 \Rightarrow \| x \| = 1.
$$
Et  
$$
J_2(x) = 0 \Rightarrow \forall n \in \mathbb{N}, \quad \frac{x_n^2}{n+1} = 0 \Rightarrow x_n = 0.
$$
Donc $x = 0$, ce qui contredit $\| x \| = 1$.  

**Conclusion :** Il n’existe pas de $x$ tel que $J(x) = 0$. L’**infimum** de $J$ **n’est pas atteint** dans $\ell^2(\mathbb{N})$.

### **3) Sur un convexe fermé (en dimension infinie)**  

On considère un **espace de Hilbert** $V$.  

### **Définition (convergence faible)**  

Une suite $(x_n)$ **converge faiblement** vers $x^*$ si  
$$
\forall y \in V, \quad \langle y, x_n \rangle \to \langle y, x^* \rangle.
$$
Notation :  
$$
x_n \rightharpoonup x^*.
$$

### **Remarque :**  
Cette notion définit une **topologie faible**, c'est-à-dire une topologie où les ouverts et fermés sont définis via la convergence faible.  
C'est la **topologie la plus grossière** permettant la continuité des applications  
$$
L_y : V \to \mathbb{R}, \quad x \mapsto \langle y, x \rangle.
$$
pour tout $y \in V$.

### **Exemple :**  

- **Ensemble ouvert :**  
  $$
  L_y^{-1}(]a, b[) = \{ x \in V \mid a < \langle y, x \rangle < b \}.
  $$

- **Ensemble fermé :**  
  $$
  L_y^{-1}(]-\infty, \alpha]) = \{ x \in V \mid \langle y, x \rangle \leq \alpha \}.
  $$

### **Exemple : convergence faible dans $L^2(]0,1[)$**  

On considère l’espace de Hilbert  
$$
V = L^2(]0,1[).
$$
La suite de fonctions  
$$
(x_n) = e^{2i\pi n x}
$$
satisfait  
$$
\| x_n \| = \sqrt{\int_0^1 |e^{2i\pi n x}|^2 \, dx} = 1.
$$

Pour tout $y \in L^2(]0,1[)$, on a  
$$
\langle y, x_n \rangle = \int_0^1 y(x) e^{2i\pi n x} \, dx.
$$
Par le **lemme de Riemann-Lebesgue**, cette intégrale tend vers 0 lorsque $n \to \infty$, ce qui implique que  
$$
x_n \rightharpoonup 0 \quad \text{(convergence faible)}.
$$

---

### **Propositions :**  

1. **Tout convexe fermé pour la topologie forte est fermé pour la topologie faible.**  
2. **Si $J$ est convexe et semi-continue inférieure pour la topologie forte, alors elle est aussi semi-continue inférieure pour la topologie faible.**

**Preuve :**  

### **1.**  
![[Pasted image 20250203152814.png]]
Tout convexe **fermé** est l’intersection des demi-espaces fermés qui le contiennent.  

$$
\text{convexe} \subset \bigcap \text{(demi-espaces qui le contiennent)}
$$

$$
\bigcap \text{(demi-espaces qui le contiennent)} \subset \text{convexe}
$$

Supposons que ce **n’est pas le cas**, que $x$ ne soit pas dans le convexe.  

D’après le **théorème de Hahn-Banach**, on peut **séparer** $x$ et le convexe fermé par un hyperplan.

![[Pasted image 20250203152830.png]]

**donc il existe un demi-espace qui contient le convexe mais pas $x$.**  

**Impossible** puisque  
$$
x \in \bigcap \text{(demi-espaces qui incluent le convexe)}
$$

**Demi-espace fermé :**  
$$
\{ x \in V \mid \langle y, x - x_0 \rangle \leq \alpha \}
$$
$$
= \{ x \in V \mid \langle y, x \rangle \leq \alpha + \langle y, x_0 \rangle \}
$$
$$
= \{ x \in V \mid \langle y, x \rangle \leq \beta \}
$$

**Fermé pour la topologie faible.**  

**Une intersection de fermés pour la topologie forte est fermée pour la topologie faible.**  

### **2.**  

$J$ **S.C.I., convexe pour la topologie forte**  

**S.C.I.**  
$$
\Rightarrow \text{ sous-ensemble de niveaux } \{ x \in V \mid J(x) \leq \alpha \} = J^{-1} (]-\infty, \alpha])
$$
fermé pour la topologie forte.  

**Convexe**  
$$
\Rightarrow \text{ sous-ensembles de niveaux sont convexes.}
$$

**les sous-ensembles de niveaux sont convexes fermés pour la topologie forte donc fermés pour la topologie faible** *(d’après 1.)*  

**donc $J$ est S.C.I. pour la topologie faible.**  

---

**Remarque :**  
$$
\| \cdot \| : V \to \mathbb{R}, \quad x \mapsto \| x \|
$$
est **convexe**  
$$
\| \lambda x + (1-\lambda) y \| \leq \lambda \| x \| + (1-\lambda) \| y \|
$$
et **continue pour la topologie forte**  

donc **S.C.I. pour la topologie forte.**  

**donc $\| \cdot \|$ est S.C.I. pour la topologie faible.**  

$$
x_n \rightharpoonup x^* \Rightarrow \liminf \| x_n \| \geq \| x^* \|
$$

**Elle n’est en général pas continue pour la topologie faible en dimension infinie.**
$$
x_n = e^{2i\pi n x} \quad \underset{n \to \infty}{\longrightarrow} \quad 0
$$

et  
$$
\| x_n \| = 1 \quad \Rightarrow \quad \| 0 \| = 0
$$
mais  
$$
\liminf \| x_n \| \geq \| 0 \| = 0
$$


