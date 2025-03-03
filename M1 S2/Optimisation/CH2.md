# Chapitre 2 : Existence et unicité

## I Existence d’un minimum  

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

"Il y a plus de suites convergentes au sens faible qu'au sens fort, donc il est plus contraignant d’être continu au sens faible qu’au sens fort."

**Théorème : (Compacité séquentielle)**  
$(x_n)$ une suite bornée dans $V$. On peut extraire une sous-suite qui converge au sens faible.  

**Remarque :**  
Autre formulation : $B(0, M)$ est séquentiellement compact pour la topologie faible.  


**Preuve :**  
On suppose que $V$ est séparable (l’ensemble $F$ est dense dans $V$).  
Alors il existe une base hilbertienne $(e_i)_{i \in \mathbb{N}^*}$.  

- $(e_i)$ est orthonormée : $(e_i, e_j) = \delta_{ij} \quad \forall i, j$.  
- $\text{Vect}(e_i) = V$.  

On a également, $\forall \mu \in V$,  
$$
\mu = \sum_{i=1}^{+\infty} (\mu, e_i) e_i, \quad ||\mu||^2 = \sum_{i=1}^{+\infty} |(\mu, e_i)|^2.
$$

$(x_n)$ étant bornée, $((x_n, e_i))_{n \in \mathbb{N}^*}$ est aussi bornée dans $\mathbb{R}$ car  
$$
|(x_n, e_i)| \leq ||x_n|| \cdot ||e_i|| = ||x_n|| \quad \forall i \in \mathbb{N}^*.
$$

On applique un procédé d’extraction diagonale :  

- Soit $(x_{j_1(n)})$ une sous-suite de $(x_n)$ telle que $(x_{j_1(n)}, e_1)_{n}$ converge vers une limite notée $\alpha_1 \in \mathbb{R}$, avec $j_1 : \mathbb{N}^* \to \mathbb{N}^*$ croissante.  
- Soit $(x_{j_2(n)})$ une sous-suite de $(x_{j_1(n)})$ telle que  
$$
(x_{j_2(n)}, e_2)
$$
converge vers une limite notée $\alpha_2 \in \mathbb{R}$, avec  
$$
j_2 : \mathbb{N}^* \to \mathbb{N}^* \text{ croissante}.
$$
- Soit $(x_{j_3(n)})$ une sous-suite de $(x_{j_2(n)})$ telle que  
$$
(x_{j_3(n)}, e_3)
$$
converge vers une limite notée $\alpha_3 \in \mathbb{R}$, avec  
$$
j_3 : \mathbb{N}^* \to \mathbb{N}^* \text{ croissante}.
$$

On considère ensuite $(x_{j_n}) = (y_n)$.  
Soit $k \in \mathbb{N}^*$. À partir d’un rang $k$, $(x_{j_k(n)}, e_k)$ est une sous-suite de $(x_{j_{k-1}(n)}, e_k)$, donc $(x_{j_k(n)}, e_k)$ converge vers $\alpha_k$.  

On montre que $(y_n)$ converge faiblement vers  
$$
\alpha = \sum_{i=1}^{+\infty} \alpha_i e_i.
$$


En effet, pour tout $N \in \mathbb{N}^*$,  

$$
\sum_{i=1}^{N} \alpha_i e_i
$$

On a :  
$$
||y_n|| \sum_{i=1}^{N} |\alpha_i| ||e_i||
$$

$$
\leq M \sum_{i=1}^{N} |\alpha_i| ||e_i||
$$

(car $(x_n)$ est bornée)  

$$
\sum_{i=1}^{N} \alpha_i (y_n, e_i) \to \alpha_i.
$$

Donc on obtient la limite :  
$$
\left( \sum_{i=1}^{N} \alpha_i^2 \right)^{1/2} \leq M.
$$

$$
\Rightarrow \left( \sum_{i=1}^{\infty} \alpha_i^2 \right)^{1/2} \leq M \Rightarrow \sum_{i=1}^{N} \alpha_i^2 \leq M^2.
$$

En prenant la limite $N \to \infty$, on obtient :  
$$
\sum_{i=1}^{\infty} \alpha_i^2 \leq M.
$$

Donc  
$$
\sum_{i=1}^{\infty} \alpha_i e_i
$$  
est bien défini et appartient à $V$.  

**Convergence faible :**  
Soit $u \in V$ et $N \in \mathbb{N}$,  

$$
| (y_n - \alpha, u) | = | (y_n - \alpha, u) |
$$

$$
= | (y_n - \sum_{i=1}^{N} \alpha_i e_i, u) + (\sum_{i=N+1}^{\infty} \alpha_i e_i, u) |.
$$

$$
\leq | (y_n - \sum_{i=1}^{N} \alpha_i e_i, u) | + | (\sum_{i=N+1}^{\infty} \alpha_i e_i, u) |.
$$

$$
\leq \sum_{i=1}^{N} | u_i (y_n - \alpha, e_i) | + \sum_{i=N+1}^{\infty} |u_i e_i|.
$$

$$
\leq ||y_n - \alpha|| \sum_{i=1}^{N} ||u_i|| ||y_n - \alpha, e_i|| + \sum_{i=N+1}^{\infty} |u_i|^2.
$$

On choisit $N \in \mathbb{N}^*$ de sorte que  
$$
\sum_{i=N+1}^{\infty} \mu_i^2 \leq \varepsilon.
$$
Puis, une fois $N$ fixé, on choisit $k \in \mathbb{N}^*$ de sorte que, pour $n \geq k$,  
$$
| (y_n, e_i) - (\alpha, e_i) | \leq \varepsilon \quad \forall i \in [1, N].
$$

$$
| (y_n, u) - (\alpha, u) | \leq \varepsilon \left( \sum_{i=1}^{N} |u_i| + M + |u| \right).
$$

$$
(y_n, u) \to (\alpha, u).
$$

**Remarque :** Si $V$ n’est pas séparable, on se place dans l’espace vectoriel engendré par $(x_n)$.

### **Théorème :**  
Soit $J: K \subset V \to \mathbb{R}$ semi-continue inférieurement (continue), convexe pour $K$ convexe fermé.  
Si $K$ n’est pas borné, on suppose $J$ coercive.  
Alors $J$ admet un minimum pour $K$.

### **Preuve :**  
Soit $(x_n)$ une suite minimisante :  
$$
J(x_n) \to \inf_K J.
$$

$(x_n)$ est bornée (soit parce que $K$ est borné, soit parce que $J$ est coercive).  

On peut extraire une sous-suite faiblement convergente $(x_{n_k})$ de limite notée $x^* \in V$ (d’après la proposition précédente).  

$J$ étant convexe et semi-continue inférieurement,  
$$
x^* \in K,
$$
car $K$ étant convexe et fermé, $K$ est fermé pour la topologie faible.

$J$ étant convexe et SCI, elle est donc SCI pour la topologie faible.  

Donc on a :  
$$
\liminf J(x_{n_k}) \geq J(x^*) \geq \inf_K J.
$$
$$
= \lim J(x_{n_k})
$$
$$
= \inf_K J.
$$

Donc  
$$
J(x^*) = \inf_K J.
$$


**Exemple :**  

**Existence d’un minimiseur pour :**  

$$
J: L^2(J_0,1) \to \mathbb{R}
$$

$$
f \mapsto \int_0^1 \rho f^2(t) dt - \int_0^1 g(t) f(t) dt.
$$

Soit  

$$
K = \{ f \in L^2(J_0,1) \mid \int_0^1 f(t) dt = 1 \}.
$$

$J$ est bien définie sur $L^2(J_0,1)$ :

$$
L^2(J_0,1) = \{ f: J_0,1 \to \mathbb{R} \mid \int_0^1 f^2(t) dt < \infty \}.
$$

Car  
$$
L^2(J_0,1) \subset L^1(J_0,1) \quad \left( \int_0^1 |f(t)| dt \leq \sqrt{\int_0^1 f^2(t) dt} \sqrt{\int_0^1 dt} = 1 \right).
$$

$$
\Rightarrow J(f) \text{ est bien définie}.
$$

$J$ est **semi-continue inférieurement** (SCI) car elle est continue.  

En effet, si  
$$
f_n \to f \text{ dans } L^2,
$$

$$
\int_0^1 f_n^2(t) dt + g(t) f_n(t) \to 0.
$$

Alors  
$$
\int_0^1 f_n^2(t) + g(t) f_n(t) dt = ||f_n||^2 - \int_0^1 g f_n dt \to 0,
$$

car  
$$
| ||f_n|| - ||f|| | \leq || f_n - f ||.
$$

**Continuité de la norme :**  

$$
\left| \int_0^1 g f_n dt - \int_0^1 g f dt \right| \leq \int_0^1 |g| |f_n - f| dt \leq || g ||_{L^2} || f_n - f ||_{L^2}.
$$

D’où  
$$
J(f_n) \to J(f).
$$

$J$ est **convexe**. En effet,  

$$
F(t) = t^2 - t \text{ est convexe, car } F''(t) = 2 \geq 0.
$$

Donc  
$$
\forall f_1, f_2 \in L^2(J_0,1),
$$

$$
F(\theta f_1 + (1-\theta) f_2) \leq \theta F(f_1) + (1-\theta) F(f_2), \quad \forall \theta \in [0,1].
$$

Donc, en intégrant, on a :

$$
J(\theta f_1 + (1-\theta) f_2) = \int_0^1 F(\theta f_1 (t) + (1-\theta) f_2 (t)) dt
$$

$$
\leq \theta \int_0^1 F(f_1 (t)) dt + (1-\theta) \int_0^1 F(f_2 (t)) dt
$$

$$
= \theta J(f_1) + (1-\theta) J(f_2).
$$

**$J$ est convexe.**  

$$
J(f) = \int_0^1 \rho f^2 (t) dt - \int_0^1 g (t) f (t) dt.
$$

$$
= \mu || f ||_{L^2}^2 - (\beta, 1)_{L^2}^2 \geq \mu || f ||_{L^2}^2 - ||\beta||_{L^2} || f ||_{L^2}.
$$

Donc  
$$
J(f) \geq \varphi (|| f ||).
$$
avec $\varphi (t) = t^2 - t$ et  
$$
\lim_{t \to +\infty} \varphi (t) = +\infty.
$$

**$K$ est convexe :**  

$$
K = \{ f \in L^2 (J_0,1) \mid \int_0^1 f (t) dt = \alpha \}.
$$

$$
\forall f_1, f_2 \in K, \forall \theta \in [0,1],
$$

$$
\int_0^1 (\theta f_1 (t) + (1-\theta) f_2 (t)) dt = \theta \int_0^1 f_1 (t) dt + (1-\theta) \int_0^1 f_2 (t) dt.
$$

$$
= \theta + (1-\theta) = 1.
$$

Donc  
$$
\theta f_1 + (1-\theta) f_2 \in K.
$$

**Remarque :** $K$ est un **espace affine**,  
$$
z + \{ g \in L^2 (J_0,1) \mid \int_0^1 g (t) dt = 0 \}.
$$

( **Espace vectoriel** )

$K$ est fermé car image réciproque de $\{ 1 \}$ par les fonctions continues.  

$$
g: f \in L^2 (J_0,1) \to \int_0^1 t f (t) dt = (f, 1)_{L^2}
$$

$g$ est linéaire (ok).  

$$
| g (f) | = \left| \int_0^1 f (t) dt \right| \leq || 1 ||_{L^2} || f ||_{L^2} = \left( \int_0^1 1^2 dt \right)^{1/2} || f ||_{L^2}.
$$

$$
g(f) = \frac{1}{\sqrt{3}} ||f||_{L^2} \quad \Rightarrow g \text{ est continue}.
$$

D’après le théorème, il existe un minimiseur.

---

## II. Unicité

#### **Propriétés :**  
Soit $J: K \subset V \to \mathbb{R}$.

1) Si $J$ est convexe et $K$ convexe, alors tout minimum local est un minimum global. De plus, l’ensemble des minimiseurs forme un ensemble convexe.  

2) Si $J$ est strictement convexe et $K$ convexe, alors il y a au plus un minimiseur. L’ensemble des minimiseurs est réduit à un singleton $\{ x^* \}$ ou est vide.

### **Preuve :**  
1) Soit $x^*$ un minimiseur local :  

$$
\exists \varepsilon > 0, \forall x \in B(x^*, \varepsilon),
$$

$$
J(x) \geq J(x^*).
$$

Soit $y^* \in K$ et $\eta > 0$ tel que  

$$
\eta x^* + (1-\eta) y \in B(x^*, \delta).
$$

En prenant  

$$
\eta = \frac{\varepsilon}{2} \frac{1}{|| y - x^* ||} \Rightarrow || \eta x^* + (1 - \eta) y - x^* ||
$$

$$
= || \eta (y - x^*) || = \eta || y - x^* || = \frac{\varepsilon}{2}.
$$

On a donc  

$$
J(x^*) \leq J(x^* + \eta (y - x^*)) \leq \eta J(y) + (1 - \eta) J(x^*).
$$

( **Convexité** )

$$
J(x^*) = \inf_{B(x^*, \varepsilon)} J.
$$

$$
\Rightarrow \eta J(x^*) \leq \eta J(y) \Rightarrow J(x^*) \leq J(y).
$$

Donc l’ensemble des minimiseurs est un $K$ convexe.

2. **Si $J$ est strictement convexe**, alors pour deux points minimaux distincts $x_1^*$ et $x_2^*$ dans $K$ et pour tout $\theta \in (0,1)$, on a  
$$
J\bigl(\theta x_1^* + (1 - \theta) x_2^*\bigr) \;<\; \theta\,J(x_1^*) \;+\; (1 - \theta)\,J(x_2^*).
$$
Étant donné que $x_1^*$ et $x_2^*$ sont tous deux des minima globaux, le membre de droite vaut $J(x_1^*)$ (ou $J(x_2^*$), ce qui contredit la définition même d’un minimum global. Par conséquent, il ne peut exister qu’un seul point minimal.