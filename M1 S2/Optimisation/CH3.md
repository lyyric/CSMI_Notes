# Chapitre 3 : Conditions d’optimalité

## I) Conditions d’optimalité (sans contraintes.)

**Proposition :** Soit $f : K \subset V \to \mathbb{R}$ avec $K$ convexe, et $f$ différentiable.  

- Si $f$ admet un minimum local en $\bar{x} \in K$,  
  alors $\langle Df(\bar{x}), y - \bar{x} \rangle \geq 0 \quad \forall y \in K$.  
- Si $f$ est convexe, alors on a équivalence :  
$$
(\bar{x} \text{ minimum global}) \iff (\bar{x} \text{ minimum local}) \iff (\nabla f(\bar{x}), y - \bar{x} ) \geq 0, \quad \forall y \in K.
$$

**Remarque :** on a  
$$
\langle \nabla f(\bar{x}), y - \bar{x} \rangle = \frac{\partial f(\bar{x})}{\partial(y - \bar{x})}
$$

C’est la dérivée directionnelle dans la direction $y - \bar{x}$.  

![[CSMI_Notes/M1 S2/Optimisation/assets/image.png|300x240]]

**Inégalité d’Euler :** $f$ est **croissante** dans toutes les directions contenues dans $K$.  

- Propriété ici énoncée dans un **Hilbert**.  
  Valide aussi dans un **espace de Banach** :  
$$
Df(\bar{x})(y - \bar{x}) \geq 0 \quad \forall y \in K.
$$

**Preuve :**  

- Soit $\bar{x}$ un minimum local de $f$ et $y \in K$.  

Alors, pour tout $t \in [0,1]$, $\bar{x} + t(y - \bar{x}) \in K$ (car $K$ est convexe).  

Comme $\bar{x}$ est un minimum local, il existe $\delta > 0$ tel que $\bar{x}$ soit un minimum de $f$ sur $B(\bar{x}, \delta) \cap K$.  
Pour tout $t < \dfrac{\delta}{2 \| y - \bar{x} \|}$, on a $\bar{x} + t(y - \bar{x}) \in K \cap B(\bar{x}, \delta)$.  

Donc :  
$$
f(\bar{x}) \leq f(\bar{x} + t(y - \bar{x}))
$$
$$
\Rightarrow 0 \leq \frac{f(\bar{x} + t(y - \bar{x})) - f(\bar{x})}{t} = \frac{\langle Df(\bar{x}), (y - \bar{x}) \rangle + o(1)}{t}
$$
$$
= \langle Df(\bar{x}), y - \bar{x} \rangle + o(1).
$$

On obtient l’inéquation d’Euler en prenant la limite  
$$
t \to 0^+.
$$

- **Si** $f$ **est convexe** et que l’inéquation d’Euler est vérifiée (en $\bar{x}$), alors :  

$$
\forall y \in K, \quad f(y) \geq f(\bar{x}) + \langle Df(\bar{x}), y - \bar{x} \rangle \geq f(\bar{x}).
$$

  - La première inégalité vient du fait que $f$ est **convexe**.  
  - La seconde inégalité provient de **l’inéquation d’Euler** :  
$$
\langle Df(\bar{x}), y - \bar{x} \rangle \geq 0.
$$

Donc, $\bar{x}$ est un **minimum global**.

**Proposition :** *(Minimisation sur un ouvert)*  

Soit $f : K \subset V \to \mathbb{R}$.  

- Si $f$ admet un minimum local en $\bar{x} \in K^\circ$, alors  
$$
\nabla f(\bar{x}) = 0 \quad \text{[Équation d’Euler]}.
$$

- Si $f$ est **convexe**, on a équivalence :  
$$
(\bar{x} \text{ minimum local}) \iff (\nabla f(\bar{x}) = 0).
$$

**Remarque :** *Version espace de Banach*  

L’équation d’Euler s’écrit :  
$$
Df(\bar{x}) = 0 \iff Df(\bar{x})(h) = 0 \quad \forall h \in V.
$$

**Preuve :**  

$\bar{x}$ est un minimum local appartenant à $K^\circ$, donc il existe $\delta > 0$ tel que $\bar{x}$ soit un minimum de $f$ sur $B(\bar{x}, \delta) \subset K^\circ \subset K$.  

On montre que pour tout $h \in V$,  
$$
\langle \nabla f(\bar{x}), h \rangle = 0.
$$

![[CSMI_Notes/M1 S2/Optimisation/assets/image-1.png|264x186]]

En effet, il existe $t > 0$ tel que $\bar{x} \pm th \in B(\bar{x}, \delta)$.  

$f$ admettant un minimum local en $\bar{x}$ sur $B(\bar{x}, \delta)$ convexe, on a l’inégalité d’Euler :  
$$
\langle \nabla f(\bar{x}), (\bar{x} \pm th) - \bar{x} \rangle \geq 0.
$$
$$
\iff \pm t \langle \nabla f(\bar{x}), h \rangle \geq 0.
$$
$$
\iff \pm \langle \nabla f(\bar{x}), h \rangle \geq 0.
$$
$$
\iff \langle \nabla f(\bar{x}), h \rangle = 0.
$$

- **Ceci est vrai pour tout** $h \in V$, **donc en particulier pour** $h = Df(\bar{x})$.  
  Donc  
$$
0 = \langle Df(\bar{x}), Df(\bar{x}) \rangle = \| Df(\bar{x}) \|^2
$$  
  donc  
$$
Df(\bar{x}) = 0.
$$

**Proposition :** *(Condition d’optimalité d’ordre 2 dans un ouvert)*  

Soit $f : K \subset V \to \mathbb{R}$, $C^1$ et **deux fois différentiable** en $\bar{x} \in K^\circ$.  

- **Si** $f$ admet un minimum local en $\bar{x} \in K^\circ$, alors  
$$
\nabla f(\bar{x}) = 0 \quad \text{et} \quad \langle D^2 f(\bar{x}) h, h \rangle \geq 0 \quad \forall h \in V.
$$

- **Réciproquement**,  
	- si $\nabla f(\bar{x}) = 0$ et  
$$
\langle D^2 f(\bar{x}) h, h \rangle \geq \alpha \| h \|^2
$$  
	    avec $\alpha > 0$,  
	    alors $\bar{x}$ est un minimum local.
    - **Si** $Df(\bar{x}) = 0$ **et**  
$$
\langle D^2 f(\bar{x}) h, h \rangle \geq 0 \quad \forall h \in V
$$
	  et $\forall x$ dans un voisinage de $\bar{x}$, alors $f$ admet un minimum local en $\bar{x}$.  

**Remarque :**  

En **dimension finie**, la condition  
$$
\langle D^2 f(\bar{x}) h, h \rangle \geq \alpha \| h \|^2
$$  
est équivalente au caractère **défini-positif** de $D^2 f(\bar{x})$.  

En effet :  
$$
\langle D^2 f(\bar{x}) h, h \rangle \geq \alpha \| h \|^2
$$
$$
\Rightarrow \langle D^2 f(\bar{x}) h, h \rangle > 0 \quad \forall h \neq 0
$$
$$
\Rightarrow D^2 f(\bar{x}) h = 0 \Rightarrow h = 0.
$$

Réciproquement :  
$$
\langle D^2 f(\bar{x}) h, h \rangle \geq \lambda_{\min} \| h \|^2
$$

avec $\lambda_{\min}$ la plus petite valeur propre de $D^2 f(\bar{x})$.

**Preuve :**  
- **Pour tout** $t$ **suffisamment petit**,  
$$
\forall h \in V, \quad 0 \leq \frac{f(\bar{x} + t h) - f(\bar{x})}{t}
$$
$$
= t \langle Df(\bar{x}), h \rangle + \frac{t^2}{2} \langle D^2 f(\bar{x}) h, h \rangle + o(t^2).
$$
- **En divisant par** $t^2 > 0$, **on obtient**  
$$
0 \leq \frac{1}{2} \langle D^2 f(\bar{x}) h, h \rangle + o(1).
$$

  **Puis la propriété vaut en prenant la limite** $t \to 0^+$.  

- **Supposons** $Df(\bar{x}) = 0$ **et**  
$$
\langle D^2 f(\bar{x}) h, h \rangle \geq \alpha \| h \|^2, \quad \forall h \in V.
$$
On a  
$$
f(\bar{x} + h) = f(\bar{x}) + \langle Df(\bar{x}), h \rangle + \frac{1}{2} \langle D^2 f(\bar{x}) h, h \rangle + o(\|h\|^2).
$$

Comme $Df(\bar{x}) = 0$, cela donne  
$$
f(\bar{x} + h) \geq f(\bar{x}) + \frac{\alpha}{2} \| h \|^2 + \| h \|^2 \varphi(\| h \|),
$$

où $o(\| h \|^2) = \| h \|^2 \varphi(\| h \|)$.  

D’où  

$$
f(\bar{x} + h) - f(\bar{x}) \geq \| h \|^2 \left( \frac{\alpha}{2} + \varphi(\| h \|) \right).
$$  

Comme $\varphi(r) \to 0$ lorsque $r \to 0$, il existe $\delta > 0$ tel que $\forall r < \delta$, on ait $|\varphi(r)| < \frac{\alpha}{4}$.  

Donc pour tout $h$, $\| h \| < \delta$,  
$$
f(\bar{x} + h) - f(\bar{x}) \geq \| h \|^2 \left( \frac{\alpha}{2} - \frac{\alpha}{4} \right) > 0.
$$
Ainsi, $\bar{x}$ est un **minimum local**.

- **Si**  
$$
\langle D^2 f(\bar{x}) h, h \rangle \geq 0 \quad \forall h \in V, \quad \forall x \text{ dans un voisinage de } \bar{x}
$$  
  alors $f$ est **convexe** sur ce voisinage, et donc l’équation d’Euler $Df(\bar{x}) = 0$ implique que $\bar{x}$ est un **minimum local** sur ce voisinage.  

## II) Applications

### 1) Minimisation d’une fonction quadratique (dimension finie)

On souhaite déterminer le minimum de  
$$
f(x) = \frac{1}{2} \langle A x, x \rangle - \langle b, x \rangle
$$
avec $A \in S_m(\mathbb{R})$ (**symétrique**) et $b \in \mathbb{R}^n$.

**Rappel :**  
$$
\nabla f(x) = A x - b
$$
$$
\nabla^2 f(x) = A.
$$

**Cas $A \in S_n^{++}(\mathbb{R})$ (définie positive)**  
$f$ est **strictement convexe**, donc en particulier **convexe**.  
Donc :  
$$
(\bar{x} \text{ minimum local}) \iff (\nabla f(\bar{x}) = 0) \iff A \bar{x} = b.
$$

Comme $A$ est **inversible**, il existe une **unique solution** à cette équation :  
$$
\bar{x} = A^{-1} b.
$$

C’est **l’unique minimum** de $f$.

**Cas $A \in S_n^{+}(\mathbb{R})$ (semi-définie positive)**  
- $f$ est **convexe**, donc  
$$
(\bar{x} \text{ minimum local}) \iff (\nabla f(\bar{x}) = 0) \iff A \bar{x} = b.
$$
- **Si** $b \notin \operatorname{Im} A$, **l’équation d’Euler n’a pas de solution**, donc $f$ **n’admet pas de minimum**.  

- **Si** $b \in \operatorname{Im} A$, **l’équation d’Euler admet une infinité de solutions** :  
  - Si $\bar{x}$ est une solution, alors  
    $$
    \bar{x} + h, \quad h \in \ker A
    $$  
    définit un **sous-espace affine de solutions** et donc de **minima**.  

**Cas $A$ non positive**  

Il existe une **valeur propre négative** $\lambda_1 < 0$.  

Notons $e_1$ un **vecteur propre associé**, on a :  

$$
f(t e_1) = \frac{t^2}{2} \langle A e_1, e_1 \rangle - t \langle b, e_1 \rangle.
$$
$$
= \frac{\lambda_1}{2} t^2 \langle e_1, e_1 \rangle - t \langle b, e_1 \rangle.
$$

Comme $\lambda_1 < 0$, on a  
$$
f(t e_1) \to -\infty \quad \text{lorsque } t \to +\infty.
$$
Donc  
$$
\inf_{x \in \mathbb{R}^n} f(x) = -\infty.
$$
$f$ **n’admet pas de minimum**.  

### **2) Minimisation d’une fonctionnelle quadratique (dimension infinie)**  

#### **Proposition (Lax-Milgram)**  

Soit $a : V \times V \to \mathbb{R}$ une **forme bilinéaire continue et coercive**, vérifiant :  

- **Continuité** :  
$$
(\exists M \geq 0), \quad |a(u, v)| \leq M \| u \| \| v \|, \quad \forall u, v \in V.
$$
- **Coercivité** :  
$$
(\exists \alpha > 0), \quad a(u, u) \geq \alpha \| u \|^2, \quad \forall u \in V.
$$

Soit $\ell : V \to \mathbb{R}$ une **forme linéaire continue**, vérifiant :  
$$
(\exists C > 0), \quad | \ell(u) | \leq C \| u \|.
$$
On considère la fonctionnelle **énergie** :  
$$
f(u) = \frac{1}{2} a(u, u) - \ell(u).
$$

Alors, $f$ **admet un unique minimiseur**, qui est **solution de** :  
$$
a(u, v) = \ell(v), \quad \forall v \in V.
$$
*(Principe des travaux virtuels)*.

### **Preuve :**  

$$
Df(u)(h) = a(u, h) - \ell(h)
$$
$$
D^2 f(u)(h, h) = a(h, h) \geq \alpha \| h \|^2.
$$

Comme $a$ est **coercive**, on en déduit que $f$ est **$\alpha$-convexe**.  

- $f$ est **semi-continue inférieurement (s.c.i)** (car continue).  
- $f$ est **convexe** (car $\alpha$-convexe).  
- $f$ est **coercive** (car $\alpha$-convexe) sur $V$ **convexe fermé**.  

Donc, **$f$ admet un minimiseur sur $V$.**  

- Le **minimiseur est unique**, car $f$ est **strictement convexe** (car $\alpha$-convexe).  
- $f$ étant **convexe**, le minimiseur est caractérisé par **l’équation d’Euler**.

$$
Df(u)(h) = 0 \quad \forall h \in V
$$
$$
\iff a(u, h) = \ell(h) \quad \forall h \in V.
$$

**Exemple : Espace de Sobolev**  
$$
H^1(]0,1[) = \{ u \in L^2(]0,1[) \mid u' \in L^2(]0,1[) \}
$$

- **Espace de Hilbert**  
  - **Produit scalaire :**  
$$
(u, v)_H = (u, v)_{L^2} + (u', v')_{L^2}
$$
  - **Norme :**  
$$
\| u \|_H = \sqrt{\| u \|_{L^2}^2 + \| u' \|_{L^2}^2}
$$

- **Espace $H^1_0(]0,1[)$**  
$$
H^1_0(]0,1[) = \{ u \in H^1(]0,1[) \mid u(0) = u(1) = 0 \} \subset H^1(]0,1[).
$$

  - **Norme :**  
$$
\| u \|_{H^1_0} = \| u' \|_{L^2}.
$$

  *(Norme équivalente à $\| u \|_H$ sur $H^1_0$).*

**Minimisation dans $H^1_0(]0,1[)$**  
$$
\forall u \in H^1_0(]0,1[), \quad f(u) = \frac{1}{2} \int_0^1 u'(x)^2 \,dx - \int_0^1 f(x) u(x) \,dx.
$$

- **Énergie élastique :**  
$$
\frac{1}{2} a(u, u).
$$
- **Énergie potentielle :**  
$$
\ell(u).
$$
![[CSMI_Notes/M1 S2/Optimisation/assets/image-2.png]]

- **$a$ est bilinéaire** ✅  
- **$a$ est continue**  
$$
|a(u, v)| = \left| \int_0^1 u' v' \right| \leq \| u' \|_{L^2} \| v' \|_{L^2}
$$
  (Inégalité de Cauchy-Schwarz)  
$$
\leq M \| u \|_{H^1_0} \| v \|_{H^1_0}.
$$

- **$a$ est coercive**  
$$
a(u, u) = \int_0^1 u'^2 = \| u' \|^2_{L^2} = \| u \|^2_{H^1_0}.
$$

- **$\ell$ est linéaire** ✅  
- **$\ell$ est continue**  
$$
|\ell(u)| = \left| \int_0^1 f u \right| \leq \| f \|_{L^2} \| u \|_{L^2}.
$$
  (Inégalité de Cauchy-Schwarz)  
$$
\leq C \| f \|_{L^2} \| u \|_{H^1_0}.
$$

**Équation d’Euler**  

$$
\int_0^1 u'(x) v'(x) \,dx = \int_0^1 f(x) v(x) \,dx, \quad \forall v \in H^1_0(]0,1[).
$$

*(Formulation variationnelle - Calcul des variations)*  

### **3) Problème aux moindres carrés**  

Soit $A \in M_{m,n}(\mathbb{R})$ et $b \in \mathbb{R}^m$.  

On souhaite résoudre le problème :  
$$
\min_{x \in \mathbb{R}^n} \| A x - b \|^2.
$$

### **Existence d’un minimiseur**  

Cela revient à chercher le **projeté de $b$ sur $\operatorname{Im} A$** :  

$$
\min_{x \in \mathbb{R}^n} \| A x - b \|^2 = \min_{y \in \operatorname{Im} A} \| y - b \|^2.
$$

Définissons la fonction :  
$$
g(y) = \| y - b \|^2.
$$

$g(y)$ est **continue et coercive** car :  
$$
\| y - b \|^2 \geq (\| y \| - \| b \|)^2 = \varphi(\| y \|).
$$

$\operatorname{Im} A$ est **fermé**, donc il **existe un minimiseur** de $g(y)$ dans $\operatorname{Im} A$.  

Ainsi, il **existe $x_0 \in \mathbb{R}^n$ tel que** :  
$$
\| A x_0 - b \|^2 = \min_{x \in \mathbb{R}^n} \| A x - b \|^2.
$$

### **Caractérisation**  

On a :  
$$
f(x + h) = \| A(x + h) - b \|^2.
$$
Développant :  
$$
= \| A x - b + A h \|^2.
$$

$$
= \| A x - b \|^2 + 2 \langle A x - b, A h \rangle + \| A h \|^2.
$$

$$
= f(x) + 2 \langle A^T (A x - b), h \rangle + \frac{1}{2} \langle A^T A h, h \rangle.
$$

### **Identification**  

$f$ étant polynomiale, par identification :  
$$
\nabla f(x) = 2 A^T (A x - b).
$$
$$
\nabla^2 f(x) = 2 A^T A.
$$

### **Convexité**  

$f$ est **convexe** car $\nabla^2 f(x)$ est **semi-définie positive**.  

En effet :  
$$
\langle \nabla^2 f(x) h, h \rangle = 2 \langle A^T A h, h \rangle = 2 \| A h \|^2 \geq 0.
$$

### **Solution du problème**  

Donc, $\bar{x}$ est un **minimum** si et seulement si :  

$$
\nabla f(\bar{x}) = 0 \iff A^T A \bar{x} = A^T b.
$$

*(Équations normales)*  

### **Existence et unicité**  

- **Il existe toujours au moins une solution** à ce problème (cf. partie existence).  

- **Cas 1 :** Si $A^T A$ est **inversible**, alors il y a une **unique solution**, donc le minimiseur est unique.  

- **Cas 2 :** Si $A^T A$ **n’est pas inversible**, alors :  
$$
\bar{x} + h, \quad h \in \ker A^T A
$$
  
  est l’ensemble des solutions.  

### **Remarque :**  
$$
\ker A^T A = \ker A.
$$

En effet,  
$$
A x = 0 \Rightarrow A^T A x = 0.
$$

Réciproquement,  
$$
A^T A x = 0 \Rightarrow x^T A^T A x = 0 \Rightarrow \| A x \|^2 = 0 \Rightarrow A x = 0.
$$

Donc,  
$$
\dim \ker A + \operatorname{rg} A = n.
$$

Donc
$$
A^T A \text{ inversible} \iff \ker A^T A = \{ 0 \}.
$$
$$
\iff \ker A = \{ 0 \}.
$$
$$
\iff \operatorname{rg} A = n.
$$

Ainsi, $A^T A$ est inversible si et seulement si $A$ est **de rang plein** ($\operatorname{rg} A = n$).

## III. Conditions d'optimalité avec contraintes d'égalité

On souhaite minimiser $J: V \to \mathbb{R}$ sur  

$$
K = \{ x \in V \mid h(x) = 0 \},
$$

avec $h: V \to \mathbb{R}^p$  

$$
x \mapsto (h_1(x), \dots, h_p(x))
$$

**Remarque :** (car $K$ est un sous-espace vectoriel).  

Si $h_i(x) = \langle a_i, x \rangle$ alors  
$$
K = \{ x \in V \mid \langle a_i, x \rangle = 0, \quad \forall i \in I \}
$$
est un sous-espace vectoriel.

$K$ est un ensemble convexe. Donc si $x^* \in K$ est un minimum local de $J$, alors  
$$
\langle \nabla J(x^*), z - x^* \rangle \geq 0 \quad \forall z \in K.
$$
Posons $y = z - x^*$, alors  
$$
\langle \nabla J(x^*), y \rangle \geq 0 \quad \forall y \in K.
$$
Or, en remplaçant $y$ par $-y$ (car $-y \in K$ si $K$ est un sous-espace vectoriel), on obtient :  
$$
\langle \nabla J(x^*), -y \rangle \geq 0.
$$
Ce qui implique :  
$$
\langle \nabla J(x^*), y \rangle = 0 \quad \forall y \in K.
$$
Donc :  
$$
\nabla J(x^*) \perp K.
$$

![[image-3.png|330x203]]

Autrement dit, la projection de $\nabla J(x^*)$ sur $K$ est nulle.  

**"La seule manière de faire décroître $J$, c'est de sortir de $K$"**  

**Théorème des extrémas liés**  

Soit $J, h_2, \dots, h_p \in \mathcal{C}^1(V)$.  

Si $J$ admet un minimum local sur $K$ en $x^* \in K$  

et si $\nabla h_2(x^*), \dots, \nabla h_p(x^*)$ forment une famille libre,  

alors il existe $\lambda_2, \dots, \lambda_p \in \mathbb{R}$ tels que :
**(Euler-Lagrange)**
$$
\nabla J(x^*) + \lambda_2 \nabla h_2(x^*) + \dots + \lambda_p \nabla h_p(x^*) = 0.
$$
$$
h(x^*) = 0.
$$

$(\lambda_i)_{i \in \{2, \dots, p\}}$ : multiplicateurs de Lagrange.

**Interprétation**
$$
\nabla J(x^*) \in \text{Vect} \left( \nabla h_i(x^*) \right)_{i \in \{2, \dots, p\}} = \left( \text{Vect} \left( \nabla h_i(x^*) \right)_{i \in \{2, \dots, p\}} \right)^\perp.
$$
$$
= \left( \bigcap_{i=1}^{p} \text{Vect} \left( \nabla h_i(x^*) \right) \right)^\perp.
$$
$$
= \left\{ h \in V \mid \langle \nabla h_i(x^*), h \rangle = 0, \quad \forall i \in \{2, \dots, p\} \right\}^\perp.
$$
*approximation linéaire de* $K = \{ x \in V \mid h_i(x) = 0 \}$.

$\nabla J(x*)$ appartient à l'orthogonal de l'espace tangent à $K$.

= l'espace normal à $K$.

![[image-4.png|369x320]] ![[image-5.png|251x320]]

Vect $\nabla h_2(x^*), \nabla h_2(x^*))$ forme un plan normal à $K$.

**Preuve**  

**Direction admissible en $x^*$**  
Un $d \in V$ est admissible s'il existe une fonction $\eta : [0, \tau[ \to V$ de classe $\mathcal{C}^1$ telle que :

$$
\begin{cases}
\eta(t) \in K, \quad \forall t \in [0, \tau[ \\
\eta(0) = x^* \\
\eta'(0) = d.
\end{cases}
$$
![[image-6.png]]

**Minimisation**  
Si $x^*$ est un minimum local de $J$ sur $K$, alors pour toute direction admissible.

En effet,  
$$
J(\eta(t)) \geq J(x^*) \quad \text{pour } t \text{ suffisamment petit}
$$
ce qui implique :
$$
J(\eta(0)) + \nabla J(\eta(0)) \cdot \eta'(0) + o(t) \geq J(x^*).
$$

$$
\Rightarrow (\nabla J(x^*), d) + o(t) \geq 0
$$
$$
\Rightarrow (\nabla J(x^*), d) \geq 0.
$$

Si $d$ et $-d$ sont admissibles, alors :

$$
(\nabla J(x^*), d) = 0.
$$

On montre que tout élément de  
$$
\bigcap_{i=1}^{p} \text{Vect}(\nabla h_i(x^*))^\perp
$$
$$
= \left( \text{Vect}(\nabla h_i(x^*)) \right)^\perp
$$
est une direction admissible.

Soit $d \in \left( \text{Vect}(\nabla h_i(x^*)) \right)^\perp$. On considère :
$$
\varphi :
\begin{cases}
\mathbb{R}^{p+1} \to \mathbb{R}^p \\
(t, y_1, \dots, y_p) \mapsto h(x^* + t d + y_1 \nabla h_2(x^*) + \dots + y_p \nabla h_p(x^*))
\end{cases}
$$

$$
\varphi(0) = h(x^*) = 0
$$

$$
\partial_t \varphi (t, y_2, \dots, y_p) = \text{Jac} \ h (x^* + t d + y_2 \nabla h_2(x^*) + \dots + y_p \nabla h_p(x^*)) d
$$

$$
(\text{Jac} \ h \circ g = \text{Jac} \ h \circ \text{Jac} \ g)
$$

où  

$$
g : t \mapsto x^* + t d + y_2 \nabla h_2(x^*) + \dots + y_p \nabla h_p(x^*)
$$

$$
\Rightarrow \partial_t \varphi (0) = \text{Jac} \ h (x^*) d
$$

$$
= \begin{bmatrix}
\nabla h_2 \\ 
\vdots \\ 
\nabla h_p
\end{bmatrix} d
$$

$$
= \begin{bmatrix}
\langle \nabla h_2(x^*), d \rangle \\
\vdots \\
\langle \nabla h_p(x^*), d \rangle
\end{bmatrix}
= 0 \quad \text{(par définition de $d$)}
$$

$$
\partial_{y_j} \varphi_i (t, y_2, \dots, y_p) = (\nabla h_i (x^* + t d + y_2 \nabla h_2(x^*) + \dots + y_p \nabla h_p(x^*)), \nabla h_j (x^*))
$$

$$
\Rightarrow \partial_{y_j} \varphi_i (0) = (\nabla h_i (x^*), \nabla h_j (x^*))
$$

$$
\Rightarrow \text{Jac}_y \varphi (0) = (\langle \nabla h_i(x^*), \nabla h_j(x^*) \rangle)
$$

$$
= (\text{Jac} \ h(x^*)) (\text{Jac} \ h(x^*))^T
$$

$$
\text{Jac}_y \varphi (0) \text{ est inversible car}
$$

$$
\text{Im} \ \text{Jac} \ h(x^*)^T = \text{Im} \ \text{Jac} \ h(x^*) \text{Jac} \ h(x^*)^T
$$

$$
\text{or} \quad \text{rg} \ \text{Jac} \ h(x^*)^T + \dim \ker \text{Jac} \ h(x^*)^T = p
$$

$$
= p
$$

$$
(\nabla h_i(x^*)) \text{ forment une famille libre}
$$

$$
\Rightarrow \dim \ker \text{Jac} \ h(x^*)^T = 0
$$

$$
\Rightarrow \ker \text{Jac} \ h(x^*) \text{Jac} \ h(x^*)^T = \{0\}
$$

*(Théorème du rang)*

**D'après le théorème des fonctions implicites, il existe $y$ :**  
$$
O_1 \to O_2
$$
avec $O_1$ voisinage de $0 \in \mathbb{R}$  
et $O_2$ voisinage de $0 \in \mathbb{R}^p$ .

**Elle vérifie**  
$$
\eta(0) = x^*
$$
$$
h(\eta(t)) = \varphi(t, y_2(t), \dots, y_p(t)) = 0, \quad \forall t \in O_1
$$
$$
\eta'(0) = d + y_2'(0) \nabla h_2(x^*) + \dots + y_p'(0) \nabla h_p(x^*)
$$
$$
= d.
$$

**Conclusion**  

$$
\forall d \in (\text{Vect} (\nabla h_i(x^*), i \in \{2, \dots, p\}))^\perp
$$
est une direction admissible, et $-d$ aussi.

Donc,
$$
\nabla J(x^*) \perp \text{Vect} (\nabla h_i(x^*), i \in \{2, \dots, p\})^\perp.
$$
Donc,
$$
\nabla J(x^*) \in \left( \text{Vect} (\nabla h_i(x^*), i \in \{2, \dots, p\}) \right)^\perp.
$$

**Exemple**  

$$
J : \mathbb{R}^2 \to \mathbb{R}
$$

$$
(x, y) \mapsto x^4 + y^4
$$

$$
K = \{(x,y) \in \mathbb{R}^2 \mid x^2 + y^2 = 1\}.
$$

- $J$ est continue (car polynomiale) et $K$ est compact (car fermé borné en dimension finie). On a donc existence d'un minimum.  

- $K = \{(x,y) \in \mathbb{R}^2 \mid h(x,y) = 0\}$ avec $h(x,y) = x^2 + y^2 - 1$.  

$$
\nabla h(x,y) =
\begin{pmatrix}
2x \\
2y
\end{pmatrix}
$$

$$
= 0 \quad \text{si } (x,y) = (0,0).
$$

- Or, $(0,0)$ n’appartient pas à $K$.  

- Donc $\nabla h(x,y) \neq 0$ pour tout $(x,y) \in K$.  

**Les contraintes sont qualifiées en tout point $(x,y) \in K$.**

- **Soit** $(x^*, y^*) \in K$ **un minimum local de** $J$ **sur** $K$.  

- D'après le théorème des extrémas liés, les contraintes étant qualifiées en $(x^*, y^*)$, il existe $\lambda \in \mathbb{R}$ tel que :

$$
\begin{cases}
\nabla J(x^*, y^*) + \lambda \nabla h(x^*, y^*) = 0 \\
h(x^*, y^*) = 0
\end{cases}
$$

avec $(x^*, y^*) \in K$.

$$
\begin{cases}
\begin{pmatrix} 
4 {x^*}^3 \\ 
4 {y^*}^3 
\end{pmatrix} 
+ \lambda
\begin{pmatrix} 
2 x^* \\ 
2 y^* 
\end{pmatrix} 
= 0
\\
{x^*}^2 + {y^*}^2 = 1
\end{cases}
$$

$\Rightarrow$

$$
\begin{cases}
2 x^* (2 {x^*}^2 + \lambda) = 0 \\
2 y^* (2 {y^*}^2 + \lambda) = 0 \\
{x^*}^2 + {y^*}^2 = 1
\end{cases}
$$

**On distingue les cas :**

**1er cas : $x^* \neq 0$ et $y^* \neq 0$**  

$$
\begin{cases}
2 {x^*}^2 + \lambda = 0 \\
2 {y^*}^2 + \lambda = 0 \\
{x^*}^2 + {y^*}^2 = 1
\end{cases}
$$
$\iff$
$$
\begin{cases}
{x^*}^2 = -\lambda /2 = 1/2 \\
{y^*}^2 = -\lambda /2 = 1/2 \\
-\lambda = 1
\end{cases}
$$
$\Rightarrow$
$$
\begin{cases}
x^* = \pm 1/\sqrt{2} \\
y^* = \pm 1/\sqrt{2} \\
\lambda = -1
\end{cases}
$$

**2e cas : $x^* = 0$ et $y^* \neq 0$**  

$$
\begin{cases}
x^* = 0 \\
y^{*2} = -\lambda /2 \\
y^{*2} = 1
\end{cases}
$$
$\Rightarrow$
$$
\begin{cases}
x^* = 0 \\
y^* = \pm 1 \\
\lambda = -2
\end{cases}
$$

**3e cas : $x^* \neq 0$ et $y^* = 0$**  

Par symétrie du problème,
$$
\begin{cases}
x^* = \pm 1 \\
y^* = 0 \\
\lambda = -2
\end{cases}
$$

**4e cas : $x^* = 0$ et $y^* = 0$**  

**Impossible**, du fait que $x^{*2} + y^{*2} = 1$.

**Les candidats possibles pour être un minimum sur $K$ sont :**  

$$
(x^*, y^*) \in \left\{ \left( \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right), \left( -\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right), \left( \frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}} \right), \left( -\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}} \right), (0,1), (0,-1), (1,0), (-1,0) \right\}
$$

On a :

$$
J \left( \frac{\pm 1}{\sqrt{2}}, \frac{\pm 1}{\sqrt{2}} \right) = \left( \frac{\pm 1}{\sqrt{2}} \right)^4 + \left( \frac{\pm 1}{\sqrt{2}} \right)^4 = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}
$$

$$
J(0, \pm 1) = 1
$$

$$
J(\pm 1, 0) = 1
$$

**Conclusion :**  
Les **minimiseurs** de $J$ sur $K$ sont :

$$
\left( \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right), \left( \frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}} \right), \left( -\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right), \left( -\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}} \right),\lambda = -1.
$$

**Remarque :**  
$J(x,y) = x^4 + y^4$ est **strictement convexe**,  

mais $K$ **n'est pas convexe**  

$\Rightarrow$ **pas de l'unicité du minimum, a priori.**

**Remarque (Multiplicateurs de Lagrange) $p = 1$**

On considère un ensemble de contraintes paramétriques :

$$
K_c = \{ x \in V \mid h(x) = c \}.
$$

On note $x^*(c)$ la solution au problème de minimisation de $J$ sur $K_c$  
(en supposant qu’elle existe et qu’elle est unique).

On a donc :
$$
h(x^*(c)) = c.
$$
et en différentiant :
$$
(\nabla h(x^*(c)), x^{*'}(c)) = 1.
$$
**Puis :**  
$$
\frac{d}{dc} \left[ J(x^*(c)) \right] = (\nabla J(x^*(c)), x^{*'}(c))
$$
$$
= -\lambda(c) (\nabla h(x^*(c)), x^{*'}(c))
$$
$$
= -\lambda(c) \times 1
$$

*(extrema liés – on suppose les hypothèses sont vérifiées)*

**Multiplicateur de Lagrange :** variation de $J$ à l’optimum lorsqu’on fait varier les contraintes.

**Application en économie :** "valeur marginale".


新课了

**Contrainte d’égalité :**  
$K = \{ x \in \mathbb{R}^n : h_i(x) = 0 \}$.

**Théorème :**  
$J, h_1, \dots, h_p \in \mathcal{C}^1(V)$.

Si $\bar{x}$ est un minimum local en $\bar{x} \in K$  
et $(Dh_i(\bar{x}))_i$ libre, alors il existe $\lambda_1, \dots, \lambda_p \in \mathbb{R}$ tels que  

$$
\nabla J(\bar{x}) + \sum \lambda_i \nabla h_i(\bar{x}) = 0
$$

$$
h(\bar{x}) = 0.
$$

---

### 1) Contraintes d’inégalités

Soit  
$$
K = \{ x \in \mathbb{R}^n : h_1(x) = \dots = h_p(x) = 0, g_1(x) \leq 0, \dots, g_q(x) \leq 0 \}.
$$

**Théorème :**  
Soit $J, h_1, \dots, h_p, g_1, \dots, g_q \in \mathcal{C}^1(V)$.  
Si $J$ admet en $\bar{x} \in K$ un minimum local,  
alors il existe $\lambda_0 \geq 0, \lambda_1, \dots, \lambda_p \in \mathbb{R}$ et $\gamma_1, \dots, \gamma_q \geq 0$ tels que :

$$
\lambda_0 \nabla J(\bar{x}) + \sum_{i=1}^{p} \lambda_i \nabla h_i(\bar{x}) + \sum_{i=1}^{q} \gamma_i \nabla g_i(\bar{x}) = 0
$$

$$
\forall i \in \{1, \dots, q\}, \quad \gamma_i g_i(\bar{x}) = 0.
$$

De plus, les multiplicateurs sont non tous nuls.

---

**Remarque : Cas $q = 1$.**  
On a $\gamma_1 g_1(\bar{x}) = 0$. Soit la contrainte est active  
($g_1(\bar{x}) = 0$), on se trouve sur le bord de la contrainte,  
soit la contrainte n'est pas active ($g_1(\bar{x}) < 0$), on se trouve à l'intérieur de l'ensemble des contraintes.


**Preuve (en dimension finie) :**  
Supposons que $\bar{x}$ soit un minimum de $J$ sur $K \cap B(\bar{x}, S)$.

On remarque que $g_i(\bar{x}) \leq 0 \iff g_i^+(\bar{x}) = \max(0, g_i(\bar{x})) = 0$.

On va utiliser une approche par pénalisation.  

Pour $k \in \mathbb{N}$, on considère :

$$
J_k(x) = J(x) + \frac{k}{2} \left( \sum_{i=1}^{p} h_i(x)^2 + \sum_{i=1}^{q} g_i^+(x)^2 \right) + \| x - \bar{x} \|^2
$$

Cette fonction est $C^1$.

On considère le problème :

$$
\inf J_k
$$

$J_k$ est continue sur $B(\bar{x}, S)$,  
elle admet donc un minimum $x_k$.

---

**Convergence de $x_k$ vers $\bar{x}$ :**  
Comme $B(\bar{x}, S/2)$ est compact,  
on peut extraire une sous-suite convergente $(x_{k_j})_j$ vers un $x^*$.

Comme $J_{k_j}(x_{k_j}) \leq J_{k_j}(\bar{x}) = J(\bar{x})$,  
on a :

$$
\left( \sum_{i=1}^{p} h_i(x_{k_j})^2 + \sum_{i=1}^{q} g_i^+(x_{k_j})^2 \right) \leq \frac{2}{k_j} \left[ J(\bar{x}) - J(x_{k_j}) - \| x_{k_j} - \bar{x} \|^2 \right]
$$

En passant à la limite à droite, on obtient $h_i(x^*) = 0 \ \forall i$, $g_i(x^*) = 0 \ \forall i$.

Ainsi $x^* \in K$.

De plus,

$$
J(x_{k_j}) + \| x_{k_j} - \bar{x} \|^2 \leq J_{k_j}(x_{k_j}) \leq J_{k_j}(\bar{x}) = J(\bar{x}) \leq J(x^*)
$$

En passant à la limite,

$$
J(x^*) + \| x^* - \bar{x} \|^2 \leq J(\bar{x})
$$

Donc $x^* = \bar{x}$. Ceci est vrai pour toute sous-suite. Donc $x_k \xrightarrow[k \to \infty]{} \bar{x}$.

---

**Condition d’optimalité :**  
Comme $x_k \to \bar{x}$, alors $x_k \in B(\bar{x}, S/2)$ à partir d’un certain rang.

Comme $x_k$ est minimum de $J_k$ sur $B(\bar{x}, S)$,

$$
0 = \nabla J_k(x_k) = \nabla J(x_k) + \sum_{i=1}^{p} k h_i(x_k) \nabla h_i(x_k) + \sum_{i=1}^{q} k g_i^+(x_k) \nabla g_i(x_k) + 2 (x_k - \bar{x})
$$

Ce qui se réécrit :

$$
0 = \lambda_0^k \nabla J(x_k) + \sum \lambda_i^k \nabla h_i(x_k) + \sum \mu_i^k \nabla g_i^+(x_k) + 2 (x_k - \bar{x})
$$

avec  

$$
\lambda_0^k = \frac{1}{\sqrt{1 + \sum k h_i(x_k)^2 + \sum k g_i^+(x_k)^2}}, \quad \lambda_i^k = \lambda_0^k k h_i(x_k), \quad \mu_i^k = \lambda_0^k k g_i^+(x_k)
$$

Donc la suite $(\lambda_0^k, -\lambda_1^k, \dots, \lambda_p^k, \mu_1^k, \dots, \mu_q^k)$ est de norme 1.  
On peut extraire une sous-suite convergente du limite $(\lambda_0, -\lambda_1, \dots, \lambda_p, \mu_1, \dots, \mu_q)$ de norme 1, donc non nulle.

En passant à la limite, on obtient la relation voulue.

De plus, $\mu_i^k \geq 0$, donc $\mu_i \geq 0$.

Si $g_i(\bar{x}) < 0$, alors à partir d’un certain rang, $g_i(x_k) < 0$,  
ce qui implique que $\mu_i^k = 0$. La limite $\mu_i$ est donc nulle. $\square$

### **2) Qualification des contraintes :**  
On note $I(\bar{x})$ l’ensemble des indices des contraintes actives en un point $\bar{x}$.

**Définition :**  
Les contraintes sont dites qualifiées en $\bar{x} \in K$ dans l’un des cas suivants :

1) **Critère d’indépendance :**  
   Si la famille $(\nabla h_i(\bar{x}))_i, (\nabla g_i(\bar{x}))_{i \in I(\bar{x})}$  
   est linéairement indépendante.

2) **Critère Mangasarian-Fromovitz :**  
   (i) $(\nabla h_i(\bar{x}))_i, (\nabla g_i(\bar{x}))_{i \in I(\bar{x})}$ sont linéairement indépendants.  
   (ii) Il existe une direction $d \in \mathbb{R}^n$ telle que :

   $$
   \langle \nabla h_i(\bar{x}), d \rangle = 0, \quad \forall i
   $$

   $$
   \langle \nabla g_i(\bar{x}), d \rangle < 0, \quad \forall i \in I(\bar{x}).
   $$

**Remarque :**  
(i) implique (ii).

#### **Théorème (Karush-Kuhn-Tucker) (KKT) :**  
Soient $J, h_1, \dots, h_p, g_1, \dots, g_q \in \mathcal{C}^1(V)$.  

Si $J$ admet un minimum local en $\bar{x} \in K$ et que les contraintes sont qualifiées en $\bar{x}$,  
alors il existe $\lambda_1, \dots, \lambda_p \in \mathbb{R}$, $\mu_1, \dots, \mu_q \geq 0$, non tous nuls, tels que :

$$
\nabla J(\bar{x}) + \sum_{i=1}^{p} \lambda_i \nabla h_i(\bar{x}) + \sum_{i=1}^{q} \mu_i \nabla g_i(\bar{x}) = 0
$$

$$
\mu_i g_i(\bar{x}) = 0, \quad \forall i.
$$

---

### **Preuve :**  
D’après le théorème précédent, il existe $\lambda_0 \geq 0$, $\lambda_1, \dots, \lambda_p \in \mathbb{R}$, $\mu_1, \dots, \mu_q \geq 0$, tels que :

$$
\lambda_0 \nabla J(\bar{x}) + \sum_{i=1}^{p} \lambda_i \nabla h_i(\bar{x}) + \sum_{i=1}^{q} \mu_i \nabla g_i(\bar{x}) = 0
$$

On souhaite montrer que $\lambda_0 \neq 0$.  
Supposons que $\lambda_0 = 0$.

---

### **Qualification 1) :**  
Par indépendance de la famille $(\nabla h_i(\bar{x}))_i, (\nabla g_i(\bar{x}))_i$, les multiplicateurs sont nuls,  
ce qui est impossible.

#### **2) Qualification 2) :**  
Soit $j^* \in I(\bar{x})$ i.e. $\mu_{j^*} > 0$, alors :

$$
0 = \langle \sum_i \lambda_i \nabla h_i(\bar{x}) + \sum_i \mu_i \nabla g_i(\bar{x}), d \rangle \leq \mu_{j^*} \langle \nabla g_{j^*}(\bar{x}), d \rangle < 0.
$$

Ce qui est impossible. Donc tous les $\mu_i$ sont nuls.  
Par indépendance de la famille $(\nabla h_i(\bar{x}))_i$, on a $\lambda_i = 0$. Ce qui est impossible. $\square$

---

### **Remarque :**  
Ce théorème englobe l’équation d’Euler et les extrémas liés.

---

### **Exemple :**  
$$
J(x, y) = x^4 + 3y^4, \quad g(x, y) = 1 - x^2 - y^2
$$

Donc 

$$
K = \{ (x, y) : 1 \leq x^2 + y^2 \}.
$$

---

### **Existence :**  
$J$ est coercive, continue et $K$ est fermé,  
on a donc existence d’un minimum $\bar{x}$.

---

### **Qualification :**  
$$
\nabla g(x) = \begin{pmatrix} -2x \\ -2y \end{pmatrix} \neq 0
$$

car $(0,1) \notin K$,  
d’où tous les points sont qualifiés.

#### **Condition d’optimalité :**  
Il existe $\mu \geq 0$ tel que :

$$
\nabla J(\bar{x}) + \mu \nabla g(\bar{x}) = 0
$$

$$
\mu g(\bar{x}) = 0
$$

$$
g(\bar{x}) \leq 0
$$

$$
\mu \geq 0
$$

---

$$
\begin{cases} 
4x^3 - 2 \mu x = 0 \\ 
12y^3 - 2 \mu y = 0 \\ 
\mu (1 - x^2 - y^2) = 0 \\ 
1 - x^2 - y^2 \leq 0 \\ 
\mu \geq 0
\end{cases}
$$

Si $\mu = 0$, alors $(\bar{x}, \bar{y}) = (0,0)$.  
Ce qui est impossible, car $(0,0) \notin K$.

Donc $\mu \neq 0$, d’où :

$$
\begin{cases} 
\frac{x^2}{4} + \frac{y^2}{12} = 1 \\ 
\mu \left( 4 x^2 - 2 \right) = 0 \\ 
\mu \left( 12 y^2 - 2 \right) = 0
\end{cases}
$$

---

- Si $x = y = 0$, alors impossible.
- Si $x = 0$ et $y \neq 0$, alors $y = \pm 1$ et $\mu = 6$.
- Si $y = 0$ et $x \neq 0$, alors $x = \pm 1$ et $\mu = 2$.
- Si $x \neq 0$ et $y \neq 0$, alors :

  $$
  \frac{x^2}{4} = \frac{y^2}{12} = \frac{1}{2}, \quad x^2 = 2, \quad y^2 = \frac{2}{3}
  $$

  $$
  x = \pm \frac{\sqrt{3}}{2}, \quad y = \pm \frac{1}{\sqrt{6}}, \quad \mu = \frac{3}{2}.
  $$

---

### **Les candidats sont :**  
$$
(0, \pm 1), \quad (\pm 1, 0), \quad \left( \pm \frac{\sqrt{3}}{2}, \pm \frac{1}{\sqrt{6}} \right).
$$

Pour lesquels $J$ prend les valeurs $1, 3$ et $\frac{3}{4}$.  

Donc $J$ atteint son minimum en  

$$
\left( \pm \frac{\sqrt{3}}{2}, \pm \frac{1}{\sqrt{6}} \right).
$$

---

### **Remarque :**  
Le théorème précédent est valide par les conditions de qualifications suivantes :

1) **Critère de limitalité :**  
   Si toutes les contraintes sont actives.

2) **Critère de convexité :**  
   (i) La famille $(\nabla h_i(\bar{x}))$ est linéairement indépendante.  
   (ii) $g_i$ convexes et $J$ quasi, i.e. :  
   $$
   \forall j \in I(\bar{x}),
   \begin{cases}
   g_j(\bar{x}) < 0, & \text{si } g_j \text{ affine} \\
   g_j(\bar{x}) < 0, & \text{sinon.}
   \end{cases}
   $$

3) **Critère général :**  
   (i) La famille $(\nabla h_i(\bar{x}))$ est linéairement indépendante.  
   (ii) Il existe une direction $d \in \mathbb{R}^n$ telle que :  

   $$
   \forall i, \quad \langle \nabla h_i(\bar{x}), d \rangle = 0
   $$

   $$
   \forall j \in I(\bar{x}),
   \begin{cases}
   \langle \nabla g_j(\bar{x}), d \rangle \leq 0, & \text{si } g_j \text{ affine} \\
   \langle \nabla g_j(\bar{x}), d \rangle < 0, & \text{sinon.}
   \end{cases}
   $$

### **3) Réciproque :**  

#### **Théorème :**  
Soient $J$ convexe, $h_1, \dots, h_p$ affines, $g_1, \dots, g_q$ convexes et toutes $\mathcal{C}^1(V)$.  

Si il existe $\bar{x}$, $\lambda_1, \dots, \lambda_p \in \mathbb{R}$, $\mu_1, \dots, \mu_q \geq 0$ vérifiant KKT,  
alors $\bar{x}$ est minimum global de $J$ sur $K$.

---

### **Preuve :**  
Soit $x \in K$. Par convexité de $J$,

$$
J(x) \geq J(\bar{x}) + \langle \nabla J(\bar{x}), x - \bar{x} \rangle.
$$

Or,

$$
\nabla J(\bar{x}) = -\sum_{i=1}^{p} \lambda_i \nabla h_i(\bar{x}) - \sum_{i \in I(\bar{x})} \mu_i \nabla g_i(\bar{x}).
$$

Donc,

$$
J(x) \geq J(\bar{x}) - \sum_{i=1}^{p} \lambda_i \langle \nabla h_i(\bar{x}), x - \bar{x} \rangle - \sum_{i \in I(\bar{x})} \mu_i \langle \nabla g_i(\bar{x}), x - \bar{x} \rangle.
$$

Or,  
$0 = h_i(x) - h_i(\bar{x}) = \langle \nabla h_i(\bar{x}), x - \bar{x} \rangle$.  
De même,  
$g_j(x) \geq g_j(\bar{x}) + \langle \nabla g_j(\bar{x}), x - \bar{x} \rangle$.

Comme $\mu_j \geq 0$, on obtient :

$$
J(x) \geq J(\bar{x}).
$$

$\square$

### **4) Dualité :**  

#### **Définition :**  
On introduit le **Lagrangien** :

$$
\mathcal{L}(x, \lambda, \mu) = J(x) + \sum_{i=1}^{p} \lambda_i h_i(x) + \sum_{i=1}^{q} \mu_i g_i(x).
$$

$$
\nabla \mathcal{L} (x, \lambda, \mu) \in V \times \mathbb{R}^p \times \mathbb{R}^q.
$$

---

### **Remarque :**  
Si on n’a pas de contrainte d’inégalité, alors les conditions des extrémas liés sont équivalentes au fait que  
$(\bar{x}, \bar{\lambda})$ est un point critique de $\mathcal{L}$.

Plus généralement :

$$
\frac{\partial}{\partial \mu_i} \mathcal{L} (\bar{x}, \bar{\lambda}, \bar{\mu}) = g_i(\bar{x}) \leq 0.
$$

Ce n’est pas un point critique mais un **point selle**.

---

#### **Définition :**  
$(\bar{x}, \bar{\lambda}, \bar{\mu})$ est un **point selle** si :

$$
\forall (x, \lambda, \mu) \in V \times \mathbb{R}^p \times \mathbb{R}^q, \quad \mathcal{L} (\bar{x}, \bar{\lambda}, \bar{\mu}) \leq \mathcal{L} (\bar{x}, \lambda, \mu) \leq \mathcal{L} (x, \bar{\lambda}, \bar{\mu}).
$$

### **Proposition :**  
S’il existe $(\bar{x}, \bar{\lambda}, \bar{\mu}) \in V \times \mathbb{R}^p \times \mathbb{R}^q$ point selle de $\mathcal{L}$,  
alors $\bar{x} \in K$ et c’est un minimum global de $J$ sur $K$, et KKT est vérifié.

Dans le cas où $J$ est convexe, $h_i$ sont affines et $g_i$ sont convexes,  
alors il y a équivalence entre $\bar{x}$ minimum global, KKT et point selle.

---

### **Preuve :**  
La première inégalité s’écrit :

$$
\sum (\lambda_i - \bar{\lambda}_i) h_i(\bar{x}) + \sum (\mu_i - \bar{\mu}_i) g_i(\bar{x}) \leq 0.
$$

En prenant $\lambda = \bar{\lambda} \pm e_i$ et $\mu = \bar{\mu}$,  
on obtient $h_i(\bar{x}) = 0$.  

Puis en prenant $\lambda = \bar{\lambda}$ et $\mu = \bar{\mu} \pm e_i$,  
on a $g_j(\bar{x}) \leq 0$.  

Donc $\bar{x} \in K$.  

En posant $\mu = \bar{\mu} \pm t e_i$ et $\mu = \bar{\mu}$,  
on obtient :

$$
\bar{\mu}_i g_i(\bar{x}) = 0.
$$

De plus,

$$
J(\bar{x}) \leq J(x) + \sum \lambda_i h_i(x) + \sum \mu_i g_i(x) \leq J(x).
$$

Ainsi, $\bar{x}$ est minimum de $J$ sur $K$.

$\bar{x}$ est un minimum de $\mathcal{L} (x, \bar{\lambda}, \bar{\mu})$ sans contrainte.  
L’égalité d’Euler associée est justement KKT.

---

2) Si $\bar{x}$ est minimum de $J$ sur $K$, alors comme $J$ est convexe, KKT implique  
que $\bar{x}$ est un minimum global de $\mathcal{L} (x, \bar{\lambda}, \bar{\mu})$,  
d’où la seconde inégalité.

La première inégalité est immédiate puisque $\bar{x} \in K$  
et que la condition de complémentarité est vérifiée. $\square$