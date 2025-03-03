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

![[image.png|300x240]]

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

![[image-1.png|264x186]]

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
![[image-2.png]]

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