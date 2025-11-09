## Chapitre 1 – Principe du Maximum de Pontryagin (PMP)

**a)** Exemple introductif

Considérons le problème d’optimisation :
$$
\inf_{u \in L^2(0,T)} J(u)
$$
où
$$
J(u) = \frac{1}{2} \int_0^T \big(x_u(s) - 1\big)^2 , ds
$$
et $x_u$ est la solution de

$$
\begin{cases}
- x_u'' + x_u = u & \text{sur } (0,1), \\
  x_u(0) = x_u(1) = 0.
\end{cases}
$$

**Calculons le gradient de $J$**,
c’est-à-dire l’unique élément de $L^2$, noté $\nabla J(u)$,
tel que
$$
DJ(u),h = \langle \nabla J(u), h \rangle_{L^2}
$$
(théorème de Riesz)

et
$$
DJ(u),h = \lim_{\varepsilon \to 0} \frac{J(u + \varepsilon h) - J(u)}{\varepsilon}.
$$
---

**Calculons la dérivée directionnelle :**

$$
\frac{J(u+\varepsilon h) - J(u)}{\varepsilon}
= \frac{1}{2} \int_0^T \frac{(x_{u+\varepsilon h}(s) - 1)^2 - (x_u(s) - 1)^2}{\varepsilon} , ds
$$
$$
= \frac{1}{2} \int_0^T \frac{(x_u(s) + \varepsilon x_h(s) - 1)^2 - (x_u(s) - 1)^2}{\varepsilon} , ds
$$
$$
= \frac{1}{2} \int_0^T \frac{2 \varepsilon x_h(s)(x_u(s) - 1) + \varepsilon^2 x_h(s)^2}{\varepsilon} , ds
$$
$$
= \frac{1}{2} \int_0^T \left(2 x_h(s)(x_u(s) - 1) + \varepsilon x_h(s)^2\right) ds
$$
Quand $\varepsilon \to 0$ :
$$
\int_0^T x_h(s)(x_u(s) - 1), ds = DJ(u) , h
$$

**Remarque :**

On définit alors l’application linéaire
$$
h \mapsto \int_0^T x_h(s)(x_u(s) - 1) , ds
$$
qui est une **forme linéaire continue**, c’est-à-dire :
$$
\exists, C > 0,\ \forall h \in L^2, \quad |DJ(u),h| \leq C \|h\|_{L^2}.
$$
Autrement dit,
$$
\Big| \int_0^T x_h(s)(x_u(s) - 1), ds \Big| \le C \|h\|_{L^2}.
$$

---

On avait :
$$
DJ(u),h = \int_0^T x_h(s),(x_u(s) - 1), ds
$$
Pour montrer que cette forme linéaire est **continue**, on utilise l’inégalité de Cauchy–Schwarz :
$$
\Big|\int_0^T x_h(s)(x_u(s) - 1), ds \Big|
\le
\sqrt{\int_0^T x_h(s)^2, ds}\times
\sqrt{\int_0^T (x_u(s) - 1)^2, ds}
$$
$$
\le C_u |x_h|_{L^2} \quad \text{où} C_u = |x_u - 1|_{L^2}.
$$
---

Or $x_h$ est la solution du problème différentiel :
$$
\begin{cases}
- x_h'' + x_h = h, \\
  x_h(0) = x_h(1) = 0.
\end{cases}
$$
On multiplie l’équation par $x_h$ et on intègre par parties :
$$
\int_0^T \big( (x_h')^2 + x_h^2 \big), ds = \int_0^T x_h h , ds.
$$
D’où :
$$
|x_h|_{L^2}^2 \le |x_h|_{H^1_0}^2 = \int_0^T x_h h \le |x_h|_{L^2}|h|_{L^2}.
$$
$$
\Rightarrow |x_h|_{L^2} \le |h|_{L^2}.
$$
---

Ainsi :
$$
|DJ(u),h| \le C_u |h|_{L^2}.
$$
Donc la forme est bien **continue sur $L^2$**.

---

**Conclusion :**

On peut donc écrire
$$
DJ(u),h = \int_0^T x_h(s),(x_u(s) - 1), ds
= \langle \text{“truc”}, h \rangle_{L^2}.
$$
C’est-à-dire que le **gradient** (au sens de Riesz) est :
$$
\nabla J(u) = \text{“truc”} = x_u - 1,
$$
où le terme $x_h$ est la réponse de l’opérateur différentiel $(-d^2/dx^2 + I)^{-1}$ appliqué à $h$.

---

**Suite du raisonnement :**

On avait à la ligne précédente :
$$
DJ(u),h = \int_0^T x_h(s),(x_u(s) - 1),ds.
$$
On veut exprimer cela **sous forme adjointe**, pour identifier le gradient.

---

**Introduction de la fonction $p$ :**

Introduisons la fonction (p) solution de

$$
\begin{cases}
\mathcal{L}^* p = g, \\
p(0) = b_1, \quad p(1) = b_2,
\end{cases}
$$
où
$$
\mathcal{L} = -\frac{d^2}{dx^2} + I,
$$
et $\mathcal{L}^*$ désigne l’**opérateur adjoint** de $\mathcal{L}$.
$Remarque sur le tableau : *opérateur de l’équation*$.

On choisit les conditions aux bords $b_1, b_2$ de façon appropriée,
et on constate que **dans ce cas particulier,**
$$
\mathcal{L}^* = \mathcal{L} = -\frac{d^2}{dx^2} + I.
$$
---

**Équation explicite pour $p$ :**
$$
\begin{cases}
- p'' + p = g, \\
  p(0) = b_1, \quad p(1) = b_2.
  \end{cases}
$$
---

**Calcul par intégration par parties :**

On multiplie l’équation de (p) par $x_h$, puis on intègre par parties :
$$
\int_0^T (-p'' + p),x_h
= \int_0^T g,x_h
$$
$$
\Rightarrow
\int_0^T p',x_h'

* p(T),x_h'(T)

- p'(0),x_h(0)
- \int_0^T p,x_h
  = \int_0^T g,x_h.
$$
Dans notre problème particulier, les conditions aux bords sont nulles : $x_h(0)=x_h(1)=0$, donc les termes de bord s’annulent.

---

**Suite : identification du gradient via l’équation adjointe**

On multiplie l’équation de $x_h$ par (p), et on intègre par parties :
$$
\int_0^T x_h'',p
= -x_h'(T)p(T) + x_h'(0)p(0) + \int_0^T x_h' p',ds
$$
et donc :
$$
\int_0^T (-x_h'' + x_h)p
= \int_0^T h,p.
$$
---

Comme $-x_h'' + x_h = h$, on obtient :
$$
\int_0^T x_h' p' + x_h p = \int_0^T h,p.
$$
Si on choisit les conditions aux bords $p(0)=p(1)=0$, les termes de bord disparaissent.

---

**Comparaison avec la définition de $DJ(u),h$**

On avait précédemment :
$$
DJ(u),h = \int_0^T x_h(x_u - 1),ds.
$$
On choisit donc :
$$
g = x_u - 1.
$$
Or, par définition de (p),
$$
* p'' + p = g = x_u - 1.
$$
---

**Conclusion :**

Ainsi :
$$
DJ(u),h = \int_0^T h,p.
$$
Donc, par identification :
$$
\nabla J(u) = p,
$$
où (p) est la solution de l’équation **adjointe** :
$$
\begin{cases}
- p'' + p = x_u - 1,\\
  p(0) = p(1) = 0.
  \end{cases}
$$
---

**Remarque — Résolution numérique**

Si on souhaite **résoudre numériquement** le problème
$$
\inf_{u \in L^2(0,T)} J(u),
$$
on utilise le fait que
$$
\nabla J(u) = p,
$$
où (p) est la **solution de l’équation adjointe**.

---

### ⚙️ **Algorithme de descente de gradient**

On peut donc appliquer une **méthode de descente de gradient** :
$$
\text{Donné : } u_0 \in L^2(0,T), \ \varepsilon > 0
$$
$$
\text{Pour } k = 0,1,2,\dots :
\begin{cases}
- x_{u_k}'' + x_{u_k} = u_k, & x_{u_k}(0)=x_{u_k}(T)=0, \\
- p_k'' + p_k = x_{u_k} - 1, & p_k(0)=p_k(T)=0, \\
  u_{k+1} = u_k - \varepsilon p_k.
  \end{cases}
$$
Ici :

* la première équation est **l’état direct**,
* la seconde est **l’état adjoint**,
* la troisième est **la mise à jour du contrôle** par descente de gradient.

---

### 💡 **Remarques à droite du tableau**

* $p$ satisfait l’**équation adjointe** :
$$
-p'' + p = x_u - 1, \quad p(0) = p(T) = 0.
$$
* Cette équation est appelée *l’équation adjointe* notation : « Rq. L’équation de $p$ s’appelle l’équation adjointe ».

---

### 🔹 **2) Problème LQ**

**On considère un système linéaire autonome**

((T > 0))
$$
\begin{cases}
\dot{x}*u = A x_u + B u,\\
x_u(0) = x_0 \in \mathbb{R}^n,
\end{cases}
$$
avec $A \in M_n(\mathbb{R}),; B \in M*{n,m}(\mathbb{R})$.

---

### **On introduit la fonction coût**
$$
J : L^2([0,T];\mathbb{R}^m) \to \mathbb{R}
$$
définie par
$$
J(u) = \frac{1}{2} \Big(
\int_0^T \big[ \langle Q x_u, x_u \rangle + \langle R u, u \rangle \big],dt

* \langle M x_u(T), x_u(T) \rangle
  \Big),
$$
  où $Q \in S^+(n),\ R \in S^{++}(m),\ M \in S^+(n)$.

---

### **avec**

* $S^+(n)$ : ensemble des matrices de $\mathbb{R}^{n\times n}$ symétriques semi-définies positives,
  c.-à-d. $\langle Mx, x\rangle \ge 0,\ \forall x \in \mathbb{R}^n$.
* $S^{++}(n)$ : ensemble des matrices symétriques définies positives,
  c.-à-d. $\langle Mx, x\rangle > 0,\ \forall x \ne 0.$

---

### **Remarque complémentaire**

On note aussi :

* $S(n)$ : ensemble des matrices symétriques réelles $n\times n$,
* $S^+(n)\subset S(n)$ : celles dont les valeurs propres sont $\ge 0$,
* $S^{++}(n)\subset S^+(n)$ : celles dont les valeurs propres sont (>0).

---

**Rappel :**

Si $M \in S^{+}(n,\mathbb{R})$, alors
$$
M = P^{T} D P,
$$
où $D = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$ et (P) est une matrice orthogonale de $\mathbb{R}^{n\times n}$.

---

Pour tout $x \in \mathbb{R}^n$ :
$$
\langle Mx, x \rangle
= \langle P^{T} D P x, x \rangle
= \langle D P x, P x \rangle
= \sum_{i=1}^n \lambda_i y_i^2,
$$
où $y = P x$.

---

Soit $M \in S^{+}(n, \mathbb{R})$.
Alors il existe $\lambda_i > 0$ pour tout (i).

Sans perte de généralité, on suppose :
$$
0 < \lambda_{\min} \le \lambda_i \le \lambda_{\max}.
$$
---

On en déduit :
$$
\lambda_{\min} \sum_i y_i^2
\le \sum_i \lambda_i y_i^2
\le \lambda_{\max} \sum_i y_i^2.
$$
Ce qui est équivalent à :
$$
\lambda_{\min} |x|^2
\le \langle Mx, x \rangle
\le \lambda_{\max} |x|^2.
$$
---

### Conclusion

Donc :
$$
\sqrt{\langle Mx, x \rangle}
$$
est **une norme équivalente** à la norme euclidienne (|x|).

*(mention sur le tableau : « la norme euclidienne » entre parenthèses)*

---

**a) Existence et unicité**

#### **Rappel :**

Une **fonction faiblement semi-continue inférieurement** (f.s.c.i.)
vérifie que pour toute suite $(u_n)*n$,
$$
\liminf_{n\to\infty} J(u_n)
\ge J\Big(\lim_{n\to\infty} u_n\Big)
\quad\text{(ou plus petit valeur inférieure).}
$$
Autrement dit :
$$
\liminf_{n\to\infty} J(u_n)
= \lim_{N\to\infty} \inf_{n\ge N} J(u_n).
$$
---

*Un petit dessin sur le tableau illustre une suite oscillante qui converge faiblement, montrant visuellement la notion de « lim inf » par rapport à la limite ordinaire.*

---

### **Théorème :**

Le **problème LQ**
$$
\begin{cases}
\inf J(u),\\
u \in L^2(0,T;\mathbb{R}^m),
\end{cases}
$$
possède **une unique solution**.

---

### **Preuve (esquisse) :**

On suit la **méthode directe du calcul des variations**.

**Méthode directe – Étapes de la preuve**

#### 1️⃣ On considère une suite minimisante $(u_k)_k$

c’est-à-dire :
$$
J(u_k) \to \inf J \quad \text{dans } L^2(0,T).
$$
---

#### 2️⃣ On montre que $(u_k)$ converge pour une certaine topologie

par exemple, la **topologie faible** de $L^2$ :
$$
u_k \rightharpoonup u \quad \text{faiblement dans } L^2
\quad \Leftrightarrow \quad
\langle u_k, \psi \rangle_{L^2} \to \langle u, \psi \rangle_{L^2}.
$$
---

#### 3️⃣ On vérifie que (J) est **semi-continue inférieurement** pour cette topologie :
$$
u_k \rightharpoonup u
\Rightarrow
\liminf_{k\to\infty} J(u_k) \ge J(u).
$$
---

### **Compacité de $u_k$**

On veut extraire une **sous-suite convergente** de $(u_k)$.

Comme $(J(u_k))_k$ est convergente,
il existe $C > 0$ tel que :
$$
\forall k, \quad |J(u_k)| \le C.
$$
---

On a, pour tout $x$ et $u$ :
$$
\langle Q x, x \rangle > 0,
\quad
\langle R u, u \rangle > 0,
$$
avec $Q, R \in S_n^{+}$.


D’où, pour tout $u \in L^2$,
$$
J(u) \ge \frac{1}{2} \int_0^T \langle R u, u \rangle.
$$
Or, pour tout $v \in \mathbb{R}^m$,
$$
\langle Rv, v \rangle \ge \lambda_{\min} |v|^2,
$$
où $\lambda_{\min}$ est la **plus petite valeur propre** de $R$.

---

Il vient donc :
$$
J(u) \ge \frac{1}{2} \lambda_{\min} \int_0^T |u(t)|^2,dt
= \frac{1}{2} \lambda_{\min} |u|_{L^2(0,T)}^2.
$$
---

On en déduit que $(u_k)$ est **bornée** dans $L^2(0,T)$.


Donc, il existe $u \in L^2$
tel que, à une sous-suite près,
$$
u_k \rightharpoonup u.
$$
---

### **Propriété de continuité de (J)**

On a la solution d’état :
$$
x_u(t) = e^{tA} x_0 + \int_0^t e^{(t-s)A} B u(s),ds.
$$
La (j)-ème composante de $x_u(t)$ est :
$$
\langle x_u(t), e_j \rangle_{\mathbb{R}^n}
= \langle e^{tA} x_0, e_j \rangle

* \int_0^t \langle u(s), B^T e^{(t-s)A^T} e_j \rangle, ds.
$$
---

$$
\Rightarrow
\langle e^{tA} x_0, e_j \rangle

* \int_0^t \langle u_k(s), B^T e^{(t-s)A^T} e_j \rangle, ds
  \longrightarrow
  \langle e^{tA} x_0, e_j \rangle
* \int_0^t \langle u(s), B^T e^{(t-s)A^T} e_j \rangle, ds
$$
  c’est-à-dire :
$$
x_{u_k}(t) \to x_u(t)
$$
---

Donc $x_{u_k}$ **converge simplement vers** $x_u$.

---

On en déduit, par le **théorème de convergence dominée** :
$$
\int_0^T \langle Q x_{u_k}, x_{u_k} \rangle
\longrightarrow
\int_0^T \langle Q x_u, x_u \rangle,
$$
et, par convergence simple :
$$
\langle M x_{u_k}(T), x_{u_k}(T) \rangle
\longrightarrow
\langle M x_u(T), x_u(T) \rangle.
$$
---


On traite le terme :
$$
\int_0^T \langle R u_k, u_k \rangle.
$$
---

Considérons :
$$
\Phi(u) = \int_0^T \langle R u, u \rangle,
$$
et $F : x \in \mathbb{R}^m \mapsto \langle R x, x \rangle \in \mathbb{R}.$

---

On a :
$$
\nabla^2 F = 2R \in S_m^{++}(\mathbb{R}).
$$
Donc (F) est **strictement convexe**.

---

Pour tout $u, v \in L^2$ et tout $\varepsilon \in (0,1)$,
on définit
$$
\Phi(\varepsilon u + (1-\varepsilon)v)
= \int_0^T F(\varepsilon u + (1-\varepsilon)v).
$$
Or,
$$
F(\varepsilon u + (1-\varepsilon)v)
\le
\varepsilon F(u) + (1-\varepsilon) F(v)
$$
car $F$ est convexe, donc
$$
\Phi(\varepsilon u + (1-\varepsilon)v)
\le
\varepsilon \Phi(u) + (1-\varepsilon) \Phi(v).
$$
---

Donc **$\Phi$ est convexe.**

---

Montrons que $\Phi$ est **continue pour la topologie faible de $L^2$**.

Soit $v_k \rightharpoonup v$ dans $L^2$.

---

On a :
$$
\Phi(v_k) - \Phi(v)
= \int_0^T \big( \langle R v_k, v_k \rangle - \langle R v, v \rangle \big)
= \int_0^T \langle R (v_k - v), v_k - v \rangle

* 2 \int_0^T \langle R (v_k - v), v \rangle.
$$
---

On obtient donc :
$$
|\Phi(v_k) - \Phi(v)|
\le M \int_0^T |v_k - v|^2

* 2C \int_0^T |v_k - v| |v|.
$$
Les deux termes tendent vers (0) lorsque $v_k \rightharpoonup v$,

Donc $\Phi$ est continue dans $L^2$.

**Conclusion :**
$\Phi$ est **semi-continue inférieurement (s.c.i.)** dans $L^2$ fort,
et comme $\Phi$ est convexe, elle est aussi **s.c.i. dans $L^2$ faible.**

---

Comme (J) est somme de fonctions :

* continues pour la topologie faible,
* et s.c.i. pour la topologie faible,

alors **(J) est s.c.i. pour la topologie faible**.

---

Donc
$$
J(u) = \lim_{k\to\infty} J(u_k)
= \liminf_{k\to\infty} J(u_k)
\Rightarrow J(u) = \inf J.
$$
On en déduit que (u) **minimise (J)**.

---

### **Unicité :**

On a
$$
G : u \mapsto \int_0^T \langle R u, u \rangle
$$
est **strictement convexe**.

De même,
$$
G : x \mapsto \int_0^T \langle Q x_u, x_u \rangle
$$
est aussi **strictement convexe**.


On a :
$$
\frac{d}{dt}(x_{u+v}) = A x_{u+v} + B(u+v)
$$
et plus généralement, pour tout $\lambda \in [0,1]$ :
$$
\frac{d}{dt}(x_{u + \lambda v}) = A x_{u + \lambda v} + B(u + \lambda v),
$$
avec $x_{u+\lambda v}(0) = x_0.$

---

De plus :
$$
\begin{cases}
\dot{x}*{u+\lambda v} = A x*{u+\lambda v} + B$u+\lambda v$,\
x_{u+\lambda v}(0) = x_0.
\end{cases}
$$
Par **unicité** de la solution du système différentiel,
on a :
$$
x_{u+\lambda v} = x_u + \lambda x_v.
$$
---


Donc $G$ est **convexe**
(par composition d’une fonction convexe avec une fonction linéaire).

---

Même argument pour montrer que :
$$
u \mapsto \langle M x_u(T), x_u(T) \rangle
$$
est **convexe**.

---

Comme **somme de fonctions convexes**,
et comme $J = G + \Phi$,
et que $\Phi$ est **strictement convexe**,
on en déduit que **$J$ est strictement convexe**.

### **c) PMP**

*Principe du Maximum de Pontryagin*

#### **Préliminaire :**

Soit $F : \mathbb{R}^m \to \mathbb{R}$ défini par
$$
F(z) = \langle Rz, z \rangle.
$$
Calculons sa différentielle :
$$
DF(z),h = \lim_{\varepsilon \to 0}
\frac{F(z + \varepsilon h) - F(z)}{\varepsilon}.
$$
---

On a :
$$
F(z + \varepsilon h) - F(z)
= \langle R(z + \varepsilon h), z + \varepsilon h \rangle

* \langle Rz, z \rangle
  = 2\varepsilon \langle Rz, h \rangle + \varepsilon^2 \langle Rh, h \rangle.
$$
D’où :
$$
DF(z),h = 2 \langle Rz, h \rangle.
$$
Ainsi :
$$
\nabla F(z) = 2Rz.
$$
---

### **Conditions d’optimalité**

On cherche maintenant les **conditions nécessaires d’optimalité** pour $J(u)$.

Soit (u) une solution du problème d’optimisation,
et $x_u$ l’état associé :
$$
\dot{x}_u = A x_u + B u, \quad x_u(0) = x_0.
$$
---

On considère une **perturbation** :
$$
v = u + \varepsilon h, \quad y = x_v - x_u,
$$
d’où :
$$
\dot{y} = A y + B h, \quad y(0) = 0.
$$
---

### **Suite du PMP (Principe du Maximum de Pontryagin)**

On considère la variation du coût :
$$
J(v) - J(u) = \frac{1}{2\varepsilon} [J(u + \varepsilon h) - J(u)].
$$
---

En développant :
$$
J(v) - J(u)
= \frac{1}{2} \int_0^T \Big(
\langle Q(x_u + \varepsilon y), (x_u + \varepsilon y) \rangle

* \langle Qx_u, x_u \rangle
  \Big),dt

- \frac{1}{2} \int_0^T
  \big( \langle R(u + \varepsilon h), (u + \varepsilon h)\rangle - \langle Ru, u \rangle \big),dt
- \frac{1}{2} \big(
  \langle D(x_u + \varepsilon y)(T), x_u(T) + \varepsilon y(T)\rangle - \langle D x_u(T), x_u(T)\rangle
  \big).
$$
---

En développant et simplifiant, on obtient :
$$
\frac{J(v) - J(u)}{\varepsilon}
= \int_0^T \big( \langle Qx_u, y\rangle + \langle R u, h\rangle \big),dt

* \langle D x_u(T), y(T)\rangle
* O(\varepsilon).
$$
---

### **Introduction de l’état adjoint :**

Soit $p_u$ la fonction **adjointe**, solution de :
$$
\begin{cases}
-\dot{p}_u - A^T p_u = Q x_u,\\
p_u(T) = D x_u(T).
\end{cases}
$$
---

On multiplie la première équation par (y) et on intègre :
$$
\int_0^T \langle \dot{p}_u, y\rangle + \langle A^T p_u, y\rangle
= - \int_0^T \langle Qx_u, y\rangle.
$$
---

### **Suite du calcul (Principe du Maximum de Pontryagin – PMP)**

On **intègre par parties** :
$$
-\int_0^T \langle \dot{p}_u, y\rangle
= -\big[\langle p_u, y\rangle \big]_0^T + \int_0^T \langle p_u, \dot{y}\rangle.
$$
D’après l’équation de (y) :
$$
\dot{y} = A y + B h, \quad y(0) = 0,
$$
et les conditions de $p_u$ :
$$
-\dot{p}_u - A^T p_u = Qx_u, \quad p_u(T) = D x_u(T),
$$
on obtient :
$$
\langle p_u(T), y(T)\rangle

* \int_0^T \langle Qx_u, y\rangle
  = \int_0^T \langle h, B^T p_u\rangle.
$$
---

### **Variation du coût**
$$
DJ(u),h
= \int_0^T \big( \langle R u, h\rangle + \langle B^T p_u, h\rangle \big),dt.
$$
D’où :
$$
\nabla J(u) = R u + B^T p_u.
$$
---

### **Conditions d’optimalité**

On veut que :
$$
DJ(u),h = 0 \quad \forall h \in L^2,
$$
c’est-à-dire :
$$
R u + B^T p_u = 0.
$$
Ainsi, la **commande optimale** est :
$$
u^* = -R^{-1} B^T p_u.
$$
---

**Suite du PMP – conclusion + exemple**

On avait :
$$
\int_0^T \langle \nabla J(u), h \rangle , dt \ge 0, \quad \forall h \in L^2.
$$
En choisissant $h = -\nabla J(u)$, on obtient :
$$
* \int_0^T |\nabla J(u)|^2 , dt \ge 0.
$$
Donc :
$$
\nabla J(u) = 0.
$$
---

### **Théorème (PMP – cas LQ)**

Si (u) est une solution du problème (LQ) et (x) la trajectoire associée,
alors (u) satisfait :
$$
u = -R^{-1} B^T p,
$$
où (p) vérifie :
$$
\begin{cases}
-\dot{p} = A^T p - Qx,\
p(T) = D x(T).
\end{cases}
$$
---

### **Exemple**

Considérons le cas :
$$
\min_u \frac{1}{2} \int_0^T (x(t)^2 + u(t)^2) , dt,
$$
sous la contrainte :
$$
\dot{x} = u, \quad x(0) = 1.
$$
---

On prend :
$$
n = 1,\quad m = 1,\quad A = (0),\quad B = (1),
$$
et :
$$
R = (1),\quad Q = (1),\quad D = (0).
$$
De plus :
$$
R, Q, D \in S_1^{++}(\mathbb{R}).
$$
---

**Application du PMP à l’exemple scalaire $cas (n=m=1)$**

D’après le **théorème**, le problème (LQ) admet une **unique solution**.

L’optimum est caractérisé par le système :
$$
\begin{cases}
\dot{x} = u,\\
\dot{p} = -x,\\
x(0) = 1,\\
p(T) = 0,
\end{cases}
$$
et la **relation d’optimalité** :
$$
u = -p.
$$
---

En particulier :
$$
\dot{x} = -p, \quad \dot{p} = -x.
$$
D’où :
$$
x'' - x = 0.
$$
---

L’**équation caractéristique** est :
$$
r^2 - 1 = 0
\quad\Longrightarrow\quad r = \pm 1.
$$
---

### **Solution générale**
$$
x(t) = \alpha e^{t} + \beta e^{-t}.
$$
---

La condition $x(0) = 1$ donne :
$$
\alpha + \beta = 1.
$$
Donc :
$$
x(t) = \alpha e^{t} + (1-\alpha)e^{-t}.
$$
---

### **Pour $p(t)$**

On a :
$$
p' = -x', \quad p'' = -x'' = -x = u = -p.
$$
D’où :
$$
p'' = -x = -(\alpha e^{t} + \beta e^{-t}),
$$
et la solution générale :
$$
p(t) = \gamma e^{t} + \delta e^{-t}.
$$
---

En utilisant $p' = -x$, on obtient la relation :
$$
\gamma e^{t} - \delta e^{-t}
= -\alpha e^{t} + (\alpha - 1)e^{-t}.
$$
---

**Suite et fin de l’exemple LQ scalaire**

On avait :
$$
p(t) = \gamma e^{t} + \delta e^{-t},
\quad\text{avec}\quad
\gamma = -\alpha,\ \delta = 1 - \alpha.
$$
---

Donc :
$$
p(t) = -\alpha e^{t} + (1-\alpha)e^{-t}.
$$
---

En utilisant la condition terminale $p(T) = 0$ :
$$
0 = -\alpha e^{T} + (1-\alpha)e^{-T}
$$
$$
\Rightarrow \alpha(e^{T} + e^{-T}) = e^{-T}
\Rightarrow \alpha = \frac{e^{-T}}{e^{T} + e^{-T}}
= \frac{e^{-T}}{2\cosh(T)}.
$$
---

Ainsi :
$$
1 - \alpha = \frac{e^{T}}{e^{T} + e^{-T}}
= \frac{e^{T}}{2\cosh(T)}.
$$
---

### **Expression finale de $p(t)$**
$$
p(t) = -\frac{e^{-T}}{2\cosh(T)} e^{t}

* \frac{e^{T}}{2\cosh(T)} e^{-t}.
$$
$$
\Rightarrow
p(t) = \frac{1}{2\cosh(T)} (e^{T-t} - e^{t-T})
= \frac{\sinh(T-t)}{\cosh(T)}.
$$
---

### **Rappel :**
$$
x = -p', \quad u = -p.
$$
---

Donc :
$$
x(t) = -p'(t) = -\frac{d}{dt}!\left(\frac{\sinh(T-t)}{\cosh(T)}\right)
= \frac{\cosh(T-t)}{\cosh(T)},
$$
et :
$$
\boxed{u(t) = -p(t) = -\frac{\sinh(T-t)}{\cosh(T)}}.
$$
---

Voici la **transcription fidèle et proprement organisée** du contenu du tableau :

---

### **Rappel : Introduction à l’Hamiltonien**

On définit la fonction d’Hamilton :
$$
H : \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^n \to \mathbb{R}
$$
par
$$
H(x, u, p) = \frac{1}{2}\langle R u, u\rangle

* \frac{1}{2}\langle Q x, x\rangle
* \langle p, A x + B u\rangle.
$$
---

### **Les conditions d’optimalité s’écrivent :**
$$
\begin{cases}
\dot{x} = \dfrac{\partial H}{\partial p}
= A x + B u, \\
\dot{p} = -\dfrac{\partial H}{\partial x}
= -Q x - A^T p.
\end{cases}
$$
et la **condition d’optimalité sur $u$** :
$$
\dfrac{\partial H}{\partial u} = 0
\quad \Longrightarrow \quad
R u + B^T p = 0
\quad \Longrightarrow \quad
u = -R^{-1} B^T p.
$$
---

### **(2) Commande en temps minimal**

Soit le système :
$$
\begin{cases}
\dot{x} = A x + B u,\\
x(0) = x_0,
\end{cases}
$$
avec $A \in M_n(\mathbb{R}),\ B \in M_{n,m}(\mathbb{R})$.

On cherche (u) dans $L^2(0,T; \mathbb{R}^m)$ $ou (u \in \mathbb{R}^m)$
tel que le système atteigne un **état final $x(T) = x_f$** en **temps minimal**.

---

**Rappel : condition pour pouvoir atteindre $x_f$ en temps (T)**

Si $U = \mathbb{R}^m$, alors
$$
\operatorname{rg}(B,|,AB,|,A^2B,|,\dots,|,A^{n-1}B) = n
$$
= **condition de contrôlabilité**.

---

Ici, $U$ **n’est pas nécessairement** égal à $\mathbb{R}^m$.

---

On considère :
$$
\inf_{u \in L^2(0,T;U)} T
\quad \text{tel que } \quad x_u(T) = x_f.
$$
---

### **Définition :**

L’**ensemble des points accessibles** à partir de $x_0$ en un temps (T > 0) est défini par :
$$
A(x_0, T)
= {,x_u(T) \mid u \in L^2(0,T;U),}.
$$
---

### **Théorème :**

On suppose que (U) est **compact, convexe, non vide** de $\mathbb{R}^m$.

Alors, pour tout (T > 0),
$$
A(x_0, T) \text{ est compact, convexe, et varie continûment par rapport à } T.
$$
---

### **Théorème (Existence)**

On suppose (U) **compact**.

Si le point $x_1$ est atteignable avec un contrôle à valeurs dans (U),
alors il existe une **trajectoire minimale** reliant $x_0$ à $x_1$.

De plus, $u^*$ est nécessairement **extrémal**, autrement dit :
$$
x^*(t) = x_{u^*}(t).
$$
---

### **Théorème (Caractérisation – Principe du Maximum de Pontryagin)**

Soit $u^* \in L^2(0,T;U)$ un contrôle qui transfère $x_0$ en $x_1$ en un temps minimal.

Alors il existe $p \neq 0$, non identiquement nul, tel que le système adjoint :
$$
p'(t) = -A^T p(t)
$$
soit satisfait, et pour presque tout $s \in [0,T]$,
le contrôle $u(s)$ **réalise instantanément le maximum du Hamiltonien** :
$$
H(x,p,u) = \langle p(s), A x(s) + B u(s) \rangle.
$$
Autrement dit :
$$
u(s) = \arg\max_{v \in U} \langle p(s), B v \rangle.
$$
---

**Remarque Contrôle Bang-Bang**

Comme nous l’avons vu, une particularité du **problème de temps minimal**
est que la **commande optimale** se trouve nécessairement
sur le **bord des contraintes** $U$.

---

Lorsque $U$ est un **intervalle de $\mathbb{R}$**,
la commande saute d’une **extrémité** à l’autre
à des instants de **commutation**.

On parle alors de **contrôle bang-bang**.

---

**Exemple :**
$$
U = [-1, 1].
$$
Schéma de commande rectangulaire alternant entre $+1$ et $-1$

---

### **Exemple : Commande optimale d’un train**
$$
\begin{cases}
x' = y,\\
y' = v,
\end{cases}
\quad
x(0) = x_0, \ y(0) = y_0,
\quad |v| \leq 1.
$$
---

Ici, on a :
$$
A =
\begin{pmatrix}
0 & 1\\
0 & 0
\end{pmatrix}
,
\qquad
B =
\begin{pmatrix}
0\\
1
\end{pmatrix}
$$.
$$
n = 2,\quad m = 1,\quad U = [-1,1].
$$
---

* Les **valeurs propres** de (A) sont de **partie réelle nulle**.
* La matrice de Kalman :
$$
[B | AB] =
  \begin{pmatrix}
  0 & 1\\
  1 & 0
  \end{pmatrix}
$$
  est de **rang 2**,
  donc le **critère de Kalman** est **vérifié**.

---

**Application du PMP au problème de commande en temps minimal (exemple du train)**

Donc, il **existe une trajectoire en temps minimal**.

Le système peut être **conduit à l’origine** en **temps fini**.

On note $T^*$ le **temps minimal**,
et $u^*$ la **commande optimale**.

---

D’après le **PMP**, il existe $p \neq 0$ telle que :
$$
\dot{p} = -A^T p.
$$
Dans notre cas :
$$
A =
\begin{pmatrix}
0 & 1\\
0 & 0
\end{pmatrix}
\quad \Longrightarrow \quad
-A^T =
\begin{pmatrix}
0 & 0\\
-1 & 0
\end{pmatrix}.
$$
---

Donc :
$$
p(s) = e^{-A^T s} p_0
= e^{\begin{pmatrix} 0 & 0 \\ -1 & 0 \end{pmatrix} s} p_0
= \begin{pmatrix} p_1(s) \\ p_2(s) \end{pmatrix}.
$$
---

Et pour presque tout (s) :
$$
u(s) = \arg\max_{v \in [-1,1]} \langle p(s), B v \rangle,
\quad B = \begin{pmatrix} 0 \\ 1 \end{pmatrix}.
$$
$$
\Rightarrow\ u(s) = \arg\max_{v \in [-1,1]} p_2(s) , v.
$$
---

### **Règle de commande :**
$$
\begin{cases}
\text{si } p_2(s) > 0 \Rightarrow u(s) = +1,\\
\text{si } p_2(s) < 0 \Rightarrow u(s) = -1.
\end{cases}
$$
---

⚠️ Si $p_2(s) = 0$ sur une partie de mesure positive,
alors cette partie du contrôle s’appelle un **arc singulier**.

---

**Question : l’ensemble des (s) tels que $p_2(s)=0$**

Peut-il être d’une **mesure positive** ?

---

On a le système adjoint :
$$
\begin{cases}
p_1' = 0,\\
p_2' = -p_1.
\end{cases}
$$
---

Donc :
$$
\exists, p_1^0, p_2^0 \in \mathbb{R} \text{ tels que }
\begin{cases}
p_1(s) = p_1^0,\\
p_2(s) = -p_1^0 s + p_2^0.
\end{cases}
$$
---

Comme $p$ est non nul, on a $(p_1^0, p_2^0) \neq (0,0)$.

---

**Étude de l’ensemble**
$$
E = \{ s \mid p_2(s) = 0 \}
= \{ s \mid -p_1^0 s + p_2^0 = 0 \}.
$$
---

C’est donc un **singleton** :
$$
E = \left\{ \frac{p_2^0}{p_1^0} \right\}
$$
Sa mesure est donc **nulle** :
$$
|E| = 0.
$$
---

**Conclusion :**
$$
u(s) =
\begin{cases}
+1, & \text{si } p_2(s) = -p_1^0 s + p_2^0 > 0,\\
-1, & \text{si } p_2(s) = -p_1^0 s + p_2^0 < 0.
\end{cases}
$$
---

On en déduit que $u$ est **bang-bang**
avec **un seul point de commutation**.

---

1️⃣ Si $u(s) = +1$ près de $T$, alors :
$$
\begin{cases}
x' = y,\\
y' = u = +1,
\end{cases}
$$
d’où :
$$
\begin{cases}
y(s) = s - T \le 0,\\
x(s) = \dfrac{1}{2}s^2 - Ts + \dfrac{1}{2}T^2
= \dfrac{1}{2}(s - T)^2.
\end{cases}
$$
$$
x(T) = y(T) = 0.
$$
---

2️⃣ Si $u(s) = -1$ près de $T$, alors :
$$
\begin{cases}
x' = y,\\
y' = u = -1,
\end{cases}
$$
donc :
$$
\begin{cases}
y(s) = -s + T,\\
x(s) = -\dfrac{1}{2}s^2 + Ts - \dfrac{1}{2}T^2
= -\dfrac{1}{2}(s - T)^2 + 1.
\end{cases}
$$
$$
x(T) = y(T) = 1.
$$
---

**Cas 1 : $u(s) = +1$ près de $0$**
$$
\begin{cases}
y'(s) = u = +1,\\
x'(s) = y,\\
x(0) = x_0,\ y(0) = y_0.
\end{cases}
$$
On en déduit :
$$
\begin{cases}
y(s) = s + y_0,\\
x(s) = \dfrac{1}{2}s^2 + y_0 s + x_0.
\end{cases}
$$
---

**Cas 2 : $u(s) = -1$ près de $0$**
$$
\begin{cases}
y'(s) = -1,\\
x'(s) = y,\\
x(0) = x_0,\ y(0) = y_0.
\end{cases}
$$
Donc :
$$
\begin{cases}
y(s) = -s + y_0,\\
x(s) = -\dfrac{1}{2}s^2 + y_0 s + x_0
= -\dfrac{1}{2}(y_0 - y)^2 + y_0 (y_0 - y) + x_0.
\end{cases}
$$
---

