### Ex 3 :
Soit $a \in \mathbb{R}^n$ non nul, $\alpha \in \mathbb{R}$, $r > 0$, $B$ symétrique définie positive.

$$
C = \{ x \in \mathbb{R}^n \mid \langle B x, x \rangle = r^2 \}
$$

et  

$$
F : \mathbb{R}^n \to \mathbb{R}, \quad x \mapsto \langle a, x \rangle + \alpha.
$$

On cherche :

$$
\inf_C F.
$$

---

### **Étapes :**  
1) **L’ensemble $C$ est une ellipsoïde.**  
2) **$C$ est compact et $F$ est continue**, donc $J$ atteint un minimum.  
3) **La contrainte s’écrit** :  

   $$
   h(x) = 0 \quad \text{où} \quad h(x) = \langle B x, x \rangle - r^2.
   $$

   On a :

   $$
   \nabla h(x) = 2 B x, \quad \forall x \neq 0, \quad \nabla h(x) \neq 0.
   $$

---

### **Condition d’optimalité (KKT) :**  
Si $\bar{x}$ est un minimum local de $F$ sur $C$, alors il existe $\lambda \in \mathbb{R}$ tel que :

$$
\nabla F(\bar{x}) + \lambda \nabla h(\bar{x}) = 0.
$$

$$
\Longleftrightarrow a + 2 \lambda B \bar{x} = 0.
$$

$$
\Longleftrightarrow \bar{x} = -\frac{1}{2 \lambda} B^{-1} a.
$$

$$
\text{Avec } \langle B \bar{x}, \bar{x} \rangle = r^2, \text{ on trouve } \lambda = \pm \frac{\sqrt{\langle a, B^{-1} a \rangle}}{2}.
$$

$$
\Longrightarrow \bar{x} = \pm \frac{B^{-1} a}{\sqrt{\langle a, B^{-1} a \rangle}} r.
$$

---

### **Valeur du minimum :**  
$$
F(\bar{x}) = \pm \sqrt{\langle a, B^{-1} a \rangle} r + \alpha.
$$

Donc $F$ admet son minimum en :

$$
x = -\frac{B^{-1} a}{\sqrt{\langle a, B^{-1} a \rangle}} r.
$$

---

### **Exemple 5 :**  

$$
\inf_{x \in C_1(\varepsilon)} G(x),
$$

où 

$$
G(x) = \frac{1}{2} \langle A x, x \rangle,
$$

avec $A$ symétrique définie positive.

$$
C_1(\varepsilon) = \{ x \in \mathbb{R}^n \mid \langle u, x \rangle = 1, \quad L e_1 x = \varepsilon \}.
$$

Avec  

$$
u = (1, -1), \quad L e_1 = (e_1, \dots, e_n),
$$

où $L e_A L = -L e_n$.

---

### **1) Montrons que $C_1(\varepsilon)$ est non vide :**  

On remarque que $u$ et $e$ sont non colinéaires.

De plus, $\langle u, x \rangle = 1$ et $L e_1(x) = \varepsilon$ décrivent des plans.  
Ces plans ne sont pas parallèles, donc $C_1(\varepsilon) \neq \varnothing$.

---

### **2) Montrons que le problème admet une unique solution :**  

$C_1(\varepsilon)$ est fermé et $G$ est coercive et continue, donc il existe un minimum.

De plus, $G$ est strictement convexe, d’où le minimum est unique.

#### **3) Montrons que $h^*$ est solution du problème de minimisation**  
Soit $x_0 \in C_1(\varepsilon)$ et $h^* = x^* - x_0$.  

Montrons que $h^*$ résout :

$$
\inf_{h \in H} J(h),
$$

où :

$$
H = \{ h \in \mathbb{R}^n \mid \langle u, h \rangle = 0, \quad L e_1 h = 0 \}.
$$

Puisque :

$$
x \in C_1(\varepsilon) \Longleftrightarrow x - x_0 \in H,
$$

on pose :

$$
J(h) = G(h + x_0).
$$

Ainsi,

$$
\forall h \in H, \quad J(h) = G(h + x_0) = G(x) \geq G(x^*) = J(x^* - x_0) = J(h^*).
$$

---

#### **4) Vérification des conditions d’optimalité**  
Il existe $\lambda, \mu$ tels que :

$$
\nabla J(h^*) = \lambda u + \mu e.
$$

On note :

$$
h_1(h) = \langle u, h \rangle, \quad h_2(h) = L e_1 h.
$$

On a $J, h_1, h_2 \in \mathcal{C}^1$.  

De plus, $(\nabla h_1(x), \nabla h_2(x))$ est une famille libre,  
car $u, e$ sont non colinéaires.

D’après les critères des extrémas liés, il existe $\lambda, \mu \in \mathbb{R}$ tels que :

$$
\nabla J(h^*) - \lambda \nabla h_1(h^*) - \mu \nabla h_2(h^*) = 0.
$$

$$
\Longrightarrow \nabla J(h^*) = \lambda u + \mu e.
$$

$$
\nabla J(h^*) = \nabla G(h^* + x_0) = \nabla G(x^*) = A x^*.
$$

Donc :

$$
x^* = \lambda A^{-1} u + \mu A^{-1} e.
$$

On a le système :

$$
\begin{cases} 
\langle u, x^* \rangle = 1 \\ 
\langle e, x^* \rangle = \varepsilon 
\end{cases}
$$

$\Rightarrow$

$$
\begin{cases} 
\langle A^{-1} u, u \rangle \lambda + \langle A^{-1} e, u \rangle \mu = 1 \\ 
\langle A^{-1} u, e \rangle \lambda + \langle A^{-1} e, e \rangle \mu = \varepsilon 
\end{cases}
$$

Le déterminant est :

$$
\Delta = \| u \|_A^2 \| e \|_A^2 - \langle u, e \rangle_A^2.
$$

où :

$$
\| u \|_A^2 = \langle A^{-1} u, u \rangle, \quad \langle u, e \rangle_A = \langle A^{-1} u, e \rangle.
$$

On a :

$$
\Delta \geq 0
$$

par **Cauchy-Schwarz**, et **$\Delta \neq 0$ car $u$ et $e$ ne sont pas colinéaires**.

On trouve :

$$
\lambda = \frac{\| e \|_A^2 - \varepsilon \langle u, e \rangle_A}{\Delta}, \quad \mu = \frac{-\langle e, u \rangle_A + \varepsilon \| u \|_A^2}{\Delta}.
$$

Ainsi,

$$
x^* = \frac{\| e \|_A^2 - \varepsilon \langle u, e \rangle_A}{\Delta} A^{-1} u + \frac{-\langle e, u \rangle_A + \varepsilon \| u \|_A^2}{\Delta} A^{-1} e.
$$

### **Exemple 7 :**  
$$
\inf_{x \in K} \frac{1}{2} \langle A x, x \rangle - \langle b, x \rangle
$$
où  
$$
K = \{ x \in \mathbb{R}^3 \mid x_3 > 1, \quad x_2 - 2x_3 = 1 \}.
$$
Données :
$$
A =
\begin{pmatrix}
-1 & -1 & 0 \\
-1 & 2 & 0 \\
0 & -1 & 3
\end{pmatrix},
\quad
b =
\begin{pmatrix}
1 \\
1 \\
-1
\end{pmatrix}.
$$

---

### **1) Expression de $J(x)$ :**  
$$
J(x) = \frac{1}{2} \langle A x, x \rangle - \langle b, x \rangle.
$$

On réécrit :
$$
\langle A x, x \rangle = \frac{1}{2} \langle A x, x \rangle + \frac{1}{2} \langle A^T x, x \rangle.
$$

Donc :
$$
J(x) = \frac{1}{2} \langle \frac{A + A^T}{2} x, x \rangle - \langle b, x \rangle.
$$

On note $\tilde{A} = \frac{A + A^T}{2}$, qui est symétrique définie positive.  
Ainsi, le problème admet une unique solution.

---

### **2) Conditions de qualification et application de KKT :**  
Les contraintes sont affines, donc tous les points de $K$ sont qualifiés.

D’après **KKT**, il existe $\lambda \in \mathbb{R}, \mu \geq 0$ tels que :

$$
\nabla J(\bar{x}) + \lambda \nabla h(\bar{x}) + \mu \nabla g(\bar{x}) = 0.
$$

$$
\mu g(\bar{x}) = 0.
$$

$$
\mu \geq 0.
$$

$$
h(\bar{x}) = 0.
$$

---

### **Calcul des gradients :**  
$$
\nabla J(\bar{x}) = A \bar{x} - b.
$$

$$
\nabla h(\bar{x}) =
\begin{pmatrix}
0 \\
1 \\
-2
\end{pmatrix}.
$$

$$
\nabla g(\bar{x}) =
\begin{pmatrix}
-1 \\
0 \\
0
\end{pmatrix}.
$$
