## I) Résolution numérique de l'équation de la chaleur (ou équation de diffusion)

### 1) Méthode de développement en série  

Inconnue $u(x,t)$  

Concentration de sucre dans un café
$$
\begin{cases}
x \in [0, L],\quad L > 0\\
t \geq 0
\end{cases}
$$
$x$ espace  
$t$ temps  

$u(x,0) = u_0(x)$  

$u_0$, fonction donnée.

$\frac{\partial u}{\partial x}$ dérivée partielle de $u$ / $x$  

$\frac{\partial u}{\partial t}$ dérivée partielle de $u$ / $t$  

$\frac{\partial u}{\partial x} = \partial_x u = u_x$

$\frac{\partial^2 u}{\partial x^2} = \partial_{xx} u = u_{xx}$

Équation de diffusion

a) $\forall (x,t) \in [0,L] \times [0,+\infty[$  

$$
\frac{\partial u}{\partial t} (x,t) - \frac{\partial^2 u}{\partial x^2} (x,t) = 0
$$  
b) $u(x,0) = u_0(x)$, $x \in [0,L]$  

c) $\frac{\partial u}{\partial x} (0,t) = \frac{\partial u}{\partial x} (L,t) = 0$  

(étanchéité)

Solutions de a) et c)

$u_i(x,t) = \cos \left( \frac{i \pi x}{L} \right) \exp \left( - \frac{i^2 \pi^2 t}{L^2} \right)$  

$i \in \mathbb{N}$  

$\partial_t u_i - \partial_{xx} u_i = 0$  

$\partial_x u_i = \exp \left( - \frac{i^2 \pi^2 t}{L^2} \right) \frac{\pi}{L} \sin \left( \frac{\pi x}{L} \right)$  

$\partial_x u_i(0,t) = 0$  

$\partial_x u_i(L,t) = 0$  

Ex0

$u(x,t) = v(x) g(t)$  
On cherche $v$ et $g$

$\partial_t u - \partial_{xx} u = 0+ cL$

$u(x,t) = \sum_{i=0}^{+\infty} C_i u(x,t)$

$u$ est solution (faible) de a) et c)

Fourier.

b) $u(x,0) = u_0(x)$

$u(x,0) = \sum_{i=0}^{+\infty} C_i \cos\left( \frac{i \pi x}{L} \right)$

$\tilde{v}_i (x) = \cos \left( \frac{i \pi x}{L} \right)$

Prop. la famille $\left( \tilde{v}_i \right)_{i=0, \dots, \infty}$ est une famille orthogonale pour le produit scalaire de $L^2([0,L])$

dém. $\langle u, v \rangle = \int_0^L u(x) v(x) dx$

$\int_0^L \tilde{v}_i(x) \tilde{v}_j(x) dx$

$= \int_0^L \cos\left( \frac{i \pi x}{L} \right) \cos\left( \frac{j \pi x}{L} \right) dx$

($\cos p \cos q = \frac{1}{2} \left[ \cos \frac{p+q}{2} + \cos \frac{p-q}{2} \right]$)

$= \int_0^L \frac{1}{2} \left[ \cos \frac{(i+j) \pi x}{2L} + \cos \frac{(i-j) \pi x}{2L} \right] dx$

$i = j =0 \Rightarrow \langle \tilde{v}_i, \tilde{v}_j \rangle = L$

$i \neq j \neq 0 \Rightarrow \langle \tilde{v}_i, \tilde{v}_j \rangle = L/2$

$i \neq j \Rightarrow \langle \tilde{v}_i, \tilde{v}_j \rangle$

$= \frac{1}{2} \left[ \frac{2L}{(i+j) \pi} \sin \left( \frac{(i+j) \pi x}{2L} \right) \right]_0^L$

$+ \frac{1}{2} \left[ \frac{2L}{(i-j) \pi} \sin \left( \frac{(i-j) \pi x}{2L} \right) \right]_0^L$

$= 0$

$\| v \|_{L^2([0,L])} = \sqrt{\langle v, v \rangle}$

$\langle \tilde{v}_0, \tilde{v}_0 \rangle = L$

$\Rightarrow \| \tilde{v}_0 \| = \sqrt{L}$

$\langle \tilde{v}_i, \tilde{v}_i \rangle = \frac{L}{2}$ \ \ $i \neq 0$

$\Rightarrow \| \tilde{v}_i \| = \sqrt{\frac{L}{2}}$

$v_i = \frac{1}{\| \tilde{v}_i \|} \tilde{v}_i$

$$
v_i(x) = \begin{cases} 
\frac{1}{\sqrt{L}} & i = 0 \\ 
\sqrt{\frac{2}{L}} \cos\left( \frac{i \pi x}{L} \right) & i \neq 0 
\end{cases}
$$

**Bon**

$u_0 = \sum C_i \tilde{v}_i$

$= \sum C_i \| \tilde{v}_i \| v_i$

$\langle u(\cdot,0), v_j \rangle = \langle \sum C_i \| \tilde{v}_i \| v_i, v_j \rangle$

$= \sum C_i \| \tilde{v}_i \| \langle v_i, v_j \rangle$

$= C_j \| \tilde{v}_j \|$

$C_i = \dfrac{1}{\| \tilde{v}_i \|} \langle u(\cdot,0), \tilde{v}_i \rangle$

$i = 0$

$C_0 = \frac{1}{L} \int_0^L u(x,0) dx$

$i \neq 0$

$C_i = \frac{2}{L} \int_0^L u(x,0) \cos \left( \frac{i \pi x}{L} \right) dx$

### 2) Méthode de différences finies.

Approximation des dérivées partielles par des opérateurs en dimension finie $N$ entier positif.

```tikz
\begin{document}
\begin{tikzpicture}[domain=-5:5] 
    % 设置坐标轴
    \draw[->] (-0.3, 0) -- (4.4, 0);
    \draw (0,1.7) -- (0,-1.7)node[below] {$x=0$};
    \draw (4,1.7) -- (4,-1.7)node[below] {$x=L$};
    \draw (1,0.7) -- (1,-0.7);
    \draw (2,0.7) -- (2,-0.7);
    \draw (3,0.7) -- (3,-0.7);    
\end{tikzpicture}
\end{document}
```

$\Delta x = h = \frac{L}{N}$ pas d’espace  

$x_i = i \Delta x + \frac{\Delta x}{2}$, $i = 0, \dots, N-1$  

$t_n = n \Delta t$  

$\Delta t$ : pas de temps  

$u(x_i, t_n) \simeq u_i^n$  

$u_i^0 = u(x_i, 0)$  

Approximation de $u_{xx}$ et $u_t$ :

$f(x + h) = f(x) + h f'(x) + \frac{h^2}{2} f''(x) + \frac{h^3}{6} f^{(3)}(x) + O(h^4)$  

$f(x - h) = f(x) - h f'(x) + \frac{h^2}{2} f''(x) - \frac{h^3}{6} f^{(3)}(x) + O(h^4)$

$f(x+h) + f(x-h)$

$= 2f(x) + h^2 f''(x) + O(h^4)$

$f''(x) = \frac{f(x-h) - 2 f(x) + f(x+h)}{h^2} + O(h^2)$

$f''(x_i) = \frac{f(x_{i-1}) - 2 f(x_i) + f(x_{i+1})}{h^2} + O(h^2)$

$\Delta t = \tau$

$g(t+\tau) = g(t) + \tau g'(t) + O(\tau^2)$

$g'(t) = \frac{g(t+\tau) - g(t)}{\tau} + O(\tau)$

(Question pour plus tard) comment poser à $O(\tau^2)$ ?

$g'(t+\tau) = \frac{g(t+\tau) - g(t)}{\tau} + O(\tau)$

$g'(t_n) = \frac{g(t_{n+1}) - g(t_n)}{\tau}$

ou $g'(t_{n+1}) = \frac{g(t_{n+1}) - g(t_n)}{\tau}$

---

Différences finies décentrées :

$g(t) = g(t + \tau - \tau)$

$= g(t+\tau) - \tau g'(t+\tau) + O(\tau^2)$

$g'(t+\tau) = \frac{g(t+\tau) - g(t)}{\tau} + O(\tau)$

$u_{xx}(x_i, t_n) \longrightarrow \dfrac{u_{i-1}^n - 2u_i^n + u_{i+1}^n}{h^2}$

$u_t(x_i, t_n) \longrightarrow \dfrac{u_i^{n+1} - u_i^n}{\tau}$

$\left( \Delta u \dfrac{u_i^n - u_{i-1}^n}{\tau} \right)$

$\frac{u_i^{n+1} - u_i^n}{\tau} + \frac{- u_{i-1}^n + 2 u_i^n - u_{i+1}^n}{h^2} = 0$

$u_i^{n+1} = u_i^n - \frac{\tau}{h^2} \left( - u_{i-1}^n + 2 u_i^n - u_{i+1}^n \right)$

---

**Schéma explicite.**

Problems aux bords :

$i = 0$ il manque $u_{-1}^n$  

$i = N - 1$ il manque $u_N^n$

$\frac{\partial u}{\partial x} \bigg|_{x = 0} = 0$

$f'(x) \bigg|_{x=0} = \frac{f(x_1) - f(x_0)}{h}$

$= \frac{f(\frac{h}{2}) - f(- \frac{h}{2})}{h} + O(h^2)$

$\frac{\partial u}{\partial x} \bigg|_{x = 0} \simeq \frac{u_0^n - u_{-1}^n}{h} = 0$

$$
\begin{cases} 
u_{-1}^n = u_0^n \\
u_N^n = u_{N-1}^n 
\end{cases}
$$

$i = 0$

$\frac{u_i^{n+1} - u_i^n}{\tau} + \frac{u_i^n - u_{i+1}^n}{h^2} = 0$

$i = N - 1$

$\frac{u_i^{n+1} - u_i^n}{\tau} + \frac{u_{i-1}^n - u_i^n}{h^2} = 0$

---

**Forme matricielle**

$$
U^n = \begin{pmatrix} 
u_0^n \\
\vdots \\
u_{N-1}^n 
\end{pmatrix}
$$

$\frac{U^{n+1} - U^n}{\tau} + \frac{1}{h^2} A U^n = 0$

$$
A = \begin{pmatrix} 
1 & -1 & 0 & \dots&0 \\
-1 & 2 & \dots & \dots &0\\
0 & \dots & \ddots & \dots&0 \\
\dots & \dots & -1 & 2 & -1 \\
0 & \dots & 0 & -1&2
\end{pmatrix}
$$

$A$ symétrique.

$$
U^0 = \begin{pmatrix} 
u(x_0, 0) \\
\vdots \\
u(x_{N-1}, 0) 
\end{pmatrix}
$$

$U^{n+1} = \left( I - \frac{\tau}{h^2} A \right) U^n$

$\frac{U^n - U^{n-1}}{\tau} + \frac{1}{h^2} A U^n = 0$

$\left( I + \frac{\tau}{h^2} A \right) U^n = U^{n-1}$

$\left( I + \frac{\tau}{h^2} A \right) U^{n+1} = U^n$

---

Pour passer de $U^n$ à $U^{n+1}$, il faut résoudre un système linéaire.

Python + Rust  
→ utiliser les matrices avec beaucoup de "space matrix"

$A \cdot v \rightarrow N$

$\rightarrow U^{n+1} = \left( I + \frac{\tau}{h^2} A \right)^{-1} U^n$

$u$ solution exacte  
$u_i^n$ est approche  

$\sup_{i,n} |u(x_i, t_n) - u_i^n| \longrightarrow 0$  
$N,P \longrightarrow \infty$

Choisir $N, P$

$h = \frac{L}{N}$ \quad $0 \leq t_n \leq T$

$\tau = \frac{T}{P}$ \quad $0 \leq x_i \leq L$

On a *** pour le schéma explicite sous une condition (Courant-Friedrichs-Lewy, CFL) :

$\tau \leq \frac{h^2}{2}$   condition de stabilité

Le schéma explicite est inconditionnellement stable, il suffit que $\tau, h \longrightarrow 0$.



$$
\frac{\partial u}{\partial t} - \frac{\partial^2 u}{\partial x^2} = 0
$$

$$
\frac{\partial u(0,t)}{\partial x} = \frac{\partial u(L,t)}{\partial x} = 0
$$

$$
u(x,0) = u_0(x)
$$

$$
u^m = \begin{pmatrix} u_0^m \\ u_1^m \\ \vdots \\ u_{N-1}^m \end{pmatrix}
$$

$$
A = \begin{pmatrix} 1 & -1 & 0 & \cdots & 0 \\ -1 & 2 & -1 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & \cdots & -1 & 2 & -1 \\ 0 & \cdots & 0 & -1 & 1 \end{pmatrix}
$$

$$
U^{m+1} - U^m + \frac{1}{h^2} AU^m = 0
$$

Explicite:

$$
U^{m+1} - U^m + \frac{1}{h^2} AU^{m+1} = 0
$$

$$
\frac{U^{m+1} - U^m}{\tau} + \frac{1}{h^2} AU^{n+1} = 0 \quad \text{(implicite)}
$$

### 3) Quelques propriétés mathématiques

$$
u(x,t) = c_0 + \sum_{i=1}^{\infty} c_i e^{-\frac{i^2 \pi^2 t}{L^2}} \cos\left(\frac{i \pi x}{L}\right)
$$

$$
c_0 = \frac{1}{L} \int_0^L u(x,0) dx
$$

$$
c_i = \frac{2}{L} \int_0^L u(x,0) \cos\left(\frac{i \pi x}{L}\right) dx \quad \text{pour} \quad i \geq 1
$$

$$
u_0(x) \in L^2([0,L])
$$

$$
\|f\|^2 = \int_0^L f^2 dx
$$

$$
|c_0| = \left| \langle u_0, e_0 \rangle \right| \leq \|u_0\|
$$

$$
|c_i| = \frac{2}{L} \left| \langle u_0, e_i \rangle \right| \leq \frac{2}{L} \|u_0\| = C
$$

$$
f_i = x \mapsto c_i e^{-\frac{i^2 \pi^2 t}{L^2}} \cos\left(\frac{i \pi x}{L}\right)
$$

$$
f_i \text{ est continue, et à } t \text{ fixé }
$$

$$
\|f_i\|_{\infty} = |c_i| e^{-\frac{i^2 \pi^2 t}{L^2}} \leq C e^{- \frac{i^2\pi^2 t}{L^2}}
$$

Espace $E = \left\{ f \in C^0([0,L]) \; | \; \|f\| < +\infty \right\}$

La série $\sum f_i$ converge uniformément dans $E$ pour $t > 0$.

$$
u(x,t) = \sum_{i=1}^{\infty} f_i(x,t)
$$

Question:

$$
\frac{\partial u}{\partial t} - \frac{\partial^2 u}{\partial x^2} = 0 ?
$$

La réponse est **oui** lorsque la série des dérivées converge.


$$
\partial_t^\alpha \partial_t^\beta f_i(x,t) = e^{-\frac{i^2 \pi^2 t}{L^2}} \times R_i(x,t)
$$

$$
R_i(x,t) = \text{puissance de } i^2
$$

$$
\times \sin\left(\frac{i \pi x}{L}\right) \quad \text{ou bien des} \quad \cos\left(\frac{i \pi x}{L}\right)
$$

$$
|R_i(x,t)| \leq P(t)
$$

$$
P \text{ est un polynôme}
$$

$$
\|\partial_t^\alpha \partial_t^\beta f_i(x,t)\|_\infty \leq e^{-\frac{i^2 \pi^2 t}{L^2}} P(t)
$$

Conclusion : on a encore une série de fractions normalement convergente pour $\| \cdot \|_{\infty}$.

On peut dériver sous le signe $\sum$ autant de fois que nécessaire, $t \geq t_0 > 0$.

$$
u(x,t) \text{ est de classe } \infty \text{ pour } t > 0
$$

Et on a bien :

$$
\frac{\partial u}{\partial t} - \frac{\partial^2 u}{\partial x^2} = 0
$$

$$
t = 0
$$

$$
c_0 + \sum_{i=1}^{\infty} c_i \cos\left(\frac{i \pi x}{L}\right) = u_0(x)
$$

$$
u_0 \in L^2 \quad \text{on a convergence dans} \; l^2
$$

(Théorie des séries de Fourier)


Si $u_0$ est de classe $C^2$ et $u_0'(L) = u_0'(0) = 0$

$$
c_i = \frac{2}{L} \int_0^L u_0(x) \cos\left(\frac{i \pi x}{L}\right) dx \quad i \geq 1
$$
$$
= - \frac{2}{L} \int_0^L u_0'(x) \frac{L}{i \pi} \sin\left(\frac{i \pi x}{L}\right) dx + \frac{2}{L} \left[ u_0(x) \frac{L}{i \pi} \sin\left(\frac{i \pi x}{L}\right) \right]_0^L
$$

$$
= - \frac{2}{L} \int_0^L \left( u_0''(x) \frac{L^2}{i^2 \pi^2} \cos\left(\frac{i \pi x}{L}\right) \right) dx
$$

$$
+ \frac{2}{L} \left[ \frac{u_0'(x)}{\cos\left(\frac{i \pi x}{L}\right)} \right]_0^L
$$

$$
c_i = -\frac{2}{L \pi^2 i^2} \int_0^L u_0''(x) \cos\left(\frac{i \pi x}{L}\right) dx
$$

$$
|c_i| \leq \frac{2}{L \pi^2 i^2} \| u_0'' \|
$$

Terme général d'une série $CV$, donc $\sum c_i$ est uniformément convergente.

$$
\frac{\partial u}{\partial t} - \frac{\partial^2 u}{\partial x^2} = 0
$$

Propriétés de la solution (quand elle est assez régulière) :

Conservation :

$$
M(t) = \int_0^L u(x,t) dx
$$

$$
\frac{dM(t)}{dt} = \int_0^L \frac{\partial u}{\partial t}(x,t) dx
$$

$$
= \int_0^L \frac{\partial^2 u}{\partial x^2}(x,t) dx
$$

$$
= \left[ \frac{\partial u}{\partial x}(x,t) \right]_0^L = 0
$$

(à cause des conditions aux limites)

Donc :

$$
\frac{dM(t)}{dt} = 0
$$

Donc :

$$
M(t) = M(0) = L c_0
$$

**Décroissance de l'énergie :**

$$
E(t) = \frac{1}{2} \int_0^L u^2(x,t) dx
$$

$$
\frac{dE(t)}{dt} = \frac{1}{2} \int_0^L \partial_t u^2(x,t)  dx
$$

$$
= \int_0^L u(x,t) \partial_t u(x,t) dx
$$

$$
= \int_0^L u(x,t) \frac{\partial^2 u}{\partial x^2}(x,t) dx
$$

$$
= - \int_0^L \left( \frac{\partial u}{\partial x}(x,t) \right)^2 dx + \left[ u \frac{\partial u}{\partial x} \right]_0^L
$$

donc, 

$$
\frac{dE(t)}{dt} \leq 0
$$

$$
E(t) \leq E(0)
$$

**Traduction sur le schéma** :

$$
u_i^n = u(x_i, t_n)
$$

$$
\begin{cases}
x_i = ih + \frac{h}{2} \\
t_n = n \tau
\end{cases}
$$
$$
M^m = h \sum_{i=0}^{N-1} u_i^n
$$

(Approximation $\int u(x,t)$ par la méthode des rectangles)

$$
U^n = \begin{pmatrix} u_0^n \\ \vdots \\ u_{N-1}^n \end{pmatrix}
$$

$$
M^n = h(1, \dots, 1) U^n
$$

$$
U^{n+1} = \left(I - \frac{\tau}{h^2} A \right) U^n
$$

$$
\begin{pmatrix} 1 & -1 & 0 & \cdots & 0 \\ -1 & 2 & -1 & \cdots & 0 \\ 0 & -1 & 2 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & \cdots & 0 & -1 & 1 \end{pmatrix}
$$

$$
(1, \dots, 1) \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}
$$

$$
(1, \dots, 1) A = \begin{pmatrix} 0 \\ \vdots \\ 0 \end{pmatrix}
$$

$$
(1, \dots, 1) U^{n+1} = (1, \dots, 1) U^n
$$

$$
M^{n+1} = M^n
$$

$$
U^{n+1} - U^n + \frac{\tau}{h^2} A U^{n+1} = 0 \quad \text{(schéma implicite)}
$$

$$
(1, \dots, 1) \quad \text{à gauche}
$$

$$
M^{n+1} - M^n = 0
$$

Les schémas implicite et explicite conservent la masse.

$$
E^n = \frac{h}{2} \sum_{i=0}^{N-1} (u_i^n)^2 \sim E(t^n)
$$

$$
E^n = \frac{h}{2} \| U^n \|_2^2
$$

**Théorème de von Neumann, Lax** :
Dit que la solution discrète converge vers la solution exacte quand $(h, \tau) \rightarrow (0,0)$, si le schéma est stable :

$$
\forall m, \quad E^{n+1} \leq E^n
$$

**Calcul des valeurs propres et vecteurs propres de** $A$ :

$$
A e_k = \lambda_k e_k
$$

$\lambda_k$ : valeurs propres de $A$  
$e_k$ : vecteurs propres

$$
A V = \lambda V
$$

$$
A = \begin{pmatrix} 
1 & -1 & 0 & \cdots & 0 \\ 
-1 & 2 & -1 & \cdots & 0 \\ 
0 & -1 & 2 & \cdots & 0 \\ 
\vdots & \vdots & \vdots & \ddots & \vdots \\ 
0 & \cdots & 0 & -1 & 1 
\end{pmatrix}
$$

$$
V = \begin{pmatrix} 
V_0 \\ 
\vdots \\ 
V_{N-1} 
\end{pmatrix}
$$

$$
v_0 - v_1 = \lambda v_0
$$

$$
-v_0 + 2v_1 - v_2 = \lambda v_1
$$

$$
-v_{i-1} + 2v_i - v_{i+1} = \lambda v_i
$$

$$
-v_{N-2} + 2v_{N-1} - v_N = \lambda v_{N-1}
$$

$$
-v_{N-1} + 2v_N - v_{N+1} = \lambda v_N
$$

**Convention :**
$$
\begin{cases}
v_{-1} = v_0 \\
v_N = v_{N-1}
\end{cases}
$$

$$
V_i = \cos(k \pi x_i)
$$

$$
V_{-1} = \cos\left(k \pi (-\frac{h}{2})\right)
$$

$$
? = \cos\left(k \pi \frac{h}{2}\right)
$$

Avec : $L=1$

$$
h = \frac{L}{N} = \frac{1}{N}
$$

$$
x_i = ih + \frac{h}{2}
$$

de même

$$
V_{N-1} = V_N
$$

$$
-\cos(k \pi x_{i-1}) + 2 \cos(k \pi x_i) - \cos(k \pi x_{i+1})
$$

$$
\cos(p) \cos(q) = 2 \cos\left(\frac{p+q}{2}\right) \cos\left(\frac{p-q}{2}\right)
$$

$$
2 \cos(k \pi x_i) - 2 \cos\left(k \pi \frac{x_{i+1} + x_{i-1}}{2}\right) \cos\left(k \pi \frac{x_{i+1} - x_{i-1}}{2}\right)
$$

$$
= 2 \cos(k \pi x_i) - 2 \cos(k \pi x_i) \cos(k \pi h)
$$

$$
= 2 \cos(k \pi x_i) \cdot 2 (1 - \cos(k \pi h))
$$

$$
= v_i \cdot 4 \sin^2\left(\frac{k \pi h}{2}\right)
$$
car, 
$$
\cos(2p) = 1 - 2 \sin^2(p)
$$

$$
\begin{cases}
v_i^{(k)} = \cos(k \pi x_i), \quad k = 0, \dots, N-1 \\
\lambda_k = 4 \sin^2\left(\dfrac{k \pi h}{2}\right)
\end{cases}
$$

$$
\lambda_0 = 0
$$

$$
\lambda_k > 0 \quad \text{pour} \quad k = 1, \dots, N-1
$$

$A$ est une matrice symétrique :

$$
A \geq 0
$$

$$
v^{(0)} = \begin{pmatrix} 1 \\ \vdots \\ 1 \end{pmatrix}
$$

$$
\ker(A) = \text{vect} \, v^{(0)} \quad \text{de dimension 1}
$$

Stabilité du schéma implicite :

$$
\left\{ v^{(k)}, \quad k = 0, \dots, N-1 \right\} \quad \text{base orthogonale de} \, \mathbb{R}^N
$$

$$
\left\{ e^{(k)}, \quad k = 0, \dots, N-1 \right\} \quad \text{BON de vecteurs propres de} \, A
$$

$$
\beta = \frac{\tau}{h^2} \leq \frac{1}{2}
$$

Condition suffisante pour la stabilité du schéma explicite :

$$
U^{n+1} = \left(I - \frac{\tau}{h^2} A\right) U^n
$$

$$
\|U^{n+1}\| \leq \|U^n\| \quad ?
$$

$$
\|U^{n+1}\| \leq \left\|I - \frac{\tau}{h^2} A\right\| \|U^n\|
$$

$$
\|A\| = \sup_{\|x\| \neq 0} \frac{\|Ax\|}{\|x\|}
$$

$$
\|M\| = \sqrt{\rho(M^T M)}
$$

Si $M$ est symétrique :

$$
\|M\| = \rho(M)=\max_k |\mu_k|
$$

$\mu_k$ : valeurs propres de $M$

$$
M = I - \frac{\tau}{h^2} A
$$

$e^{(k)}$ sont vecteurs propres de $M$ :

$$
\mu_k = 1 - \frac{\tau}{h^2} 4 \sin^2\left(\frac{k \pi}{2N}\right), \quad k = 0, \dots, N-1
$$


$$
1 - \frac{4\tau}{h^2} \leq \mu_k \leq 1
$$

$$
\frac{\tau}{h^2} \leq \frac{1}{2}
$$

Donc :

$$
-1 \leq \mu_k \leq 1
$$

Donc :

$$
\rho(M) \leq 1
$$

Donc, le schéma est stable.

$$
U^{n+1} = \left(I + \frac{\tau}{h^2} A\right)^{-1} U^n
$$

$$
\left\|\left(I + \frac{\tau}{h^2} A\right)^{-1}\right\| \leq 1 \quad ?
$$

$$
M = \left(I + \frac{\tau}{h^2} A\right)^{-1}
$$

$$
\mu_k = \frac{1}{1 + \dfrac{\tau}{h^2} 4 \sin^2\left(\dfrac{k \pi}{2N}\right)}
$$

On voit que :

$$
0 \leq \mu_k \leq 1
$$

Indépendamment de $\beta = \dfrac{\tau}{h^2}$

Donc :

$$
\rho(M) \leq 1
$$

Le schéma implicite est inconditionnellement stable.

**θ-schéma (Crank-Nicolson)** :

$$
(1 - \theta) \times \frac{U^{n+1} - U^n}{\tau} + \frac{A}{h^2} U^n = 0
$$

$$
\theta \times \frac{U^{n+1} - U^n}{\tau} + \frac{A}{h^2} U^{n+1} = 0
$$

$$
\frac{U^{n+1} - U^n}{\tau} + \frac{\tau A}{h^2} \theta \, U^{n+1} + \frac{\tau A}{h^2} (1 - \theta) \, U^n = 0
$$

$$
\left(I + \theta \frac{\tau A}{h^2}\right) U^{n+1} = \left(I - (1 - \theta) \frac{\tau A}{h^2}\right) U^n
$$

$$
\theta = 0 \quad \text{(schéma explicite)}
$$

$$
\theta = 1 \quad \text{(schéma implicite)}
$$

$$
\theta = \frac{1}{2} \quad \text{(Crank-Nicolson)}
$$

$$
\left(I + \frac{\tau A}{2 h^2}\right)U^{n+1} = \left(I - \frac{\tau A}{2 h^2}\right) U^n
$$

$$
U^{n+1} = M U^n
$$

$$
M = \left(I + \frac{\tau A}{2 h^2}\right)^{-1} \left(I - \frac{\tau A}{2 h^2}\right)
$$

$$
\mu_k = \frac{1 - \frac{\tau}{2h^2} 4 \sin^2\left(\frac{k \pi}{2N}\right)}{1 + \frac{\tau}{2h^2} 4 \sin^2\left(\frac{k \pi}{2N}\right)}=\frac{1 - x}{1 + x}
$$

Avec $x \geq 0$.

$$
0 \leq \mu_k \leq 1
$$

Donc, le schéma est inconditionnellement stable.

$$
\partial_t u(x_i, t_n + \frac{\tau}{2}) - \partial_{xx} u(x_i, t_n + \frac{\tau}{2}) = 0
$$

$$
\frac{u(x_i, t_{n+1}) - u(x_i, t_n)}{\tau} = \partial_{xx} \left(\frac{u(x_i, t_{n+1}) + u(x_i, t_n)}{2}\right) + O(\tau^2)
$$

$$
\partial_{xx} \left( \frac{u(x_i, t_{m+1}) + u(x_i, t_m)}{2} \right) = \frac{u(x_{i+1}, t_{m+1}) - 2 u(x_i, t_{m+1}) + u(x_{i-1}, t_{m+1})}{h^2}
$$

$$
+ \frac{1}{2h^2} \left( u(x_{i+1}, t_m) - 2u(x_i, t_m) + u(x_{i-1}, t_m) \right) = -R
$$


Il y a $R$

$$
au \ point \ (x_i, t_m + \frac{\tau}{2})
$$

$$
= \dots \partial_{xx} u(x_i, t_m + \frac{\tau}{2})
$$

$$
+ O(\tau^2) + O(h^2)
$$
