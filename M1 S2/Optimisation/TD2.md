# **TD 2 : Existence et unicité**  
## **Exercice 1 (Inégalité de Young)**  
1. Soient $x$ et $y$, deux réels strictement positifs. Soit $p$, un nombre entier naturel. On appelle $q$ son conjugué, c’est-à-dire le nombre vérifiant :  
   $$
   \frac{1}{p} + \frac{1}{q} = 1.
   $$
   Démontrer que pour tous $(x,y) \in \mathbb{R}_+^2$ :
   $$
   xy \leq \frac{x^p}{p} + \frac{y^q}{q}.
   $$
   **Indication :** on utilisera pour cela la convexité d’une fonction judicieusement choisie.  

2. Soient $f$ et $g$, deux fonctions continues et coercives de $\mathbb{R}$ dans $\mathbb{R}$. Démontrer que la fonction  
   $$
   \varphi : \mathbb{R}^2 \to \mathbb{R}, \quad (x,y) \mapsto f(x) + g(y)
   $$
   est coercive.

3. Étudier l’existence de solutions pour le problème :
   $$
   \inf_{(x,y) \in \mathbb{R}^2} x^4 + |y| \sqrt{|y|} - xy.
   $$

---

## **Exercice 2**  
Soit $\alpha$, un paramètre réel. On introduit la fonction :
$$
f : \mathbb{R}^2 \to \mathbb{R}, \quad (x,y) \mapsto x^4 + y^4 + 2 \alpha xy.
$$
1. Pour quelles valeurs de $\alpha$ la fonction $f$ est-elle convexe ? concave ?  
2. Dans le cas où $\alpha = 1$, résoudre le problème :
   $$
   \inf_{(x,y) \in \mathbb{R}^2} f(x,y).
   $$
   **Indication :** on demande d’étudier l’existence de solutions puis de les déterminer à l’aide de conditions d’optimalité. On calculera la valeur du minimum.

---

## **Exercice 3 (Ensemble convexe)**  
1. Montrer que pour tout $(a,b) \in (\mathbb{R}_+^*)^2$ et tout $\theta \in [0,1]$, on a :
   $$
   a^\theta b^{1-\theta} \leq \theta a + (1-\theta)b.
   $$
   **On rappellera la définition de $a^\theta$ en préliminaire.**  

2. L’ensemble  
   $$
   H_n = \{(x_1, \dots, x_n) \in \mathbb{R}_+^n \mid \prod_{i=1}^{n} x_i \geq 1\}
   $$
   est-il convexe ?  

3. Soit $x = (x_i)_{1 \leq i \leq n} \in \mathbb{R}^n$ et $(\alpha_i)_{1 \leq i \leq n}$ une famille de nombres réels strictement positifs. Le problème d’optimisation suivant :
   $$
   \inf_{y = (y_i)_{1 \leq i \leq n} \in H_n} \sum_{i=1}^{n} \alpha_i (x_i - y_i)^2
   $$
   possède-t-il une solution ?

---

## **Exercice 4 (Semi-continuité inférieure)**  
Soit $E$, un espace vectoriel normé.

1. Soit $f : E \to \mathbb{R}$ semi-continue inférieurement (s.c.i.). Montrer que :
   $$
   \forall \alpha \in \mathbb{R}, \quad \{ f \leq \alpha \} \text{ est fermé dans } E.
   $$

2. Soit $f : E \to \mathbb{R}$ telle que $\forall \alpha \in \mathbb{R}$, $f \leq \alpha$ est fermé dans $E$. Soit $x \in E$. Montrer que pour tout $\varepsilon > 0$, il existe un voisinage $V_x$ de $x$ tel que  
$$
f(x) \geq \inf_{y \in V_x} f(y) - \varepsilon.
$$
   En déduire que $f$ est semi-continue inférieurement (s.c.i.).  

3. Montrer que toute fonction continue est semi-continue inférieurement et donner un exemple de fonction semi-continue inférieurement qui n’est pas continue.  

4. Soit $I$, un sous-ensemble quelconque de $\mathbb{R}$, et $(f_j)_{j \in I}$ une famille de fonctions linéaires de $\mathbb{R}^n$ dans $\mathbb{R}$. Définir  
$$
f_0(x) = \sup \{ f_j(x), j \in I \}, \quad x \in \mathbb{R}^n.
$$
   Montrer que le problème  
$$
\inf_{x \in K} f_0(x)
$$
   possède (au moins) une solution, où $K$ est un compact de $\mathbb{R}^n$.

---

## **Exercice 5**  
On définit la famille des points $\{ u_i \}_{i=0,\dots,N+1}$ par  
$$
u_i = ih, \quad \text{où } h = \frac{1}{N+1}.
$$  
On se donne un nuage de points de $\mathbb{R}^2$ :  
$$
(u_i, x_i)_{i=0,\dots,N+1}, \quad N \in \mathbb{N}^*.
$$  
On suppose par ailleurs que  
$$
x_0 = 0, \quad x_{N+1} = 1.
$$  
Posons $x = (x_1, \dots, x_N)$. On appelle $f(x)$, la longueur de la courbe affine par morceaux passant par les points $(u_i, x_i)$ :

$$
f(x) = \sum_{i=0}^{N} \sqrt{ 1 + \left( \frac{x_{i+1} - x_i}{h} \right)^2 }.
$$

1. Montrer que pour tout $x \in \mathbb{R}^N$, $f(x)$ est bien définie.  
2. Étudier l’existence et l’unicité de solutions pour le problème  
   $$
   \inf_{x \in \mathbb{R}^N} f(x).
   $$

---

## **Exercice 6**  
Soit $J$, la fonctionnelle définie sur $L^4([0,1])$ par :
$$
J(u) = \int_0^1 u(x)^4 dx - \int_0^1 x u(x) dx.
$$

1. $J$ est-elle bien définie ? Étudier la convexité de $J$.  
2. Soient $u$ et $h$, deux éléments de $L^4([0,1])$. Montrer que $J$ est différentiable au point $u$ dans la direction $h$ et calculer $DJ(u) \cdot h$.  
3. En déduire que le problème  
   $$
   \inf_{u \in L^4([0,1])} J(u)
   $$
   possède une solution et résoudre ce problème.

---

## **Exercice 7 (Optimisation du remplissage d’un réservoir d’eau)**  
On considère un réservoir d’eau dont la hauteur d’eau au temps $s$ est notée $y(s)$ et qui subit une perte d’eau linéaire en temps et auquel on peut ajouter de l’eau au cours du temps.  
On modélise le système par l’équation différentielle commandée :
$$
\begin{cases}
y'(s) = u(s) - \gamma y(s), & s \in [0,T], \\
y(0) = 0.
\end{cases}
$$
où $\gamma > 0$ est une constante donnée, $T > 0$ est l’horizon de temps (fixé), la hauteur d’eau au temps de départ est nulle et la fonction $u$ est la commande modélisant l’ajout d’eau.  
On suppose que le coût d’ajout d’eau est donné par  
$$
\int_0^T u(s)^2 ds.
$$
On souhaite qu’au temps final, la hauteur d’eau du réservoir soit la plus proche possible de la hauteur $h > 0$。  
Pour déterminer une stratégie optimale de remplissage du réservoir, on considère le problème suivant :

$$
\min_{u} J(u) = \frac{1}{2} \int_0^T u(s)^2 ds + \frac{M}{2} (y(T) - h)^2.
$$

où $M > 0$ est une constante fixée。

1. Étudier l’existence de solutions pour le problème ci-dessus, en utilisant la méthode directe du calcul des variations.  
2. Démontrer que la fonctionnelle $J$ est strictement convexe.  
3. En déduire que l’unique solution $u^*$ de ce problème est constante et la déterminer.  
   Donner l’expression de la fonction "hauteur d’eau" $y^*$ associée à $u^*$.  
   Que se passe-t-il lorsque $M \to +\infty$ ?

---

## **Exercice 8 (Espace de Sobolev et optimisation variationnelle)**  
Dans l'exercice qui suit, si $\Omega$ désigne un ouvert de $\mathbb{R}^d$, on appelle $H^1(\Omega)$ l’espace de Hilbert (appelé espace de Sobolev) des fonctions $u \in L^2(\Omega)$ telles qu’il existe $f_1, \dots, f_d \in L^2(\Omega)$ vérifiant :
$$
\int_{\Omega} u \frac{\partial \varphi}{\partial x_i} = - \int_{\Omega} f_i \varphi, \quad \forall \varphi \in C_c^{\infty} (\Omega), \quad \forall i = 1, \dots, d.
$$

L’espace $H^1(\Omega)$ est équipé de la norme définie par :
$$
\| u \|_{H^1(\Omega)} = \left( \| u \|_{L^2(\Omega)}^2 + \sum_{i=1}^{d} \left\| \frac{\partial u}{\partial x_i} \right\|_{L^2(\Omega)}^2 \right)^{1/2}.
$$

Par ailleurs, on admettra que :
- Si $\Omega$ est un ouvert dont le bord est **Lipschitz**, l’application trace $\gamma : D(\Omega) \to C^0(\partial \Omega)$ se prolonge par continuité en une application linéaire (continue) de $H^1(\Omega)$ dans $L^2(\partial \Omega)$.
- Si $d = 1$, alors $H^1(\Omega)$ **s’injecte continûment** dans $C^0(\overline{\Omega})$, c’est-à-dire qu’il existe une constante $C > 0$ telle que :
  $$
  \forall u \in H^1(\Omega), \quad \max_{x \in \Omega} |u(x)| \leq C \| u \|_{H^1(\Omega)}.
  $$

On considère le cas $\Omega = (0,1)$ et $f \in L^2(\Omega)$. Soit $J$ la fonctionnelle définie sur $H^1(\Omega)$ par :
$$
J(u) = \frac{1}{2} \int_0^1 ( u'(x)^2 + u(x)^2 ) dx + \frac{\alpha}{2} ( u(0)^2 + u(1)^2 ) - \int_0^1 f(x) u(x) dx.
$$

1. **Montrer que $J$ est différentiable dans $H^1(\Omega)$** et calculer son gradient directionnel $DJ(u) \cdot h$, où $h \in H^1(\Omega)$.
2. **Montrer que $J$ est convexe**. En déduire que le problème :
   $$
   \inf_{u \in H^1(\Omega)} J(u)
   $$
   admet une **unique solution $u^*$**.
3. On admet que la **dérivée** de $u^*$ est également dans $H^1(\Omega)$.  
   **Montrer que $u^*$ est solution d’une équation différentielle** et préciser cette équation.

---
