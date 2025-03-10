## **TD 1 : Introduction**  

### **Exercice 1 (Calcul explicite de différentielles)**  
Calculer la différentielle à l'origine de l'application  
$$
f : \mathbb{R}^3 \to \mathbb{R}, \quad (x, y, z) \mapsto f(x, y, z) = xyz \sin(xy) + 2x + 5.
$$

---

### **Exercice 2 (Différentiabilité de la norme, du produit scalaire)**  

1. Soit $E$, un espace vectoriel réel muni d’un produit scalaire $\langle \cdot, \cdot \rangle$.  
   - Étudier la continuité puis la différentiabilité et calculer la différentielle de l’application **produit scalaire**  
   $$
   \Phi : E^2 \to \mathbb{R}, \quad (x, y) \mapsto \Phi(x, y) = \langle x, y \rangle
   $$
   pour tout $x, y \in E$.  
   - On munira $E^2$ de la norme  
   $$
   \|(x, y)\|_{E^2} = \max\{\|x\|, \|y\|\}
   $$
   où $\|\cdot\|$ désigne la norme induite par le produit scalaire.  

2. Soit $A \in M_{n,m}(\mathbb{R})$ avec $n, m \in \mathbb{N}^*$. L’application  
   $$
   G : \mathbb{R}^m \to \mathbb{R}, \quad G(X) = \|AX\|
   $$
   est-elle différentiable sur $\mathbb{R}^m$ ?  

---

### **Exercice 3 (Différentiabilité d’une fonction définie à l’aide d’un max)**  
On définit la fonction $f$ sur $\mathbb{R}^2$ par  
$$
f(x, y) = \max(x, y).
$$  
Étudier la continuité et l’existence de dérivées partielles de $f$ sur $\mathbb{R}^2$.  

---

### **Exercice 4 (Régularité d’une fonction de plusieurs variables)**  
Soit $f$, la fonction définie de $\mathbb{R}^2$ dans $\mathbb{R}$ par :  
$$
f(x, y) = x^2 y \sin \left( \frac{y}{x} \right).
$$  

1. Montrer que l’on peut définir un prolongement par continuité de $f$. On appellera encore $f$ ce prolongement.  
2. $f$ admet-elle des dérivées partielles sur $\mathbb{R}^2$ ?  
3. $f$ est-elle différentiable ? De classe $C^1$ sur $\mathbb{R}^2$ ?  
4. Calculer $\frac{\partial^2 f}{\partial x \partial y} (0,0)$ et $\frac{\partial^2 f}{\partial y \partial x} (0,0)$.  

**Remarque :** Attention de donner un sens convenable aux deux expressions ci-dessus avant de les calculer. Rien ne certifie que $f$ soit de classe $C^2$ en $(0,0)$.  

---

### **Exercice 5 (Différentiabilité en dimension infinie)**  
L’application  
$$
J : u \in L^2(0,1) \mapsto \int_0^1 \sin(u(t)) dt
$$
est-elle bien définie et différentiable en tout point de $L^2(0,1)$ ?  

---

### **Exercice 6 (Exemples de problèmes d’optimisation)**  
Étudier l’existence de solution aux problèmes d’optimisation suivants et les résoudre lorsque c’est possible.  

1. **Problème de dimension finie.**  

   $$
   \inf_{(x,y,z) \in \mathbb{R}^3 \setminus \{(0,0,0)\}} \frac{2x - 3y + z}{\sqrt{x^2 + y^2 + z^2}}
   $$
   et  
   $$
   \sup_{x^2 + y^2 + y = 2} x + y.
   $$  

   **Indication :** Pour le premier problème, penser à l'inégalité de Cauchy-Schwarz.  

2. **Problème en dimension infinie.**  
   La notation $AC([-1,1])$ désigne l’ensemble des fonctions absolument continues sur $[-1,1]$.  

$$
\inf_{f \in C^0(0,1)} \int_0^1 (f(t) - 1)^2 dt + \int_0^1 f(t)^2 dt + \int_0^1 f(t) dt,
$$

$$
\inf_{f \in AC([-1,1]) \atop f(-1) = f(1) = 0} \int_{-1}^{1} (1 - |f'(t)|)^2 dt,
$$

$$
\inf_{f \in C^0(-1,1)} \int_{-1}^{1} |f(t) - \varphi(t)|^2 dt \quad \text{avec } \varphi(t) = \mathbb{1}_{[0,1]}(t).
$$

---

### **Exercice 7 (Un problème d’optimisation linéaire)**  
Un pâtissier vend des cornets de glace :  
- certains à **une boule**,  
- d’autres à **deux boules**.  

On cherche à déterminer le **bénéfice maximal** qu’il peut espérer faire en un jour, compte tenu des contraintes suivantes :  

- **Bénéfice** :  
  - 0.15 euros par cornet à une boule,  
  - 0.22 euros par cornet à deux boules.  
- **Contraintes** :  
  - Le marchand dispose de **60 cornets** à une boule et **60 cornets** à deux boules.  
  - Il ne peut vendre **au plus 100 cornets** par jour.  
  - Il dispose de **suffisamment de glace pour faire 150 boules** par jour.  

#### **Questions**  
1. Soit $x$ le nombre de cornets à une boule vendus et $y$ le nombre de cornets à deux boules vendus.  
   - **Représenter graphiquement** l’ensemble des contraintes sur le plan $(x, y)$.  
2. Montrer que le problème d’optimisation a **une solution** et que cette solution se situe sur la **frontière** de l’ensemble des contraintes.  
   - **Résoudre** le problème d’optimisation.

---

### **Exercice 8 (Coercivité)**  
Les fonctions $J$ suivantes définies sur $\mathbb{R}^2$ sont-elles **coercives** ?  
Si oui, déterminer une fonction $\varphi : \mathbb{R} \to \mathbb{R}$ telle que  
$$
J(x_1, x_2) \geq \varphi(\|(x_1, x_2)\|)
$$
pour tout $(x_1, x_2) \in \mathbb{R}^2$, avec  
$$
\lim_{t \to +\infty} \varphi(t) = +\infty.
$$

#### **Les fonctions étudiées**  
1. $J(x_1, x_2) = x_1^2 x_2^2 - x_1^3$.  
2. $J(x_1, x_2) = x_1^2 + 2x_2^2 - a x_1 - b x_2 - c$, avec $a, b, c \in \mathbb{R}$.  
3. $J(x_1, x_2) = x_1^2 - x_2^2$.  
4. $J(x_1, x_2) = 2x_1^2 + x_2^3 + 2x_2^2$.  
5. $J(x_1, x_2) = x_1^4 + (x_2 - 1)^2$.  

---

### **Exercice 9 (Propriété de type coercivité)**  
1. Soit $\Omega \subset \mathbb{R}^n$, un **ouvert borné**, et $f \in C^0(\Omega, \mathbb{R})$.  
   On suppose que  
   $$
   \forall y \in \partial \Omega, \quad \lim_{x \to y, x \in \Omega} f(x) = +\infty.
   $$
   Montrer que le problème  
   $$
   \inf_{x \in \Omega} f(x)
   $$
   possède une **solution**.  

2. Résoudre le problème d’optimisation :  
   $$
   \inf_{(x,y) \in (\mathbb{R}^+)^2} \frac{1}{x} + \frac{1}{y} + xy.
   $$
---
