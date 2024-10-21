
**Année universitaire 2024-2025** 
**Algorithmique et graphes**  
**Exercices**

## Exercice 1.

Quel est le plus grand entier $k$ tel que, s’il est possible de multiplier des matrices $3 × 3$ à l’aide de $k$ multiplications, alors il est possible de multiplier des matrices $n × n$ à l’aide de $o(n^{\log_2 7})$ multiplications ? Quelle serait la complexité de cet algorithme ?

## Exercice 2.

Il est possible de multiplier des matrices $68 × 68$ (resp. $70 × 70$, $72 × 72$) à l’aide de $132464$ multiplications (resp. $143640$ multiplications, $155424$ multiplications). On rappelle qu’il est possible de multiplier des matrices $2 × 2$ à l’aide de $7$ multiplications. Nous pouvons donc en déduire $4$ algorithmes pour multiplier deux matrices de taille $n × n$. Lequel d’entre eux est le plus rapide ?

## Exercice 3.

Soient $z_0, z_1,...,z_{n-1}$ des nombres entiers, pas nécessairement distincts. Expliquer comment trouver les coefficients du polynôme $(X - z_0) ×...× (X - z_{n-1})$ en temps $O(n log² n)$.

## Exercice 4. [Examen, 2024]

Soient $U$ et $V$ deux fonctions croissantes définies sur $\mathbb{R}_{>0}$ et telles que

$$
U(x) = V(x) = 1, \quad \text{pour} \ 0 < x ≤ 4,
$$

et

$$
U(n) ≤ 4V\left(\frac{n}{4}\right) + \frac{n}{\log n}, \quad V(n) ≤ 4V\left(\frac{n}{4}\right) + \frac{n}{(\log n)^2}, \quad \text{pour tout entier} \ n ≥ 3.
$$

Majorer (le plus précisément possible) $U(n)$ pour tout entier $n$. Majorer (le plus précisément possible) V(n) pour tout entier n. On pourra commencer par traiter le cas où $n$ est de la forme $4^h$.

## Exercice 5. [Examen, 2024]

Soit k ≥ 1 un entier. Un tableau T[1...n] à n éléments est dit k-trié si, pour tout i = 1, 2,...,n - k, on a

$$
\frac{\sum_{j=i}^{i+k-1} T[j]}{k} \leq \frac{\sum_{j=i+1}^{i+k} T[j]}{k}
$$

1. Que cela signifie-t-il pour un tableau d’être 1-trié ?
2. Donner une permutation des nombres 1, 2,...,6 qui soit 3-triée mais pas 2-triée.
3. Montrer qu’un tableau T[1...n] à n éléments est k-trié si, et seulement si, T[i] ≤ T[i + k] pour tout i = 1, 2,...,n - k.
4. Décrire un algorithme qui k-trie un tableau à n éléments en un temps O(n log(n/k)).

## Exercice 6. [Examen, 2024]

On considère trois opérations sur une pile S :
- Empiler(S, x), dont le coût est O(1), [On empile l’élément x]
- Dépiler(S), dont le coût est O(1), [On dépile le dernier élément empilé]
- Multipop(S, k), définie par tant que la pile S n’est pas vide et k > 0 faire Dépiler(S) ; k := k - 1, et dont le coût est O(min{k, s}), où s est le nombre d’éléments de la pile S avant l’appel de Multipop(S, k).

Ces opérations sont respectivement notées E, D et M. On veut déterminer le coût maximal C(n) d’une suite de n instructions E, D ou M en partant d’une pile vide. Comme on ne retire jamais rien d’une pile vide, la première instruction est donc nécessairement E. Soient, respectivement, e, d et m le nombre de fois où E, D et M sont exécutées. Pour i = 1,...,m, soit k_i le nombre d’éléments dépilés lors de la i-ème exécution de Multipop.

1. Démontrer l’estimation 

$$
C(n) = O \left( e + d + \sum_{i=1}^{m} k_i \right)
$$

et en déduire que C(n) = O(n²).

Une analyse plus fine permet d’améliorer cette majoration.

2. Démontrer l’inégalité

$$
e ≥ d + \sum_{i=1}^{m} k_i
$$

et en déduire que C(n) = O(n).

---
