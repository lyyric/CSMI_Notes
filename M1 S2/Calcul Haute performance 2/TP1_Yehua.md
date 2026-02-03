# TP1 — HPC, introduction à OpenMP

Machine : MacBook Air
Compilation : `g++-15`
Options : `-O3 -march=native -fopenmp -std=c++11`

---

## Compilation (Makefile)

J’ai utilisé un Makefile pour compiler facilement plusieurs versions.

Extrait :

```makefile
CXX      = g++-15
CXXFLAGS = -O3 -march=native -fopenmp -std=c++11

SRC  = grid2d.cpp leapfrog2d.cpp
HDR  = grid2d.hpp

all: leapfrog_std leapfrog_std_v2 tile16 tile32 tile64

leapfrog_std: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

leapfrog_std_v2: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -DOMP_V2 -o $@ $(SRC)

tile16: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -DLOOP_TILING -DTILE_SIZE=16 -o $@ $(SRC)

tile32: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -DLOOP_TILING -DTILE_SIZE=32 -o $@ $(SRC)

tile64: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -DLOOP_TILING -DTILE_SIZE=64 -o $@ $(SRC)
```

Compilation :

```bash
make
```

---

## 1) Performances (différentes tailles de maillage)

Commandes :

```bash
time ./leapfrog_std 1024
time ./leapfrog_std 2048
time ./leapfrog_std 4096
time ./leapfrog_std 8192
```

### Résultats

| Grid size | Mémoire (3 grilles) | Execution time (s) | time real (s) |  CPU |
| --------- | ------------------: | -----------------: | ------------: | ---: |
| 1024×1024 |            0.023 GB |              0.044 |         0.436 |  91% |
| 2048×2048 |            0.094 GB |              0.200 |         0.371 | 382% |
| 4096×4096 |            0.375 GB |              0.809 |         1.678 | 412% |
| 8192×8192 |              1.5 GB |              3.609 |         7.574 | 424% |

Logs (extraits) :

```text
Grid size: 1024x1024
Estimated memory for 3 grids: 0.0234375 GB
Execution time: 0.04431 s
./leapfrog_std 1024  0.38s user 0.02s system 91% cpu 0.436 total

Grid size: 2048x2048
Estimated memory for 3 grids: 0.09375 GB
Execution time: 0.200115 s
./leapfrog_std 2048  1.39s user 0.03s system 382% cpu 0.371 total

Grid size: 4096x4096
Estimated memory for 3 grids: 0.375 GB
Execution time: 0.80936 s
./leapfrog_std 4096  6.69s user 0.24s system 412% cpu 1.678 total

Grid size: 8192x8192
Estimated memory for 3 grids: 1.5 GB
Execution time: 3.60924 s
./leapfrog_std 8192  31.13s user 1.00s system 424% cpu 7.574 total
```

Petit commentaire : quand on double la taille (1024 -> 2048 -> 4096 -> 8192), le temps augmente beaucoup, c’est normal car il y a $nx \times ny$ points (stencil 2D).

---

## 1.4) OpenMP : 2 versions

### Code

Version standard :

```cpp
#pragma omp parallel for
for (int i = 0; i < nx; ++i)
  for (int j = 0; j < ny; ++j)
    unp1(i, j) = ...;
```

Version v2 :

```cpp
#pragma omp parallel
{
  #pragma omp for
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      unp1(i, j) = ...;
}
```

Commandes :

```bash
export OMP_NUM_THREADS=4
time ./leapfrog_std 4096
time ./leapfrog_std_v2 4096
```

Résultats :

| Version | Execution time (s) | time real (s) |  CPU |
| ------- | -----------------: | ------------: | ---: |
| std     |              1.099 |         2.279 | 208% |
| v2      |              1.049 |         2.252 | 215% |

Logs :

```text
Grid size: 4096x4096
Estimated memory for 3 grids: 0.375 GB
Execution time: 1.09931 s
./leapfrog_std 4096  4.64s user 0.11s system 208% cpu 2.279 total

Grid size: 4096x4096
Estimated memory for 3 grids: 0.375 GB
Execution time: 1.04852 s
./leapfrog_std_v2 4096  4.77s user 0.09s system 215% cpu 2.252 total
```

Ici les 2 versions sont assez proches. Je n’ai pas vu une grosse différence.

---

## 2) Tiling

### 2.1) Tiling sur les boucles

J’ai fait un loop tiling avec une taille de tuile `TILE_SIZE` (16/32/64).

Extrait :

```cpp
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

#pragma omp parallel for collapse(2) schedule(static)
for (int ii = 0; ii < nx; ii += TILE_SIZE) {
  for (int jj = 0; jj < ny; jj += TILE_SIZE) {

    int i_end = std::min(ii + TILE_SIZE, nx);
    int j_end = std::min(jj + TILE_SIZE, ny);

    for (int i = ii; i < i_end; ++i) {
      for (int j = jj; j < j_end; ++j) {
        unp1(i, j) = ...; // stencil leapfrog
      }
    }
  }
}
```

---

### 2.2) Tests (8192×8192)

Commandes :

```bash
export OMP_NUM_THREADS=4
time ./tile16 8192
time ./tile32 8192
time ./tile64 8192
```

Résultats :

| Version | TILE_SIZE | Execution time (s) | time real (s) |  CPU |
| ------- | --------: | -----------------: | ------------: | ---: |
| tile16  |        16 |              2.588 |         6.920 | 196% |
| tile32  |        32 |              2.522 |         7.254 | 198% |
| tile64  |        64 |              2.652 |         7.680 | 195% |

Logs :

```text
Grid size: 8192x8192
Estimated memory for 3 grids: 1.5 GB
Execution time: 2.58751 s
./tile16 8192  13.11s user 0.46s system 196% cpu 6.920 total

Grid size: 8192x8192
Estimated memory for 3 grids: 1.5 GB
Execution time: 2.52186 s
./tile32 8192  14.21s user 0.18s system 198% cpu 7.254 total

Grid size: 8192x8192
Estimated memory for 3 grids: 1.5 GB
Execution time: 2.65185 s
./tile64 8192  14.76s user 0.24s system 195% cpu 7.680 total
```

On voit que `TILE_SIZE=32` est le meilleur ici.

---

## Conclusion

* Plus le maillage est grand, plus c’est lent (beaucoup de points).
* OpenMP marche mais l’accélération n’est pas énorme sur les grandes tailles.
* Avec le tiling, `TILE_SIZE=32` est un peu plus rapide que 16 et 64.