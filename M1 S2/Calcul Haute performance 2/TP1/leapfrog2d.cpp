#include "grid2d.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

// openmp header
#include <omp.h>

// Paramètres et constantes
const double L = 1.0;

int nx = 1024;
int ny = 1024;

double dx = 0.0;
double dy = 0.0;
double H  = 0.0;

const double c = 1.0;
const double cfl = 0.4;

double dt = 0.0;
double bx = 0.0;
double by = 0.0;

const double r = 1.0;
const double gam = (1.0 - r) / (1.0 + r);


// Fonction pour la condition initiale
double peak(double x) {
  double r2 = x * x;
  double eps = 0.2;
  double eps2 = eps * eps;
  return (r2 / eps2 < 1.0) ? std::pow(1.0 - r2 / eps2, 4) : 0.0;
}

// Solution exacte
double exact_sol(const std::vector<double> &xy, double t) {
  double x = xy[0], y = xy[1];
  double r_val = std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
  return peak(r_val);
}

// Fonction source
double source(double x, double y, double t) { return 0.0; }

// Initialisation des solutions
void init_sol(Grid2D &un, Grid2D &unm1) {
  // #pragma omp parallel for
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      double x = i * dx;
      double y = j * dy;
      un(i, j) = exact_sol({x, y}, 0.0);
      unm1(i, j) = un(i, j);
    }
  }
}

// Pas de temps Leapfrog
void leapfrog_step(Grid2D &un, Grid2D &unm1, Grid2D &unp1, double t) {
  static const int dir[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// ========== TP2 : loop tiling ==========
#ifdef LOOP_TILING

#pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < nx; ii += TILE_SIZE) {
    for (int jj = 0; jj < ny; jj += TILE_SIZE) {

      int i_end = std::min(ii + TILE_SIZE, nx);
      int j_end = std::min(jj + TILE_SIZE, ny);

      for (int i = ii; i < i_end; ++i) {
        for (int j = jj; j < j_end; ++j) {

          double a = 1.0;
          if (i == 0 || i == nx - 1)
            a = 1.0 / (1 + bx * gam);
          if (j == 0 || j == ny - 1)
            a = 1.0 / (1 + by * gam);

          double u[4] = {0.0, 0.0, 0.0, 0.0};
          for (int d = 0; d < 4; ++d) {
            int iR = i + dir[d][0];
            int jR = j + dir[d][1];

            if (iR == -1)
              iR = 1;
            else if (iR == nx)
              iR = nx - 2;

            if (jR == -1)
              jR = 1;
            else if (jR == ny)
              jR = ny - 2;

            u[d] = un(iR, jR);
          }

          double s = source(i * dx, j * dy, t);

          unp1(i, j) =
              (1 - 2 * a) * unm1(i, j) +
              2 * a * (1 - bx * bx - by * by) * un(i, j) +
              a * bx * bx * (u[0] + u[1]) +
              a * by * by * (u[2] + u[3]) -
              dt * dt * a * s;
        }
      }
    }
  }

// ========== TP1 : no tiling (std / v2 OpenMP placement) ==========
#else

#ifdef OMP_V2
#pragma omp parallel
{
#pragma omp for
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {

      double a = 1.0;
      if (i == 0 || i == nx - 1)
        a = 1.0 / (1 + bx * gam);
      if (j == 0 || j == ny - 1)
        a = 1.0 / (1 + by * gam);

      double u[4] = {0.0, 0.0, 0.0, 0.0};
      for (int d = 0; d < 4; ++d) {
        int iR = i + dir[d][0];
        int jR = j + dir[d][1];

        if (iR == -1)
          iR = 1;
        else if (iR == nx)
          iR = nx - 2;

        if (jR == -1)
          jR = 1;
        else if (jR == ny)
          jR = ny - 2;

        u[d] = un(iR, jR);
      }

      double s = source(i * dx, j * dy, t);

      unp1(i, j) =
          (1 - 2 * a) * unm1(i, j) +
          2 * a * (1 - bx * bx - by * by) * un(i, j) +
          a * bx * bx * (u[0] + u[1]) +
          a * by * by * (u[2] + u[3]) -
          dt * dt * a * s;
    }
  }
}
#else

#pragma omp parallel for
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {

      double a = 1.0;
      if (i == 0 || i == nx - 1)
        a = 1.0 / (1 + bx * gam);
      if (j == 0 || j == ny - 1)
        a = 1.0 / (1 + by * gam);

      double u[4] = {0.0, 0.0, 0.0, 0.0};
      for (int d = 0; d < 4; ++d) {
        int iR = i + dir[d][0];
        int jR = j + dir[d][1];

        if (iR == -1)
          iR = 1;
        else if (iR == nx)
          iR = nx - 2;

        if (jR == -1)
          jR = 1;
        else if (jR == ny)
          jR = ny - 2;

        u[d] = un(iR, jR);
      }

      double s = source(i * dx, j * dy, t);

      unp1(i, j) =
          (1 - 2 * a) * unm1(i, j) +
          2 * a * (1 - bx * bx - by * by) * un(i, j) +
          a * bx * bx * (u[0] + u[1]) +
          a * by * by * (u[2] + u[3]) -
          dt * dt * a * s;
    }
  }

#endif // OMP_V2
#endif // LOOP_TILING
}


void update_params()
{
    dx = L / (nx - 1);
    dy = dx;
    H  = ny * dy;

    dt = cfl * std::sqrt(dx * dx + dy * dy) / c;
    bx = c * dt / dx;
    by = c * dt / dy;
}


int main(int argc, char** argv) {
  // lecture argv
  if (argc == 2) {
    nx = ny = std::atoi(argv[1]);
  } else if (argc == 3) {
    nx = std::atoi(argv[1]);
    ny = std::atoi(argv[2]);
  } else if (argc != 1) {
    std::cerr << "Usage:\n"
              << "  " << argv[0] << "              (default 1024x1024)\n"
              << "  " << argv[0] << " nx           (square nx x nx)\n"
              << "  " << argv[0] << " nx ny        (rectangular nx x ny)\n";
    return 1;
  }

  update_params();


  // Calcul de la mémoire totale
  double total_mem_gb =
      3.0 * (double(nx) * ny * sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
  std::cout << "Grid size: " << nx << "x" << ny << std::endl;
  std::cout << "Estimated memory for 3 grids: " << total_mem_gb << " GB"
            << std::endl;

  // Initialisation des tableaux
  Grid2D unm1(nx, ny);
  Grid2D un(nx, ny);
  Grid2D unp1(nx, ny);
  init_sol(un, unm1);
  // un.plot2d(); 

  // Simulation
  double t = 0.0;
  const double tmax = 10.0 * dt;

  double t_start = omp_get_wtime();
  while (t < tmax) {
    leapfrog_step(un, unm1, unp1, t);
    std::swap(un, unm1);
    std::swap(unp1, un);
    t += dt;
  }
  double t_end = omp_get_wtime();
  std::cout << "Execution time: " << t_end - t_start << " s" << std::endl;

  // un.plot2d();
  return 0;
}