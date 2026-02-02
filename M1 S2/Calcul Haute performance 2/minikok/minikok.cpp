#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <cassert>
#include <Kokkos_Core.hpp>

#ifdef OPMP
#define SPACE Kokkos::HostSpace
using ExecutionSpace = Kokkos::OpenMP;
#else
#define SPACE Kokkos::CudaSpace
using ExecutionSpace = Kokkos::Cuda;
#endif

using namespace std;

// Fonction pour la condition initiale
KOKKOS_INLINE_FUNCTION
float peak(float x)
{
  float r2 = x * x;
  float eps = 0.2;
  float eps2 = eps * eps;
  return (r2 / eps2 < 1.0) ? std::pow(1.0 - r2 / eps2, 4) : 0.0;
}

// Solution exacte
KOKKOS_INLINE_FUNCTION
float exact_sol(Kokkos::Array<float, 2> xy, float t)
{
  float x = xy[0], y = xy[1];
  float r_val = std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
  return peak(r_val);
}

// Fonction source
KOKKOS_INLINE_FUNCTION
float source(Kokkos::Array<float, 2> xy, float t)
{
  return 0.0;
}

void initialize_grid(Kokkos::View<float**, SPACE> unm1,
                     Kokkos::View<float**, SPACE> un,
                     Kokkos::View<float**, SPACE> unp1,
                     int nx, int ny, float dx, float dy)
{
  Kokkos::parallel_for("initialize_grid",
    Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {nx, ny}),
    KOKKOS_LAMBDA(int i, int j)
    {
      float x = i * dx;
      float y = j * dy;
      Kokkos::Array<float, 2> xy = {x, y};
      unm1(i, j) = exact_sol(xy, 0.0);
      un(i, j) = unm1(i, j);
      unp1(i, j) = unm1(i, j);
    });
}

void leapfrog_step(Kokkos::View<float**, SPACE> &unm1,
                   Kokkos::View<float**, SPACE> &un,
                   Kokkos::View<float**, SPACE> &unp1,
                   int nx, int ny, float dt, float dx, float dy)
{
  float c = 1.0;
  float bx = c * dt / dx;
  float by = c * dt / dy;
  float r = 1.0;
  float gam = (1 - r) / (1 + r);

  const int dir[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  Kokkos::parallel_for("leapfrog_step",
    Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {nx, ny}),
    KOKKOS_LAMBDA(int i, int j) {
      float a = 1.0;
      if (i == 0 || i == nx - 1) a = 1.0 / (1 + bx * gam);
      if (j == 0 || j == ny - 1) a = 1.0 / (1 + by * gam);

      Kokkos::Array<float, 4> u = {0.0};
      for (int d = 0; d < 4; ++d) {
        int iR = i + dir[d][0];
        int jR = j + dir[d][1];

        if (iR < 0) iR = 1;
        else if (iR >= nx) iR = nx - 2;
        if (jR < 0) jR = 1;
        else if (jR >= ny) jR = ny - 2;

        u[d] = un(iR, jR);
      }

      float s = 0.0; // À remplacer par un appel Kokkos-compatible de source()
      unp1(i, j) = (1 - 2*a) * unm1(i, j)
                  + 2*a * (1 - bx*bx - by*by) * un(i, j)
                  + a * bx*bx * (u[0] + u[1])
                  + a * by*by * (u[2] + u[3])
                  - dt * dt * a * s;
    });

  auto temp = unm1;
  unm1 = un;
  un = unp1;
  unp1 = temp;
}

void solve(Kokkos::View<float**, SPACE> unm1,
           Kokkos::View<float**, SPACE> un,
           Kokkos::View<float**, SPACE> unp1,
           int nx, int ny, float dt, float tmax)
{
  float t = 0.0;
  while (t < tmax)
  {
    leapfrog_step(unm1, un, unp1, nx, ny, dt, 1.0 / (nx - 1), 1.0 / (ny - 1));
    t += dt;
  }
}

void plot2d(const Kokkos::View<float**, SPACE> un, int nx, int ny)
{
  auto h_data = Kokkos::create_mirror_view(un);
  Kokkos::deep_copy(h_data, un);

  std::ofstream file("plot.dat");
  for (int i = 0; i < nx; i++)
  {
    for (int j = 0; j < ny; j++)
    {
      file << i << " " << j << " " << h_data(i, j) << std::endl;
    }
    file << std::endl;
  }
  file.close();

  std::ofstream gp("plot.gp");
  gp << "set terminal png" << std::endl;
  gp << "set output 'plot.png'" << std::endl;
  gp << "set pm3d map" << std::endl;
  gp << "set size ratio -1" << std::endl;
  gp << "splot 'plot.dat' with pm3d" << std::endl;
  gp.close();

  auto a = system("gnuplot plot.gp");
}

int main()
{
  const int blocksize = 32;
  const int nx = 4 * 30 * blocksize;
  const int ny = 4 * 30 * blocksize;
  const float cfl = 0.4;
  float L = 1.0;
  float dx = L / (nx - 1);
  float dy = dx;
  float c = 1.0;
  float dt = cfl * std::sqrt(dx * dx + dy * dy) / c;

  Kokkos::initialize();
  {
    Kokkos::View<float**, SPACE> unm1("unm1", nx, ny);
    Kokkos::View<float**, SPACE> un("un", nx, ny);
    Kokkos::View<float**, SPACE> unp1("unp1", nx, ny);

    initialize_grid(unm1, un, unp1, nx, ny, dx, dy);
    solve(unm1, un, unp1, nx, ny, dt, 1.);
    //plot2d(un, nx, ny);
  }
  Kokkos::finalize();

  return 0;
}