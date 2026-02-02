#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include "grid2d.hpp"

// openmp header
//#include <openmp.h>


// Paramètres et constantes
const double L = 1.0;
const int nx = 128;
const int ny = 128;
const double dx = L / (nx - 1);
const double dy = dx;
const double H = ny * dy;
const double c = 1.0;
const double cfl = 0.4;
const double dt = cfl * std::sqrt(dx*dx + dy*dy) / c;
const double bx = c * dt / dx;
const double by = c * dt / dy;
const double r = 1.0;
const double gam = (1 - r) / (1 + r);

// Fonction pour la condition initiale
double peak(double x) {
    double r2 = x * x;
    double eps = 0.2;
    double eps2 = eps * eps;
    return (r2 / eps2 < 1.0) ? std::pow(1.0 - r2 / eps2, 4) : 0.0;
}

// Solution exacte
double exact_sol(const std::vector<double>& xy, double t) {
    double x = xy[0], y = xy[1];
    double r_val = std::sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5));
    return peak(r_val);
}


// Fonction source
double source(const std::vector<double>& xy, double t) {
    return 0.0;
}

// Initialisation des solutions
void init_sol(Grid2D& un, Grid2D& unm1) {
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double x = i * dx;
            double y = j * dy;
            un(i,j) = exact_sol({x, y}, 0.0);
            unm1(i,j) = un(i,j);
        }
    }
}

// Pas de temps Leapfrog
void leapfrog_step(Grid2D& un,
                   Grid2D& unm1,
                   Grid2D& unp1,
                   double t) {
    const std::vector<std::vector<int>> dir = {{-1,0}, {1,0}, {0,-1}, {0,1}};


    // open mp paraller loop
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double a = 1.0;
            if (i == 0 || i == nx-1) a = 1.0 / (1 + bx * gam);
            if (j == 0 || j == ny-1) a = 1.0 / (1 + by * gam);

            std::array<double, 4> u = {0.0, 0.0, 0.0, 0.0};
            for (int d = 0; d < 4; ++d) {
                int iR = i + dir[d][0];
                int jR = j + dir[d][1];

                if (iR == -1) iR = 1;
                else if (iR == nx) iR = nx - 2;
                if (jR == -1) jR = 1;
                else if (jR == ny) jR = ny - 2;

                u[d] = un(iR, jR);
            }

            double s = source({i*dx, j*dy}, t);
            unp1(i, j) = (1 - 2*a)*unm1(i, j)
                        + 2*a*(1 - bx*bx - by*by)*un(i, j)
                        + a*bx*bx*(u[0] + u[1])
                        + a*by*by*(u[2] + u[3])
                        - dt*dt*a*s;
        }
    }
}

int main() {
    // Initialisation des tableaux
    Grid2D unm1(nx, ny);
    Grid2D un(nx, ny);
    Grid2D unp1(nx, ny);
    init_sol(un, unm1);
    //un.plot2d();

    // Simulation
    double t = 0.0;
    const double tmax = 0.0;
    
    while (t < tmax) {
        leapfrog_step(un, unm1, unp1, t);
        std::swap(un, unm1);
        std::swap(unp1, un);
        t += dt;
    }

    un.plot2d();
    return 0;
}