#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>

#define TILE_SIZE 32


const double L = 1.0;
const double c = 1.0;
const double cfl = 0.4;

int nx = 128;
int ny = 128;

double dx, dy, dt;
double bx, by;

double r0 = 1.0;
double gam = (1.0 - r0) / (1.0 + r0);

inline double peak(double x)
{
    double eps = 0.2;
    double r2 = x * x;
    double eps2 = eps * eps;
    if (r2 / eps2 < 1.0)
        return std::pow(1.0 - r2 / eps2, 4);
    else
        return 0.0;
}

inline double exact_sol(double x, double y)
{
    double r = std::sqrt((x - 0.5) * (x - 0.5)
                       + (y - 0.5) * (y - 0.5));
    return peak(r);
}

inline double source(double, double, double)
{
    return 0.0;
}

void init_sol(std::vector<double>& un,
              std::vector<double>& unm1)
{
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            double x = i * dx;
            double y = j * dy;
            double u0 = exact_sol(x, y);
            un[i * ny + j]   = u0;
            unm1[i * ny + j] = u0;
        }
}

void leapfrog_step(const std::vector<double>& un,
                   const std::vector<double>& unm1,
                   std::vector<double>& unp1,
                   double t)
{
    const int dir[4][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    };
    
    #pragma omp parallel for
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            double a = 1.0;

            if (i == 0 || i == nx - 1)
                a = 1.0 / (1.0 + bx * gam);
            if (j == 0 || j == ny - 1)
                a = 1.0 / (1.0 + by * gam);

            double u[4];

            for (int d = 0; d < 4; ++d)
            {
                int iR = i + dir[d][0];
                int jR = j + dir[d][1];

                if (iR == -1)     iR = 1;
                if (iR == nx)     iR = nx - 2;
                if (jR == -1)     jR = 1;
                if (jR == ny)     jR = ny - 2;

                u[d] = un[iR * ny + jR];
            }

            double s = source(i * dx, j * dy, t);

            int id = i * ny + j;

            unp1[id] =
                (1.0 - 2.0 * a) * unm1[id]
                + 2.0 * a * (1.0 - bx * bx - by * by) * un[id]
                + a * bx * bx * (u[0] + u[1])
                + a * by * by * (u[2] + u[3])
                - dt * dt * a * s;
        }
}

void leapfrog_step_tiled(const std::vector<double>& un,
                         const std::vector<double>& unm1,
                         std::vector<double>& unp1,
                         double t)
{
    static const int dir[4][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    };

    #pragma omp parallel for collapse(2) schedule(static)
    //#pragma omp parallel for collapse(2) schedule(static)
    //#pragma omp parallel for collapse(2) schedule(static, 1)
    //#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < nx; ii += TILE_SIZE)
    {
        for (int jj = 0; jj < ny; jj += TILE_SIZE)
        {
            const int i_end = std::min(ii + TILE_SIZE, nx);
            const int j_end = std::min(jj + TILE_SIZE, ny);

            for (int i = ii; i < i_end; ++i)
            {
                for (int j = jj; j < j_end; ++j)
                {
                    double a = 1.0;

                    if (i == 0 || i == nx - 1)
                        a = 1.0 / (1.0 + bx * gam);
                    if (j == 0 || j == ny - 1)
                        a = 1.0 / (1.0 + by * gam);

                    double u[4];

                    for (int d = 0; d < 4; ++d)
                    {
                        int iR = i + dir[d][0];
                        int jR = j + dir[d][1];

                        if (iR == -1) iR = 1;
                        if (iR == nx) iR = nx - 2;
                        if (jR == -1) jR = 1;
                        if (jR == ny) jR = ny - 2;

                        u[d] = un[iR * ny + jR];
                    }

                    const double s = source(i * dx, j * dy, t);
                    const int id = i * ny + j;

                    unp1[id] =
                        (1.0 - 2.0 * a) * unm1[id]
                        + 2.0 * a * (1.0 - bx * bx - by * by) * un[id]
                        + a * bx * bx * (u[0] + u[1])
                        + a * by * by * (u[2] + u[3])
                        - dt * dt * a * s;
                }
            }
        }
    }
}


int main(int argc, char** argv)
{
    if (argc == 2)
        nx = ny = std::atoi(argv[1]);

    dx = L / (nx - 1);
    dy = dx;
    dt = cfl * std::sqrt(dx * dx + dy * dy) / c;
    bx = c * dt / dx;
    by = c * dt / dy;

    std::vector<double> unm1(nx * ny, 0.0);
    std::vector<double> un(nx * ny, 0.0);
    std::vector<double> unp1(nx * ny, 0.0);

    init_sol(un, unm1);

    double t = 0.0;
    double tmax = 1.0;

    auto t_start = std::chrono::high_resolution_clock::now();

    while (t < tmax)
    {
        //leapfrog_step(un, unm1, unp1, t);
        leapfrog_step_tiled(un, unm1, unp1, t);
        unm1.swap(un);
        un.swap(unp1);
        t += dt;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;

    std::cout << "nx = ny = " << nx
              << " | time = " << elapsed.count() << " s\n";

    return 0;
}
