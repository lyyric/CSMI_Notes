// simple class for managing 2d grid data
// only double precision data

#ifndef GRID2D_HPP
#define GRID2D_HPP
#include <vector>
#include <iostream>
#include <cassert>
#include <memory>
#include <Kokkos_Core.hpp>

class Grid2D
{

private:
    int _nx;
    int _ny;
    Kokkos::View<double **> _data;

public:
    Grid2D(int nx, int ny);
    ~Grid2D()= default;

    // accesseurs constants et non constant
    // Accessors

    int getNx() const { return _nx; }
    int getNy() const { return _ny; }

    KOKKOS_INLINE_FUNCTION
    double& operator() (int i, int j) { return _data(i, j); }

    KOKKOS_INLINE_FUNCTION
    const double& operator() (int i, int j) const { return _data(i, j); }
    // 2d plot by creating a gnuplot file
    // and calling gnuplot

    Kokkos::View<double**> getData() { return _data; }

    void plot2d() const;
};

#endif