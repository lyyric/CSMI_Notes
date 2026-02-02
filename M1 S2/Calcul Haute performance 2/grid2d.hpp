// simple class for managing 2d grid data
// only double precision data

#ifndef GRID2D_HPP
#define GRID2D_HPP
#include <vector>
#include <iostream>
#include <cassert>
#include <memory>

class Grid2D
{

private:
    int _nx;
    int _ny;
    std::shared_ptr<double[]> _data;

public:
    Grid2D(int nx, int ny);
    ~Grid2D();

    // accesseurs constants et non constant
    double &operator() (int i, int j) { return _data[i + j*_nx]; }
    const double& operator() (int i, int j) const { return _data[i + j*_nx]; }

    // 2d plot by creating a gnuplot file
    // and calling gnuplot
    void plot2d() const;

};

# endif