// simple class for managing 2d grid data
// only double precision data

#ifndef GRID2D_HPP
#define GRID2D_HPP
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

class Grid2D {

private:
  size_t _nx;
  size_t _ny;
  std::shared_ptr<double[]> _data;

#ifdef TILING
  static const size_t TILE_SIZE = 32;
  size_t _nbx; // number of blocks in x direction
#endif

public:
  Grid2D(size_t nx, size_t ny);
  ~Grid2D();

  // accesseurs constants et non constant
#ifdef TILING
  double &operator()(size_t i, size_t j) {
    size_t bx = i / TILE_SIZE;
    size_t by = j / TILE_SIZE;
    size_t lx = i % TILE_SIZE;
    size_t ly = j % TILE_SIZE;
    return _data[(bx + by * _nbx) * (TILE_SIZE * TILE_SIZE) + lx +
                 ly * TILE_SIZE];
  }
  const double &operator()(size_t i, size_t j) const {
    size_t bx = i / TILE_SIZE;
    size_t by = j / TILE_SIZE;
    size_t lx = i % TILE_SIZE;
    size_t ly = j % TILE_SIZE;
    return _data[(bx + by * _nbx) * (TILE_SIZE * TILE_SIZE) + lx +
                 ly * TILE_SIZE];
  }
#else
  double &operator()(size_t i, size_t j) { return _data[i + j * _nx]; }
  const double &operator()(size_t i, size_t j) const {
    return _data[i + j * _nx];
  }
#endif

  // 2d plot by creating a gnuplot file
  // and calling gnuplot
  void plot2d() const;
};

#endif