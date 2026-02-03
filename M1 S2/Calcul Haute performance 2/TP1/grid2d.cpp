#include "grid2d.hpp"
#include <fstream>
#include <iostream>
using namespace std;

Grid2D::Grid2D(size_t nx, size_t ny) : _nx(nx), _ny(ny) {
#ifdef TILING
  _nbx = (nx + TILE_SIZE - 1) / TILE_SIZE;
  size_t nby = (ny + TILE_SIZE - 1) / TILE_SIZE;
  size_t total_size = _nbx * nby * TILE_SIZE * TILE_SIZE;
#else
  size_t total_size = nx * ny;
#endif

  try {
    _data = std::shared_ptr<double[]>(new double[total_size]);
  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory allocation failed for " << nx << "x" << ny << " grid ("
              << (total_size * sizeof(double)) / (1024.0 * 1024.0 * 1024.0)
              << " GB)" << std::endl;
    throw;
  }
  for (size_t i = 0; i < total_size; i++)
    _data[i] = 0.0;
}

// destructeur

Grid2D::~Grid2D() {
  // delete[] _data;
}

// 2d plot by creating a gnuplot file
// and calling gnuplot
void Grid2D::plot2d() const {
  ofstream file("plot.dat");
  for (size_t i = 0; i < _nx; i++) {
    for (size_t j = 0; j < _ny; j++) {
      file << i << " " << j << " " << (*this)(i, j) << endl;
    }
    file << endl;
  }
  file.close();

  // gnuplot option (keep aspect ratio)
  ofstream gp("plot.gp");
  gp << "set pm3d map" << endl;
  gp << "set size ratio -1" << endl;
  gp << "splot 'plot.dat' with pm3d" << endl;
  gp.close();

  system("gnuplot -p plot.gp");
}
