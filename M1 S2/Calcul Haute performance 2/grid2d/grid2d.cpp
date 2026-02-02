#include "grid2d.hpp"
#include <fstream>
#include <iostream>
using namespace std;

Grid2D::Grid2D(int nx, int ny) : _nx(nx), _ny(ny), _data(new double[nx * ny])
{
    for (int i = 0; i < nx * ny; i++)
        _data[i] = 0.0;
}

// destructeur

Grid2D::~Grid2D()
{
    // delete[] _data;
}

// 2d plot by creating a gnuplot file
// and calling gnuplot
void Grid2D::plot2d() const
{
    ofstream file("plot.dat");
    for (int i = 0; i < _nx; i++)
    {
        for (int j = 0; j < _ny; j++)
        {
            file << i << " " << j << " " << _data[i + j * _nx] << endl;
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
