#include "grid2d.hpp"
#include <fstream>
#include <iostream>
using namespace std;

Grid2D::Grid2D(int nx, int ny) : _nx(nx), _ny(ny), _data("kfrog data", nx, ny)
{
    Kokkos::parallel_for("initialize_grid", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_nx, _ny}), KOKKOS_LAMBDA(int i, int j) { _data(i, j) = 0.; });
}

#include <fstream>
#include <Kokkos_Core.hpp>

void Grid2D::plot2d() const
{
    // Create a host mirror of the Kokkos::View
    auto h_data = Kokkos::create_mirror_view(_data);
    Kokkos::deep_copy(h_data, _data); // Copy data from device to host

    // Write data to file
    std::ofstream file("plot.dat");
    for (int i = 0; i < _nx; i++)
    {
        for (int j = 0; j < _ny; j++)
        {
            file << i << " " << j << " " << h_data(i, j) << std::endl;
        }
        file << std::endl;  // Separate rows for gnuplot
    }
    file.close();

    // Write gnuplot commands to a script
    std::ofstream gp("plot.gp");
    gp << "set terminal png" << std::endl;
    gp << "set output 'plot.png'" << std::endl;
    gp << "set pm3d map" << std::endl;
    gp << "set size ratio -1" << std::endl;
    gp << "splot 'plot.dat' with pm3d" << std::endl;
    gp.close();

    // Execute gnuplot to generate the plot
    system("gnuplot plot.gp");
}

