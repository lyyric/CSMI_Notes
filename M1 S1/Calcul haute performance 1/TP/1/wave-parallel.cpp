#include <grid_utilities.hpp>

#include <mpi.h>
#include <cmath>

//! \brief calcul du champ de pression initial
//! \param[out] pressure0 champ de pression au temps time
//! \param[out] pressure1 champ de pression au temps time+dt

void initial_conditions( FunctionSpace::Element & pressure0, FunctionSpace::Element & pressure1 );

int main(int argc, char *argv[])
    {
    //------------------------------------------------//
    // init the MPI environment
    //------------------------------------------------//
    MPI_Init (&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size (MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

    //------------------------------------------------//
    // simulation parameters
    //------------------------------------------------//
    double LX = 10;//1.0;
    double LY = 10;//1.0;
    int NX = 500;
    int NY = 500;
    double C=1.0;
    double TFINAL=20;//2;
    double CFL=0.9;
    //------------------------------------------------//
    // export parameters
    //------------------------------------------------//
    int NOUT=250;
    // specify the directory of export
    std::string exportDir = "results";
    std::string prefix = "wave2D_";

    //---------------------------------------------------------------//
    // define the 2d grid and the partitioning
    //---------------------------------------------------------------//
    GridStructured grid( LX, LY, NX, NY, world_size, world_rank );

    //---------------------------------------------------------------//
    // define the function space and approximation (solution of pde)
    //---------------------------------------------------------------//
    FunctionSpace space( grid );
    // pressure at time n
    FunctionSpace::Element pressure0 = space.element();
    // pressure at time n-1
    FunctionSpace::Element pressure1 = space.element();
    // pressure at time n+1
    FunctionSpace::Element pressure = space.element();

    //---------------------------------------------------------------//
    // some usefull variables
    //---------------------------------------------------------------//
    double time = 0.0;  // temps courant de la simulation
    double dt = CFL/( C*std::sqrt(1./std::pow(grid.delta(0),2)+1./std::pow(grid.delta(1),2)) );
    double dtout = TFINAL / NOUT; // pas de temps pour l'écriture des fichiers
    int mout = 0; // numéro de l'itération d'écriture des fichiers


    //---------------------------------------------------------------//
    // update the initial conditions
    //---------------------------------------------------------------//
    initial_conditions( pressure0, pressure1 );

    //---------------------------------------------------------------//
    // export initial condition on the disk
    //---------------------------------------------------------------//
    data_dump( exportDir, prefix, pressure1, time, mout );

    //---------------------------------------------------------------//
    // loop in time
    //---------------------------------------------------------------//

    while (time<TFINAL) {
    time += dt;
    if ( world_rank == 0 )
        std::cout << "time : " << time << std::endl;

    //------------------------------------------------//
    // compute solution at current time from the discretisation of wave equation
    //------------------------------------------------//

    double beta_x = C * dt / grid.delta(0);
    double beta_y = C * dt / grid.delta(1);
    int nx = space.nLocalDofByDirection(0);
    int ny = space.nLocalDofByDirection(1);

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            pressure(i, j) = -pressure0(i, j)
                                + 2 * (1 - beta_x * beta_x - beta_y * beta_y) * pressure1(i, j)
                                + beta_x * beta_x * (pressure1(i - 1, j) + pressure1(i + 1, j))
                                + beta_y * beta_y * (pressure1(i, j - 1) + pressure1(i, j + 1));
        }
    }



    //------------------------------------------------//
    // export solutions on the disk
    //------------------------------------------------//
    if ((mout+1)*dtout<time)
    {
        mout +=1;
        data_dump( exportDir, prefix, pressure, time, mout );
    }

    //------------------------------------------------//
    // go to the next time step
    //------------------------------------------------//
    pressure0 = pressure1;
    pressure1 = pressure;
    }
    MPI_Finalize();
    return 0;
    }


    void initial_conditions( FunctionSpace::Element & pressure0, FunctionSpace::Element & pressure1 )
    {
    double xcenter = 0.5;//0.25;//0.5
    double ycenter = 0.5;//0.25;//0.5;
    double sigma = 0.01;
    double ampli = 1.0;
    double x0 = xcenter;// + C*time;
    double y0 = ycenter;// + C*time;

    FunctionSpace const& space = pressure0.functionSpace();
    for (int i=0;i<space.nLocalDofByDirection( 0 );++i )
    {
        for (int j=0;j<space.nLocalDofByDirection( 1 );++j )
        {
            Point pt = space.localDofToPoint(i,j);
            double r2 = std::pow(pt.x()-x0, 2) + std::pow(pt.y()-y0, 2);
            pressure0(i,j) = ampli*exp(-r2/sigma);
            pressure1(i,j) = pressure0(i,j);
        }
    }
}