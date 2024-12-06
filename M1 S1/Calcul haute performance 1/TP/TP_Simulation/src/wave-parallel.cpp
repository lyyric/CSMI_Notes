#include <grid_utilities.hpp>

#include <mpi.h>
#include <cmath>

#include <vector>

//! \brief calcul du champ de pression initial
//! \param[out] pressure0 champ de pression au temps time
//! \param[out] pressure1 champ de pression au temps time+dt
void initial_conditions( FunctionSpace::Element & pressure0, FunctionSpace::Element & pressure1 );

int main(int argc, char *argv[]) {
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
    double time = 0.0;    // temps courant de la simulation
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
        // compute solution at current time
        //------------------------------------------------//
        for (int i = 1; i < space.nLocalDofByDirection(0) - 1; ++i) {
            for (int j = 1; j < space.nLocalDofByDirection(1) - 1; ++j) {
                pressure(i, j) = -pressure0(i, j)
                                + 2.0 * (1 - beta_x * beta_x - beta_y * beta_y) * pressure1(i, j)
                                + beta_x * beta_x * (pressure1(i - 1, j) + pressure1(i + 1, j))
                                + beta_y * beta_y * (pressure1(i, j - 1) + pressure1(i, j + 1));
            }
        }

        //------------------------------------------------//
        // apply periodic boundary conditions
        //------------------------------------------------//
        for (int j = 1; j < space.nLocalDofByDirection(1) - 1; ++j) {
            pressure(0, j) = pressure1(space.nLocalDofByDirection(0) - 2, j);
            pressure(space.nLocalDofByDirection(0) - 1, j) = pressure1(1, j);
        }

        for (int i = 1; i < space.nLocalDofByDirection(0) - 1; ++i) {
            pressure(i, 0) = pressure1(i, space.nLocalDofByDirection(1) - 2);
            pressure(i, space.nLocalDofByDirection(1) - 1) = pressure1(i, 1);
        }

        //------------------------------------------------//
        // MPI communication for ghost cells
        //------------------------------------------------//

        int left_neighbor = grid.neighbor(0, -1);
        int right_neighbor = grid.neighbor(0, 1);
        int bottom_neighbor = grid.neighbor(1, -1);
        int top_neighbor = grid.neighbor(1, 1);

        MPI_Status status;

        if (left_neighbor != MPI_PROC_NULL) {
            MPI_Sendrecv(
                &pressure1(1, 0), space.nLocalDofByDirection(1), MPI_DOUBLE, left_neighbor, 0,
                &pressure1(0, 0), space.nLocalDofByDirection(1), MPI_DOUBLE, left_neighbor, 0,
                MPI_COMM_WORLD, &status
            );
        }

        if (right_neighbor != MPI_PROC_NULL) {
            MPI_Sendrecv(
                &pressure1(space.nLocalDofByDirection(0) - 2, 0), space.nLocalDofByDirection(1), MPI_DOUBLE, right_neighbor, 0,
                &pressure1(space.nLocalDofByDirection(0) - 1, 0), space.nLocalDofByDirection(1), MPI_DOUBLE, right_neighbor, 0,
                MPI_COMM_WORLD, &status
            );
        }

        if (bottom_neighbor != MPI_PROC_NULL) {
            MPI_Sendrecv(
                &pressure1(0, 1), space.nLocalDofByDirection(0), MPI_DOUBLE, bottom_neighbor, 0,
                &pressure1(0, 0), space.nLocalDofByDirection(0), MPI_DOUBLE, bottom_neighbor, 0,
                MPI_COMM_WORLD, &status
            );
        }

        if (top_neighbor != MPI_PROC_NULL) {
            MPI_Sendrecv(
                &pressure1(0, space.nLocalDofByDirection(1) - 2), space.nLocalDofByDirection(0), MPI_DOUBLE, top_neighbor, 0,
                &pressure1(0, space.nLocalDofByDirection(1) - 1), space.nLocalDofByDirection(0), MPI_DOUBLE, top_neighbor, 0,
                MPI_COMM_WORLD, &status
            );
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