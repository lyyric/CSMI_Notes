#include <grid_utilities.hpp>

#include <mpi.h>
#include <cmath>

//! \brief calcul du champ de pression initial
//! \param[out] pressure0 champ de pression au temps time
//! \param[out] pressure1 champ de pression au temps time+dt
void initial_conditions( FunctionSpace::Element & pressure0, FunctionSpace::Element & pressure1 );

//! \brief calcul du champ de pression dans le domaine (hors conditions aux bords)
//! \param[out] pressure0 champ de pression au temps n-1
//! \param[out] pressure1 champ de pression au temps n
void update_solution( double C, double dt, FunctionSpace::Element const& pressure0, FunctionSpace::Element & pressure1, FunctionSpace::Element & pressure, MPI_Datatype mpiTypeRow, MPI_Datatype mpiTypeCol );


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
  int npx = 1, npy = world_size; // change these parameters for have another type of partitioning
  GridStructured grid( LX, LY, NX, NY, world_size, world_rank, BlockPartioning{npx,npy} );

  //---------------------------------------------------------------//
  // define the function space and approximation (solution of pde)
  //---------------------------------------------------------------//
  FunctionSpace space( grid );
  // pressure at time n-1
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
  // create MPI DataType
  //---------------------------------------------------------------//
  int ndx = space.nLocalDofByDirection( 0 );
  int ndy = space.nLocalDofByDirection( 1 );
  MPI_Datatype mpiTypeRow;
  MPI_Type_contiguous(ndx-2, MPI_DOUBLE ,&mpiTypeRow );
  MPI_Type_commit(&mpiTypeRow );
  MPI_Datatype mpiTypeCol;
  MPI_Type_vector(ndy-2,1,ndx, MPI_DOUBLE, &mpiTypeCol );
  MPI_Type_commit( &mpiTypeCol );

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
    update_solution( C, dt, pressure0, pressure1, pressure, mpiTypeRow, mpiTypeCol );

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


void update_solution( double C, double dt, FunctionSpace::Element const& pressure0, FunctionSpace::Element & pressure1, FunctionSpace::Element & pressure, MPI_Datatype mpiTypeRow, MPI_Datatype mpiTypeCol )
{
  FunctionSpace const& space = pressure.functionSpace();
  GridStructured const& grid = space.grid();

  double betax2 = std::pow(C*dt/grid.delta(0),2);
  double betay2 = std::pow(C*dt/grid.delta(1),2);

  int ndx = space.nLocalDofByDirection( 0 );
  int ndy = space.nLocalDofByDirection( 1 );
  int rankLeft = grid.neighbour( NeighbourPosition::Left );
  int rankRight = grid.neighbour( NeighbourPosition::Right );
  int rankBottom = grid.neighbour( NeighbourPosition::Bottom );
  int rankTop = grid.neighbour( NeighbourPosition::Top );

  MPI_Request reqs[8];
  int cptRequest = 0;
  int tag1=123;
  int tag2=tag1+1;

#if 0
  // VERSION 1 : les comm ne sont pas recouvertes par du calcul
  for (int i=1;i<space.nLocalDofByDirection( 0 )-1;++i )
    for (int j=1;j<space.nLocalDofByDirection( 1 )-1;++j )
      {
        pressure( i,j )= -pressure0(i,j) + 2*(1-betax2-betay2)*pressure1(i,j) + betax2*(pressure1(i-1,j) + pressure1(i+1,j)) + betay2*(pressure1(i,j-1) + pressure1(i,j+1));
      }


  if ( rankLeft == grid.mpiRank() )
    {
      for (int j=1;j<(ndy-1);++j )
        {
          pressure( 0,j ) = pressure( ndx-3,j );
          pressure( ndx-1,j ) = pressure( 2,j );
        }
    }
  else
    {
      // MPI COMM
      // TODO
    }


  if ( rankBottom == grid.mpiRank() )
    {
      for (int i=1;i<(ndx-1);++i )
        {
          pressure( i,0 ) = pressure( i,ndy-3 );
          pressure( i,ndy-1 ) = pressure( i,2 );
        }
    }
  else
    {
      // MPI COMM
      int tag=123;
      // send row to the top
      MPI_Isend( &pressure(1,ndy-3), ndx-2, MPI_DOUBLE, rankTop, tag1, MPI_COMM_WORLD, &reqs[cptRequest++] );
      // recv row from bottom
      MPI_Irecv( &pressure(1,0), ndx-2, MPI_DOUBLE, rankBottom, tag1, MPI_COMM_WORLD, &reqs[cptRequest++] );

      // send row to the bottom
      MPI_Isend( &pressure(1,2), ndx-2, MPI_DOUBLE, rankBottom, tag2, MPI_COMM_WORLD, &reqs[cptRequest++] );
      // recv row from bottom
      MPI_Irecv( &pressure(1,ndy-1), ndx-2, MPI_DOUBLE, rankTop, tag2, MPI_COMM_WORLD, &reqs[cptRequest++] );
    }

  if ( cptRequest > 0 )
    MPI_Waitall(cptRequest, reqs,MPI_STATUSES_IGNORE);


#else

  if ( rankLeft == grid.mpiRank() )
    {
      for (int j=1;j<(ndy-2);++j )
        {
          pressure1( 0,j ) = pressure1( ndx-3,j );
          pressure1( ndx-1,j ) = pressure1( 2,j );
        }
    }
  else
    {
      // MPI COMM
      // send col to the left
      MPI_Isend( &pressure1(2,1), 1, mpiTypeCol, rankLeft, tag1, MPI_COMM_WORLD, &reqs[cptRequest++] );
      // recv row from right
      MPI_Irecv( &pressure1(ndx-1,1), 1, mpiTypeCol, rankRight, tag1, MPI_COMM_WORLD, &reqs[cptRequest++] );

      // send row to the right
      MPI_Isend( &pressure1(ndx-3,1), 1, mpiTypeCol, rankRight, tag2, MPI_COMM_WORLD, &reqs[cptRequest++] );
      // recv row from left
      MPI_Irecv( &pressure1(0,1), 1, mpiTypeCol, rankLeft, tag2, MPI_COMM_WORLD, &reqs[cptRequest++] );
    }


  if ( rankBottom == grid.mpiRank() )
    {
      for (int i=1;i<(ndx-2);++i )
        {
          pressure1( i,0 ) = pressure1( i,ndy-3 );
          pressure1( i,ndy-1 ) = pressure1( i,2 );
        }
    }
  else
    {
      // MPI COMM
      // send row to the top
      MPI_Isend( &pressure1(1,ndy-3), 1, mpiTypeRow, rankTop, tag1, MPI_COMM_WORLD, &reqs[cptRequest++] );
      // recv row from bottom
      MPI_Irecv( &pressure1(1,0), 1, mpiTypeRow, rankBottom, tag1, MPI_COMM_WORLD, &reqs[cptRequest++] );

      // send row to the bottom
      MPI_Isend( &pressure1(1,2), 1, mpiTypeRow, rankBottom, tag2, MPI_COMM_WORLD, &reqs[cptRequest++] );
      // recv row from bottom
      MPI_Irecv( &pressure1(1,ndy-1), 1, mpiTypeRow, rankTop, tag2, MPI_COMM_WORLD, &reqs[cptRequest++] );
    }

  for (int i=2;i<space.nLocalDofByDirection( 0 )-2;++i )
    for (int j=2;j<space.nLocalDofByDirection( 1 )-2;++j )
      {
        pressure( i,j )= -pressure0(i,j) + 2*(1-betax2-betay2)*pressure1(i,j) + betax2*(pressure1(i-1,j) + pressure1(i+1,j)) + betay2*(pressure1(i,j-1) + pressure1(i,j+1));
      }

  if ( cptRequest > 0 )
    MPI_Waitall(cptRequest, reqs,MPI_STATUSES_IGNORE);


  int tab_i[2] = { 1, space.nLocalDofByDirection( 0 )-2 };
  int tab_j[2] = { 1, space.nLocalDofByDirection( 1 )-2 };
  for (int k=0;k<2;++k )
    {
      int i = tab_i[k];
      for (int j=1;j<space.nLocalDofByDirection( 1 )-1;++j )
        {
          pressure( i,j )= -pressure0(i,j) + 2*(1-betax2-betay2)*pressure1(i,j) + betax2*(pressure1(i-1,j) + pressure1(i+1,j)) + betay2*(pressure1(i,j-1) + pressure1(i,j+1));
        }
    }
  for (int i=1;i<space.nLocalDofByDirection( 0 )-1;++i )
    {
      for (int k=0;k<2;++k )
      {
        int j = tab_j[k];
        pressure( i,j )= -pressure0(i,j) + 2*(1-betax2-betay2)*pressure1(i,j) + betax2*(pressure1(i-1,j) + pressure1(i+1,j)) + betay2*(pressure1(i,j-1) + pressure1(i,j+1));
      }
    }

#endif
}