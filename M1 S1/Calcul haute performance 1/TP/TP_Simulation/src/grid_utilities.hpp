#include <string>
#include <iostream>
#include <assert.h>
#include <mpi.h>
#include <hdf5.h>
#include <sys/stat.h>
#include <fstream>
#include <cstring>
#include <sstream>
#include <iomanip>

#define FAIL -1


enum class NeighbourPosition { Bottom=0, Right, Top, Left };

class Point
{
public :
    Point( double x=0, double y=0 )
    {
        M_coord[0] = x;
        M_coord[1] = y;
    }

    double x() const { return M_coord[0]; }
    double y() const { return M_coord[1]; }

private :
    double M_coord[2];
};

class GridStructured
{
public :
    GridStructured( double lx, double ly, int nx, int ny, int worldSize, int rank )
        :
        M_mpiRank( rank )
    {
        M_nPoints[0] = nx;
        M_nPoints[1] = ny;
        M_lenght[0] = lx;
        M_lenght[1] = ly;

        for (int d=0;d<2;++d )
            {
                M_nCells[d] = M_nPoints[d]-1;
                M_delta[d] = M_lenght[d]/(M_nCells[d]);
            }

        for (int i=0;i<4;++i)
            M_neighbour[i] = -1;

        for (int d=0;d<2;++d )
            {
                M_nCellsByProc[d] = new int[worldSize];
                M_startCellId[d] = new int[worldSize];
                M_nPointsByProc[d] = new int[worldSize];
                M_startPointId[d] = new int[worldSize];
            }

        // defined a row partitioning
        if ( true )
            {
                int d1 = 0;
                int d2 = 1;
                for ( int p=0;p<worldSize;++p )
                    {
                        M_nCellsByProc[d1][p] = M_nCells[d1];
                        M_startCellId[d1][p] = 0;
                    }

                // compute number of element in each processus
                int nSameEltByProc = M_nCells[d2]/worldSize;
                int nAdditionalElt = M_nCells[d2]%worldSize;
                for ( int p=0;p<worldSize;++p )
                    M_nCellsByProc[d2][p] = nSameEltByProc;
                for ( int k=0;k<nAdditionalElt;++k )
                    M_nCellsByProc[d2][k] += 1;
                M_startCellId[d2][0] = 0;
                for ( int p=1;p<worldSize;++p )
                    M_startCellId[d2][p] = M_startCellId[d2][p-1] + M_nCellsByProc[d2][p-1];


                for (int d=0;d<2;++d )
                    {
                        for ( int p=0;p<worldSize;++p )
                            {
                                M_startPointId[d][p] = M_startCellId[d][p];
                                M_nPointsByProc[d][p] = M_nCellsByProc[d][p]+1;
                            }
                    }

                if ( rank > 0 )
                    this->setNeighbour( NeighbourPosition::Bottom, rank-1);
                else
                    this->setNeighbour( NeighbourPosition::Bottom, worldSize-1);
                if ( rank < (worldSize-1) )
                    this->setNeighbour( NeighbourPosition::Top, rank+1);
                else
                    this->setNeighbour( NeighbourPosition::Top, 0);

                this->setNeighbour( NeighbourPosition::Left, rank);
                this->setNeighbour( NeighbourPosition::Right, rank);

            }
    }


    // return the rank of the current processus
    int mpiRank() const { return M_mpiRank; }

    // return the number of point in the direction d in the whole grid
    int nPoints( int d ) const { return M_nPoints[d]; }
    // return the number of point in the direction d in the sub grid associated to the processus rank
    int nPoints( int d, int rank ) const { return M_nPointsByProc[d][rank]; }
    // return the number of point in the direction d in the sub grid associated to the current processus
    int nLocalPoints( int d ) const { return this->nPoints( d, M_mpiRank ); }
    // return the first point index in the direction d in the sub grid associated to the processus rank
    int startPointId( int d, int rank ) const { return M_startPointId[d][rank]; }
    // return the first point index in the direction d in the sub grid associated to the current processus
    int startPointId( int d ) const { return this->startPointId(d,M_mpiRank); }

    // return the number of cell in the direction d in the whole grid
    int nCells( int d ) const { return M_nCells[d]; }
    // return the number of cell in the direction d in the sub grid associated to the processus rank
    int nCells( int d, int rank ) const { return M_nCellsByProc[d][rank]; }
    // return the number of cell in the direction d in the sub grid associated to the current processus
    int nLocalCells( int d ) const { return this->nCells(d,M_mpiRank); }
    // return the first cell index in the direction d in the sub grid associated to the processus rank
    int startCellId( int d, int rank ) const { return M_startCellId[d][rank]; }
    // return the first cell index in the direction d in the sub grid associated to the current processus
    int startCellId( int d ) const { return this->startCellId(d,M_mpiRank); }

    // return the mesh size in the direction d (Delta x, Delta y)
    double delta( int d ) const { return M_delta[d]; }

    // return the point at index i,j
    Point point( int i, int j ) const { return Point( 0.0 + M_delta[0]*i,
                                                                                                        0.0 + M_delta[1]*j ); }
    // return the rank of a neighbour at the given position
    int neighbour( NeighbourPosition pos ) const { return M_neighbour[(int)pos]; }
    // return true if a neigbour is defined at the given position
    bool hasNeighbour( NeighbourPosition pos ) const { return (neighbour( pos ) >= 0); }
private :
    void setNeighbour( NeighbourPosition pos, int p ) { if ( p < 0 || p > 5 ) std::cout << "["<< M_mpiRank <<"]aieie : " << p << " with "    << (int)pos << "\n";M_neighbour[(int)pos] = p; }

private :
    int M_mpiRank;
    double M_lenght[2];
    int M_nPoints[2];
    int * M_nPointsByProc[2];
    int * M_startPointId[2];
    int M_nCells[2];
    int * M_nCellsByProc[2];
    int * M_startCellId[2];
    double M_delta[2];
    int M_neighbour[4];
};

class FunctionSpace
{
public :
    FunctionSpace (GridStructured const& grid )
        :
        M_grid(grid)
    {
        M_nLocalDofWithoutGhost = grid.nLocalPoints(0)*grid.nLocalPoints(1);
        M_nLocalDofByDirection[0] = grid.nLocalPoints(0);
        M_nLocalDofByDirection[1] = grid.nLocalPoints(1);
        M_nLocalGhost = 0;
        if ( grid.hasNeighbour( NeighbourPosition::Left ) )
            ++M_nLocalDofByDirection[0];
        if ( grid.hasNeighbour( NeighbourPosition::Right ) )
                ++M_nLocalDofByDirection[0];
        if ( grid.hasNeighbour( NeighbourPosition::Top ) )
                ++M_nLocalDofByDirection[1];
        if ( grid.hasNeighbour( NeighbourPosition::Bottom ) )
                ++M_nLocalDofByDirection[1];

        M_nLocalGhost = M_nLocalDofByDirection[0]*M_nLocalDofByDirection[1] - M_nLocalDofWithoutGhost;
    }

    // return the gird
    GridStructured const& grid() const { return M_grid; }

    // number of degree of freedom inside the current processus (including ghosts)
    int nLocalDofWithGhost() const { return M_nLocalDofWithoutGhost + M_nLocalGhost; }

    // number of degree of freedom by direction inside the current processus (including ghosts)
    int nLocalDofByDirection( int d ) const { return M_nLocalDofByDirection[d]; }

    // return the point in the grid associated to the degree of freedom (i,j)
    Point localDofToPoint( int i,int j ) const
    {
        int pointId[2];

        pointId[0] = this->grid().startPointId(0) + i;
        if ( this->grid().hasNeighbour( NeighbourPosition::Left ) )
            --pointId[0];
        pointId[1] = this->grid().startPointId(1) + j;
        if ( this->grid().hasNeighbour( NeighbourPosition::Bottom ) )
            --pointId[1];

        return this->grid().point( pointId[0],pointId[1] );
    }

    class Element
    {
    public :
        Element( FunctionSpace const& functionSpace )
            :
            M_functionSpace( functionSpace ),
            M_data( new double[functionSpace.nLocalDofWithGhost()] )
        {
            for (int k=0;k<functionSpace.nLocalDofWithGhost();++k)
                M_data[k] = 0;
        }
        ~Element() { delete [] M_data; }

        Element& operator=( Element const& u )
        {
            for (int k=0;k<this->functionSpace().nLocalDofWithGhost();++k)
                M_data[k] = u.data()[k];
            return *this;
        }

        // return the function space
        FunctionSpace const& functionSpace() const { return M_functionSpace; }

        // return the value at degree of freedom (i,j)
        double & operator()(int i,int j) { return M_data[j*M_functionSpace.nLocalDofByDirection(0)+i]; }
        // return the value at degree of freedom (i,j)
        double operator()(int i,int j) const { return M_data[j*M_functionSpace.nLocalDofByDirection(0)+i]; }

        // return the array of values
        double * data() const { return M_data; }

    private :
        FunctionSpace const& M_functionSpace;
        double * M_data;
    };


    Element element() const { return Element(*this); }

private :
    GridStructured const& M_grid;
    int M_nLocalDofByDirection[2];
    int M_nLocalDofWithoutGhost, M_nLocalGhost;
};



//! \brief écriture des fichiers de sortie en format HDF5
//! \param[in] exportDir repertoire des exports xdmf/hdf5
//! \param[in] prefix prefix des fichiers ecrits
//! \param[in] pressureField champ de pression au temps time
//! \param[in] time temps de la simulation
//! \param[in] m numéro de l'itération d'écriture des fichiers
void data_dump( std::string const& exportDir, std::string const& prefix, FunctionSpace::Element const& pressureField, double time, int m )
{

    // create directory at first export
    if ( m == 0 )
        mkdir( exportDir.c_str(),ACCESSPERMS);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;

    GridStructured const& grid = pressureField.functionSpace().grid();
    double * pressure_tab = pressureField.data();

    int world_rank = grid.mpiRank();

    std::ostringstream osIndex;
    osIndex<<std::setfill('0')<<std::setw(4)<<m;
    std::string xmfname= prefix + osIndex.str() + ".xmf";
    std::string h5name= prefix + osIndex.str() + ".h5";
    std::ostringstream osIndexInitial;
    osIndexInitial<<std::setfill('0')<<std::setw(4)<<0;
    std::string h5nameInitial= prefix + osIndexInitial.str() + ".h5";


    /* ------------------------------------------------
     * écriture des fichiers xdmf (utiles pour de la visualisation
     * avec Paraview)
        -------------------------------------------------*/
    if (world_rank==0){
        std::string xmfpath = std::string(exportDir) + "/" + std::string(xmfname);
        std::ofstream ofs;
        ofs.open ( xmfpath.c_str()/*xmfname*/, std::ofstream::out);

        ofs << "<?xml version=\"1.0\" ?> "    << std::endl;
        ofs << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []> " << std::endl;
        //ofs << "<Xdmf Version=\"2.0\" xmlns:xi=\"[http://www.w3.org/2001/XInclude]\"> " << std::endl;
        ofs << "<Xdmf Version=\"2.0\"> " << std::endl;
        ofs << " <Domain> " << std::endl;
        ofs << "    <Grid Name=\"TheMesh\" GridType=\"Uniform\"> " << std::endl;
        ofs << "     <Time Value=\""<< time << "\"/> " << std::endl;
        ofs << "     <Topology TopologyType=\"2DCORECTMesh\" Dimensions=\""<< grid.nPoints(1) <<" "<< grid.nPoints(0) <<"\" /> " << std::endl;
        ofs << "     <Geometry GeometryType=\"ORIGIN_DXDY\" > " << std::endl;
        ofs << "        <DataItem Name =\"Origin\" Dimensions=\"2\" NumberType=\"Float\" Format=\"XML\"> " << std::endl;
        ofs << "         0 0 " << std::endl;
        ofs << "        </DataItem> " << std::endl;
        ofs << "        <DataItem Name =\"Spacing\" Dimensions=\"2\" NumberType=\"Float\" Format=\"XML\"> " << std::endl;
        ofs << "         "<< grid.delta(1) <<" "<< grid.delta(0) <<" " << std::endl;
        ofs << "        </DataItem> " << std::endl;
        ofs << "     </Geometry> " << std::endl;
        ofs << "     <Attribute Name=\"pressure\" Center=\"Node\" > " << std::endl;
        ofs << "        <DataItem Format=\"hdf5\" NumberType=\"Float\" Dimensions=\"" << grid.nPoints(1) << " " << grid.nPoints(0) << " \"> " << std::endl;
        ofs << "         " << h5name << "://pressure" << std::endl;
        ofs << "        </DataItem> " << std::endl;
        ofs << "     </Attribute> " << std::endl;
        ofs << "     <Attribute Name=\"pid\" Center=\"Cell\" > " << std::endl;
        ofs << "        <DataItem Format=\"hdf5\" NumberType=\"Float\" Dimensions=\"" << grid.nCells(1) << " " << grid.nCells(0) << " \"> " << std::endl;
        ofs << "         " << h5nameInitial << "://pid" << std::endl;
        ofs << "        </DataItem> " << std::endl;
        ofs << "     </Attribute> " << std::endl;
        ofs << "    </Grid> "<< std::endl;
        ofs << " </Domain> "<< std::endl;
        ofs << "</Xdmf> "<< std::endl;

        ofs.close();
    }


    /* ------------------------------------------------
     * écriture des fichiers hdf5
        -------------------------------------------------*/

    hid_t fid1;			/* HDF5 file IDs */
    hid_t acc_tpl1;		/* File access templates */
    hid_t sid1;     		/* Dataspace ID */
    hid_t file_dataspace;	/* File dataspace ID */
    hid_t mem_dataspace;	/* memory dataspace ID */
    hid_t dataset1, dataset2;	/* Dataset ID */
    hsize_t sizex = static_cast <hsize_t> (grid.nPoints(1));
    hsize_t sizey = static_cast <hsize_t> (grid.nPoints(0));
    hsize_t dims1[2] = { sizex,    sizey};	 /* dataspace dim sizes */

    hsize_t start[2];			/* for hyperslab setting */
    hsize_t count[2], stride[2], block[2];	/* for hyperslab setting */

    herr_t ret;                 	/* Generic return value */

    /* setup file access template with parallel IO access. */
    acc_tpl1 = H5Pcreate (H5P_FILE_ACCESS);
    assert(acc_tpl1 != FAIL);
    // std::cout << "H5Pcreate access succeed" << std::endl;

    /* set Parallel access with communicator */
    ret = H5Pset_fapl_mpio(acc_tpl1, comm, info);
    assert(ret != FAIL);

    /* create the file collectively */
    std::string h5path = std::string(exportDir) + "/" + std::string(h5name);
    fid1 = H5Fcreate(h5path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_tpl1);
    assert(fid1 != FAIL);

    /* Release file-access template */
    ret = H5Pclose(acc_tpl1);
    assert(ret != FAIL);

    /*    define the dimensions of the overall datasets
     * and the slabs local to the MPI process.
     */
    /* setup dimensionality object */
    sid1 = H5Screate_simple(2, dims1, NULL);
    assert (sid1 != FAIL);

    /* create a dataset collectively */
    dataset1 = H5Dcreate2(fid1, "pressure", H5T_NATIVE_DOUBLE, sid1,
                                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dataset1 != FAIL);

    /* set up dimensions of the slab this process accesses */
    start[0] = grid.startPointId( 1, world_rank );
    start[1] = grid.startPointId( 0, world_rank );
    count[0] = 1;
    count[1] = 1;
    stride[0] = 1;
    stride[1] = 1;
    block[0] = grid.nPoints(1,world_rank);
    block[1] = grid.nPoints(0,world_rank);

    // create a file dataspace independently
    file_dataspace = H5Dget_space (dataset1);
    assert(file_dataspace != FAIL);

    ret = H5Sselect_hyperslab( file_dataspace, H5S_SELECT_SET, start, stride,
                                                         count, block/*NULL*/);
    assert(ret != FAIL);

    // create a memory dataspace independently
    block[0] = pressureField.functionSpace().nLocalDofByDirection( 1 );
    block[1] = pressureField.functionSpace().nLocalDofByDirection( 0 );
    mem_dataspace = H5Screate_simple (2, block/*count*/, NULL);
    assert (mem_dataspace != FAIL);

    // warning, need to shift is has ghosts
    start[0] = 0;
    start[1] = 0;
    if ( grid.hasNeighbour( NeighbourPosition::Bottom ) )
        start[0] = 1;
    if ( grid.hasNeighbour( NeighbourPosition::Left ) )
        start[1] = 1;
    block[0] = grid.nPoints(1,world_rank);
    block[1] = grid.nPoints(0,world_rank);

    ret = H5Sselect_hyperslab( mem_dataspace, H5S_SELECT_SET, start, stride,
                                                         count, block/*NULL*/);

    /* write data independently */
    ret = H5Dwrite(dataset1, H5T_NATIVE_DOUBLE, mem_dataspace, file_dataspace,
                                 H5P_DEFAULT, pressure_tab);
    assert(ret != FAIL);

    /* release dataspace ID */
    H5Sclose(file_dataspace);

    /* close dataset collectively */
    ret=H5Dclose(dataset1);
    assert(ret != FAIL);

    /* release all IDs created */
    H5Sclose(sid1);

    // write the pid
    if ( m == 0 )
        {
            dims1[0] = grid.nCells(1);
            dims1[1] = grid.nCells(0);
            hid_t sid_pid = H5Screate_simple(2, dims1, NULL);
            assert (sid_pid != FAIL);
            // create a dataset collectively
            hid_t dataset_pid = H5Dcreate2(fid1, "pid", H5T_NATIVE_DOUBLE, sid_pid,
                                                                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            assert(dataset_pid != FAIL);
            // set up dimensions of the slab this process accesses
            start[0] = grid.startCellId( 1, world_rank );
            start[1] = grid.startCellId( 0, world_rank );
            count[0] = 1;
            count[1] = 1;
            stride[0] = 1;
            stride[1] = 1;
            block[0] = grid.nCells(1,world_rank);
            block[1] = grid.nCells(0,world_rank);

            /* create a file dataspace independently */
            hid_t file_dataspace_pid = H5Dget_space(dataset_pid);
            assert(file_dataspace_pid != FAIL);

            ret = H5Sselect_hyperslab( file_dataspace_pid, H5S_SELECT_SET, start, stride,
                                                                 count, block/*NULL*/);
            assert(ret != FAIL);


            hid_t mem_dataspace_pid = H5Screate_simple (2, block/*count*/, NULL);
            assert (mem_dataspace_pid != FAIL);

            double* pid_tab = new double[block[0]*block[1]];
            for (int k=0;k<block[0]*block[1];++k)
                pid_tab[k] = world_rank;

            /* write data independently */
            ret = H5Dwrite(dataset_pid, H5T_NATIVE_DOUBLE, mem_dataspace_pid, file_dataspace_pid,
                                         H5P_DEFAULT, pid_tab);
            assert(ret != FAIL);

            /* release dataspace ID */
            H5Sclose(file_dataspace_pid);

            /* close dataset collectively */
            ret=H5Dclose(dataset_pid);
            assert(ret != FAIL);

            /* release all IDs created */
            H5Sclose(sid_pid);

            delete [] pid_tab;
        }

    /* close the file collectively */
    H5Fclose(fid1);
}