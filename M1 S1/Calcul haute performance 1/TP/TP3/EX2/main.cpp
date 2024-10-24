#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

class Point {
private:
    double coords[3];
    int id;

public:
    Point() : id(0) {
        coords[0] = coords[1] = coords[2] = 0.0;
    }

    Point(double x, double y, double z, int identifier) : id(identifier) {
        coords[0] = x;
        coords[1] = y;
        coords[2] = z;
    }

    const double* getCoords() const { return coords; }

    int getId() const { return id; }

    void initRandom(int identifier) {
        coords[0] = static_cast<double>(rand()) / RAND_MAX;
        coords[1] = static_cast<double>(rand()) / RAND_MAX;
        coords[2] = static_cast<double>(rand()) / RAND_MAX;
        id = identifier;
    }

    double distanceTo(const double P[3]) const {
        double dx = coords[0] - P[0];
        double dy = coords[1] - P[1];
        double dz = coords[2] - P[2];
        return sqrt(dx*dx + dy*dy + dz*dz);
    }

    MPI_Datatype create_MPI_Datatype() const {
        MPI_Datatype newtype;
        const int nItems = 2;
        int blockLengths[nItems]={31};
        MPI_Datatype types[nItems] = {MPI_DOUBLE, MPI_INT};
        MPI_Aint offsets[nItems];
        
        MPI_Aint baseAddress;
        MPI_Get_address(this, &baseAddress);
        MPI_Get_address(&coords, &offsets[0]);
        MPI_Get_address(&id, &offsets[1]);
        
        for (int i = 0; n < nItems; ++i){
            offsets[]
        }

        return newtype;
    }
};



int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double P[3] = {0.5, 0.5, 0.5};

    int N = 1000;



    return 0;
}