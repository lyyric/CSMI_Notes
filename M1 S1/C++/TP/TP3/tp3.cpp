#include <iostream>
#include "randomnumber.hpp"
class Vehicule
{
public :

    Vehicule( int nPlaceTotal, double poidsVehicule, int nPlaceOccupe = 0 )
        :
        M_nPlaceTotal( nPlaceTotal ),
        M_nPlaceOccupe( nPlaceOccupe ),
        M_poidsVehicule( poidsVehicule )
    {}
    virtual ~Vehicule() {}

    void setNombreDePlaceOccupe( int n )
    {
        if ( n <= M_nPlaceTotal )
            M_nPlaceOccupe = n;
        else
            std::cout << "error le NombreDePlaceOccupe est trop grand\n" ;
    }
    int nombreDePlaceRestante() const { return M_nPlaceTotal - M_nPlaceOccupe; }
    int nPlaceTotal() const { return M_nPlaceTotal; }
    double poidsTotal() const { return M_poidsVehicule +  M_nPlaceOccupe*75; }
private :
    int M_nPlaceTotal, M_nPlaceOccupe;
    double M_poidsVehicule;
};

class Voiture : public Vehicule
{
public :
    Voiture( int nPlaceOccupe = 0 ) : Vehicule( 5,1000,nPlaceOccupe ) {}
};

class Moto : public Vehicule
{
public :
    Moto( int nPlaceOccupe = 0 ) : Vehicule( 2,500,nPlaceOccupe ) {}
};

class Camion : public Vehicule
{
public :
    Camion( int nPlaceOccupe = 0 ) : Vehicule( 3,4000,nPlaceOccupe ) {}
};

class Bus : public Vehicule
{
public :
    Bus( int nPlaceOccupe = 0 ) : Vehicule( 40,5000,nPlaceOccupe ) {}
};



double poidsTotal( Vehicule** tab, int n )
{
    double res = 0;
    for (int k=0;k<n;++k )
        res += tab[k]->poidsTotal();
    return res;
}

int nombreDePlaceRestante( Vehicule** tab, int n )
{
    int res = 0;
    for (int k=0;k<n;++k )
        res += tab[k]->nombreDePlaceRestante();
    return res;
}

int nombreDeVoiture( Vehicule** tab, int n )
{
    int res = 0;
    for ( int k=0;k<n;++k )
    {
        if ( dynamic_cast<Voiture*>( tab[k] ) )
            ++res;
    }
    return res;
}


int main(int argc, char** argv)
{
    // test voiture
    Voiture v1;
    v1.setNombreDePlaceOccupe(3);
    std::cout << "[voiture] nombreDePlaceRestante : " << v1.nombreDePlaceRestante()
              << " poids total : " << v1.poidsTotal() << "\n";
    // test bus
    Bus b1;
    b1.setNombreDePlaceOccupe(35);
    std::cout << "[bus] nombreDePlaceRestante : " << b1.nombreDePlaceRestante()
              << " poids total : " << b1.poidsTotal() << "\n";

    int nVehicules = 10;
    if ( argc > 1 )
        nVehicules = std::stoi( argv[1] );
    Vehicule** tab = new Vehicule*[nVehicules];
    RandomNumber<int> rnType(0,3);
    RandomNumber<int> rnOcc(0,100);
    for (int k=0;k<nVehicules;++k)
    {
        int type = rnType();
        switch( type )
        {
        default:
        case 0: tab[k] = new Voiture(); break;
        case 1: tab[k] = new Moto(); break;
        case 2: tab[k] = new Camion(); break;
        case 3: tab[k] = new Bus(); break;
        }
        tab[k]->setNombreDePlaceOccupe( rnOcc() % tab[k]->nPlaceTotal() );
    }
    double poids = poidsTotal(tab,nVehicules);
    std::cout << "[tab] poids total des vehicules : " << poids << "\n";

    int nPlaceRestante =nombreDePlaceRestante(tab,nVehicules);
    std::cout << "[tab] nPlaceRestante : " << nPlaceRestante << "\n";

    int nVoiture = nombreDeVoiture(tab,nVehicules);
    std::cout << "[tab] nVoiture : " << nVoiture << "\n";

    return 0;
}
