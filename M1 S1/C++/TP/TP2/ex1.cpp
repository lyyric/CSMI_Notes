
#include <iostream>

/**
 * class Date représentant une date composée du jour/mois/année
 */
class Date    
{    
public:    

  // constructeur
  /* version basic mais sans verification
    Date(int j,int m ,int a)
    :
    M_jour( j ), M_mois( m ), M_annee( m )
  {}
  */
  //version avec verification
  Date(int j,int m ,int a)
    :
    M_jour( 1 ), M_mois( 1 ), M_annee( 1 )
  {
    if ( !estValide(j,m,a) )
      std::cout << "date " << j << "/" << m << "/" << a << " invalide \n";
    else
      {
        M_jour = j; M_mois = m; M_annee = a;
      }
  }

  // accesseurs
  int jour() const { return M_jour; }    
  int mois() const { return M_mois; }    
  int annee() const { return M_annee; }

  //une méthode statique vérifiant la validité des jours et mois
  static bool estValide(int j,int m,int a)
  {
    if ( m > 12 || m < 1 )
      return false;
    if ( j < 1 || j > nJourParMois[m-1] )
      return false;
    return true;
  }

  // mutateurs
  void setJour( int j )
  {
    if ( estValide(j,M_mois,M_annee) )
      M_jour = j;
    else
      std::cout << "erreur : la date " << j << "/" << M_mois << "/" << M_annee << " est incorrect" << std::endl;
  }
  void setMois( int m )
  {
    if ( estValide(M_jour,m,M_annee) )
      M_mois = m;
    else
      std::cout << "erreur : la date " << M_jour << "/" << m << "/" << M_annee << " est incorrect" << std::endl;
  }
  void setAnnee( int a )
  {
    if ( estValide(M_jour,M_mois,a) )
      M_annee = a;
    else
      std::cout << "erreur : la date " << M_jour << "/" << M_mois << "/" << a << " est incorrect" << std::endl;
  }
  
  // une méthode statique convertissant un nombre de jour en objet
  static Date nJourEnDate(int njour)
  {
    // calcul annee
    int a = njour/365;
    int jrest = njour%365;
    if ( jrest == 0 )
      return Date(31,12,a-1);
    // calculs mois
    int cpt = 0;
    int m = 1;
    for ( ; m<=12 ; ++m)
      {
        if ( (cpt+nJourParMois[m-1]) >= jrest )
          break;
        cpt += nJourParMois[m-1];
      }
    // calcul jour
    int j = jrest-cpt;
    return Date(j,m,a);
  }

private:    
  // attributs
  int M_jour, M_mois, M_annee;
  static const int nJourParMois[12];
}; 

const int Date::nJourParMois[12]={ 31,28,31,30,31,30,31,31,30,31,30,31 };

// surcharge de l’opérateur de flux de sortie
std::ostream & operator<< ( std::ostream & o , const Date & d ) {
  return o << d.jour() <<"/"<< d.mois()<<"/"<<d.annee();
}




int main()
{
  Date d1(12,10,2017);

  // test accesseurs
  std::cout << "d1.jour()="<<d1.jour()<<"\n";
  std::cout << "d1.mois()="<<d1.mois()<<"\n";
  std::cout << "d1.annee()="<<d1.annee()<<"\n";

  // test opérateur de flux de sortie
  std::cout << "d1="<<d1<<"\n";

  // test static méthode estValide
  if ( Date::estValide(21,02,2017) && !Date::estValide(31,02,2017) )
    std::cout << "TEST estValide OK\n";
  else
    std::cout << "TEST NON Valide\n";

  // test mutateurs
  d1.setJour( 3 );
  d1.setMois( 8 );
  d1.setAnnee( 2018 );
  std::cout << "d1 up1="<<d1<<"\n";
  d1.setMois( 14 );
  std::cout << "d1 up2="<<d1<<"\n";
    
  // test static méthode nJourEnDate
  Date d2 = Date::nJourEnDate(31);
  std::cout <<  d2 << "\n";
  Date d3 = Date::nJourEnDate(32);
  std::cout <<  d3 << "\n";
  Date d4 = Date::nJourEnDate(31+28);
  std::cout <<  d4 << "\n";
  Date d5 = Date::nJourEnDate(31+29);
  std::cout <<  d5 << "\n";
  Date d6 = Date::nJourEnDate(365);
  std::cout <<  d6 << "\n";
  Date d7 = Date::nJourEnDate(366);
  std::cout <<  d7 << "\n";
  Date d8 = Date::nJourEnDate(365*2017+293);
  std::cout <<  d8 << "\n";
}
