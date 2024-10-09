#include <cmath>
#include "randomnumber.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

class Complexe
{
public :
  Complexe(double re=0,double im=0)
    :
    M_re(re),M_im(im)
  {}
  double re() const { return M_re; }
  double im() const { return M_im; }
  void setRe(double re) { M_re=re; }
  void setIm(double im) { M_im=im; }
  double norm() { return std::sqrt(std::pow(M_re,2)+std::pow(M_im,2)); }
private :
  double M_re,M_im;
};

std::ostream & operator<<( std::ostream & o , Complexe const & c ) {
  return o << "(" << c.re() << "," << c.im()<< ")";
}
std::istream & operator>>( std::istream & i , Complexe & c ) {
  char t ; double im , re ;
  i >> t >> re >> t  >> im >> t;
  c.setRe( re );
  c.setIm( im );
  return i ;
}


int main()
{
  // generation du fichier nombrescomplexes.txt
  if ( false )
    {
      int nComplexe=2000;
      std::ofstream ofile("nombrescomplexes.txt");
      ofile << nComplexe << std::endl;
      RandomNumber < double > rnd (-1000 ,1000);
      double mean = 0;
      for (int i=0;i<nComplexe;++i)
        {
          double re = rnd();
          double im = rnd();
          Complexe c(re,im);
          mean+=c.norm();
          ofile << std::setprecision(10) << c << "\n";
        }
      ofile.close();
      mean/=nComplexe;
      std::cout << std::setprecision(10) << "mean="<<mean<<"\n";
    }

  if ( true )
    {
      // exercice 2
      std::ifstream ifile("nombrescomplexes.txt");
      int nComplexe=0;
      ifile >> nComplexe;
      Complexe * tab = new Complexe[nComplexe];
      std::cout << "nComplexe="<<nComplexe << "\n";
      double mean = 0;
      for (int i=0;i<nComplexe;++i)
        {
          ifile >> tab[i];
          mean+=tab[i].norm();
        }
      ifile.close();
      mean/=nComplexe;
      std::cout << std::setprecision(10) << "mean="<<mean<<"\n";        

      // execrice 3
      std::cout << "entre un seuil minimal : ";
      double seuil;
      std::cin >> seuil;

      std::ostringstream filename_sup; filename_sup << "nombrescomplexes_seuil_" << seuil << "_sup.txt";
      std::ostringstream filename_inf; filename_inf << "nombrescomplexes_seuil_" << seuil << "_inf.txt";
      std::ofstream ofile_sup(filename_sup.str());
      std::ofstream ofile_inf(filename_inf.str());
      for (int i=0;i<nComplexe;++i)
        {
          double norm = tab[i].norm();
          if ( norm >= seuil )
            ofile_sup << std::setprecision(10) << norm << " " << tab[i] << "\n";
          else
            ofile_inf << std::setprecision(10) << norm << " " << tab[i] << "\n";
        }
      ofile_sup.close();
      ofile_inf.close();
    }
  
  return 0;
}
