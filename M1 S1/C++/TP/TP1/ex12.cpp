#include <iostream>

int somme( int tab[], int size )
{
  int res = 0;
  for ( int k=0;k<size;++k )
    res+=tab[k];
  return res;
}
double somme( double tab[], int size )
{
  double res = 0;
  for ( int k=0;k<size;++k )
    res+=tab[k];
  return res;
}

int main()
{
  int nVal = 12;
  std::cout << "Entrer le nombre d'élément : "; std::cin >> nVal;

  int * itab = new int[nVal];
  double * dtab = new double[nVal];
  for ( int k=0;k<nVal;++k )
  {
    itab[k]=k+1;
    dtab[k]=k+1;
  }
  std::cout<<"n*(n+1)/2="<<nVal*(nVal+1)/2<<"\n";
  std::cout<<"somme itab="<<somme(itab,nVal)<<"\n";
  std::cout<<"somme dtab="<<somme(dtab,nVal)<<"\n";
  return 0;
}
