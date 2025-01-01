#include <cmath>
#include <iostream>

double * myfunc(double min, double max, int size)
{
  double step = (max-min)/(size-1);
  double * tab = new double[size];
  double x = min;
  for (int k=0;k<size;++k )
    {
      tab[k] = cos(x);
      x+=step;
    }
  return tab;
}

double carre ( double x ) { return x * x ;}
double inverse ( double x ) { return 1/ x ;}
double racine ( double x ) { return sqrt ( x );}

double * myfunc(double min, double max, int size, double (* f )( double ) )
{
  double step = (max-min)/(size-1);
  double * tab = new double[size];
  double x = min;
  for (int k=0;k<size;++k )
    {
      tab[k] = f(x);
      x+=step;
    }
  return tab;
}

int main()
{
  double min=0;
  double max=3.14151629;
  int size=10;
  std :: cout << "Entrer le min : ";
  std :: cin >> min ;
  std :: cout << "Entrer le max : ";
  std :: cin >> max ;
  std :: cout << "Entrer la taille : ";
  std :: cin >> size ;

  double * tab = myfunc(min,max,size);
  for (int k=0;k<size;++k )
    std::cout << "cos()="<<tab[k]<<"\n";
  std::cout <<"-------------\n";

  int type = 2 ;
  std :: cout << "Entrer le type de fonction : 1= carre ,2= inverse ,3= racine, 4=cos :";
  std :: cin >> type ;
  double (* monPointeur )( double );
  switch ( type ){ // on d é fini le pointeur sur la fonction choisie
  case 1: monPointeur = carre ; break ;
  case 2: monPointeur = inverse ; break ;
  case 3: monPointeur = racine ; break ;
  case 4: monPointeur = cos ; break ;
  }
  double * tab2 = myfunc(min,max,size,monPointeur);
  for (int k=0;k<size;++k )
    std::cout << "cos()="<<tab2[k]<<"\n";
  delete [] tab2;
  return 0;
}
         
