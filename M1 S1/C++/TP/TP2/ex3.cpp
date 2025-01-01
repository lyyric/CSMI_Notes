
#include <iostream>

class Vecteur
{
public :
  // constructeur
  Vecteur( int size )
    :
    M_size( size ),
    M_vec( new double[size] )
  {
    for ( int k=0;k<M_size;++k )
      M_vec[k] = 0;
  }

  // destructeur
  ~Vecteur() { delete M_vec; }

  // constructeur par copie
  Vecteur( Vecteur const& v )
    :
    M_size( v.M_size ),
    M_vec( new double[M_size] )
  {
    for ( int k=0;k<M_size;++k )
      M_vec[k] = v.M_vec[k];
  }

  // opérateur d'affectation
  Vecteur& operator=( Vecteur const& v )
  {
    if (this != &v )
      {
        if ( M_size != v.M_size )
          {
            std::cout << "invalid affectation\n";
            return *this;
          }
        for ( int k=0;k<M_size;++k )
          M_vec[k] = v.M_vec[k];        
      }
    return *this;
  }

  // accesseur
  int size() const { return M_size; }
  
  void setConstant( double val )
  {
    for ( int k=0;k<M_size;++k )
      M_vec[k] = val;
  }
  void zero()
  {
    this->setConstant( 0 );
  }

  double somme() const
  {
    double res=0;
    for ( int k=0;k<M_size;++k )
      res += M_vec[k];
    return res;
  }

  // opérateurs
  double & operator()(int k) { return M_vec[k]; }
  double const& operator()(int k) const { return M_vec[k]; }
  double & operator[](int k) { return M_vec[k]; }
  double const& operator[](int k) const { return M_vec[k]; }

  Vecteur & operator+=(Vecteur const& v)
  {
    if ( M_size != v.M_size )
      {
        std::cout << "invalid operator+=\n";
        return *this;
      }
    for ( int k=0;k<M_size;++k )
      M_vec[k] += v.M_vec[k];
  }
  Vecteur & operator-=(Vecteur const& v)
  {
    if ( M_size != v.M_size )
      {
        std::cout << "invalid operator+=\n";
        return *this;
      }
    for ( int k=0;k<M_size;++k )
      M_vec[k] -= v.M_vec[k];
  }

  Vecteur operator+(Vecteur const& v)
  {
    Vecteur res( M_size );
    res = *this;
    res += v;
    return res;
  }
  Vecteur operator-(Vecteur const& v)
  {
    Vecteur res( M_size );
    res = *this;
    res -= v;
    return res;
  }
  
private :
  int M_size;
  double * M_vec;
};


// surcharge de l’opérateur de flux de sortie
std::ostream & operator << ( std::ostream & o , Vecteur const& v ) {
  for (int k=0;k<v.size();++k)
    o << v(k) << " ";
  return o;
}

double inner_product( Vecteur const& v1, Vecteur const& v2 )
{
  double res = 0;
  for ( int k=0;k<v1.size();++k )
    res += v1(k)*v2(k);
  return res;
}

int main()
{
  Vecteur v1(10);
  v1.setConstant(6.2);
  std::cout << "v1 : "<< v1 << "\n";
  // test constructeur par copie
  Vecteur v2(v1);
  std::cout << "v2 : "<< v2 << "\n";

  // test opérateur d'affectation
  Vecteur v3a(10);
  v1 = v3a;
  Vecteur v3b(15);
  v1 = v3b;

  // test opérateur () et []
  v1(0)=3.13; v1(1)=2.3;
  std::cout << "v1(0)=" << v1(0) << "\n";
  v1[0]=6.2; v1[1]=3.7;
  std::cout << "v1[0]=" << v1[0] << "\n";

  // test operator+=
  v1.setConstant(3);
  v2.setConstant(2);
  v1+=v2;
  std::cout << "v1 : "<< v1 << "\n";

  // test operator+
  Vecteur v4a = v1+v2;
  std::cout << "v4a : "<< v4a << "\n";
  Vecteur v4b = v1-v2;
  std::cout << "v4b : "<< v4b << "\n";

  // test inner_product
  v1.setConstant(3);
  v2.setConstant(2);
  double ip = inner_product(v1,v2);
  std::cout << "inner_product="<<ip<< " (should be=" << v1.size()*3*2 << ")\n";

  return 0;
}
