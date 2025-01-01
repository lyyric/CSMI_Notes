#include <iostream>
     
class Complexe
{
public :
  // constructeurs
  Complexe() : Complexe(0,0) {}
  Complexe( double re, double im );
  Complexe( double re) : Complexe(re,0) {}

  // accesseurs
  double re() const { return M_re; }
  double im() const { return M_im; }

  // mutateurs
  void setRe( double re ) { M_re = re; }
  void setIm( double im ) { M_im = im; }

  // operateurs
  bool operator==( Complexe const& c ) { return (M_re == c.M_re) && (M_im == c.M_im); }
  bool operator!=( Complexe const& c ) { return !(*this==c); }

  Complexe &operator+=(const Complexe &);
  Complexe &operator-=(const Complexe &);
  Complexe &operator*=(const Complexe &);
  Complexe &operator/=(const Complexe &);
private :
  double M_re,M_im;
};

Complexe::Complexe( double re, double im )
  :
  M_re( re ), M_im( im )
{}

// surcharge de l’opérateur de flux de sortie
std::ostream & operator << ( std::ostream & o , Complexe const& d ) {
  return o << d.re() <<"+"<< d.im()<<"i";
}

Complexe& Complexe::operator+=(const Complexe &c)
{
  M_re += c.M_re;
  M_im += c.M_im;
  return *this;
}

Complexe& Complexe::operator-=(const Complexe &c)
{
  M_re -= c.M_re;
  M_im -= c.M_im;
  return *this;
}

Complexe& Complexe::operator*=(const Complexe &c)
{
  double temp = M_re*c.M_re -M_im*c.M_im;
  M_im = M_re*c.M_im + M_im*c.M_re;
  M_re = temp;
  return *this;
}

Complexe& Complexe::operator/=(const Complexe &c)
{
  double norm = c.M_re*c.M_re + c.M_im*c.M_im;
  double temp = (M_re*c.M_re + M_im*c.M_im) / norm;
  M_im = (-M_re*c.M_im + M_im*c.M_re) / norm;
  M_re = temp;
  return *this;
}

Complexe operator+(const Complexe &c1, const Complexe &c2)
{
  Complexe result = c1;
  return result += c2;
}

Complexe operator-(const Complexe &c1, const Complexe &c2)
{
  Complexe result = c1;
  return result -= c2;
}

Complexe operator*(const Complexe &c1, const Complexe &c2)
{
  Complexe result = c1;
  return result *= c2;
}

Complexe operator/(const Complexe &c1, const Complexe &c2)
{
  Complexe result = c1;
  return result /= c2;
}


int main()
{
  Complexe c1(3.4,2.3);
  std::cout << "c1=" << c1 << "\n";
  Complexe c2(5.3,1.2);
  std::cout << "c2=" << c2 << "\n";
  Complexe c3(3.2,6.4);
  std::cout << "c3=" << c3 << "\n";

  if ( c1 == c2 ) std::cout <<"c1==c2\n";
  if ( c1 != c2 ) std::cout <<"c1!=c2\n";
  if ( c1 == c3 ) std::cout <<"c1==c3\n";
  if ( c1 != c3 ) std::cout <<"c1!=c3\n";

  c1+=c2;
  std::cout << "c1+=c2 : " << c1 << "\n";
  c1*=c2;
  std::cout << "c1*=c2 : " << c1 << "\n";
  
  Complexe e1(0,1); 
  std::cout << "e1*e1=" << e1*e1 << "\n";

  Complexe f1 = 2+c1;
  std::cout << "f1=" << f1 << "\n";
  Complexe f2 = c1+2;
  std::cout << "f2=" << f2 << "\n";
  
  return 0;
}
