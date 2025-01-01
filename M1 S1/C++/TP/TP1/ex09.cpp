#include<iostream>
using namespace std;
#include<cmath>

double distance(double xa, double ya, double xb, double yb)
{
  double dx,dy;
  dx=xa-xb;
  dy=ya-yb;
  return sqrt(dx*dx+dy*dy);
}

int main()
{
  double x1,y1,x2,y2,d;

  cout<<"Tapez l'abscisse de A : ";cin>>x1;
  cout<<"Tapez l'ordonnée de A : ";cin>>y1;
  cout<<"Tapez l'abscisse de B : ";cin>>x2;
  cout<<"Tapez l'ordonnée de B : ";cin>>y2;

  d=distance(x1,y1,x2,y2);

  cout<<"La distance AB vaut : "<<d<<endl;
  return 0;
}
