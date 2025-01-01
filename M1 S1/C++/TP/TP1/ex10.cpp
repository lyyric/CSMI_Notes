#include<iostream>
using namespace std;

void swap(int &x, int &y)
{
  int temp;
  temp=x;
  x=y;
  y=temp;
}

int main()
{
  int a,b;	
  cout<<"Tapez a :";cin>>a;
  cout<<"Tapez b :";cin>>b;
  swap(a,b);
  cout<<"a vaut : "<<a<<endl;
  cout<<"b vaut : "<<b<<endl;

  return 0;
}
