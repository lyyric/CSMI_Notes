#include<iostream>
using namespace std;

long int factorielle(long int n)
{
  if ( n>1 )
    return n*factorielle(n-1);
  else
    return 1;
}

int main()
{
  int n;
  cout<<"Tapez un entier > 0 :";cin>>n;
  long int res = factorielle(n);
  cout<< n << "! vaut : "<<res<<endl;

  return 0;
}
