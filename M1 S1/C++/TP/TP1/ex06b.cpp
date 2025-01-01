#include<iostream>
using namespace std;

int main()
{
  int a,n,u,M,amax,nmax;
  cout<<"Tapez la valeur de M : ";cin>>M;
  amax=2;
  nmax=2;

  for(a=3;a<=M;a++)
    {
      n=0;
      u=a;
      while(u!=1)
        {
          if(u%2==0)
            u=u/2;
          else
            u=3*u+1;
          n++;
        }
      if(n>nmax) {
        amax=a;nmax=n;
      }
    }
  cout<<"La valeur de A est :"<<amax<<endl;
  cout<<"La valeur de N correspondante est :"<<nmax<<endl;

  return 0;
}
