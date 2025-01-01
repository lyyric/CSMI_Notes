#include<iostream>
using namespace std;

const int N=10;

int main()
{
  int t[N],i,j,V;
  bool trouve;
  for(i=0;i<N;i++) {
    cout<<"Tapez un entier ";cin>>t[i];
  }
  cout<<"Tapez la valeur de V : ";cin>>V;

  trouve=false;
  i=0;
  while(!trouve && i<N)
    {
      if(t[i]==V)
        trouve=true;
      else
        i++;
    }

  if(trouve)
    {
      for(j=i;j<N-1;j++)
        t[j]=t[j+1];
      t[N-1]=0;
    }
  for(i=0;i<N;i++)
    cout<<t[i]<<endl;

  return 0;
}
