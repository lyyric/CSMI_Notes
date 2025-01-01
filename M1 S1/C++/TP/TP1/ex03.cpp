#include<iostream>
using namespace std;

int main()
{
    int a,b,temp;

    cout<<"Tapez la valeur de a : ";
    cin>>a;
    cout<<"Tapez la valeur de b : ";
    cin>>b;

    temp=a;
    a=b;
    b=temp;

    cout<<"La valeur de a est "<<a<<endl;
    cout<<"La valeur de b est "<<b<<endl;

    cout << "Appuyez sur une touche pour continuer ..." << endl;
    cin.ignore();
    cin.get();

    return EXIT_SUCCESS;
}
