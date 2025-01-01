#include<iostream>
using namespace std;
int main()
{
    int a;
    double s=0;

    cout<<"Tapez la valeur numero 1 : "<<a;
    cin>>a;
    s=s+a;
    cout<<"Tapez la valeur numero 2 : ";
    cin>>a;
    s=s+a;
    cout<<"Tapez la valeur numero 3 : ";
    cin>>a;
    s=s+a;
    cout<<"Tapez la valeur numero 4 : ";
    cin>>a;
    s=s+a;
    cout<<"Tapez la valeur numero 5 : ";
    cin>>a;
    s=s+a;

    s=s/5.0;
    cout<<"La moyenne vaut : "<<s<<endl;

    cout << "Appuyez sur une touche pour continuer ..." << endl;
    cin.ignore();
    cin.get();

    return EXIT_SUCCESS;
}
