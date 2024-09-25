#include <iostream>

int main() {
    int a;
    int b;   
    std::cout << "Entrez un entier u(0)= ";
    std::cin >> a;
    int A = a;
    std::cout << "u(0)=" << a << std::endl;
    for (int i=0;a!=1;i++){
        if(a%2==0) a=a/2;
        else a=3*a+1;
        std::cout << "u(" << i+1 << ")=" << a << std::endl;
    }
    int M;
    int N=2;
    std::cout << "Entrez un entier M = ";
    std::cin >> M;
    a = A;
    for (int i=2;i<M+2;i++){
        if(a%2==0) a=a/2;
        else a=3*a+1;
        if (a > A) {
            A = a;
            N = i-1;
        }
        if (a==1) break;
    }
    std::cout << "Maxi u(" << N << ")=" << A << std::endl;
    return 0;
}