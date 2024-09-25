#include <iostream>

int main() {
    int n;
    int a[3]={0, 1, 1};
    std::cout << "Entrez un entier n: ";
    std::cin >> n;
    if (n<0){
        std::cout << "Entrez un entier > 0 " << std::endl;
    } 
    else{
        if (n<3){

        }
        else{
            for (int i = 3; i < n+1; i++){
                a[i%3]=a[(i-1)%3]+a[(i-2)%3];
            }
        }
        std::cout << a[n%3] << std::endl;
    }
    return 0;
}