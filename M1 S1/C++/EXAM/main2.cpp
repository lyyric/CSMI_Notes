#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>



int buildSuite(int a){
    std::vector<int> u;
    int i = 0;
    u[0] = a;
    while (u[i]!=1){
        if(u[i]%2 == 1) u[i+1] = 3*u[i]+1;
        else u[i+1] = u[i] / 2;
        i = i + 1;
    }
    return i;
}

int main() {
    int suite15 = buildSuite(15);
    std::cout << suite15 << std::endl;
    return 0;
}
