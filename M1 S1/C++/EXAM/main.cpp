#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>

template <typename T> 
std::vector<T> buildSuite(T a){
    std::vector<T> u(100);
    int i = 0;
    u[0] = a;
    while (u[i]!=1){
        if(u[i]%2 == 1) u[i+1] = 3*u[i]+1;
        else u[i+1] = u[i] / 2;
        i = i + 1;
    }
    return u;
}

std::vector<T> evalProperties(std::vector<T> u){
    int t = u.size();
    int t_max = 0;
    T max = u[0];
    for(int i = 0; i < t; ++i){
        if (!(max > u[i]))
        {
            max = u[i];
            t_max = i;
        }
        
    }
    std::cout << "temps de vol :" << t << std::endl;
    std::cout << "temps de vol en altitude :" << t_max << std::endl;
    std::cout << "altitude maximale :" << max << std::endl;
    std::vector<T> ttt(3);
    ttt[0] = t;
    ttt[1] = t_max;
    ttt[2] = max;
    return ttt;
}

template <typename T> 
void records_max(T n){
    std::vector<T> max_a(3);
    std::vector<T> max_t(3);
    for(int j = 0; j < 3; ++j){
        max_a[j] = 0;
        max_t[j] = 0;
    }
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < 3; ++j){
            if(!(evalProperties(buildSuite<T>(n))[j] > max_t[j])){
            max_t[j] = evalProperties(buildSuite<T>(n))[j];
            max_a[j] = i;
            }
        }
    }
    std::cout << "record de temps de vol :" << max_t[0] << "pour a = " << max_a[0] << std::endl;
    std::cout << "record de temps de vol en altitude :" << max_t[1] << "pour a = "  << max_a[1] << std::endl;
    std::cout << "record de altitude maximale :" << max_t[2] << "pour a = "  << max_a[2] << std::endl;
}


int main() {
    auto suite15 = buildSuite<int> (27);
    std::cout << suite15.size() << std::endl;
    return 0;
}
