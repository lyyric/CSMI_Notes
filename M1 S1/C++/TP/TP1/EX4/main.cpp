#include <iostream>

int main() {
    for (int i = 8; i < 24; i++ ) {
        std::cout << i <<" ";
    }
    std::cout << std::endl;
    for (int i = 8; i < 24; i++ ) {
        if (i%2 == 0) std::cout << i <<" ";
    }
    std::cout << std::endl;
    return 0;
}